"""YOLOTracker — detection-based single-object tracker.

Implements the BaseTracker interface. Uses a YOLO11s detector (ONNX or RKNN)
with a Kalman + IoU associator. The state machine drives three modes:

    SEARCHING   — full-frame detection every frame
    TRACKING    — crop around Kalman-predicted position
    LOST        — full-frame detection, bridging with Kalman prediction
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from core.bbox import BBox
from core.frame import Frame
from trackers.base import BaseTracker, TrackResult
from trackers.yolo_detection.associator import SOTAssociator
from trackers.yolo_detection.detector import YOLODetector
from trackers.yolo_detection.postprocess import Detection


class YOLOTracker(BaseTracker):
    """Detection-based single-object tracker using YOLO11s + Kalman associator.

    Configuration is read from ``cfg["tracker"]["yolo"]``.  See the project
    spec for a full description of the state machine and parameter rationale.
    """

    def __init__(self, cfg: dict) -> None:
        # --------------------------------------------------------------
        # Resolve configuration
        # --------------------------------------------------------------
        trk_cfg = cfg.get("tracker", {})
        ycfg   = trk_cfg.get("yolo", {})

        model_path = ycfg.get("model_path", "models/yolo/yolo11s_uav.rknn")
        input_size = tuple(ycfg.get("input_size", [640, 640]))
        conf_thr   = float(ycfg.get("conf_threshold", 0.25))
        device     = cfg.get("inference", {}).get("device", "cpu").strip().lower()

        self._detector = YOLODetector(
            model_path=model_path,
            input_size=input_size,
            conf_threshold=conf_thr,
            device=device,
        )
        self._associator = SOTAssociator(
            max_lost=int(ycfg.get("max_lost", 10)),
            iou_threshold=float(ycfg.get("iou_threshold", 0.3)),
        )
        self._full_frame_every = int(ycfg.get("full_frame_every", 10))
        self._crop_scale       = float(ycfg.get("crop_scale", 3.0))

        # State
        self._state: str       = "searching"   # searching | tracking | lost
        self._roi: tuple[float, float, float, float] | None = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public interface (BaseTracker)
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "yolo_sot"

    def init(self, frame: Frame, bbox: BBox) -> None:
        """Manual lock-on.  Seeds the associator with a synthetic
        high-confidence detection at *bbox* and forces TRACKING state.

        Corresponds to QuadGuide ``lockon/cmd``.
        """
        self._associator.reset()
        synth = Detection(bbox=bbox, confidence=1.0, class_id=0)
        self._associator.update([synth])
        self._state = "tracking"
        self._roi = self._compute_roi(bbox)
        self._frame_count = 0

    def update(self, frame: Frame) -> TrackResult:
        """Per-frame update.  Implements the full state machine:

        1. Kalman predicts → drives crop region (TRACKING only).
        2. Detector runs (full-frame or crop).
        3. Associator matches detections → state transition.
        """
        t0 = time.perf_counter()

        # --- Step 0: decide detection mode --------------------------------
        use_full_frame = (
            self._state in ("searching", "lost")
            or self._frame_count % self._full_frame_every == 0
        )

        # --- Step 1: Kalman predict → drive crop (TRACKING) ---------------
        if not use_full_frame and self._state == "tracking":
            predicted = self._associator.predict()
            if predicted is not None:
                self._roi = self._compute_roi(predicted)

        # --- Step 2: Detector inference -----------------------------------
        if use_full_frame:
            detections = self._detector.detect(frame.image, roi=None)
            # When searching with full-frame, let the associator use all detections
        else:
            detections = self._detector.detect(frame.image, roi=self._roi)
            # In crop mode, detections are already in full-frame coordinates
            # (the detector maps them back). We trust these detections
            # implicitly because the crop already constrains the search area.

        # --- Step 3: Associator match → state transition ------------------
        bbox = self._associator.update(detections)

        if bbox is None:
            self._state = "searching"
            self._roi = None
        elif self._associator.is_lost:
            self._state = "lost"
            self._roi = None  # fall back to full-frame for recovery
        else:
            self._state = "tracking"

        self._frame_count += 1

        # --- Step 4: confidence heuristics ---------------------------------
        if bbox is None:
            confidence = 0.0
        elif self._associator.is_lost:
            confidence = 0.3  # bridging — lower confidence
        elif self._state == "tracking" and self._frame_count <= 2:
            confidence = 0.85  # freshly initialized — high but not perfect
        else:
            # Use the matched detection's confidence if available
            # (the associator doesn't expose the matched detection's score,
            # so we use the last detection's score as a proxy).
            if detections and not self._associator.is_lost:
                confidence = max(d.confidence for d in detections)
            else:
                confidence = 1.0

        return TrackResult(
            bbox=bbox or BBox(cx=0.0, cy=0.0, w=1.0, h=1.0),
            confidence=confidence,
            latency_s=time.perf_counter() - t0,
            source="yolo_sot",
        )

    def close(self) -> None:
        """Release detector resources."""
        self._detector.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_roi(self, bbox: BBox) -> tuple[float, float, float, float]:
        """Compute a crop region around *bbox*, large enough to contain
        the target at the next frame given its expected velocity.

        Returns
        -------
        (cx, cy, w, h)
            Cropping region in image coordinates.
        """
        w = max(bbox.w * self._crop_scale, 64.0)
        h = max(bbox.h * self._crop_scale, 64.0)
        return (bbox.cx, bbox.cy, w, h)
