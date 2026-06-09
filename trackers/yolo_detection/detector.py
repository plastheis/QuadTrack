"""YOLO11 inference wrapper — ONNX Runtime + RKNN backends.

Backend selection (same pattern as nanotrack_tracker.py):
  cfg["inference"]["device"] == "npu" → try RKNN first, fall back to ONNX CPU
  cfg["inference"]["device"] == "cuda" → ONNX Runtime CUDA
  cfg["inference"]["device"] == "cpu" → ONNX Runtime CPU

Supports both full-frame and crop-mode inference. When a crop ROI is provided,
the frame is cropped to the ROI, letterbox-resized to input_size, and inference
runs on that sub-region. Output bounding boxes are transformed back to
full-frame image coordinates.
"""

from __future__ import annotations

import time
import warnings

import cv2
import numpy as np

from core.bbox import BBox
from trackers.yolo_detection.preprocess import crop_roi, letterbox, to_blob
from trackers.yolo_detection.postprocess import Detection, detections_from_output


class YOLODetector:
    """YOLO11 inference wrapper for ONNX Runtime (CUDA/CPU) and RKNN (NPU)."""

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to .rknn (SBC) or .onnx (dev) file.
        input_size : (int, int)
            (width, height) the model expects.
        conf_threshold : float
            Minimum confidence for a detection.
        device : str
            'cuda', 'npu', or 'cpu'. Matches cfg["inference"]["device"].
        """
        self._input_size = input_size
        self._conf_threshold = conf_threshold
        self._device = device.strip().lower()
        self._backend: str | None = None  # "rknn" | "onnx"

        self._rknn = None
        self._session = None

        # Try RKNN first for NPU device
        if self._device == "npu":
            if self._try_rknn(model_path):
                self._backend = "rknn"
                print("[YOLODetector] running on RKNN (NPU)")
            else:
                warnings.warn(
                    "[YOLODetector] RKNN unavailable — falling back to CPU ONNX Runtime",
                    stacklevel=2,
                )
                self._device = "cpu"

        if self._backend is None:
            self._load_onnx(model_path)
            self._backend = "onnx"

    # ------------------------------------------------------------------
    # RKNN loading
    # ------------------------------------------------------------------

    def _try_rknn(self, model_path: str) -> bool:
        """Try to load model via RKNN. Returns True on success."""
        _RKNNCls = None
        _is_lite = False
        try:
            from rknnlite.api import RKNNLite

            _RKNNCls = RKNNLite
            _is_lite = True
        except ImportError:
            pass
        if _RKNNCls is None:
            try:
                from rknn.api import RKNN

                _RKNNCls = RKNN
            except ImportError:
                pass
        if _RKNNCls is None:
            return False

        try:
            m = _RKNNCls()
            if m.load_rknn(model_path) != 0:
                raise RuntimeError(f"RKNN load failed: {model_path}")
            ret = m.init_runtime() if _is_lite else m.init_runtime(target=None)
            if ret != 0:
                raise RuntimeError(f"RKNN init_runtime failed: {model_path}")
            self._rknn = m
            return True
        except Exception as exc:
            warnings.warn(
                f"[YOLODetector] RKNN init failed: {exc}", stacklevel=3
            )
            return False

    # ------------------------------------------------------------------
    # ONNX loading
    # ------------------------------------------------------------------

    def _load_onnx(self, model_path: str) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )
        self._onnx_in_name = self._session.get_inputs()[0].name
        self._onnx_out_name = self._session.get_outputs()[0].name

        active = self._session.get_providers()
        on_gpu = active[0] == "CUDAExecutionProvider"
        provider = "GPU (CUDA)" if on_gpu else "CPU"
        print(f"[YOLODetector] running on {provider}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        roi: tuple[float, float, float, float] | None = None,
    ) -> list[Detection]:
        """Run inference and return detections.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3) as uint8.
        roi : (cx, cy, w, h) or None
            Optional crop region in image coordinates. When None, the full
            frame is processed.

        Returns
        -------
        list[Detection]
            Detections sorted by confidence descending.
        """
        orig_h, orig_w = frame.shape[:2]

        if roi is not None:
            cx, cy, rw, rh = roi
            # Ensure minimum crop size
            side = max(int(rw), int(rh), 64)
            cropped, offset_x, offset_y = crop_roi(frame, cx, cy, float(side), float(side))
            # Letterbox to model input size
            prepped, scale, pad_l, pad_t = letterbox(cropped, self._input_size)
            blob = to_blob(prepped)
        else:
            offset_x, offset_y = 0, 0
            prepped, scale, pad_l, pad_t = letterbox(frame, self._input_size)
            blob = to_blob(prepped)

        # Run model
        output = self._infer(blob)

        # Convert normalized output to image coordinates
        if roi is not None:
            # Detections are in crop coordinate space.
            # 1. Denormalize to crop (letterboxed) space
            # 2. Map to cropped image space (account for padding)
            # 3. Map to full-frame space (account for offset)
            detections = detections_from_output(
                output,
                orig_size=(self._input_size[0], self._input_size[1]),
                conf_threshold=self._conf_threshold,
            )

            mapped: list[Detection] = []
            for det in detections:
                # Step 1+2: remove padding, then scale to crop image size
                crop_h, crop_w = cropped.shape[:2]
                cx_crop = (det.bbox.cx - pad_l) / scale
                cy_crop = (det.bbox.cy - pad_t) / scale
                w_crop = det.bbox.w / scale
                h_crop = det.bbox.h / scale
                # Step 3: map to full-frame coordinates
                cx_full = cx_crop + offset_x
                cy_full = cy_crop + offset_y
                bbox = BBox(
                    cx=cx_full,
                    cy=cy_full,
                    w=w_crop,
                    h=h_crop,
                )
                mapped.append(
                    Detection(bbox=bbox, confidence=det.confidence, class_id=det.class_id)
                )
            return mapped
        else:
            return detections_from_output(
                output,
                orig_size=(orig_w, orig_h),
                conf_threshold=self._conf_threshold,
            )

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        """Run a single inference pass. Blob is (1, 3, H, W) float32 [0,1]."""
        if self._backend == "rknn":
            if self._rknn is None:
                raise RuntimeError("RKNN model not loaded")
            out = self._rknn.inference(inputs=[blob])
            return out[0]
        else:
            if self._session is None:
                raise RuntimeError("ONNX session not initialized")
            return self._session.run([self._onnx_out_name], {self._onnx_in_name: blob})[0]

    def release(self) -> None:
        """Release NPU / ONNX resources."""
        if self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:
                pass
            self._rknn = None
        self._session = None

    @property
    def backend(self) -> str | None:
        """Return the active backend name: 'rknn' or 'onnx'."""
        return self._backend
