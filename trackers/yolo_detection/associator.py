"""Single-target Kalman filter with max-IoU matching.

An ~50-line hand-rolled 6-state constant-velocity Kalman filter. No external
dependencies beyond numpy. No Hungarian algorithm, no ReID, no appearance
features — just max-IoU matching. Designed for the YOLO11s → Kalman pipeline
where the detector provides per-frame detections and the associator handles
temporal continuity.

State vector:     [cx, cy, w, h, vx, vy]^T
Measurement:      [cx, cy, w, h]^T
"""

from __future__ import annotations

import numpy as np

from core.bbox import BBox
from core.iou import bbox_iou


# ---------------------------------------------------------------------------
# Hand-rolled 6-state constant-velocity Kalman filter
# ---------------------------------------------------------------------------

class _KalmanFilter:
    """6-state CV Kalman filter. No external dependencies.

    State layout:  [cx, cy, w, h, vx, vy]  (position + velocity)
    Measurement:   [cx, cy, w, h]
    """

    def __init__(self) -> None:
        self.x = np.zeros((6, 1), dtype=np.float64)  # state
        self.P = np.eye(6, dtype=np.float64) * 10.0  # initial covariance

        # State transition: x += vx, y += vy
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        # Measurement function: only position is observed
        self.H = np.eye(4, 6, dtype=np.float64)  # (4, 6)

        # Process noise — low on velocity (const-vel assumption)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01])

        # Measurement noise — moderate detection jitter
        self.R = np.eye(4, dtype=np.float64) * 5.0

        self._initialized = False

    def predict(self) -> np.ndarray:
        """Kalman predict one step forward. Returns state vector (6,1)."""
        if not self._initialized:
            return self.x.copy()

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Kalman update with measurement z = [cx, cy, w, h] (4,1).

        If not yet initialized, sets the state directly from z.
        """
        z = np.asarray(z, dtype=np.float64).reshape(4, 1)

        if not self._initialized:
            self.x[:4] = z
            self.x[4:] = 0.0
            self._initialized = True
            return self.x.copy()

        y = z - self.H @ self.x           # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x.copy()

    def reset(self) -> None:
        """Reset the filter to uninitialized state."""
        self.x.fill(0.0)
        self.P = np.eye(6, dtype=np.float64) * 10.0
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# SOTAssociator
# ---------------------------------------------------------------------------


class SOTAssociator:
    """Single-object tracking associator: Kalman filter + max-IoU matching.

    Parameters
    ----------
    max_lost : int
        Frames without a match before the track is declared lost and cleared.
    iou_threshold : float
        Minimum IoU for a detection to be considered a match.
    """

    def __init__(self, max_lost: int = 10, iou_threshold: float = 0.3) -> None:
        self._kf = _KalmanFilter()
        self._max_lost = max_lost
        self._iou_threshold = iou_threshold
        self._track: BBox | None = None
        self._lost_count: int = 0

    def predict(self) -> BBox | None:
        """Kalman-predicted position without updating from detections.

        Returns
        -------
        BBox | None
            The predicted bounding box, or None if no track exists.
        """
        if not self._kf.is_initialized:
            return self._track

        state = self._kf.predict()
        cx, cy, w, h = float(state[0, 0]), float(state[1, 0]), float(state[2, 0]), float(state[3, 0])
        self._track = BBox(cx=max(cx, 0), cy=max(cy, 0), w=max(w, 1), h=max(h, 1))
        return self._track

    def update(self, detections: list) -> BBox | None:
        """Match detections to current track. Auto-initialize if no track yet.

        Parameters
        ----------
        detections : list[Detection]
            Detections from the YOLO detector for this frame.

        Returns
        -------
        BBox | None
            The matched bounding box (or Kalman prediction if lost, or None
            if no track at all).
        """
        # Auto-initialize on first detection
        if not self._kf.is_initialized:
            if detections:
                best = max(detections, key=lambda d: d.confidence)
                z = np.array([[best.bbox.cx], [best.bbox.cy], [best.bbox.w], [best.bbox.h]])
                self._kf.update(z)
                self._track = best.bbox
                self._lost_count = 0
                return self._track
            return None

        # Predict where the target should be
        predicted = self.predict()  # Updates self._track to predicted position

        # Match: find detection with highest IoU to predicted bbox
        best_iou = 0.0
        best_det = None
        for det in detections:
            iou = bbox_iou(det.bbox, predicted)
            if iou > best_iou:
                best_iou = iou
                best_det = det

        if best_det is not None and best_iou >= self._iou_threshold:
            # Match found — Kalman update
            z = np.array([[best_det.bbox.cx], [best_det.bbox.cy],
                          [best_det.bbox.w], [best_det.bbox.h]])
            self._kf.update(z)
            self._track = best_det.bbox
            self._lost_count = 0
            return self._track

        # No match — increment lost counter
        self._lost_count += 1

        if self._lost_count > self._max_lost:
            self.reset()
            return None

        # Bridge with Kalman prediction
        return predicted

    def reset(self) -> None:
        """Clear the current track."""
        self._kf.reset()
        self._track = None
        self._lost_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_tracking(self) -> bool:
        """True when a track exists and is not lost."""
        return self._kf.is_initialized and self._lost_count == 0

    @property
    def is_lost(self) -> bool:
        """True when a track exists but no match for >= 1 frame."""
        return self._kf.is_initialized and self._lost_count > 0

    @property
    def track(self) -> BBox | None:
        """The current tracked bounding box, or None."""
        return self._track
