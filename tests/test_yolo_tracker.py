"""Unit tests for YOLOTracker state machine.

Tests the SEARCHING → TRACKING → LOST → SEARCHING state transitions
without requiring an actual YOLO model (mocked detector).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
import numpy as np
from core.bbox import BBox
from core.frame import Frame
from trackers.yolo_detection.postprocess import Detection


def _mock_frame():
    return Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0)


def _det(cx, cy, w, h, conf=0.9):
    return Detection(bbox=BBox(cx=cx, cy=cy, w=w, h=h), confidence=conf, class_id=0)


class TestYOLOTrackerStateMachine:
    """Tests the YOLOTracker with a mocked detector."""

    @staticmethod
    def _make_cfg():
        return {
            "tracker": {
                "yolo": {
                    "model_path": "fake.pt",
                    "input_size": [640, 640],
                    "conf_threshold": 0.25,
                    "crop_scale": 3.0,
                    "full_frame_every": 10,
                    "max_lost": 10,
                    "iou_threshold": 0.3,
                }
            },
            "inference": {"device": "cpu"},
        }

    def test_name(self):
        """Tracker name is 'yolo_sot'."""
        with patch("trackers.yolo_detection.tracker.YOLODetector"):
            from trackers.yolo_detection.tracker import YOLOTracker
            t = YOLOTracker(self._make_cfg())
            assert t.name() == "yolo_sot"

    def test_init_forces_tracking(self):
        """init() should seed the associator and set state to TRACKING."""
        with patch("trackers.yolo_detection.tracker.YOLODetector"):
            from trackers.yolo_detection.tracker import YOLOTracker
            t = YOLOTracker(self._make_cfg())
            frame = _mock_frame()
            bbox = BBox(cx=320, cy=240, w=50, h=50)

            t.init(frame, bbox)
            assert t._state == "tracking"
            assert t._associator.is_tracking

    def test_update_searching_to_tracking(self):
        """When a detection is found in SEARCHING, transition to TRACKING."""
        with patch("trackers.yolo_detection.tracker.YOLODetector") as MockDet:
            mock_det = MockDet.return_value
            mock_det.detect.return_value = [
                _det(320, 240, 50, 50, 0.95),
            ]

            from trackers.yolo_detection.tracker import YOLOTracker
            t = YOLOTracker(self._make_cfg())
            frame = _mock_frame()

            # First update: should find detection and transition to tracking
            result = t.update(frame)
            assert result is not None
            assert result.confidence > 0
            assert t._state == "tracking"

    def test_update_no_detections_stays_searching(self):
        """When no detections and no track, stay in SEARCHING."""
        with patch("trackers.yolo_detection.tracker.YOLODetector") as MockDet:
            mock_det = MockDet.return_value
            mock_det.detect.return_value = []  # no detections

            from trackers.yolo_detection.tracker import YOLOTracker
            t = YOLOTracker(self._make_cfg())
            frame = _mock_frame()

            result = t.update(frame)
            assert result.confidence == 0.0
            assert t._state == "searching"

    def test_lost_after_no_match(self):
        """When tracking and detections stop matching, transition to LOST."""
        with patch("trackers.yolo_detection.tracker.YOLODetector") as MockDet:
            mock_det = MockDet.return_value
            # First frame: detection found
            mock_det.detect.return_value = [_det(320, 240, 50, 50, 0.95)]
            from trackers.yolo_detection.tracker import YOLOTracker
            t = YOLOTracker(self._make_cfg())
            frame = _mock_frame()

            result = t.update(frame)
            assert t._state == "tracking"

            # Second frame: no detections — should go LOST
            mock_det.detect.return_value = []
            result = t.update(frame)
            assert t._state == "lost"
            # Should still get a result (Kalman prediction)
            assert result.confidence == 0.3  # bridging confidence

    def test_full_lost_eventually_returns_to_searching(self):
        """After max_lost frames without match, return to SEARCHING."""
        with patch("trackers.yolo_detection.tracker.YOLODetector") as MockDet:
            mock_det = MockDet.return_value
            # Initialize
            mock_det.detect.return_value = [_det(320, 240, 50, 50, 0.95)]
            from trackers.yolo_detection.tracker import YOLOTracker

            cfg = self._make_cfg()
            cfg["tracker"]["yolo"]["max_lost"] = 2  # small for test
            t = YOLOTracker(cfg)
            frame = _mock_frame()

            t.update(frame)  # → TRACKING
            assert t._state == "tracking"

            mock_det.detect.return_value = []
            t.update(frame)  # → LOST (lost_count=1)
            assert t._state == "lost"

            t.update(frame)  # → LOST (lost_count=2)
            assert t._state == "lost"

            t.update(frame)  # → SEARCHING (lost_count=3 > max_lost=2)
            assert t._state == "searching"

    def test_full_frame_every_n(self):
        """Every N frames, full-frame detection runs even in TRACKING."""
        with patch("trackers.yolo_detection.tracker.YOLODetector") as MockDet:
            mock_det = MockDet.return_value
            mock_det.detect.return_value = [_det(320, 240, 50, 50, 0.95)]

            from trackers.yolo_detection.tracker import YOLOTracker

            cfg = self._make_cfg()
            cfg["tracker"]["yolo"]["full_frame_every"] = 3
            t = YOLOTracker(cfg)
            frame = _mock_frame()

            # Update 0: searching → full-frame, frame_count → 1
            t.update(frame)

            # Update 1: tracking, frame_count=1→2, 1%3≠0 → crop
            t.update(frame)
            assert mock_det.detect.call_args[1]["roi"] is not None  # crop

            # Update 2: tracking, frame_count=2→3, 2%3≠0 → crop
            t.update(frame)
            assert mock_det.detect.call_args[1]["roi"] is not None  # crop

            # Update 3: tracking, frame_count=3→4, 3%3==0 → full-frame (periodic)
            t.update(frame)
            assert mock_det.detect.call_args[1]["roi"] is None  # full frame (periodic)

    def test_close_releases_detector(self):
        """close() should call detector.release()."""
        with patch("trackers.yolo_detection.tracker.YOLODetector") as MockDet:
            mock_det = MockDet.return_value
            from trackers.yolo_detection.tracker import YOLOTracker
            t = YOLOTracker(self._make_cfg())

            t.close()
            mock_det.release.assert_called_once()

    def test_compute_roi_scales_bbox(self):
        """_compute_roi should produce a region scaled by crop_scale."""
        with patch("trackers.yolo_detection.tracker.YOLODetector"):
            from trackers.yolo_detection.tracker import YOLOTracker
            cfg = self._make_cfg()
            cfg["tracker"]["yolo"]["crop_scale"] = 2.5
            t = YOLOTracker(cfg)

            roi = t._compute_roi(BBox(cx=100, cy=200, w=30, h=20))
            cx, cy, w, h = roi
            assert cx == 100
            assert cy == 200
            assert w == 75.0  # 30 * 2.5
            assert h == max(20 * 2.5, 64)  # scaled or min 64

    def test_result_has_required_fields(self):
        """TrackResult must have bbox, confidence, latency_s, source."""
        with patch("trackers.yolo_detection.tracker.YOLODetector") as MockDet:
            mock_det = MockDet.return_value
            mock_det.detect.return_value = [_det(320, 240, 50, 50, 0.95)]

            from trackers.yolo_detection.tracker import YOLOTracker
            t = YOLOTracker(self._make_cfg())
            frame = _mock_frame()

            result = t.update(frame)
            assert result.bbox is not None
            assert 0.0 <= result.confidence <= 1.0
            assert result.latency_s >= 0.0
            assert result.source == "yolo_sot"
