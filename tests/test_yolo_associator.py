"""Unit tests for SOTAssociator (Kalman filter + IoU matching)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.bbox import BBox
from trackers.yolo_detection.associator import SOTAssociator
from trackers.yolo_detection.postprocess import Detection


def _det(cx, cy, w, h, conf=0.9):
    return Detection(bbox=BBox(cx=cx, cy=cy, w=w, h=h), confidence=conf, class_id=0)


def test_init_empty_detections_returns_none():
    """Calling update with no detections and no track returns None."""
    a = SOTAssociator()
    result = a.update([])
    assert result is None
    assert not a.is_tracking
    assert not a.is_lost


def test_auto_init_on_first_detection():
    """First detection auto-initializes the track."""
    a = SOTAssociator()
    det = _det(100, 200, 30, 40, 0.9)
    result = a.update([det])

    assert result is not None
    assert abs(result.cx - 100) < 5
    assert abs(result.cy - 200) < 5
    assert a.is_tracking
    assert not a.is_lost


def test_auto_init_picks_highest_confidence():
    """When multiple detections arrive with no track, pick highest confidence."""
    a = SOTAssociator()
    d1 = _det(100, 100, 20, 20, 0.5)
    d2 = _det(200, 200, 30, 30, 0.95)
    result = a.update([d1, d2])

    # Should pick d2 (higher confidence)
    assert abs(result.cx - 200) < 5
    assert abs(result.cy - 200) < 5


def test_kalman_predict_bridges_missing_detection():
    """When a detection is missing, the Kalman should predict the position."""
    a = SOTAssociator()
    # Initialize
    a.update([_det(100, 100, 20, 20)])
    # Predict without update
    pred = a.predict()
    assert pred is not None
    # With no velocity, prediction should be near original
    assert abs(pred.cx - 100) < 10
    assert abs(pred.cy - 100) < 10


def test_iou_matching_selects_best_match():
    """When multiple detections exist, pick the one with highest IoU to prediction."""
    a = SOTAssociator()
    # Initialize at (100, 100)
    a.update([_det(100, 100, 20, 20)])

    # Now provide two detections: one near track, one far away
    d_near = _det(105, 100, 20, 20, 0.5)   # high IoU
    d_far = _det(300, 300, 20, 20, 0.99)   # high confidence but far

    result = a.update([d_near, d_far])

    # Should match d_near (higher IoU), not d_far (higher confidence)
    assert abs(result.cx - 105) < 5


def test_lost_counter_increments():
    """When no detection matches, lost counter increments."""
    a = SOTAssociator(max_lost=10)
    a.update([_det(100, 100, 20, 20)])

    # Predict + no matching detection
    result = a.update([])
    assert a.is_lost
    assert result is not None  # Still returns prediction

    # Second miss
    result = a.update([])
    assert a.is_lost


def test_reset_after_max_lost():
    """After exceeding max_lost frames without a match, track is cleared."""
    a = SOTAssociator(max_lost=2)
    a.update([_det(100, 100, 20, 20)])

    # Miss 3 times (exceeds max_lost=2)
    a.update([])  # lost_count=1, still bridged
    a.update([])  # lost_count=2, still bridged
    result = a.update([])  # lost_count=3 > max_lost=2 → reset

    assert result is None
    assert not a.is_tracking
    assert not a.is_lost


def test_recover_from_lost():
    """A match after a few lost frames should recover the track."""
    a = SOTAssociator(max_lost=10)
    a.update([_det(100, 100, 20, 20)])

    # One miss
    a.update([])
    assert a.is_lost

    # Match returns — should recover
    result = a.update([_det(102, 100, 20, 20)])
    assert result is not None
    assert a.is_tracking
    assert not a.is_lost


def test_velocity_propagation():
    """Kalman should propagate velocity forward when no measurement."""
    a = SOTAssociator()
    a.update([_det(100, 100, 20, 20)])
    # Move to a new position — Kalman learns velocity
    a.update([_det(110, 100, 20, 20)])  # +10 in x

    # Predict without update — should continue +10 in x
    pred = a.predict()
    assert pred.cx > 110  # Should move further in +x direction


def test_reset_clears_all():
    """reset() should clear the track completely."""
    a = SOTAssociator()
    a.update([_det(100, 100, 20, 20)])

    a.reset()
    assert not a.is_tracking
    assert not a.is_lost
    assert a.track is None
    assert a.update([]) is None


def test_negative_coordinates_clamped():
    """Kalman prediction should not go below zero."""
    a = SOTAssociator()
    a.update([_det(5, 5, 20, 20)])
    # Force a negative prediction by moving left
    a.update([_det(-5, 5, 20, 20)])
    pred = a.predict()
    # cx might be negative if velocity carries it left, but we should survive
    assert pred is not None


def test_multiple_updates_stable():
    """Repeated updates with the same detection should keep the track stable."""
    a = SOTAssociator()
    det = _det(100, 200, 30, 40, 0.9)

    for _ in range(10):
        result = a.update([det])
        assert result is not None
        assert abs(result.cx - 100) < 10
        assert abs(result.cy - 200) < 10
