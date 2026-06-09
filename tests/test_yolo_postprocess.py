"""Unit tests for YOLO postprocessing (NMS-free bbox conversion)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from trackers.yolo_detection.postprocess import Detection, detections_from_output
from core.bbox import BBox


def test_nms_free_format_1_N_6():
    """Test (1, N, 6) NMS-free format with two detections."""
    # Each row: [cx, cy, w, h, conf, cls_id] in normalized [0,1]
    output = np.array([[
        [0.5, 0.5, 0.2, 0.2, 0.9, 0.0],   # high conf, class 0
        [0.3, 0.3, 0.1, 0.1, 0.4, 0.0],   # lower conf, class 0
    ]], dtype=np.float32)
    dets = detections_from_output(output, orig_size=(640, 480), conf_threshold=0.25)

    assert len(dets) == 2
    # Sorted by confidence descending
    assert abs(dets[0].confidence - 0.9) < 0.01
    assert abs(dets[1].confidence - 0.4) < 0.01

    # Check coordinate denormalization
    b0 = dets[0].bbox
    assert abs(b0.cx - 320.0) < 1.0   # 0.5 * 640
    assert abs(b0.cy - 240.0) < 1.0   # 0.5 * 480
    assert abs(b0.w - 128.0) < 1.0    # 0.2 * 640
    assert abs(b0.h - 96.0) < 1.0     # 0.2 * 480


def test_nms_free_format_1_6_N():
    """Test (1, 6, N) transposed NMS-free format."""
    # Transposed: each column is [cx, cy, w, h, conf, cls_id]
    output = np.array([[
        [0.5, 0.3],
        [0.5, 0.3],
        [0.2, 0.1],
        [0.2, 0.1],
        [0.9, 0.4],
        [0.0, 0.0],
    ]], dtype=np.float32)
    dets = detections_from_output(output, orig_size=(640, 480), conf_threshold=0.25)

    assert len(dets) == 2
    assert abs(dets[0].confidence - 0.9) < 0.01


def test_conf_threshold_filters():
    """Detections below conf_threshold should be discarded."""
    output = np.array([[
        [0.5, 0.5, 0.2, 0.2, 0.9, 0.0],
        [0.3, 0.3, 0.1, 0.1, 0.1, 0.0],   # below threshold
        [0.7, 0.7, 0.3, 0.3, 0.2, 0.0],   # below threshold
    ]], dtype=np.float32)
    dets = detections_from_output(output, orig_size=(640, 480), conf_threshold=0.25)

    assert len(dets) == 1
    assert abs(dets[0].confidence - 0.9) < 0.01


def test_nonzero_class_filtered():
    """Only class_id 0 should be kept for single-class mode."""
    output = np.array([[
        [0.5, 0.5, 0.2, 0.2, 0.9, 0.0],
        [0.3, 0.3, 0.1, 0.1, 0.8, 1.0],   # class 1 — should be filtered
    ]], dtype=np.float32)
    dets = detections_from_output(output, orig_size=(640, 480), conf_threshold=0.1)

    assert len(dets) == 1
    assert dets[0].class_id == 0


def test_sorted_by_confidence():
    """Results must be sorted by confidence descending."""
    output = np.array([[
        [0.1, 0.1, 0.1, 0.1, 0.3, 0.0],
        [0.2, 0.2, 0.1, 0.1, 0.7, 0.0],
        [0.3, 0.3, 0.1, 0.1, 0.5, 0.0],
    ]], dtype=np.float32)
    dets = detections_from_output(output, orig_size=(640, 480), conf_threshold=0.25)

    assert abs(dets[0].confidence - 0.7) < 0.01
    assert abs(dets[1].confidence - 0.5) < 0.01
    assert abs(dets[2].confidence - 0.3) < 0.01


def test_detection_dataclass():
    """Detection dataclass should store fields correctly."""
    bbox = BBox(cx=100.0, cy=200.0, w=50.0, h=60.0)
    det = Detection(bbox=bbox, confidence=0.85, class_id=0)
    assert det.confidence == 0.85
    assert det.class_id == 0
    assert det.bbox.cx == 100.0


def test_empty_output():
    """N=0 detections should return empty list."""
    output = np.zeros((1, 0, 6), dtype=np.float32)
    dets = detections_from_output(output, orig_size=(640, 480))
    assert dets == []
