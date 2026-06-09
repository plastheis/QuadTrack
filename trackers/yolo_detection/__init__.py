"""YOLO detection-based tracking framework.

Public API:
    YOLODetector      — YOLO11 ONNX/RKNN inference wrapper
    SOTAssociator     — Kalman filter + IoU single-target matching
    YOLOTracker       — Detection-based tracker (BaseTracker subclass)
"""

from trackers.yolo_detection.detector import YOLODetector
from trackers.yolo_detection.associator import SOTAssociator
from trackers.yolo_detection.tracker import YOLOTracker

__all__ = ["YOLODetector", "SOTAssociator", "YOLOTracker"]
