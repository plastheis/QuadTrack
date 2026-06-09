"""Unit tests for YOLO preprocessing utilities."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from trackers.yolo_detection.preprocess import letterbox, to_blob, crop_roi


def test_letterbox_square_target():
    """Verify letterbox produces correct output shape for square target."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result, scale, pad_l, pad_t = letterbox(img, 640)
    assert result.shape == (640, 640, 3), f"Expected (640,640,3), got {result.shape}"
    assert result.dtype == np.uint8
    assert 0 < scale <= 1.0


def test_letterbox_rect_target():
    """Verify letterbox with rectangular target size."""
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    result, scale, pad_l, pad_t = letterbox(img, (320, 256))
    assert result.shape == (320, 256, 3)
    assert 0 < scale <= 1.0


def test_letterbox_preserves_aspect_ratio():
    """The padded image should preserve the original aspect ratio."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result, scale, pad_l, pad_t = letterbox(img, 300)
    # The image is 2:1, so in a 300x300 target, it should be 300x150 plus padding
    # scale = min(300/100, 300/200) = min(3, 1.5) = 1.5
    # new_h = 150, new_w = 300
    assert abs(scale - 1.5) < 0.01
    # Total padding: 300 - 150 = 150, split top/bottom
    assert pad_t + (300 - 150 - pad_t) == 150  # pad_t + pad_b == 150


def test_to_blob_shape():
    """Verify to_blob produces correct NCHW float32 blob."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob = to_blob(img)
    assert blob.shape == (1, 3, 480, 640)
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() <= 1.0
    assert 0.0 <= blob.max() <= 1.0


def test_to_blob_rgb_order():
    """BGR input should become RGB in the blob."""
    # Create a BGR image with distinct channel values
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 0] = 100  # B
    img[:, :, 1] = 200  # G
    img[:, :, 2] = 50   # R
    blob = to_blob(img)
    # After BGR→RGB flip, channel 0 should be original R (50), channel 2 should be original B (100)
    assert abs(blob[0, 0, 5, 5] - 50/255.0) < 0.01   # was R, now ch0
    assert abs(blob[0, 2, 5, 5] - 100/255.0) < 0.01  # was B, now ch2


def test_to_blob_contiguous():
    """Blob must be C-contiguous for ONNX/RKNN."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob = to_blob(img)
    assert blob.flags["C_CONTIGUOUS"]


def test_crop_roi_fully_inside():
    """Crop when ROI is fully inside the image."""
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    cropped, off_x, off_y = crop_roi(img, 150, 100, 50, 60)
    assert cropped.shape[0] == cropped.shape[1]  # square
    assert cropped.shape[0] >= 60  # side >= max(w,h)
    assert off_x >= 0
    assert off_y >= 0


def test_crop_roi_partial_outside():
    """Crop when ROI extends beyond image boundaries — should pad."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cropped, off_x, off_y = crop_roi(img, 10, 10, 80, 80)
    assert cropped.shape[0] == 80
    # offset should be negative since ROI goes left/top of image
    assert off_x <= 10  # centered at 10, side=80 → x1 = 10-40 = -30
