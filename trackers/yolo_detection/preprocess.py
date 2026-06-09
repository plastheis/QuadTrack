"""YOLO detection preprocessing utilities.

Pure functions for image preprocessing commonly used in YOLO-based object
detection and tracking pipelines: letterbox resizing, blob conversion, and
ROI cropping with boundary handling.
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    target_size: int | tuple[int, int],
) -> tuple[np.ndarray, float, int, int]:
    """Resize *image* to *target_size* with padding to preserve aspect ratio.

    The image is scaled so that its larger dimension fits exactly within
    *target_size*, then padded with the per-channel mean colour to reach the
    requested dimensions.  Padding is distributed evenly (left / right,
    top / bottom).

    Parameters
    ----------
    image : np.ndarray
        Input image as a uint8 H×W×C array (typically BGR from OpenCV).
    target_size : int or (int, int)
        Desired output size.  A single integer produces a square output;
        a ``(height, width)`` tuple is also accepted.

    Returns
    -------
    resized : np.ndarray
        The padded image of shape ``(target_h, target_w, C)``, dtype uint8.
    scale : float
        Scale factor applied to the original image during resizing.
    pad_left : int
        Number of padding columns added to the left side.
    pad_top : int
        Number of padding rows added to the top side.

    Notes
    -----
    If *target_size* is a single int *s*, the output shape is ``(s, s, C)``.
    """
    h, w = image.shape[:2]

    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    # Scale factor — fit the larger side, keep aspect ratio
    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Per-channel mean for padding colour
    mean_color = image.mean(axis=(0, 1)).astype(np.float64)

    # Padding to reach target dimensions (distribute evenly)
    dw = target_w - new_w
    dh = target_h - new_h
    pad_left = dw // 2
    pad_top = dh // 2
    pad_right = dw - pad_left
    pad_bottom = dh - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=mean_color,
    )

    return padded, float(scale), pad_left, pad_top


def to_blob(image: np.ndarray) -> np.ndarray:
    """Convert an image into a standard neural-network input blob.

    Performs the following transformations:

    * HxWxC layout is transposed to CxHxW.
    * A batch dimension is prepended to (1, C, H, W).
    * BGR channel order is reversed to RGB.
    * Pixel values are normalised from [0, 255] uint8 to [0.0, 1.0] float32.

    The caller must provide a 3-channel BGR image (H, W, 3). Monochrome
    conversion is handled upstream by the camera pipeline, not here.

    Parameters
    ----------
    image : np.ndarray
        Input image as uint8 (H, W, 3) in BGR order.

    Returns
    -------
    np.ndarray
        Blob with shape (1, 3, H, W), dtype float32, values in [0, 1].
    """
    # BGR to RGB, float32, [0, 1]
    blob = image[:, :, ::-1].astype(np.float32) / 255.0

    # HWC to CHW to NCHW
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])

    return blob


def crop_roi(
    image: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> tuple[np.ndarray, int, int]:
    """Extract a square crop centred on an axis-aligned ROI.

    The crop side length is ``max(w, h)`` — large enough to fully contain the
    ROI.  Regions that fall outside *image* are filled with the per-channel
    mean colour of the original image.

    Parameters
    ----------
    image : np.ndarray
        Input image (H×W×C, uint8).
    cx, cy : float
        Centre coordinates of the ROI (pixels, sub-pixel precision allowed).
    w, h : float
        Width and height of the ROI (pixels).

    Returns
    -------
    cropped : np.ndarray
        Square crop of shape ``(side, side, C)``, dtype uint8.
    offset_x : int
        X-coordinate in the original image that corresponds to column 0 of
        the crop.  May be **negative** when the ROI extends past the left
        image boundary (padding was added).
    offset_y : int
        Y-coordinate in the original image that corresponds to row 0 of
        the crop.  May be **negative** when padding was added at the top.

    Notes
    -----
    Use ``offset_x`` / ``offset_y`` to map coordinates back to the original
    image::

        >>> orig_x = crop_x + offset_x
        >>> orig_y = crop_y + offset_y
    """
    side = int(max(w, h))

    # Top-left of the (unpadded) square in image coordinates
    x1 = int(cx) - side // 2
    y1 = int(cy) - side // 2

    img_h, img_w = image.shape[:2]
    mean_color = image.mean(axis=(0, 1)).astype(np.float64)

    # Padding amounts for each side
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x1 + side - img_w)
    pad_bottom = max(0, y1 + side - img_h)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=mean_color,
        )
    else:
        padded = image

    # Crop origin in the (possibly padded) image
    sx = x1 + pad_left
    sy = y1 + pad_top

    cropped = padded[sy : sy + side, sx : sx + side]

    return cropped, x1, y1
