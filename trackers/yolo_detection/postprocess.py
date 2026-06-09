"""YOLO11 postprocessing — NMS-free bbox coordinate conversion.

YOLO11 NMS-free output is already final bounding boxes. The only conversion
needed: normalized [0,1] model output → image pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.bbox import BBox


@dataclass(frozen=True)
class Detection:
    """A single detection from the YOLO detector.

    Attributes
    ----------
    bbox : BBox
        Bounding box in canonical (cx, cy, w, h) format, in image coordinates.
    confidence : float
        Detection confidence in [0, 1].
    class_id : int
        Class index (always 0 for single-class UAV detector).
    """

    bbox: BBox
    confidence: float
    class_id: int


def detections_from_output(
    output: np.ndarray,
    orig_size: tuple[int, int],
    conf_threshold: float = 0.25,
) -> list[Detection]:
    """Convert raw YOLO11 NMS-free model output to Detection objects.

    YOLO11 NMS-free mode outputs shape ``(1, N, 6)`` where the last dimension
    is ``[cx, cy, w, h, confidence, class_id]`` in normalized [0, 1] space.

    Alternatively, some export formats use ``(1, 84, 8400)`` — this function
    handles both shapes automatically.

    Parameters
    ----------
    output : np.ndarray
        Raw model output tensor. Accepted shapes:
        - ``(1, N, 6)`` — NMS-free, each row is ``[cx, cy, w, h, conf, cls]``.
        - ``(1, 6, N)`` — transposed NMS-free (seen in some ONNX exports).
        - ``(1, 84, 8400)`` — standard YOLO output (80-class COCO). Only
          class 0 is kept (single-class mode). Bounding boxes are decoded
          on-the-fly.
    orig_size : (int, int)
        Original image size as ``(width, height)`` in pixels.
    conf_threshold : float
        Minimum confidence score. Detections below this are discarded.

    Returns
    -------
    list[Detection]
        Detections sorted by confidence descending.
    """
    orig_w, orig_h = orig_size
    detections: list[Detection] = []

    if output.ndim == 3:
        # (1, N, 6) or (1, 6, N) — NMS-free format
        if output.shape[1] == 6 and output.shape[2] > 0:
            # (1, 6, N) — transpose to (1, N, 6)
            output = output.transpose(0, 2, 1)

        # Now (1, N, 6)
        rows = output[0]  # (N, 6)
        for row in rows:
            cx, cy, w, h, conf, cls_id = row
            conf = float(conf)
            if conf < conf_threshold:
                continue
            cls_id = int(cls_id)
            if cls_id != 0:
                continue  # single-class: only keep class 0
            # Denormalize: [0, 1] → pixel coordinates
            bbox = BBox(
                cx=float(cx) * orig_w,
                cy=float(cy) * orig_h,
                w=float(w) * orig_w,
                h=float(h) * orig_h,
            )
            detections.append(Detection(bbox=bbox, confidence=conf, class_id=0))

    elif output.ndim == 2 and output.shape[0] == 84:
        # (84, 8400) — standard YOLO decode format
        output = output[np.newaxis, :, :]  # → (1, 84, 8400)

    if output.ndim == 3 and output.shape[1] == 84:
        # (1, 84, 8400) with raw predictions (needs decode)
        dets = _decode_yolo_raw(output[0], orig_w, orig_h)  # (N, 6)
        for row in dets:
            conf = float(row[4])
            if conf < conf_threshold:
                continue
            cls_id = int(row[5])
            if cls_id != 0:
                continue
            bbox = BBox(
                cx=float(row[0]),
                cy=float(row[1]),
                w=max(float(row[2]), 1.0),
                h=max(float(row[3]), 1.0),
            )
            detections.append(Detection(bbox=bbox, confidence=conf, class_id=0))

    # Sort by confidence descending
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _decode_yolo_raw(
    preds: np.ndarray,
    img_w: int,
    img_h: int,
    nc: int = 1,
) -> np.ndarray:
    """Decode raw YOLO output ``(84, N)`` into ``(M, 6)`` pixel-coordinate rows.

    Handles the standard YOLO stride-based anchor-free output format:
    - First 4 channels per anchor: ``[tx, ty, tw, th]`` (distance-to-bbox edges
      in stride-normalized space).
    - Next ``nc`` channels: class logits.
    - The function computes confidence via sigmoid, picks argmax class,
      and converts to pixel coordinates.

    Parameters
    ----------
    preds : np.ndarray
        Shape ``(4 + nc, N)`` — raw model output.
    img_w, img_h : int
        Original image dimensions in pixels.
    nc : int
        Number of classes (default 1 for UAV).

    Returns
    -------
    np.ndarray
        ``(M, 6)`` array of ``[cx, cy, w, h, conf, cls_id]`` in pixel coords.
    """
    # For a single-class model, there's only one class channel
    # The output layout depends on the model architecture.
    # YOLO11 NMS-free typically outputs direct bboxes, so the raw decode path
    # is rarely needed. We handle it as a fallback.

    nc_dim = 4 + nc  # channels per anchor: 4 bbox + nc class

    # If preds has exactly nc_dim rows, it's a single detection head
    if preds.shape[0] == nc_dim:
        # Simple case: (4+nc, N) → direct decode
        bbox_raw = preds[:4, :]  # (4, N)
        cls_raw = preds[4:, :]  # (nc, N)

        # Compute confidence
        cls_scores = 1.0 / (1.0 + np.exp(-cls_raw))  # sigmoid
        if nc == 1:
            conf = cls_scores[0, :]
            cls_ids = np.zeros_like(conf, dtype=np.int32)
        else:
            cls_ids = np.argmax(cls_scores, axis=0)
            conf = cls_scores[cls_ids, np.arange(cls_ids.size)]

        # Decode bounding boxes (center to corners, then to pixel)
        cx = bbox_raw[0, :] * img_w / 640.0
        cy = bbox_raw[1, :] * img_h / 640.0
        w = bbox_raw[2, :] * img_w / 640.0
        h = bbox_raw[3, :] * img_h / 640.0

        return np.stack([cx, cy, w, h, conf, cls_ids.astype(np.float32)], axis=1)

    # For multi-stride outputs (e.g., 84 channels for 80 classes + 4 bbox),
    # we need to handle the standard YOLO output format.
    # This is the simplest decode path:
    if preds.shape[0] > nc_dim:
        # Take per-anchor slices if shape is (num_anchors * nc_dim, N)
        # This is a fallback; most YOLO11 exports use NMS-free direct output
        n_anchors = preds.shape[0] // nc_dim
        all_dets = []
        for i in range(n_anchors):
            start = i * nc_dim
            chunk = preds[start : start + nc_dim, :]
            decoded = _decode_yolo_raw(chunk, img_w, img_h, nc)
            all_dets.append(decoded)
        if all_dets:
            return np.concatenate(all_dets, axis=0)
        return np.empty((0, 6), dtype=np.float32)

    return np.empty((0, 6), dtype=np.float32)
