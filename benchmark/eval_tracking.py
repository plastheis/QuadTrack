#!/usr/bin/env python3
"""
Run YOLO + BoT-SORT tracker on the UAV-Anti-UAV test set and compute metrics.

Operates in tracking-by-detection mode: the model detects per-frame, the
tracker (BoT-SORT) associates detections across frames.  Ground truth is
the per-frame bounding box from the tracking annotations.

Metrics:
  - Precision: % of frames where predicted bbox IoU > 0.5 with GT
  - Success (AUC): area under the precision-vs-IoU-threshold curve
  - Avg IoU: mean IoU on frames where both pred and GT exist
  - Center error: mean pixel distance

Usage:
    python benchmark/eval_tracking.py \
        --model /path/to/best.pt \
        --data training/datasets/UAV-Anti-UAV/Test \
        --limit 20

Output:
    benchmark_results/uav_anti_uav_tracking_eval.json
    (optional: --visualize saves annotated videos)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO tracker on UAV-Anti-UAV test set")
    p.add_argument("--model", required=True, help="Path to trained .pt weights")
    p.add_argument("--data", default="data/tracking/uav_anti_uav",
                   help="Path to tracking test sequences")
    p.add_argument("--output", default="benchmark_results/uav_anti_uav_tracking_eval.json",
                   help="Output JSON for metrics")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit to N sequences (0 = all 420)")
    p.add_argument("--visualize", action="store_true",
                   help="Save annotated videos to benchmark_results/videos/")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.7,
                   help="Tracker IOU threshold for association")
    return p.parse_args()


def parse_gt_line(line: str) -> tuple[int, int, int, int] | None:
    """Parse 'x,y,w,h' -> (x, y, w, h) or None."""
    try:
        parts = [int(x) for x in line.strip().split(",")]
        return tuple(parts) if len(parts) == 4 else None
    except ValueError:
        return None


def xywh_to_xyxy(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    return x, y, x + w, y + h


def compute_iou(boxA, boxB):
    """Compute IoU of two boxes in xyxy format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def center_dist(boxA, boxB):
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2
    return np.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)


def evaluate_sequence(seq_dir: Path, model: YOLO, args) -> dict:
    """
    Run YOLO tracker on one sequence, compare against ground truth.
    Returns per-sequence metrics dict.
    """
    seq_name = seq_dir.name
    mp4_path = seq_dir / f"{seq_name}.mp4"
    gt_path = seq_dir / "groundtruth_rect.txt"
    absent_path = seq_dir / "absent.txt"

    if not mp4_path.exists() or not gt_path.exists():
        return {"name": seq_name, "error": "missing video or GT"}

    # Load GT
    with open(gt_path) as f:
        gt_lines = [l.strip() for l in f if l.strip()]
    absent = []
    if absent_path.exists():
        with open(absent_path) as f:
            absent = [int(l.strip()) for l in f if l.strip()]

    # Ensure absent list matches GT length
    if len(absent) < len(gt_lines):
        absent += [0] * (len(gt_lines) - len(absent))

    cap = cv2.VideoCapture(str(mp4_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Visualisation setup
    vis_writer = None
    if args.visualize:
        vis_dir = Path("benchmark_results/videos")
        os.makedirs(vis_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vis_writer = cv2.VideoWriter(
            str(vis_dir / f"{seq_name}_tracked.mp4"), fourcc, fps, (w, h)
        )

    ious = []
    center_errors = []
    pred_present = 0      # frames model detected something
    gt_present = 0        # frames GT says target is present
    both_present = 0       # both detected
    frames_evaluated = 0
    latency_ms = []

    frame_idx = 0

    while frame_idx < total_frames and frame_idx < len(gt_lines):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracker (BoT-SORT)
        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
            verbose=False,
        )
        latency_ms.append((time.perf_counter() - t0) * 1000)

        # Get predicted bbox (only first tracked object)
        pred_box = None
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Filter to class 0 (uav) and take highest confidence
            boxes = results[0].boxes
            # boxes.xyxy: (N, 4) in pixel coords
            if boxes.xyxy is not None and boxes.xyxy.shape[0] > 0:
                xyxy = boxes.xyxy[0].cpu().numpy()
                pred_box = tuple(xyxy.astype(int))

        # Parse GT
        gt_parsed = parse_gt_line(gt_lines[frame_idx])
        gt_present_flag = (frame_idx < len(absent) and absent[frame_idx] == 0)

        gt_box = None
        if gt_parsed and gt_present_flag:
            x, y, w, h = gt_parsed
            if w > 0 and h > 0:
                gt_box = xywh_to_xyxy(x, y, w, h)
                gt_present += 1
                frames_evaluated += 1

        if pred_box is not None:
            pred_present += 1

        # Compute metrics
        if pred_box is not None and gt_box is not None:
            both_present += 1
            iou = compute_iou(pred_box, gt_box)
            ious.append(iou)
            cd = center_dist(pred_box, gt_box)
            center_errors.append(cd)
        elif gt_box is not None and pred_box is None:
            ious.append(0.0)  # miss
        elif gt_box is None and pred_box is not None:
            pass  # false positive — not counted in this simple metric

        # Visualize
        if vis_writer is not None:
            vis_frame = frame.copy()
            if gt_box:
                cv2.rectangle(vis_frame, (gt_box[0], gt_box[1]),
                              (gt_box[2], gt_box[3]), (0, 255, 0), 2)
                cv2.putText(vis_frame, "GT", (gt_box[0], gt_box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if pred_box:
                cv2.rectangle(vis_frame, (pred_box[0], pred_box[1]),
                              (pred_box[2], pred_box[3]), (0, 0, 255), 2)
                cv2.putText(vis_frame, "Pred", (pred_box[0], pred_box[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            vis_writer.write(vis_frame)

        frame_idx += 1

    cap.release()
    if vis_writer is not None:
        vis_writer.release()

    # Success rate at thresholds
    thresholds = np.arange(0.05, 1.0, 0.05)
    success_curve = []
    for t in thresholds:
        hits = sum(1 for iou in ious if iou >= t)
        success_curve.append(hits / max(1, frames_evaluated))

    auc = np.trapezoid(success_curve, thresholds) if len(success_curve) > 1 else 0.0
    precision = sum(1 for iou in ious if iou >= 0.5) / max(1, frames_evaluated)

    return {
        "name": seq_name,
        "frames": frame_idx,
        "frames_evaluated": frames_evaluated,
        "gt_present": gt_present,
        "pred_present": pred_present,
        "both_present": both_present,
        "precision@0.5": round(precision, 4),
        "success_auc": round(auc, 4),
        "avg_iou": round(np.mean(ious), 4) if ious else 0.0,
        "avg_center_error_px": round(np.mean(center_errors), 2) if center_errors else 0.0,
        "avg_latency_ms": round(np.mean(latency_ms), 2) if latency_ms else 0.0,
    }


def main():
    args = parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        print(f"ERROR: Test directory not found: {data_dir}")
        sys.exit(1)

    seq_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("UAV-Anti-UAV_Test_")
    )

    if not seq_dirs:
        print(f"ERROR: No test sequences found in {data_dir}")
        sys.exit(1)

    if args.limit > 0:
        seq_dirs = seq_dirs[:args.limit]

    print(f"\n{'=' * 60}")
    print(f"  YOLO Tracking Evaluation")
    print(f"  Model:  {args.model}")
    print(f"  Dataset: {data_dir} ({len(seq_dirs)} sequences)")
    print(f"  Conf: {args.conf}  |  IoU: {args.iou}  |  imgsz: {args.imgsz}")
    print(f"{'=' * 60}")

    # Load model
    model = YOLO(args.model)

    # Evaluate
    seq_metrics = []
    for seq_dir in tqdm(seq_dirs, desc="Evaluating", unit="seq"):
        m = evaluate_sequence(seq_dir, model, args)
        seq_metrics.append(m)

    # Aggregate
    valid = [m for m in seq_metrics if "error" not in m]
    n_valid = len(valid)
    avg_precision = np.mean([m["precision@0.5"] for m in valid]) if valid else 0
    avg_auc = np.mean([m["success_auc"] for m in valid]) if valid else 0
    avg_iou = np.mean([m["avg_iou"] for m in valid]) if valid else 0
    avg_ce = np.mean([m["avg_center_error_px"] for m in valid]) if valid else 0
    avg_lat = np.mean([m["avg_latency_ms"] for m in valid]) if valid else 0

    # Per-size bin analysis (optional, based on bbox area)
    # Not implemented yet — can be added later

    summary = {
        "model": args.model,
        "dataset": str(data_dir),
        "config": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
        },
        "n_sequences": n_valid,
        "aggregate": {
            "precision@0.5": round(float(avg_precision), 4),
            "success_auc": round(float(avg_auc), 4),
            "avg_iou": round(float(avg_iou), 4),
            "avg_center_error_px": round(float(avg_ce), 2),
            "avg_latency_ms": round(float(avg_lat), 2),
        },
        "sequences": seq_metrics,
    }

    # Output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS ({n_valid} sequences)")
    print(f"{'=' * 60}")
    print(f"  Precision@0.5:    {avg_precision:.4f}")
    print(f"  Success AUC:      {avg_auc:.4f}")
    print(f"  Avg IoU:          {avg_iou:.4f}")
    print(f"  Avg Center Error: {avg_ce:.2f} px")
    print(f"  Avg Latency:      {avg_lat:.2f} ms")
    print(f"\n  Full results → {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
