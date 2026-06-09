#!/usr/bin/env python3
"""
Convert UAV-Anti-UAV tracking dataset to YOLO detection format.

Each sequence is a video with per-frame tracking annotations (x,y,w,h).
Frames are extracted at a configurable stride and treated as independent
detection samples.  Train/val split is done at the sequence level to
prevent temporal leakage between splits.

Output structure:
  data/detection/uav_anti_uav/
    images/train/      (frames extracted from Train sequences 1–1260)
    images/val/        (frames from Train sequences 1261–1400)
    labels/train/      (YOLO .txt)
    labels/val/
    data.yaml

Usage:
    python training/convert_uav_anti_uav.py                    # defaults
    python training/convert_uav_anti_uav.py --stride 5         # every 5th frame
    python training/convert_uav_anti_uav.py --val-ratio 0.15   # 85/15 split
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


# --- CONFIGURATION ---
SRC_DIR = Path("data/raw/UAV-Anti-UAV")
DST_DIR = Path("data/detection/uav_anti_uav")
CLASSES = ["uav"]  # single class — matches README convention


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert UAV-Anti-UAV to YOLO detection format"
    )
    p.add_argument("--stride", type=int, default=10,
                   help="Extract every Nth frame (default: 10, ~3fps from 30fps source)")
    p.add_argument("--val-ratio", type=float, default=0.10,
                   help="Fraction of sequences for validation (default: 0.10)")
    p.add_argument("--val-seed", type=int, default=42,
                   help="Seed for reproducible sequence split")
    return p.parse_args()


def load_sequence_gt(seq_dir):
    """
    Load ground truth from a sequence directory.
    Returns: [(frame_idx, x, y, w, h, img_w, img_h), ...]
    Skips frames where absent=1.
    """
    gt_path = seq_dir / "groundtruth_rect.txt"
    absent_path = seq_dir / "absent.txt"

    with open(gt_path) as f:
        gt_lines = [l.strip() for l in f if l.strip()]

    absent = []
    if absent_path.exists():
        with open(absent_path) as f:
            absent = [int(l.strip()) for l in f if l.strip()]
    # Pad absent list if shorter than GT lines
    if len(absent) < len(gt_lines):
        absent.extend([0] * (len(gt_lines) - len(absent)))

    return gt_lines, absent


def parse_gt_line(line):
    """Parse a ground truth line: 'x,y,w,h' -> (x, y, w, h)."""
    parts = line.split(",")
    if len(parts) != 4:
        return None
    return tuple(int(p) for p in parts)


def gt_to_yolo(x, y, w, h, img_w, img_h):
    """Convert pixel (x,y,w,h) top-left to YOLO normalized (cx,cy,w,h)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def extract_frame(video_path, frame_idx):
    """Extract a single frame from video. Returns (success, numpy_array)."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return ret, frame


def process_sequence(seq_dir, seq_name, stride, out_img_dir, out_lbl_dir, split_tag):
    """
    Process one sequence: extract frames at stride, convert GT, write outputs.
    Returns stats dict.
    """
    mp4_path = seq_dir / f"{seq_name}.mp4"
    if not mp4_path.exists():
        print(f"  WARNING: video not found for {seq_name}")
        return {"frames": 0, "written": 0, "absent_skipped": 0, "errors": 0}

    # Get video dimensions once
    cap = cv2.VideoCapture(str(mp4_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    gt_lines, absent = load_sequence_gt(seq_dir)

    stats = {"frames": 0, "written": 0, "absent_skipped": 0, "errors": 0}

    cap = cv2.VideoCapture(str(mp4_path))

    for frame_idx in range(0, total_frames, stride):
        stats["frames"] += 1

        # Skip if absent
        if frame_idx < len(absent) and absent[frame_idx] == 1:
            stats["absent_skipped"] += 1
            continue

        # Parse GT
        if frame_idx >= len(gt_lines):
            stats["errors"] += 1
            continue

        bbox = parse_gt_line(gt_lines[frame_idx])
        if bbox is None:
            stats["errors"] += 1
            continue
        x, y, w, h = bbox

        # Skip degenerate boxes
        if w <= 0 or h <= 0:
            stats["absent_skipped"] += 1
            continue

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            stats["errors"] += 1
            continue

        # Actual frame dimensions (may differ from reported)
        actual_h, actual_w = frame.shape[:2]
        if actual_w != img_w or actual_h != img_h:
            img_w, img_h = actual_w, actual_h

        # Unique filename: sequence_name_frameXXXXXX.jpg
        img_name = f"{seq_name}_{frame_idx:06d}.jpg"
        lbl_name = f"{seq_name}_{frame_idx:06d}.txt"

        img_path = out_img_dir / img_name
        lbl_path = out_lbl_dir / lbl_name

        # Write image
        cv2.imwrite(str(img_path), frame)

        # Write YOLO label
        cx, cy, nw, nh = gt_to_yolo(x, y, w, h, img_w, img_h)
        with open(lbl_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        stats["written"] += 1

    cap.release()
    return stats


def create_data_yaml(train_path, val_path):
    """Write the data.yaml descriptor."""
    yaml_path = DST_DIR / "data.yaml"
    content = {
        "path": str(DST_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: CLASSES[0]},
        "nc": 1,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    print(f"\nCreated {yaml_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("  UAV-Anti-UAV → YOLO Detection Format")
    print(f"  Stride: every {args.stride}th frame  |  Val ratio: {args.val_ratio}")
    print("=" * 60)

    # --- Collect sequences ---
    train_src = SRC_DIR / "Train"
    seq_names = sorted(
        d.name for d in train_src.iterdir()
        if d.is_dir() and d.name.startswith("UAV-Anti-UAV_Train_")
    )

    if not seq_names:
        print("ERROR: No training sequences found")
        sys.exit(1)

    print(f"\nFound {len(seq_names)} training sequences")

    # --- Sequence-level train/val split ---
    n_val = max(1, int(len(seq_names) * args.val_ratio))
    n_train = len(seq_names) - n_val
    train_seqs = seq_names[:n_train]
    val_seqs = seq_names[n_train:]

    print(f"Train: {len(train_seqs)} sequences  |  Val: {len(val_seqs)} sequences")

    # --- Create output directories ---
    for d in [
        DST_DIR / "images" / "train",
        DST_DIR / "images" / "val",
        DST_DIR / "labels" / "train",
        DST_DIR / "labels" / "val",
    ]:
        os.makedirs(d, exist_ok=True)

    # --- Process ---
    all_stats = {"train": {}, "val": {}}

    for split_name, seq_list in [("train", train_seqs), ("val", val_seqs)]:
        print(f"\n--- Processing {split_name} ({len(seq_list)} sequences) ---")
        img_dir = DST_DIR / "images" / split_name
        lbl_dir = DST_DIR / "labels" / split_name

        total_written = 0
        total_absent = 0
        total_errors = 0
        total_frames = 0

        pbar = tqdm(seq_list, unit="seq", desc=split_name)
        for seq_name in pbar:
            seq_dir = train_src / seq_name
            stats = process_sequence(
                seq_dir, seq_name, args.stride, img_dir, lbl_dir, split_name
            )
            total_written += stats["written"]
            total_absent += stats["absent_skipped"]
            total_errors += stats["errors"]
            total_frames += stats["frames"]
            pbar.set_postfix(
                written=total_written, absent=total_absent, err=total_errors
            )

        all_stats[split_name] = {
            "sequences": len(seq_list),
            "frames_checked": total_frames,
            "frames_written": total_written,
            "absent_skipped": total_absent,
            "errors": total_errors,
        }

    # --- Create data.yaml ---
    create_data_yaml(
        DST_DIR / "images" / "train",
        DST_DIR / "images" / "val",
    )

    # --- Cleanup tracking results ---
    # Test videos stay untouched for tracking evaluation

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)
    for split_name in ["train", "val"]:
        s = all_stats[split_name]
        print(f"\n  {split_name}:")
        print(f"    Sequences:      {s['sequences']}")
        print(f"    Frames checked: {s['frames_checked']}")
        print(f"    Frames written: {s['frames_written']}")
        print(f"    Absent skipped: {s['absent_skipped']}")
        if s["errors"]:
            print(f"    Errors:         {s['errors']}")
    print(f"\n  Output: {DST_DIR}/")
    print(f"    images/train/  ({all_stats['train']['frames_written']} images)")
    print(f"    images/val/    ({all_stats['val']['frames_written']} images)")
    print(f"    labels/train/  (.txt)")
    print(f"    labels/val/    (.txt)")
    print(f"    data.yaml")
    print(f"\n  Test videos preserved at: {SRC_DIR}/Test/ ({420} sequences)")
    print("=" * 60)


if __name__ == "__main__":
    main()
