#!/usr/bin/env python3
"""
Stage 1 — UAV Appearance (Det-Fly)

Fine-tunes COCO-pretrained YOLO11s with a P2 detection head on the Det-Fly
air-to-air UAV dataset. The P2 head adds stride-4 detection for very small
targets (down to ~8×8 px vs ~16×16 px without P2).

Single-class detection — the model learns general "UAV-ness" features that
will transfer to other domains in later stages.

Usage:
    python training/scripts/stage1_detfly.py                        # defaults
    python training/scripts/stage1_detfly.py --epochs 150 --batch 8 # override
    python training/scripts/stage1_detfly.py --help                 # all options

Output:
    training/runs/stage1_detfly/weights/best.pt
"""

import argparse
import os
from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: Fine-tune YOLO11s-P2 on Det-Fly dataset"
    )
    p.add_argument("--cfg", default=f"{REPO}/training/configs/yolo11s-p2.yaml")
    p.add_argument("--weights", default=f"{REPO}/models/pretrained/yolo11s.pt")
    p.add_argument("--data", default=f"{REPO}/data/detection/detfly/data.yaml")
    p.add_argument("--project", default=f"{REPO}/training/runs")
    p.add_argument("--name", default="stage1_detfly")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (default: 8 for 8GB VRAM with P2)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="0")

    p.add_argument("--optimizer", default="auto")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--warmup_epochs", type=float, default=3)
    p.add_argument("--cos_lr", action="store_true", default=True)

    p.add_argument("--mixup", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--hsv_h", type=float, default=0.015)
    p.add_argument("--hsv_s", type=float, default=0.7)
    p.add_argument("--hsv_v", type=float, default=0.4)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--close_mosaic", type=int, default=10)
    p.add_argument("--fliplr", type=float, default=0.5)

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--single_cls", action="store_true", default=True)
    p.add_argument("--rect", action="store_true", default=False)
    p.add_argument("--cache", default=None,
                   help="Cache images: 'ram' or 'disk' (default: none)")

    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.cfg).load(args.weights)
    model.train(
        data=args.data, epochs=args.epochs, batch=args.batch,
        imgsz=args.imgsz, workers=args.workers, device=args.device,
        optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        mixup=args.mixup, scale=args.scale, translate=args.translate,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        mosaic=args.mosaic, close_mosaic=args.close_mosaic, fliplr=args.fliplr,
        amp=args.amp, single_cls=args.single_cls, rect=args.rect, cache=args.cache,
        project=args.project, name=args.name,
    )


if __name__ == "__main__":
    main()
