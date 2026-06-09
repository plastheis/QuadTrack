# Implementation Plan — Detection-Based Tracking Framework

Date: 2026-06-07
Parent spec: docs/superpowers/specs/2026-06-07-detection-tracker-design.md

---

## 1. Repository Layout

All new code lives under `trackers/yolo_detection/`. Training scripts live under
`training/`. Nothing outside these two directories is created or modified except
one import line in `trackers/factory.py` and one config block in `config.yaml`.

```
QuadTrack/                              # existing repo root
│
├── config.yaml                         # MODIFY: add yolo block + yolo_sot algo
├── trackers/
│   ├── factory.py                      # MODIFY: one import + one dict entry
│   └── yolo_detection/                 # NEW directory
│       ├── __init__.py                 # re-exports public API
│       ├── detector.py                 # YOLO26 ONNX/RKNN inference wrapper
│       ├── preprocess.py              # letterbox, normalize, CHW conversion
│       ├── postprocess.py             # bbox denormalization (NMS-free)
│       ├── associator.py              # Kalman filter + IoU matching
│       └── tracker.py                 # YOLOTracker (BaseTracker subclass)
│
├── training/                           # NEW directory (not a Python package)
│   ├── README.md                       # how to run the training pipeline
│   ├── stage1_dronesod.py             # Phase 1 training script
│   ├── stage2_anti_uav.py             # Phase 2 training script
│   ├── stage3_uav_anti_uav.py         # Phase 3 training script
│   ├── stage4_p2.py                   # Phase 4 training script (conditional)
│   ├── stage5_custom.py               # Phase 5 training script (optional)
│   ├── export_rknn.py                 # RKNN INT8 export script
│   ├── common.py                      # shared config, paths, argparse helpers
│   └── datasets/                      # YOLO dataset config YAMLs
│       ├── dronesod.yaml
│       ├── anti_uav.yaml
│       └── uav_anti_uav.yaml
│
└── scripts/                            # existing directory
    └── convert_anti_uav.py            # NEW: Anti-UAV JSON → YOLO txt
    └── convert_uav_anti_uav.py        # NEW: UAV-Anti-UAV → YOLO txt
```

---

## 2. Training Pipeline Design

### 2.1 Principle: Each stage is a standalone script

Every training stage takes a model path as input, produces a model path as
output, and saves checkpoints to a predictable directory. Stages are run
sequentially by a human (or a shell script). No stage imports another stage.

```
stage1_dronesod.py → runs/detect/stage1_dronesod/weights/best.pt
                           │
                           ▼
stage2_anti_uav.py  → runs/detect/stage2_anti_uav/weights/best.pt
                           │
                           ▼
stage3_uav_anti_uav.py → runs/detect/stage3_uav_anti_uav/weights/best.pt
                           │
                           ▼ (conditional)
stage4_p2.py        → runs/detect/stage4_p2/weights/best.pt
                           │
                           ▼ (optional)
stage5_custom.py    → runs/detect/stage5_custom/weights/best.pt
                           │
                           ▼
export_rknn.py      → models/yolo/yolo26n_uav.rknn
```

### 2.2 Rerunning a stage

To rerun Stage 2 with different data or hyperparameters:

```bash
python training/stage2_anti_uav.py \
    --weights runs/detect/stage1_dronesod/weights/best.pt \
    --data training/datasets/anti_uav_v2.yaml \
    --epochs 150 \
    --lr0 0.0003
```

The script accepts CLI arguments for everything tunable. Defaults match the
spec values. The `--data` flag points to a YOLO dataset YAML, so you can
point it at a different dataset without changing code.

### 2.3 Shared module: `training/common.py`

Provides:
- `DATA_DIR` — root path for all datasets (configurable via env var `QUADTRACK_DATA`)
- `RUNS_DIR` — root path for training outputs (default: `./runs/detect`)
- `MODELS_DIR` — root path for exported models (default: `./models/yolo`)
- `build_parser(description)` — returns an argparse.ArgumentParser with common
  flags (--weights, --data, --epochs, --batch, --lr0, --device, --name)
- `resolve_paths(args)` — creates output directories, validates input paths

No training logic lives here. Just path management and argument parsing.

---

## 3. Component Implementation Order

The Pi coding agent should implement these in dependency order. Each component
is testable independently before the next one is started.

### Step 1: `preprocess.py` (no dependencies)

Pure functions. No ONNX, no RKNN, no PyTorch. Just numpy + OpenCV.

```python
def letterbox(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize with padding to target_size, preserving aspect ratio."""

def to_blob(image: np.ndarray) -> np.ndarray:
    """BGR HWC [0,255] → RGB CHW [0,1] float32 with batch dim."""
```

**Test:** Pass a random 640×480 BGR image, verify output shape is `(1, 3, 640, 640)`,
verify values are in [0,1], verify channel order is RGB.

### Step 2: `postprocess.py` (no dependencies)

Pure functions. NMS-free — just coordinate denormalization.

```python
@dataclass
class Detection:
    bbox: BBox   # (cx, cy, w, h) in image coordinates
    confidence: float
    class_id: int

def detections_from_output(output: np.ndarray, orig_size: tuple[int, int],
                           conf_threshold: float = 0.25) -> list[Detection]:
    """Convert raw YOLO26 model output to Detection objects.
    
    YOLO26 NMS-free output is already final bounding boxes.
    Only conversion needed: normalized [0,1] → image pixel coordinates.
    """
```

**Test:** Pass a mock model output tensor, verify bounding boxes are correctly
scaled from normalized to pixel coordinates.

### Step 3: `detector.py` (depends on preprocess, postprocess)

Two backends behind one interface. The constructor tries RKNN, falls back to ONNX.

```python
class YOLODetector:
    def __init__(self, model_path: str, input_size: tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.25):
        """Load model. Auto-detect backend from file extension or available libs."""

    def detect(self, frame: np.ndarray,
               roi: tuple[float, float, float, float] | None = None
               ) -> list[Detection]:
        """Run inference. If roi is given, crop frame to roi first."""

    def release(self) -> None:
        """Free NPU/ONNX resources."""
```

Backend selection logic (same pattern as `nanotrack_tracker.py`):
- `.rknn` extension or RKNN library available → try `rknnlite` (on-device) or `rknn-toolkit2` (PC sim)
- `.onnx` extension → ONNX Runtime CPU
- Fallback: raise RuntimeError with actionable message

The `detect()` method handles monochrome input by stacking to 3-channel:
```python
if len(frame.shape) == 2:
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
```

**Test:** Load the COCO-pretrained YOLO26n ONNX, run on a sample image, verify
detections are returned. Then test with a crop ROI — verify only the cropped
region is processed and output coordinates are in full-frame space.

### Step 4: `associator.py` (no dependencies beyond numpy)

~50 lines. Pure math.

```python
class SOTAssociator:
    def __init__(self, max_lost: int = 10, iou_threshold: float = 0.3)

    def predict(self) -> BBox | None
        """Kalman predict one step forward. Returns predicted position."""

    def update(self, detections: list[Detection]) -> BBox | None
        """Match detections to current track. Auto-init if no track exists."""

    def reset(self) -> None
        """Clear the current track."""

    @property
    def is_tracking(self) -> bool
    @property
    def is_lost(self) -> bool
```

Uses `filterpy.kalman.KalmanFilter` or a hand-rolled 6-state Kalman. The
hand-rolled version is ~20 lines and removes a dependency:

```python
class KalmanFilter:
    """6-state constant-velocity Kalman filter. [cx, cy, w, h, vx, vy]."""
    def __init__(self):
        self.x = np.zeros((6, 1))     # state
        self.P = np.eye(6) * 10       # covariance
        self.F = np.eye(6)            # state transition
        self.F[0, 4] = 1; self.F[1, 5] = 1  # x += vx, y += vy
        self.H = np.eye(4, 6)         # measurement function
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01])  # process noise
        self.R = np.eye(4) * 5        # measurement noise

    def predict(self): ...
    def update(self, z): ...
```

**Test:** Create an associator. Call `update([])` — should return None. Call
`update([det])` — should return the detection bbox. Call `update([])` again —
should return the Kalman-predicted position. Verify `is_lost` flips correctly.

### Step 5: `tracker.py` (depends on detector, associator)

Implements `BaseTracker`. The state machine from the spec.

```python
class YOLOTracker(BaseTracker):
    def __init__(self, cfg: dict) -> None
    def name(self) -> str                    # "yolo_sot"
    def init(self, frame: Frame, bbox: BBox) -> None
    def update(self, frame: Frame) -> TrackResult
    def close(self) -> None
```

**Test:** Wire into `testing/main.py` (one config change). Track an object in
a webcam feed or recorded video. Verify the state transitions: searching →
tracking → lost → searching.

### Step 6: `__init__.py` + factory registration

```python
# trackers/yolo_detection/__init__.py
from trackers.yolo_detection.detector import YOLODetector
from trackers.yolo_detection.associator import SOTAssociator
from trackers.yolo_detection.tracker import YOLOTracker

__all__ = ["YOLODetector", "SOTAssociator", "YOLOTracker"]
```

One line in `trackers/factory.py`:
```python
from trackers.yolo_detection.tracker import YOLOTracker
# ... in _ALGO_MAP: "yolo_sot": YOLOTracker
```

---

## 4. Training Script Template

Each training script follows this pattern:

```python
"""Stage N: <description>.

Usage:
    python training/stageN_xxx.py [--weights PATH] [--data PATH] [--epochs N] ...

Default hyperparameters match the spec. Override via CLI flags.
"""
import sys
from pathlib import Path
from ultralytics import YOLO

# So we can run from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import DATA_DIR, RUNS_DIR, build_parser, resolve_paths


def main():
    parser = build_parser(description="Stage N: <description>")
    # Stage-specific arguments
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    # ... more stage-specific args ...

    args = parser.parse_args()
    resolve_paths(args)

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        name=args.name,
        exist_ok=True,

        optimizer="auto",
        lr0=args.lr0,
        lrf=0.01,
        weight_decay=0.0005,

        mosaic=1.0,
        close_mosaic=10,
        mixup=args.mixup,
        scale=args.scale,
        translate=getattr(args, "translate", 0.1),
        hsv_h=getattr(args, "hsv_h", 0.015),
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,

        warmup_epochs=3,
        cos_lr=True,
        amp=True,
        single_cls=True,
        rect=False,
    )

    best = Path(RUNS_DIR) / args.name / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")


if __name__ == "__main__":
    main()
```

Each stage script has its own defaults (different `lr0`, `mixup`, etc.) set
via `parser.add_argument(default=...)`. The CLI allows overriding any value.
The defaults ARE the spec values — running a script with no arguments should
produce the exact training run described in the spec.

---

## 5. Dataset Conversion Scripts

### `scripts/convert_anti_uav.py`

Converts Anti-UAV Challenge format (JSON per sequence with `gt_rect` and
`exist` arrays) to YOLO detection format. Strips sequence IDs — each frame
becomes an independent sample.

### `scripts/convert_uav_anti_uav.py`

Converts UAV-Anti-UAV format to YOLO detection format. Exact schema TBD
(dataset not yet publicly downloaded). Same principle: extract frames, discard
sequence IDs, write one `.txt` per frame.

Both scripts:
- Accept `--input-dir` and `--output-dir` CLI arguments
- Skip frames where the target is absent (`exist=0` or equivalent)
- Normalize bounding boxes to YOLO format (class x_center y_center w h, all 0-1)
- Create `images/` and `labels/` directories with train/val splits
- Generate the corresponding dataset YAML config

---

## 6. What the Pi Coding Agent Should Implement

### Order of work (each step produces a reviewable, testable unit):

| Step | File | Lines (est.) | Dependencies |
|------|------|-------------|--------------|
| 1 | `preprocess.py` | ~40 | numpy, cv2 |
| 2 | `postprocess.py` | ~50 | core.bbox |
| 3 | `detector.py` | ~120 | preprocess, postprocess, numpy, cv2, onnxruntime or rknn |
| 4 | `associator.py` | ~60 | numpy |
| 5 | `tracker.py` | ~100 | detector, associator, trackers.base, core.bbox, core.frame |
| 6 | `__init__.py` | ~5 | tracker |
| 7 | `training/common.py` | ~80 | pathlib, argparse |
| 8 | `training/stage1_dronesod.py` | ~60 | ultralytics, common |
| 9 | `training/stage2_anti_uav.py` | ~60 | ultralytics, common |
| 10 | `training/stage3_uav_anti_uav.py` | ~60 | ultralytics, common |
| 11 | `training/stage4_p2.py` | ~60 | ultralytics, common |
| 12 | `training/stage5_custom.py` | ~60 | ultralytics, common |
| 13 | `training/export_rknn.py` | ~30 | ultralytics, common |
| 14 | `scripts/convert_anti_uav.py` | ~80 | pathlib, json, cv2 |
| 15 | `scripts/convert_uav_anti_uav.py` | ~80 | pathlib, json, cv2 |

### Pre-written (human provides):

- `training/common.py` — the shared argument parser and path manager. The
  agent doesn't need to design the CLI interface; it just imports and uses it.
- `training/datasets/*.yaml` — dataset config files. Just YAML stubs with
  paths to be filled in when datasets are downloaded.
- `trackers/base.py`, `core/bbox.py`, `core/frame.py` — already exist.

### What the agent should NOT touch:

- `config.yaml` — human edits this (one block, well-defined)
- `trackers/factory.py` — human adds one import + one dict entry
- Any existing tracker files (nanotrack_tracker.py, kcf_tracker.py, etc.)
- `benchmark/` — existing dataset loaders unchanged
- `testing/main.py` — unchanged (just reads config)

---

## 7. Human-Run Training Procedure

### Prerequisites (one-time setup):

```bash
# 1. Install dependencies
pip install ultralytics>=8.4.0

# 2. Set data directory
export QUADTRACK_DATA=/data/quadtrack_datasets

# 3. Download and convert datasets
python scripts/convert_anti_uav.py \
    --input-dir $QUADTRACK_DATA/anti_uav_challenge/raw \
    --output-dir $QUADTRACK_DATA/anti_uav_challenge/yolo

python scripts/convert_uav_anti_uav.py \
    --input-dir $QUADTRACK_DATA/uav_anti_uav/raw \
    --output-dir $QUADTRACK_DATA/uav_anti_uav/yolo

# DroneSOD-30K: check format — may already be YOLO-compatible
```

### Training run (manual, sequential):

```bash
# Stage 1: UAV appearance (DroneSOD-30K)
python training/stage1_dronesod.py

# Stage 2: Domain adaptation (Anti-UAV Challenge)
python training/stage2_anti_uav.py

# Stage 3: Air-to-air (UAV-Anti-UAV)
python training/stage3_uav_anti_uav.py

# --- DECISION GATE ---
# Run benchmark on Stage 3 model. If recall at <16px < 70%:

# Stage 4: P2 small-object head (conditional)
python training/stage4_p2.py

# Stage 5: Custom footage (optional)
python training/stage5_custom.py \
    --data training/datasets/my_quadrotor.yaml \
    --epochs 30

# Export for SBC deployment
python training/export_rknn.py \
    --weights runs/detect/stage5_custom/weights/best.pt
```

### Rerunning a stage with different settings:

```bash
# Rerun Stage 2 with different learning rate and more epochs
python training/stage2_anti_uav.py \
    --lr0 0.0003 \
    --epochs 150

# Rerun Stage 2 on a different dataset split
python training/stage2_anti_uav.py \
    --data training/datasets/anti_uav_v2.yaml

# Rerun Stage 3 with different augmentation
python training/stage3_uav_anti_uav.py \
    --translate 0.3 \
    --scale 0.4
```

---

## 8. Testing Strategy

### Unit tests (agent or human writes alongside implementation):

| Test file | What it tests |
|-----------|--------------|
| `tests/test_yolo_preprocess.py` | `letterbox()` output shape/values, `to_blob()` channel order |
| `tests/test_yolo_postprocess.py` | `detections_from_output()` coordinate scaling |
| `tests/test_yolo_associator.py` | Kalman predict/update, IoU matching, lost tracking, init from detections |
| `tests/test_yolo_tracker.py` | State transitions: SEARCHING→TRACKING→LOST→SEARCHING |

### Integration tests (human runs):

| Test | How |
|------|-----|
| Detector on sample image | `python -c "from trackers.yolo_detection import YOLODetector; ..."` with a known image |
| Full tracker on webcam | Change `config.yaml` to `algorithm: yolo_sot`, run `testing/main.py` |
| Benchmark on Anti-UAV | Run `testing/benchmark.py` with yolo_sot against existing trackers |

---

## 9. File Contents Reference

### `training/common.py`

```python
"""Shared utilities for training scripts.

Usage:
    from training.common import DATA_DIR, RUNS_DIR, build_parser, resolve_paths

    parser = build_parser(description="Stage 1: ...")
    parser.add_argument("--mixup", type=float, default=0.1)  # stage-specific
    args = parser.parse_args()
    resolve_paths(args)
"""

import argparse
import os
from pathlib import Path

# Overridable via environment variables
DATA_DIR = Path(os.environ.get("QUADTRACK_DATA", "/data/quadtrack_datasets"))
RUNS_DIR = Path("runs/detect")
MODELS_DIR = Path("models/yolo")


def build_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser with common training flags.
    
    All flags have defaults matching the spec. Override via CLI.
    """
    p = argparse.ArgumentParser(description=description)
    
    # Required-ish
    p.add_argument("--weights", type=str, default="yolo26n.pt",
                   help="Input model weights (pretrained or from previous stage)")
    p.add_argument("--data", type=str, required=True,
                   help="Path to dataset YAML config")
    
    # Training duration
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0",
                   help="GPU index, 'cpu', or '0,1' for multi-GPU")
    
    # Optimization
    p.add_argument("--lr0", type=float, default=0.001,
                   help="Initial learning rate")
    
    # Output
    p.add_argument("--name", type=str, default=None,
                   help="Experiment name (default: derived from script name)")
    
    # Augmentation (stage-specific — set defaults in each script)
    p.add_argument("--mixup", type=float, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    p.add_argument("--hsv_h", type=float, default=None)
    
    return p


def resolve_paths(args) -> None:
    """Create output directories, set default experiment name."""
    if args.name is None:
        import sys
        stem = Path(sys.argv[0]).stem
        args.name = stem
    
    out_dir = RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(args.weights).exists() and not args.weights.startswith("yolo"):
        print(f"WARNING: weights file not found: {args.weights}")
```

### `training/datasets/dronesod.yaml`

```yaml
# DroneSOD-30K — Ground-to-air UAV detection
# Fill in actual paths after downloading the dataset.
path: /data/quadtrack_datasets/dronesod
train: images/train
val: images/val
names:
  0: uav
nc: 1
```

### `training/datasets/anti_uav.yaml`

```yaml
# Anti-UAV Challenge — Ground-to-air UAV tracking → detection
# Generated by scripts/convert_anti_uav.py
path: /data/quadtrack_datasets/anti_uav_challenge/yolo
train: images/train
val: images/val
names:
  0: uav
nc: 1
```

### `training/datasets/uav_anti_uav.yaml`

```yaml
# UAV-Anti-UAV — Air-to-air, moving platform
# Generated by scripts/convert_uav_anti_uav.py
path: /data/quadtrack_datasets/uav_anti_uav/yolo
train: images/train
val: images/val
names:
  0: uav
nc: 1
```

---

## 10. QuadTrack Module Compatibility

The YOLOTracker integrates with all existing QuadTrack modules without
modification. Here is the interface contract it satisfies and why each
existing module works:

### 10.1 Benchmark runner (`benchmark/runner.py`)

The runner calls `tracker.init(frame0, bbox0)` with a `Frame` and `BBox`,
then calls `tracker.update(frame)` for every subsequent frame, collects
`TrackResult`, and passes results through the fusion layer.

**YOLOTracker behavior:**
- `init(frame, bbox)`: Seeds the associator with a synthetic high-confidence
  detection at the ground-truth bbox. Sets state to TRACKING. The detector
  does NOT run on the init frame — the bbox is authoritative.
- `update(frame)`: Runs the state machine. Returns `TrackResult` with a valid
  `BBox` (or `BBox.zero()` when lost) and a confidence in [0, 1].
- The benchmark's init frame is always a ground-truth visible frame
  (`seq.init_frame()` filters for `exist=True`), so the seed bbox is valid.

**Latency measurement:** The benchmark wraps `update()` with its own
`time.perf_counter()` and adds the tracker's internal `latency_s`. This
correctly captures both tracker-internal time (detector NPU + associator CPU)
and framework overhead (fusion, metrics computation).

**Single-tracker mode:** When `config.yaml` lists only `yolo_sot` as the
algorithm, the factory uses `PassthroughFusion` which returns the single
`TrackResult` unchanged.

### 10.2 Factory (`trackers/factory.py`)

The `YOLOTracker` constructor takes `cfg: dict` and reads from
`cfg["tracker"]["yolo"]`. This matches the existing pattern used by
`NanoTracker`, `KCFTracker`, etc. No changes to the factory beyond
one import and one dict entry.

### 10.3 Config (`config.yaml`)

The `yolo` config block is namespaced under `tracker:` alongside existing
`nanotrack_backbone`, `nanotrack_head`, etc. The factory passes the full
config dict to each tracker constructor; each tracker reads its own section.

### 10.4 Fusion algorithms

`PassthroughFusion` (single tracker) and `IoUFusion` (multi-tracker) both
expect `TrackResult` objects with `.bbox` and `.confidence`. Our tracker
returns these. If used in a multi-tracker config with KCF or NanoTrack,
the fusion layer treats `yolo_sot` results identically.

### 10.5 Ground station GUI (`ground_station/gui.py`)

`draw_overlay(frame.image, bbox, tracking, roi_half, fps)` takes a raw numpy
image and a `BBox`. The `testing/main.py` loop calls this with the fused
`TrackResult.bbox`. The YOLOTracker's output is a standard `BBox` — the GUI
doesn't know or care which tracker produced it.

### 10.6 Live test harness (`testing/main.py`)

The main loop calls `tracker.init()` on SPACE press (centered ROI bbox) and
`tracker.update()` every frame. The YOLOTracker handles both. On SPACE press,
`init()` forces TRACKING at the ROI. On 'r' release, the existing code sets
`tracking = False` but doesn't call `reset()` on the tracker — the YOLOTracker
continues in its current state. The next SPACE press calls `init()` again,
which resets and reseeds the associator.

**Note:** The existing `testing/main.py` doesn't call a tracker `reset()` or
`close()` method on 'r' release. This is fine — the YOLOTracker simply
continues tracking (or searching) in the background. The next `init()` call
resets state. For a production system (QuadGuide), the `lockon/cmd` with a
zero-size bbox triggers `reset()`.

### 10.7 Monochrome camera

The monochrome-to-3-channel conversion belongs in QuadGuide's camera pipeline,
not in the tracker. The camera worker's `run()` loop converts frames before
writing to the shared memory buffer:

```python
# quadguide/src/quadguide/perception/camera/worker.py — in run()
frame, ts = source.read()
if len(frame.shape) == 2:
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
frame_buffer.write_frame(frame, ts)
```

Three lines. Every downstream consumer (tracker, guidance, ground station) sees
standard 3-channel BGR frames. The YOLOTracker's `detect()` method receives 3-channel
input and needs no monochrome-specific code. The `to_blob()` function in the
preprocess module retains 1-channel support for dev/testing flexibility but it
becomes dead code in the QuadGuide deployment path.

### 10.8 What does NOT need to change

| Module | Status |
|--------|--------|
| `core/bbox.py` | Unchanged — YOLOTracker uses `BBox` throughout |
| `core/frame.py` | Unchanged — `Frame.image` is the input to `detect()` |
| `core/iou.py` | Unchanged — `bbox_iou()` used in associator (same function) |
| `benchmark/runner.py` | Unchanged — calls `init()`/`update()` via BaseTracker interface |
| `benchmark/metrics/standard.py` | Unchanged — reads `TrackResult.confidence` for failure rate |
| `benchmark/datasets/anti_uav.py` | Unchanged — existing loader works as-is |
| `benchmark/datasets/base.py` | Unchanged — `YOLOTracker` is a `BaseTracker` subclass |
| `fusion_algs/*.py` | Unchanged — Passthrough/IoU/IoUKF fusion all read `TrackResult` |
| `ground_station/gui.py` | Unchanged — reads `BBox` from `TrackResult` |
| `testing/main.py` | Unchanged — config-driven tracker selection |
| `testing/benchmark.py` | Unchanged — uses `BenchmarkRunner` |
