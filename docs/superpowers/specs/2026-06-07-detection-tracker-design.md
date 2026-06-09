# Detection-Based Tracking Framework — Design Spec

Date: 2026-06-07
Status: draft
Updated: 2026-06-07 — switched from SDD-YOLO to YOLO26n (Ultralytics)

---

## 1. Overview

Add a detection-based single-object tracker to the QuadTrack dev/test repo,
with a clear path to deployment inside QuadGuide's tracker worker. The tracker
uses a **YOLO26n detector fine-tuned on Anti-UAV datasets** running on
Rockchip NPU (RK3588), with a lightweight Kalman + IoU associator on CPU.

### 1.1 Why YOLO26n

YOLO26 (Ultralytics, arXiv:2606.03748) is the latest YOLO generation and is
**fully integrated into the Ultralytics ecosystem** (58K+ GitHub stars). Key
features directly relevant to this use case:

- **NMS-free end-to-end inference.** No NMS post-processing on CPU — cleaner
  deployment pipeline, no variable-latency bottleneck.
- **DFL-free detection head.** Lighter than YOLOv8/v11 heads, faster ONNX
  inference (43% CPU speedup over YOLO11n reported).
- **P2 small-object variant.** `yolo26-p2.yaml` adds a stride-4 detection
  head for objects down to ~8×8 px. No pretrained weights, but trainable.
- **Built-in RKNN export.** `model.export(format='rknn', int8=True)` —
  one command, INT8 quantized, ready for RK3588.
- **STAL label assignment.** Guarantees positive coverage for small objects
  during training — critical for distant UAV detection.
- **Five scales (n/s/m/l/x).** The nano variant (YOLO26n) is 2.4M params,
  5.4 GFLOPs, 1.7ms T4 TensorRT latency — ideal for edge NPU.

### 1.2 What this spec covers

- New tracker implementation satisfying QuadTrack's `BaseTracker` interface:
  a YOLO26n detector running on RK3588 NPU with a Kalman + IoU associator on
  CPU. The Kalman filter predicts target motion and drives a crop region so
  the detector runs only on the relevant portion of the frame.
- Five-stage training pipeline using Ultralytics on verified Anti-UAV datasets
  (DroneSOD-30K → Anti-UAV Challenge → UAV-Anti-UAV → optional P2 head →
  custom footage). All stages train as pure per-frame detection.
- Dataset preparation for all stages, with tracking datasets converted to
  detection format (sequence IDs stripped).
- QuadGuide integration via the structural tracker protocol, including
  blind-fire auto-acquisition mode.

### 1.3 What this spec does NOT cover

- Multi-object tracking, segmentation, or pose estimation.
- Changes to QuadGuide bus, guidance, control, or link modules.
- YOLOE-26 open-vocabulary extension (future work).

---

## 2. Architecture

### 2.1 Data flow

```
Camera frame
    │
    ▼
┌─────────────────────────────────┐
│ SOTAssociator (CPU)              │
│   Kalman predict → where target  │
│   should be this frame           │
│   → predicted bbox               │
└──────────────┬──────────────────┘
               │ predicted bbox
               ▼
        ┌──────────────┐
        │ compute crop  │  scale × predicted size, min 64×64
        └──────┬───────┘
               │ roi (cx, cy, w, h)
               ▼
┌─────────────────────────────────┐
│ YOLODetector (NPU)               │
│   YOLO26n INT8 RKNN              │
│   - full-frame (searching/lost)  │
│   - crop mode (tracking)         │
│   → N detections {bbox, conf}    │
│   (NMS-free — direct output)     │
└──────────────┬──────────────────┘
               │ detections
               ▼
┌─────────────────────────────────┐
│ SOTAssociator (CPU)              │
│   max-IoU match → Kalman update  │
│   → tracked bbox or None         │
└──────────────┬──────────────────┘
               │
               ▼
         TrackResult
```

The Kalman predict step runs BEFORE detection, and its output directly determines
the crop region. This is the core efficiency mechanism: instead of scanning the
full 1920×1080 frame, the detector only processes a ~300×300 crop around where
the target is expected. The periodic full-frame scan (every N frames) guards
against the Kalman prediction drifting outside the crop.

### 2.2 States

```
                 ┌──────────┐
                 │ SEARCHING │  full-frame detection every frame
                 └─────┬────┘
                       │ first detection found → init track
                       ▼
                 ┌──────────┐
                 │ TRACKING  │  crop around Kalman-predicted position
                 └─────┬────┘
                       │ no match for 1 frame
                       ▼
                 ┌──────────┐
                 │   LOST    │  full-frame detection, bridge with prediction
                 └─────┬────┘
                       │ match found → back to TRACKING
                       │ no match for >10 frames → back to SEARCHING
                       ▼
                 ┌──────────┐
                 │ SEARCHING │  (re-acquisition)
                 └──────────┘
```

**State behavior:**

| State | Detection mode | Kalman | Crop source |
|-------|---------------|--------|-------------|
| SEARCHING | Full frame every frame | No track | None (full frame) |
| TRACKING | Crop (full frame every N) | Predict + update | Kalman prediction |
| LOST | Full frame every frame | Predict only (no update) | None (full frame) |

In TRACKING state, the Kalman predicts the target position BEFORE the detector
runs, and that prediction drives the crop region. This is the core efficiency
mechanism. The detector only sees the relevant ~300×300 region, not the full
1920×1080 frame.

In LOST state, the detector falls back to full-frame scanning while the Kalman
continues predicting (bridging). If a detection matches within `max_lost`
frames, the track recovers to TRACKING. If not, the track is cleared and the
system returns to SEARCHING.

Blind-fire launch: tracker starts in SEARCHING. Full-frame detection runs
every frame until a UAV is detected. Auto-initializes on first detection —
no manual lock-on required.

Manual lock-on (QuadGuide `lockon/cmd`): `init(frame, bbox)` seeds the
associator with a synthetic high-confidence detection at the given bbox
and transitions directly to TRACKING.

### 2.3 Files

All new code under `trackers/yolo_detection/`:

```
trackers/yolo_detection/
    __init__.py           # public API: YOLODetector, SOTAssociator, YOLOTracker
    detector.py           # YOLO26 RKNN/ONNX inference wrapper
    associator.py         # Kalman filter + IoU matching (~40 lines)
    tracker.py            # YOLOTracker (BaseTracker subclass) — state machine
    preprocess.py         # frame crop, resize, letterbox for YOLO input
    postprocess.py        # bbox conversion (NMS-free — just scale to image coords)
```

Training code lives separately in `training/`:

```
training/
    train_yolo26.py       # Ultralytics training script
    datasets/
        uav_anti_uav.yaml # Dataset config for UAV-Anti-UAV
        anti_uav.yaml     # Dataset config for Anti-UAV challenge
        dronesod.yaml     # Dataset config for DroneSOD-30K
    README.md             # Training instructions
```

### 2.4 Config

Add to `config.yaml`:

```yaml
tracker:
  algorithms:
    - algorithm: yolo_sot
      async: false
  yolo:
    model_path: models/yolo/yolo26n_uav.rknn
    input_size: [640, 640]
    conf_threshold: 0.25
    crop_scale: 3.0             # multiplier on bbox size for crop region
    full_frame_every: 10        # run full-frame detection every N frames
    max_lost: 10                # frames without match before declaring lost
```

---

## 3. Component Specifications

### 3.1 YOLODetector (`detector.py`)

Thin wrapper around YOLO26 inference. Backend-agnostic — RKNN on SBC,
ONNX Runtime on dev machine.

```python
class YOLODetector:
    """YOLO26 inference wrapper. Supports full-frame and crop modes."""

    def __init__(self, model_path: str, input_size: tuple[int,int] = (640,640),
                 conf_threshold: float = 0.25):
        """
        Args:
            model_path: Path to .rknn (SBC) or .onnx (dev) file.
            input_size: (width, height) the model expects.
            conf_threshold: Minimum confidence for a detection.
        """

    def detect(self, frame: np.ndarray,
               roi: tuple[float,float,float,float] | None = None
               ) -> list[Detection]:
        """
        Run inference and return detections.

        Args:
            frame: BGR image (H, W, 3).
            roi: Optional (cx, cy, w, h) crop region. When None, full frame.

        Returns:
            List of Detection, sorted by confidence descending.
            NMS-free — detections are direct model outputs.
        """

    def release(self) -> None:
        """Release NPU/ONNX resources."""

@dataclass
class Detection:
    bbox: BBox          # canonical (cx, cy, w, h) in image coordinates
    confidence: float   # [0, 1]
    class_id: int       # always 0 for UAV (single-class detector)
```

**Backend selection:** Same pattern as `nanotrack_tracker.py`. Try RKNN first
(`rknnlite` on-device), fall back to ONNX Runtime CPU. The `detect()` method
is backend-agnostic.

**Preprocessing:** Standard YOLO letterbox — resize with padding to
`input_size`, BGR→RGB, normalize to [0,1], CHW layout. Matches what
Ultralytics RKNN export produces.

**Postprocessing:** NMS-free. The model outputs final bounding boxes directly.
Only conversion needed: normalized [0,1] model output → image pixel coordinates.

**Full-frame vs crop:** When `roi` is provided, crop the frame to the ROI
(with padding to maintain aspect ratio), letterbox-resize to `input_size`,
run inference. Detected bbox coordinates are transformed back to full-frame
image coordinates. A 300×300 crop letterboxed to 640×640 runs ~3× faster
than full 1920×1080 frame.

### 3.2 SOTAssociator (`associator.py`)

Single-target Kalman filter with max-IoU matching. No Hungarian algorithm,
no ReID model, no appearance features. ~40 lines.

```python
class SOTAssociator:
    """Single-object tracking associator: Kalman filter + max-IoU matching."""

    def __init__(self, max_lost: int = 10, iou_threshold: float = 0.3):
        self._kf = KalmanFilter(dim_x=6, dim_z=4)
        # State: [cx, cy, w, h, vx, vy]
        # Measurement: [cx, cy, w, h]
        self._track: BBox | None = None
        self._lost: int = 0

    def update(self, detections: list[Detection]) -> BBox | None:
        """Match detections to current track. Auto-initialize if no track."""

    def predict(self) -> BBox | None:
        """Kalman-predicted position without updating from detections."""

    def reset(self) -> None:
        """Clear the current track."""

    @property
    def is_tracking(self) -> bool: ...
    @property
    def is_lost(self) -> bool: ...
```

**Kalman parameters (hardcoded):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| dim_x | 6 | [cx, cy, w, h, vx, vy] — constant velocity |
| dim_z | 4 | [cx, cy, w, h] — direct measurement |
| dt | 1.0 | Normalized to 1 frame |
| Process noise Q | diag(1,1,1,1,0.01,0.01) | Low position noise, very low velocity |
| Measurement noise R | diag(5,5,5,5) | Moderate detection jitter |
| Initial P | diag(10,…,10) | High initial uncertainty, converges fast |

**Matching logic:**

```
if no track and no detections → return None
if no track and detections exist → init on highest-confidence detection
if track exists:
    predict Kalman one step forward
    find detection with highest IoU to predicted bbox
    if max_iou > 0.3 → update Kalman, reset lost counter
    if no match → increment lost counter, return Kalman prediction
    if lost > max_lost → clear track, return None
```

Computational cost: ~5 microseconds per frame. The Kalman is a 6×6 matrix
multiply. IoU is 4 float ops per detection. At 5 detections per frame, that's
~20 FLOPs total.

### 3.3 YOLOTracker (`tracker.py`)

Implements `BaseTracker` from `trackers/base.py`. Registered in factory as
`"yolo_sot"`.

```python
class YOLOTracker(BaseTracker):
    """Detection-based single-object tracker using YOLO26n + Kalman associator."""

    def __init__(self, cfg: dict) -> None:
        ycfg = cfg["tracker"]["yolo"]
        self._detector = YOLODetector(
            model_path=ycfg["model_path"],
            input_size=tuple(ycfg.get("input_size", [640, 640])),
            conf_threshold=ycfg.get("conf_threshold", 0.25),
        )
        self._associator = SOTAssociator(
            max_lost=ycfg.get("max_lost", 10),
        )
        self._full_frame_every = ycfg.get("full_frame_every", 10)
        self._crop_scale = ycfg.get("crop_scale", 3.0)
        self._state = "searching"
        self._roi = None
        self._frame_count = 0

    def name(self) -> str:
        return "yolo_sot"

    def init(self, frame: Frame, bbox: BBox) -> None:
        """Manual lock-on from QuadGuide lockon/cmd. Forces TRACKING at bbox."""
        self._associator.reset()
        self._associator.update([Detection(bbox=bbox, confidence=1.0, class_id=0)])
        self._state = "tracking"
        self._roi = self._compute_roi(bbox)
        self._frame_count = 0

    def update(self, frame: Frame) -> TrackResult:
        """Per-frame update. Kalman predicts crop → detector runs → associator matches."""
        t0 = time.perf_counter()

        # Determine whether to use full frame or crop
        use_full_frame = (
            self._state in ("searching", "lost")
            or self._frame_count % self._full_frame_every == 0
            or self._roi is None
        )

        if use_full_frame:
            detections = self._detector.detect(frame.image, roi=None)
        else:
            # Kalman predicts where target should be → drives the crop
            predicted = self._associator.predict()
            if predicted is not None:
                self._roi = self._compute_roi(predicted)
            detections = self._detector.detect(frame.image, roi=self._roi)

        bbox = self._associator.update(detections)

        # State transitions
        if bbox is None:
            self._state = "lost"
            self._roi = None
        elif self._associator.is_lost:
            self._state = "lost"  # bridging with Kalman prediction
        else:
            self._state = "tracking"

        self._frame_count += 1

        if bbox is None:
            confidence = 0.0
        elif self._associator.is_lost:
            confidence = 0.3
        else:
            confidence = 1.0

        return TrackResult(
            bbox=bbox or BBox.zero(),
            confidence=confidence,
            latency_s=time.perf_counter() - t0,
            source="yolo_sot",
        )

    def close(self) -> None:
        self._detector.release()

    def _compute_roi(self, bbox: BBox) -> tuple[float, float, float, float]:
        """Crop region around bbox, sized to contain target at next frame."""
        w = max(bbox.w * self._crop_scale, 64)
        h = max(bbox.h * self._crop_scale, 64)
        return (bbox.cx, bbox.cy, w, h)
```

### 3.4 Factory registration

One line added to `trackers/factory.py`:

```python
from trackers.yolo_detection.tracker import YOLOTracker

_ALGO_MAP: dict[str, type[BaseTracker]] = {
    ...
    "yolo_sot": YOLOTracker,
}
```

---

## 4. Training Pipeline

### 4.1 Training framework: Ultralytics

YOLO26 is part of the official Ultralytics package. Training is a few lines
of Python. No external training framework needed — Ultralytics handles
everything: data loading, augmentation, optimizer (MuSGD), progressive loss,
STAL label assignment, validation, and export.

```bash
pip install ultralytics>=8.4.0
```

### 4.2 Training strategy

The model is a **per-frame detector, not a temporal tracker.** The Kalman
filter in the associator handles motion continuity — the model's only job is
accurate per-frame bounding box regression. This means detection datasets
(independent images) are equally valid as tracking datasets (video sequences)
for training. In fact, detection datasets may be better: they lack temporal
correlations that can cause overfitting to smooth motion priors.

**Why tracking datasets can be used as detection data:** SOT datasets like
Anti-UAV Challenge provide bounding boxes for every frame of every sequence.
By ignoring the sequence IDs and treating each frame as an independent sample,
the model learns robust per-frame detection without learning to rely on "the
target was here last frame." The Kalman filter provides that context at
inference time — it should not be baked into the model weights.

**Training paradigm:** All stages train as pure detection. Bounding boxes
from tracking datasets are extracted frame-by-frame with no sequence continuity.
The loss is detection mAP — spatial accuracy only. No temporal loss terms.

Five progressively specialized stages. Each stage uses lower learning rates
because we are fine-tuning, not training from scratch. The COCO-pretrained
YOLO26n backbone already has rich general visual features; each stage adapts
those features to a narrower domain.

| Stage | Dataset | Type | Role | Epochs | lr0 |
|-------|---------|------|------|--------|-----|
| 1 | DroneSOD-30K | Detection | UAV appearance: teach per-frame UAV detection across diverse weather | 100 | 0.001 |
| 2 | Anti-UAV Challenge | Tracking→Detection | Domain adaptation: UAVs against sky/clutter, varied ranges | 100 | 0.0005 |
| 3 | UAV-Anti-UAV | Tracking→Detection | Air-to-air: dual-motion dynamics, sky backgrounds, moving platform | 100 | 0.0005 |
| 4 | YOLO26n-P2 | Architecture | Small-object head: stride-4 P2 (ONLY if Stage 3 recall at <16px is poor) | 50 | 0.001 |
| 5 | Custom footage | Detection | Instance specialization: your specific UAV type | 30 | 0.0002 |

**Stage 4 decision gate:** After Stage 3, run benchmark on the UAV-Anti-UAV
test set and measure recall at different target size bins. If recall at
8-16 px targets is below 70%, proceed to Stage 4. Otherwise, the standard
YOLO26n stride-8 head is sufficient and the P2 overhead (55% more GFLOPs,
~20% slower inference) is not justified.

**YOLO26 automatic features:** MuSGD optimizer, Progressive Loss, and STAL
label assignment are baked into the YOLO26 architecture. When you train with
`optimizer="auto"` on a YOLO26 model, Ultralytics automatically selects MuSGD.
Progressive Loss (shifting supervision from the auxiliary one-to-many head to
the inference one-to-one head during training) and STAL (guaranteeing positive
label coverage for small objects) are part of the model's internal training
logic — active by default, no configuration needed.

### 4.3 Dataset preparation

Datasets must be in YOLO format. **All datasets are converted to single-frame
detection format** — each row is an independent sample with no sequence
continuity information:

```
dataset_root/
    images/
        train/
            seq001_00001.jpg
            seq001_00002.jpg
            ...
        val/
            ...
    labels/
        train/
            seq001_00001.txt   # one row per object: class x_center y_center w h
            seq001_00002.txt   # (all normalized 0-1)
            ...
        val/
            ...
```

For tracking datasets (Anti-UAV Challenge, UAV-Anti-UAV), frames are extracted
from video sequences and sequence IDs are discarded. Each frame's bounding box
becomes an independent detection label.

Dataset YAML config (`training/datasets/uav_anti_uav.yaml`):

```yaml
path: /data/uav_anti_uav
train: images/train
val: images/val

names:
  0: uav

# Single class — we only care about UAV detection
nc: 1
```

### 4.4 Dataset sources and priorities

All datasets verified: **UAV is the annotation target class**, not the camera
platform. Full verification details in `ANTI_UAV_DATASETS.md`.

| Dataset | Type | Size | Domain | Priority | Stage |
|---------|------|------|--------|----------|-------|
| **DroneSOD-30K** (arXiv:2603.25218) | Detection | 30,000 images | Ground-to-air, diverse weather | HIGHEST | 1 |
| **Anti-UAV Challenge** | Tracking→Detection | 300+ sequences | Ground-to-air, static + moving | HIGH | 2 |
| **UAV-Anti-UAV** (arXiv:2512.07385) | Tracking→Detection | 1,810 videos | Air-to-air, moving platform | HIGHEST | 3 |
| **YOLO26n-P2** (architecture only) | — | — | Stride-4 small-object head | CONDITIONAL | 4 |
| **Custom quadrotor footage** | Detection | 500-1000 frames | Your exact engagement geometry | HIGH | 5 |

**Excluded datasets** (UAV as camera platform, not target):
VisDrone, UAVDT, UAV123, DTB70, DOTA, GOT-10k.

### 4.5 Training recipe

All stages train as pure per-frame detection. Tracking dataset frames are
extracted as independent samples with no sequence continuity. The model
learns spatial accuracy; the Kalman filter provides temporal continuity.

#### Stage 1 — UAV appearance (DroneSOD-30K, 30,000 images)

The foundation. DroneSOD-30K is the largest detection-only UAV dataset with
explicitly diverse weather conditions. Train the model to find UAVs in single
images across rain, fog, sun, and cloud. This is pure per-frame detection
optimization — no temporal context, no motion priors, just spatial accuracy.

```python
"""Stage 1: UAV appearance — per-frame detection on DroneSOD-30K."""
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # COCO pretrained

model.train(
    data="training/datasets/dronesod.yaml",
    epochs=100,
    imgsz=640,
    batch=16,

    # Optimizer
    optimizer="auto",        # MuSGD (YOLO26 default)
    lr0=0.001,               # fine-tuning, not from scratch
    lrf=0.01,                # final LR = 1e-5
    weight_decay=0.0005,

    # Augmentation — tuned for detection-only training
    mosaic=1.0,              # 4 images per sample — more small objects
    close_mosaic=10,         # disable mosaic last 10 epochs
    mixup=0.1,               # light mixup — diverse weather backgrounds
    scale=0.5,               # simulates range variation
    translate=0.1,           # simulates camera jitter
    hsv_h=0.015,             # hue — weather/lighting
    hsv_s=0.7,               # saturation — cloud cover
    hsv_v=0.4,               # brightness — sun angle
    fliplr=0.5,

    # Training dynamics
    warmup_epochs=3,
    cos_lr=True,
    amp=True,

    # Detection-only
    single_cls=True,         # only "uav"
    rect=False,              # square inputs for small objects

    device=0,
    name="yolo26n_uav_stage1",
    exist_ok=True,
)
```

**Parameter rationale (differences from Ultralytics defaults):**

| Parameter | Default | Ours | Reason |
|-----------|---------|------|--------|
| `lr0` | 0.01 | 0.001 | Fine-tuning COCO weights, not training from scratch. |
| `mixup` | 0.0 | 0.1 | Detection datasets have no temporal correlation — MixUp creates synthetic backgrounds safely. |
| `single_cls` | False | True | Single UAV class. Multi-class loss wastes compute. |
| `rect` | varies | False | Square inputs preserve small-object FOV. |
| `scale` | 0.5 | 0.5 | Default is correct — scale IS range simulation. |

#### Stage 2 — Domain adaptation (Anti-UAV Challenge, 300+ sequences → detection frames)

The Anti-UAV dataset provides UAVs against sky and clutter backgrounds at
various ranges. Frames are extracted from sequences with sequence IDs discarded.
This teaches the model the ground-to-air UAV appearance domain — different
backgrounds, lighting, and target scales than DroneSOD-30K.

```python
"""Stage 2: Domain adaptation — Anti-UAV Challenge as per-frame detection."""
model = YOLO("runs/detect/yolo26n_uav_stage1/weights/best.pt")

model.train(
    data="training/datasets/anti_uav.yaml",
    epochs=100,
    imgsz=640,
    batch=16,

    optimizer="auto",
    lr0=0.0005,              # lower — model already detects UAVs well
    lrf=0.01,                # final LR = 5e-6
    weight_decay=0.0005,

    # Augmentation — moderate, domain shift is from detection→UAV-sky
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.05,              # VERY light — sky backgrounds are consistent
    scale=0.5,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,

    warmup_epochs=3,
    cos_lr=True,
    amp=True,
    single_cls=True,
    rect=False,

    device=0,
    name="yolo26n_uav_stage2",
    exist_ok=True,
)
```

**Stage 1 → 2 changes:**

| Parameter | Stage 1 | Stage 2 | Reason |
|-----------|---------|---------|--------|
| `lr0` | 0.001 | 0.0005 | Model already detects UAVs. Lower LR prevents overfitting. |
| `mixup` | 0.1 | 0.05 | Reduced — sky backgrounds are more consistent than diverse weather. |

#### Stage 3 — Air-to-air domain (UAV-Anti-UAV, 1,810 videos → detection frames)

THE critical stage. UAV-Anti-UAV is the only dataset where both camera and
target are flying UAVs. Frames are extracted from pursuer-UAV video with
sequence IDs discarded. This teaches the model the dual-motion domain — the
target UAV seen from a moving platform against sky backgrounds.

```python
"""Stage 3: Air-to-air domain — UAV-Anti-UAV as per-frame detection."""
model = YOLO("runs/detect/yolo26n_uav_stage2/weights/best.pt")

model.train(
    data="training/datasets/uav_anti_uav.yaml",
    epochs=100,
    imgsz=640,
    batch=16,

    optimizer="auto",
    lr0=0.0005,              # same as Stage 2 — refining domain, not learning new class
    lrf=0.01,
    weight_decay=0.0005,

    # Augmentation — REDUCED because domain shift is within UAV detection
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.0,               # NO MixUp — air-to-air backgrounds are sky
    scale=0.3,               # REDUCED — air-to-air ranges are more consistent
    translate=0.2,           # HIGHER — dual-motion means larger frame-to-frame shifts
    hsv_h=0.01,              # reduced — sky has less hue variation
    hsv_s=0.5,
    hsv_v=0.3,
    fliplr=0.5,

    warmup_epochs=3,
    cos_lr=True,
    amp=True,
    single_cls=True,
    rect=False,

    device=0,
    name="yolo26n_uav_stage3",
    exist_ok=True,
)
```

**Stage 2 → 3 changes and why:**

| Parameter | Stage 2 | Stage 3 | Reason |
|-----------|---------|---------|--------|
| `lr0` | 0.0005 | 0.0005 | Same rate — both are domain refinement, not new concept learning. |
| `mixup` | 0.05 | 0.0 | Air-to-air is exclusively sky backgrounds. MixUp with ground scenes adds noise. |
| `scale` | 0.5 | 0.3 | Ground-to-air has diverse ranges (UAVs at 50m and 500m). Air-to-air engagements have more consistent closing geometries. |
| `translate` | 0.1 | 0.2 | Both platforms are moving. Higher translation teaches the model to handle dual-motion frame-to-frame displacement. |
| `hsv_h` | 0.015 | 0.01 | Sky has less hue variation than ground scenes (buildings, trees, roads). |

#### Stage 4 — P2 small-object head (ONLY if warranted by Stage 3 results)

Adds a stride-4 P2 detection head for objects down to ~8×8 px. Only run if
Stage 3 benchmark shows recall below 70% at 8-16 px target sizes.

```python
"""Stage 4: P2 small-object head (conditional on Stage 3 recall at <16px)."""
model = YOLO("yolo26n-p2.yaml").load(
    "runs/detect/yolo26n_uav_stage3/weights/best.pt"
)

model.train(
    data="training/datasets/uav_anti_uav.yaml",
    epochs=50,
    imgsz=640,
    batch=12,                # REDUCED — P2 needs more GPU memory per sample

    optimizer="auto",
    lr0=0.001,               # higher — P2 head trains from scratch
    lrf=0.01,
    weight_decay=0.0005,

    mosaic=1.0,
    close_mosaic=5,          # shorter — P2 benefits from mosaic longer
    mixup=0.0,
    scale=0.3,
    translate=0.2,
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
    fliplr=0.5,

    warmup_epochs=2,
    cos_lr=True,
    amp=True,
    single_cls=True,

    device=0,
    name="yolo26n_uav_p2",
    exist_ok=True,
)
```

**P2 overhead vs benefit:**

| Metric | YOLO26n | YOLO26n-P2 | Delta |
|--------|---------|------------|-------|
| Parameters | 2.57M | 2.66M | +3.5% |
| GFLOPs | 6.1 | 9.5 | +55% |
| Detection head strides | P3/8, P4/16, P5/32 | P2/4, P3/8, P4/16, P5/32 | +1 head |
| Minimum detectable target | ~16×16 px | ~8×8 px | 2× range extension |
| NPU inference (est.) | ~14ms | ~17ms | +20% |

The P2 head adds a detection branch at 4× downsampling from the input.
In the architecture YAML, this is a single additional upsample + concat +
C3k2 block plus the corresponding Detect output. The backbone is unchanged.

#### Stage 5 — Instance specialization (your own quadrotor footage)

Fine-tune on 500-1000 annotated frames of your specific target UAV at
engagement-relevant ranges. This is the final polish — the model already
detects UAVs in air-to-air scenarios; this stage narrows to YOUR UAV.

```python
"""Stage 5: Instance specialization on custom quadrotor footage."""
model = YOLO("runs/detect/yolo26n_uav_stage3/weights/best.pt")

model.train(
    data="training/datasets/quadrotor_custom.yaml",
    epochs=30,
    imgsz=640,
    batch=8,                 # smaller batch for small custom dataset

    optimizer="auto",
    lr0=0.0002,              # VERY low — just nudging toward your specific UAV
    lrf=0.01,
    weight_decay=0.0005,

    # Augmentation — REDUCED for small custom dataset
    mosaic=0.5,              # reduced — small dataset, less synthetic diversity needed
    close_mosaic=5,
    mixup=0.0,
    scale=0.2,
    translate=0.1,
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.2,
    fliplr=0.5,

    warmup_epochs=1,
    cos_lr=True,
    amp=True,
    single_cls=True,

    device=0,
    name="yolo26n_quadrotor_final",
    exist_ok=True,
)
```

**Why augmentation is reduced for custom data:** A small dataset (500-1000
frames) has limited viewpoint diversity. Heavy augmentation would generate
synthetic training samples that may not represent real engagement conditions.
Lower `mosaic=0.5`, `scale=0.2`, and reduced HSV augmentation keep the model
close to the real data distribution while still providing some regularization.

### 4.6 Conversion scripts for datasets

Scripts to convert all datasets to YOLO detection format. Tracking datasets
have sequence IDs stripped — each frame becomes an independent sample.

```
training/scripts/
    convert_dronesod.py       # DroneSOD-30K → YOLO txt (detection format)
    convert_anti_uav.py       # Anti-UAV JSON → YOLO txt (strips sequence IDs)
    convert_uav_anti_uav.py   # UAV-Anti-UAV → YOLO txt (strips sequence IDs)
```

### 4.7 RKNN export

After training, export to RKNN INT8 on an x86 Linux machine:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/yolo26n_quadrotor_final/weights/best.pt")

# INT8 quantized RKNN for RK3588
model.export(
    format="rknn",
    int8=True,
    data="training/datasets/uav_anti_uav.yaml",  # calibration dataset
    imgsz=640,
)
# Output: yolo26n_quadrotor_final_int8.rknn
```

One function call. Ultralytics handles ONNX → RKNN internally using
`rknn-toolkit2`. The calibration dataset is sampled from the data YAML.

---

## 5. Dataset Loaders for Benchmarking

### 5.1 Existing loader

`benchmark/datasets/anti_uav.py` — already implemented. Supports Anti-UAV
challenge format (IR + RGB, JSON labels per sequence). No changes needed.

### 5.2 New loaders

**UAV-Anti-UAV** (`benchmark/datasets/uav_anti_uav.py`):

Follows the same `BaseSequence` / `BaseDataset` pattern. 1,810 videos with
bounding box annotations, language prompts, and tracking attributes. Primary
benchmark dataset — this IS the air-to-air domain.

**DroneSOD-30K** (`benchmark/datasets/dronesod.py`):

30,000 annotated images (detection-only, no track identities). For benchmark,
treat each image as a 1-frame sequence: initialize on ground-truth bbox,
produce one TrackResult. Useful for evaluating detector recall at various
target sizes, but not for tracking continuity metrics.

**Dataset loader registration:**

All loaders register in a module-level dict so the benchmark runner can
discover them by name. Same pattern as the existing `AntiUAVDataset`.

---

## 6. Integration with QuadGuide

### 6.1 Architecture: separate `quadtrack` package

The tracker code lives in QuadTrack for dev and benchmarking. For QuadGuide
deployment, it becomes a standalone package `quadtrack` installed on the SBC.

Rationale: QuadTrack has PyTorch, Ultralytics, dataset loaders. QuadGuide
has systemd services, SHM IPC, UART link. The tracker algorithm is the same;
the dependencies and runtime context are different. A separate package keeps
QuadGuide dependency-light (only `numpy`, `opencv-python`, and the RKNN
runtime).

### 6.2 `quadtrack` package structure

```
quadtrack/
    pyproject.toml
    src/quadtrack/
        __init__.py           # exports YOLOTracker
        _detector.py          # YOLODetector (same as QuadTrack version)
        _associator.py        # SOTAssociator (same as QuadTrack version)
        _preprocess.py        # frame crop, resize, letterbox
        _postprocess.py       # bbox coordinate conversion
        tracker.py            # YOLOTracker → satisfies QuadGuide protocol
    models/
        yolo26n_uav.rknn      # (git-ignored, deployed separately to /opt/quadguide/models/)
```

### 6.3 QuadGuide protocol compliance

The `quadtrack.YOLOTracker` satisfies the QuadGuide structural protocol
defined in the tracker refactor spec (2026-05-29):

| Method | Implementation |
|--------|---------------|
| `name()` | Returns `"yolo_sot"` |
| `init(frame, bbox)` | Forces TRACKING at bbox. Converts QuadGuide normalized coords (0-1) to pixel coords. |
| `update(frame)` | Returns object with `.bbox` (normalized 0-1), `.confidence` (0-1), `.health` ("nominal"/"uncertain"/"lost"/"no_lock") |
| `reset()` | Clears track, returns to SEARCHING |
| `close()` | Releases NPU resources |

### 6.4 QuadGuide config

One-line change in `configs/config.yaml`:

```yaml
tracker:
  import: quadtrack:YOLOTracker
  params:
    model_path: /opt/quadguide/models/yolo26n_uav.rknn
    input_size: [640, 640]
    conf_threshold: 0.25
    crop_scale: 3.0
    full_frame_every: 10
    max_lost: 10
```

No changes to `tracker_worker.py`, the bus, or any other QuadGuide module.

### 6.5 Blind-fire mode

The tracker starts in SEARCHING on construction. Full-frame detection runs
every frame until a UAV is detected, then auto-initializes. No `lockon/cmd`
message is needed.

If a `lockon/cmd` arrives (manual target selection), `init()` is called and
overrides the auto-detected track. If the cmd has zero-size bbox, `reset()`
is called → back to SEARCHING.

QuadGuide blind-fire launch works without changes — don't send `lockon/cmd`,
and the tracker auto-acquires.

---

## 7. Implementation Phases

### Phase 1: Training pipeline + model validation (5-7 days)

**Goal:** Trained YOLO26n model through Stage 1-3, validated on all datasets.

**Deliverable:** `yolo26n_uav_stage3.pt` with benchmark metrics.

**Steps:**
1. Download DroneSOD-30K, Anti-UAV Challenge, and UAV-Anti-UAV datasets.
2. Convert all datasets to YOLO detection format:
   - DroneSOD-30K: detection format (already per-frame images).
   - Anti-UAV Challenge: extract frames from video, discard sequence IDs.
   - UAV-Anti-UAV: extract frames from video, discard sequence IDs.
3. Run Stage 1 training (DroneSOD-30K, 100 epochs, ~4 hours on GPU).
4. Run Stage 2 training (Anti-UAV Challenge, 100 epochs, ~4 hours).
5. Run Stage 3 training (UAV-Anti-UAV, 100 epochs, ~4 hours).
6. Validate at each stage: mAP, recall-at-size, FPS on dev machine (ONNX CPU).
7. Decision gate for Stage 4: if Stage 3 recall at 8-16 px targets < 70%,
   proceed to P2 training.

**Dependencies:** GPU with 8GB+ VRAM, Ultralytics 8.4.0+, all datasets downloaded.
**Fallback if UAV-Anti-UAV unavailable:** Stop after Stage 2. Anti-UAV Challenge
alone provides ground-to-air domain adaptation. Air-to-air fine-tuning is
preferred but not blocking.

### Phase 2: Detector + associator implementation (2-3 days)

**Goal:** YOLOTracker satisfying BaseTracker, testable in QuadTrack live harness.

**Deliverable:** `testing/main.py` shows real-time tracking with YOLO26n.

**Steps:**
1. Implement `detector.py` (ONNX backend for dev, RKNN stub for later).
2. Implement `associator.py` (Kalman + IoU, ~40 lines).
3. Implement `tracker.py` (state machine with SEARCHING/TRACKING/LOST, BaseTracker subclass).
4. Register in factory, add config.
5. Test with `testing/main.py` using webcam or recorded video.

### Phase 3: Benchmark on Anti-UAV datasets (2-3 days)

**Goal:** Quantitative results: success rate, precision, FPS, latency.

**Deliverable:** Benchmark report on all datasets.

**Steps:**
1. Implement `benchmark/datasets/uav_anti_uav.py` loader (if not already done).
2. Run benchmark runner with `yolo_sot` tracker on Anti-UAV Challenge and
   UAV-Anti-UAV datasets.
3. Compare against existing trackers (NanoTrack, KCF, MOSSE).
4. Analyze failure cases: bin detections by target size (0-8, 8-16, 16-32,
   32-64, 64+ px) and measure recall per bin. This data drives the Stage 4
   (P2) decision.

### Phase 4: RKNN export + SBC deployment (2-3 days)

**Goal:** INT8 RKNN model running on RK3588, benchmarked on-device.

**Deliverable:** `yolo26n_uav.rknn` with on-device FPS + accuracy metrics.

**Steps:**
1. Export trained model to RKNN INT8 (`model.export(format='rknn', int8=True)`).
2. Copy `.rknn` to SBC, test with standalone inference script.
3. Validate accuracy vs ONNX (expect <1% mAP drop from INT8 quantization).
4. Measure NPU inference time (target: <15ms at 640×640, <8ms on crop).

### Phase 5: QuadGuide integration (1 day)

**Goal:** `quadtrack` package on SBC, QuadGuide HIL tests passing.

**Deliverable:** QuadGuide running with `tracker.import: quadtrack:YOLOTracker`.

**Steps:**
1. Extract `quadtrack` package from QuadTrack code.
2. `pip install` on SBC.
3. Change one config line in QuadGuide.
4. Run HIL test suite with `constant_vel` scenario.
5. Verify blind-fire auto-acquisition.

### Phase 6: Optimization + hardening (3-5 days)

**Goal:** 50+ FPS real-world tracking with auto-recovery.

**Deliverable:** Production tracker on Orange Pi 5.

**Work:**
- Crop size tuning (balance NPU speed vs search coverage).
- Full-frame interval tuning (every 5 vs 10 vs 20 frames).
- Kalman parameter tuning with real engagement data.
- Multi-scale test: detection range through terminal.
- Re-acquisition latency measurement.

---

## 8. Key Design Decisions

### 8.1 Why YOLO26n over YOLOv8n

- NMS-free: simpler deployment, no CPU post-processing variability.
- DFL-free: lighter head, faster inference (43% CPU speedup reported).
- STAL: better small-object coverage during training.
- Built-in RKNN export in Ultralytics.
- P2 variant available for sub-20px targets if needed.
- YOLOv8n is the fallback if RKNN export issues arise — same Ultralytics API.

### 8.2 Why Kalman + IoU, not ByteTrack/OC-SORT/DeepSORT

- Single target: O(N) max-IoU, not O(N³) Hungarian.
- No ReID model: saves 2-5ms per frame, no second model to quantize.
- No camera motion compensation: the camera IS on the interceptor — target
  motion in image coords IS the guidance signal.
- ~40 lines, ~5 microseconds per frame.

### 8.3 Why fine-tune COCO weights, not train from scratch

- YOLO26n COCO backbone has rich visual features (trained on 118K images).
- Anti-UAV datasets are much smaller (1,810 videos = ~100K frames with
  annotations, but limited viewpoint diversity).
- Fine-tuning adapts the existing feature hierarchy to UAV-specific
  appearance while preserving general edge/texture/shape detectors.
- Training from scratch on UAV-only data risks overfitting to limited
  backgrounds and viewpoints.

### 8.4 Why P2 variant is conditional (not default)

- YOLO26n-P2 has no COCO pretrained weights available. However, the Stage 4
  recipe loads the Stage 3 weights into the P2 architecture
  (`YOLO("yolo26n-p2.yaml").load("stage3/best.pt")`). The backbone weights
  transfer directly; only the new P2 neck/head branches are randomly initialized
  and trained. This is far cheaper than training from scratch (~50 epochs vs
  ~300) and leverages the existing UAV-tuned backbone.
- The standard YOLO26n with stride-8 detection head detects objects down to
  ~16×16 px reliably. At 1080p/90° HFOV, that's a UAV at ~500m — well within
  blind-fire acquisition range.
- P2 adds a stride-4 head for ~8×8 px detection. This extends detection range
  to ~1000m but costs 55% more GFLOPs and ~20% more inference time.
- Decision: start with standard YOLO26n (Stages 1-3). Only add P2 (Stage 4)
  if Stage 3 benchmark shows recall below 70% at 8-16 px target sizes.

### 8.5 Why separate QuadTrack training code from QuadGuide deployment code

QuadTrack has PyTorch, Ultralytics, GPU training, dataset loaders, benchmark
metrics. QuadGuide has systemd services, SHM IPC, UART link, runs headless on
ARM. The tracker algorithm is identical, but dependencies differ. A separate
`quadtrack` package means QuadGuide never imports PyTorch or Ultralytics.

---

## 9. Performance Targets

| Metric | Target | Measured on |
|--------|--------|-------------|
| Detector NPU latency (full frame 1080p) | <15ms | RK3588 single NPU core |
| Detector NPU latency (300×300 crop) | <8ms | RK3588 single NPU core |
| Associator latency | <0.1ms | RK3588 CPU |
| End-to-end tracker latency (crop mode) | <10ms | RK3588 |
| End-to-end tracker latency (full-frame mode) | <18ms | RK3588 |
| Tracking FPS (crop mode, real-world) | >80 FPS | RK3588 |
| Tracking FPS (sustained, with periodic full-frame) | >50 FPS | RK3588 |
| Detection recall @ 20×20 px target | >90% | UAV-Anti-UAV test set |
| Detection recall @ 10×10 px target | >70% | UAV-Anti-UAV test set |
| INT8 quantization accuracy drop | <2% mAP | vs FP16 ONNX |
| Re-acquisition latency after track loss | <200ms | After next full-frame scan |

---

## 10. Open Questions

1. **UAV-Anti-UAV dataset availability.** The paper (arXiv:2512.07385) says
   "dataset and codes will be available at https://github.com/983632847/
   Awesome-Multimodal-Object-Tracking". Confirm the dataset is downloadable
   before Phase 1 Stage 3. Fallback: Stages 1-2 (DroneSOD-30K + Anti-UAV
   Challenge) provide a strong ground-to-air model. Air-to-air fine-tuning
   is the ideal final domain adaptation but the system is functional without it.

2. **YOLO26n vs YOLO26n-P2 for our target sizes.** After Stage 3, measure
   the distribution of target sizes in the UAV-Anti-UAV test set. If >20% of
   targets are <16×16 px, invest in Stage 4 P2 training. If most targets
   are >16×16 px, standard YOLO26n stride-8 head is sufficient. The P2 head
   costs 55% more GFLOPs — only pay for it if the data says you need it.

3. **Frame sampling for tracking datasets.** Stages 2 and 3 convert tracking
   sequences to detection format. Using every frame may retain temporal
   correlations (consecutive frames are nearly identical). Consider sampling
   every 2nd or 5th frame from tracking sequences to decorrelate training
   samples. This is a hyperparameter to tune — start with every frame, reduce
   if validation shows overfitting to temporal smoothness.

4. **Multi-class vs single-class detector.** Single-class ("uav") is simpler
   and faster. If you need to distinguish friendly vs hostile UAVs, or UAV
   types, add a second class. Multi-class with two visually similar classes
   (quadrotor vs fixed-wing) is harder than UAV vs background — consider
   whether classification is better done as a separate stage after detection.

5. **Thermal support.** The Anti-UAV datasets include IR. A separate IR model
   (`yolo26n_uav_ir.rknn`) can be trained on IR-only data and selected at
   runtime based on sensor mode. Same architecture, different weights.
   Out of scope for Phase 1 but the training pipeline is identical.
