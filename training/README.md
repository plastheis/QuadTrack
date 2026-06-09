# QuadTrack Training Pipeline

All stages train YOLO11s as a per-frame detector using Ultralytics on PyTorch
CUDA. Tracking datasets are converted to detection format — sequence IDs are
discarded and each frame becomes an independent sample. The Kalman filter in
the associator handles temporal continuity at inference time; the model learns
only spatial accuracy.

## Quick Start

```bash
cd /home/plas/Projects/QuadTrack
source .venv/bin/activate

# Download and convert datasets (see DATASETS section below)
python scripts/convert_anti_uav.py --input-dir ... --output-dir ...
python scripts/convert_uav_anti_uav.py --input-dir ... --output-dir ...

# Train all stages sequentially
python training/stage1_detfly.py
python training/stage2_anti_uav.py
python training/stage3_uav_anti_uav.py

# Optional: P2 small-object head (only if Stage 3 recall at <16px is <70%)
python training/stage4_p2.py

# Optional: fine-tune on custom footage
python training/stage5_custom.py --data training/datasets/quadrotor_custom.yaml

# Export for SBC deployment
python training/export_rknn.py --weights runs/detect/stage5_custom/weights/best.pt
```

## Dataset Setup

Place datasets under `/data/quadtrack_datasets/` (override with `QUADTRACK_DATA`
environment variable). Each dataset needs a YAML config in `training/datasets/`.

```bash
export QUADTRACK_DATA=/data/quadtrack_datasets
```

### Det-Fly (13K images, air-to-air detection)

| | |
|---|---|
| Paper | IEEE RA-L 2021 |
| Size | 13,000+ images, DJI Mavic 2 filming another UAV |
| Domain | Air-to-air — sky, urban, field, mountain backgrounds |
| Download | [OneDrive (annotations)](https://westlakeu-my.sharepoint.com/:f:/g/personal/zhengye_westlake_edu_cn/Em9kWPCMYm1KpFHnhVOTn9UBQfQ25m0lH67xIcULuYXYHw) + [OneDrive (images)](https://westlakeu-my.sharepoint.com/:f:/g/personal/zhengye_westlake_edu_cn/EqFYguroD9lEnVDTRkyoOJQBrHvndTQGa5f8EQurGRFUqQ) |
| Baidu mirror | [Annotations](https://pan.baidu.com/s/1v30h53xu0PmgwnarAaFK7A) (pwd: `4dcc`) + [Images](https://pan.baidu.com/s/1A6D9j_PzcWLeYe9GQttypQ) (pwd: `qjyt`) |
| Convert | Format likely VOC/COCO — write `scripts/convert_detfly.py` to produce YOLO txt |

### Anti-UAV Challenge (300-600 sequences, ground-to-air)

| | |
|---|---|
| Paper | 3rd Anti-UAV Workshop & Challenge (CVPR 2023) |
| Size | Anti-UAV300/410/600 variants, IR + RGB |
| Domain | Ground-to-air, static and moving cameras |
| Download | [Google Drive (300)](https://drive.google.com/file/d/1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr/view) or [Baidu (410)](https://pan.baidu.com/s/1PbINXhxc-722NWoO8P2AdQ) (pwd: `wfds`) or [ModelScope (600)](https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV/files) |
| Convert | `python scripts/convert_anti_uav.py --input-dir ... --output-dir ... --modality RGB` |

### UAV-Anti-UAV (1,820 videos, air-to-air)

| | |
|---|---|
| Paper | arXiv:2512.07385 |
| Size | 1,820 videos (1,400 train, 420 test), multi-modal |
| Domain | Air-to-air — pursuer UAV tracking target UAV, moving platform |
| Download | [Google Drive](https://drive.google.com/drive/folders/1Rvd7HcYirOEclB1xcnPNA_mL3fpvwgWI) or [Baidu](https://pan.baidu.com/s/139xn-nKY4KbTOupCn2XDyg) (pwd: `UAVU`) |
| Convert | `python scripts/convert_uav_anti_uav.py --input-dir ... --output-dir ...` |

## Training Stages

All stages use `optimizer=auto`, `cos_lr=True`, `amp=True`,
`single_cls=True`, `rect=False`, `imgsz=640`. All are fine-tuning from
COCO-pretrained YOLO11s, not training from scratch.

### Stage 1 — UAV Appearance (Det-Fly)

```bash
python training/stage1_detfly.py
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 100 | |
| batch | 16 | |
| lr0 | 0.001 | Fine-tuning from COCO, not from scratch |
| lrf | 0.01 | Final LR = 1e-5 |
| weight_decay | 0.0005 | |
| mixup | 0.1 | Detection dataset — no temporal correlation, MixUp safe |
| scale | 0.5 | Range simulation — targets from 5% to 100% of frame |
| translate | 0.1 | |
| hsv_h | 0.015 | |
| hsv_s | 0.7 | |
| hsv_v | 0.4 | |
| mosaic | 1.0 | 4 images per sample — more small objects |
| close_mosaic | 10 | Disable mosaic last 10 epochs |
| fliplr | 0.5 | |
| warmup_epochs | 3 | |

### Stage 2 — Domain Adaptation (Anti-UAV Challenge)

```bash
python training/stage2_anti_uav.py
```

Weights loaded from: `runs/detect/stage1_detfly/weights/best.pt`

| Parameter | Value | Change from Stage 1 |
|-----------|-------|---------------------|
| epochs | 100 | |
| batch | 16 | |
| lr0 | 0.0005 | Lower — model already detects UAVs |
| lrf | 0.01 | |
| weight_decay | 0.0005 | |
| mixup | 0.05 | Reduced — sky backgrounds are more consistent than diverse weather |
| scale | 0.5 | |
| translate | 0.1 | |
| hsv_h | 0.015 | |
| hsv_s | 0.7 | |
| hsv_v | 0.4 | |
| mosaic | 1.0 | |
| close_mosaic | 10 | |
| fliplr | 0.5 | |
| warmup_epochs | 3 | |

### Stage 3 — Air-to-Air Domain (UAV-Anti-UAV)

```bash
python training/stage3_uav_anti_uav.py
```

Weights loaded from: `runs/detect/stage2_anti_uav/weights/best.pt`

| Parameter | Value | Change from Stage 2 |
|-----------|-------|---------------------|
| epochs | 100 | |
| batch | 16 | |
| lr0 | 0.0005 | Same — refining domain, not learning new concept |
| lrf | 0.01 | |
| weight_decay | 0.0005 | |
| mixup | 0.0 | No MixUp — air-to-air backgrounds are sky, blending with ground adds noise |
| scale | 0.3 | Reduced — air-to-air ranges are more consistent |
| translate | 0.2 | Higher — dual-motion means larger frame-to-frame shifts |
| hsv_h | 0.01 | Reduced — sky has less hue variation |
| hsv_s | 0.5 | |
| hsv_v | 0.3 | |
| mosaic | 1.0 | |
| close_mosaic | 10 | |
| fliplr | 0.5 | |
| warmup_epochs | 3 | |

**Decision gate after Stage 3:** Run benchmark on UAV-Anti-UAV test set.
Bin detections by target size (8-16, 16-32, 32-64, 64+ px). If recall at
8-16 px is below 70%, proceed to Stage 4.

### Stage 4 — P2 Small-Object Head (CONDITIONAL)

```bash
python training/stage4_p2.py
```

Weights loaded from: `runs/detect/stage3_uav_anti_uav/weights/best.pt`
Architecture: `yolo11s-p2.yaml` (adds stride-4 P2 detection head).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 50 | Shorter — backbone is frozen, only training new P2 head |
| batch | 12 | Reduced — P2 needs more GPU memory |
| lr0 | 0.001 | Higher — P2 head trained from scratch |
| lrf | 0.01 | |
| weight_decay | 0.0005 | |
| mixup | 0.0 | |
| scale | 0.3 | |
| translate | 0.2 | |
| hsv_h | 0.01 | |
| hsv_s | 0.5 | |
| hsv_v | 0.3 | |
| mosaic | 1.0 | |
| close_mosaic | 5 | Shorter — P2 benefits from mosaic longer |
| fliplr | 0.5 | |
| warmup_epochs | 2 | |

**P2 overhead:** +55% GFLOPs (6.1 → 9.5), +20% NPU inference time, minimum
detectable target ~8×8 px (vs ~16×16 px without P2).

### Stage 5 — Instance Specialization (OPTIONAL)

```bash
python training/stage5_custom.py --data training/datasets/quadrotor_custom.yaml
```

Weights loaded from: `runs/detect/stage3_uav_anti_uav/weights/best.pt`
(or Stage 4 if P2 was used)

Requires 500-1000 annotated frames of your specific target UAV at
engagement-relevant ranges.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 30 | |
| batch | 8 | Smaller — custom datasets are small |
| lr0 | 0.0002 | Very low — just nudging toward your specific UAV |
| lrf | 0.01 | |
| weight_decay | 0.0005 | |
| mixup | 0.0 | |
| scale | 0.2 | Reduced for small custom dataset |
| translate | 0.1 | |
| hsv_h | 0.01 | |
| hsv_s | 0.3 | Reduced |
| hsv_v | 0.2 | Reduced |
| mosaic | 0.5 | Reduced — small dataset, less augmentation needed |
| close_mosaic | 5 | |
| fliplr | 0.5 | |
| warmup_epochs | 1 | |

## Export to RKNN

```bash
python training/export_rknn.py \
    --weights runs/detect/stage5_custom/weights/best.pt \
    --data training/datasets/uav_anti_uav.yaml
```

Uses Ultralytics built-in RKNN export with INT8 quantization. Must be run on
an x86 Linux machine with `rknn-toolkit2` installed. The calibration dataset
is automatically sampled from the data YAML.

Output: `models/yolo/yolo11s_uav_int8.rknn`

## YOLO11 Automatic Features

These are baked into the architecture and activate automatically — no
configuration needed:

- **MuSGD optimizer** — hybrid Muon+SGD, selected when `optimizer="auto"`
- **Progressive Loss** — shifts supervision from auxiliary one-to-many head
  to inference one-to-one head during training
- **STAL label assignment** — guarantees positive coverage for small objects

## Hardware

- **Training:** NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **Each stage:** ~4 hours on RTX 3070 at batch=16, imgsz=640
- **Inference:** RK3588 NPU, INT8 quantized, ~73 FPS full-frame, ~100+ FPS crop mode

## Single-Class Training

All datasets use a single class `uav`. The model learns a general "UAV-ness"
feature — it fires on quadcopters, fixed-wing, hexacopters, and any other
UAV type present in the training data. Multi-class training (separating UAV
types) would reduce recall and is not needed for the interceptor use case.

## Rerunning a Stage

Every training script accepts CLI overrides for all hyperparameters:

```bash
python training/stage2_anti_uav.py \
    --data training/datasets/anti_uav_v2.yaml \
    --epochs 150 \
    --lr0 0.0003 \
    --batch 8
```

Run `python training/stage2_anti_uav.py --help` for all options.
