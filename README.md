# QuadTrack

Exploratory workspace for developing and benchmarking vision-based tracker fusion strategies targeting SBC edge devices. The goal is to find a fusion configuration that is accurate and efficient enough to run onboard a drone.

Pratical implementation for proportional navigation and flight controller integration lives in the separate [quadguide](https://github.com/plastheis/quadguide) repo. QuadTrack feeds validated tracker designs into that system.

## Architecture

Five modules with distinct responsibilities:

- **tracker/** — Core tracking engine. `camera.py` handles frame capture; `kcf_tracker.py` and `nanotrack_tracker.py` implement two tracking algorithms (KCF and NanoTrack respectively); `fusion.py` fuses multiple tracking inputs; `utils.py` holds shared helpers.
- **ground-station/** — Operator-facing layer. `gui.py` is the control interface; `stream.py` handles video/telemetry streaming.
- **benchmarks/** — Offline tooling. `benchmark.py` evaluates tracker performance; `calibrate-camera.py` calibrates camera intrinsics.
- **models/** — Pre-trained weights. `models/YOLO/` for YOLO detection models; `models/nanotrack/` for NanoTrack ONNX weights (git-ignored via `*.onnx`).

## Data Flow

```
Camera frames → tracker/camera.py → kcf_tracker.py ──────────────────┐
                                  → nanotrack_tracker.py ─────────────→ fusion.py → ground-station/{stream,gui}.py → Operator
───────────────────┘
```

## Configuration

`config.yaml` at the repo root is the single configuration file for runtime parameters (camera index, serial port, tracker selection, etc.).

## Dependencies

- Python 3
- OpenCV (`cv2`)
- PySerial
- ONNX Runtime

## Changelog

### 2026-04-07
- **Added `nanotrack_accel` tracker** (`trackers/nanotrack_accel_tracker.py`) — full NanoTrack inference via ONNX Runtime, bypassing `cv2.TrackerNano` to enable explicit device control:
  - `device: cuda` → ONNX Runtime `CUDAExecutionProvider` (requires `onnxruntime-gpu`)
  - `device: npu` → RKNN via `rknnlite` (on Rockchip SBC) or `rknn-toolkit2` (PC simulation); auto-falls back to CPU if RKNN libraries are absent
  - `device: cpu` → ONNX Runtime `CPUExecutionProvider`
  - Reuses existing `nanotrack_backbone_sim.onnx` / `nanotrack_head_sim.onnx` for CUDA/CPU; separate `.rknn` files (converted externally) for NPU
  - Select via `config.yaml`: `tracker.algorithm: nanotrack_accel`
