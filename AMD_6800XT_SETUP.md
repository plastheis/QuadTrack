# Switching Training to AMD Radeon RX 6800 XT

The RX 6800 XT (16 GB VRAM, RDNA 2 / gfx1030) is supported via ROCm on Linux.
Compared to the RTX 3070 (8 GB), it offers 2× the VRAM and roughly 2× the
compute throughput, halving per-stage training time.

## Prerequisites

- Physical card swap (power off → swap → boot)
- Ubuntu 24.04 (Noble)
- AMDGPU kernel module loads automatically on boot when the card is present

---

## 1. Install ROCm 6.2.4

PyTorch ROCm wheels target ROCm 6.x — the system ROCm 7.2 must be downgraded.

```bash
# Remove existing ROCm 7.2
sudo apt remove --purge rocm-* hip-* miopen-* miopengemm-* rccl-* half-*

# Install ROCm 6.2.4
wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/noble/amdgpu-install_6.2.60204-1_all.deb
sudo apt install ./amdgpu-install_6.2.60204-1_all.deb
sudo amdgpu-install -y --usecase=rocm
sudo usermod -aG render,video $USER
```

**Log out and back in** for the group changes to take effect.

## 2. Verify GPU Detection

```bash
rocminfo | grep -E "Name:|VRAM|gfx"
# Expected: Name: gfx1030, VRAM: 16368 MB
```

## 3. Install ROCm PyTorch

```bash
cd /home/plas/Projects/QuadTrack
source .venv/bin/activate

# Remove CUDA PyTorch
pip uninstall torch torchvision torchaudio -y

# Install ROCm build
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/rocm6.2
```

## 4. Verify

```bash
python -c "
import torch
print(f'PyTorch:     {torch.__version__}')
print(f'ROCm usable: {torch.cuda.is_available()}')
print(f'GPU:         {torch.cuda.get_device_name(0)}')
print(f'VRAM:        {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'Arch:        {torch.cuda.get_device_capability(0)}')
"
```

**Note:** ROCm presents itself as CUDA to PyTorch — `torch.cuda.is_available()` returns `True`
and Ultralytics sees it as a standard CUDA device.

## 5. Train

No code changes needed — Ultralytics works identically with ROCm:

```bash
python training/scripts/stage1_detfly.py --batch 24   # up from 8 on 3070
python training/scripts/stage2_uav_anti_uav.py --batch 24
```

## Performance Comparison

| | RTX 3070 | RX 6800 XT |
|---|---|---|
| **VRAM** | 8 GB | 16 GB |
| **FP16 TFLOPS** | 20.3 | 41.5 |
| **Typical batch (P2)** | 8 | 16–24 |
| **Stage 1 time** | ~4 hrs | ~2 hrs |
| **Stage 2 time** | ~4 hrs | ~2 hrs |

## Troubleshooting

### `rocminfo` shows no GPU
- Verify the card is physically seated and power cables connected
- Check kernel module: `lsmod | grep amdgpu`
- Check dmesg: `dmesg | grep -i amdgpu`

### `torch.cuda.is_available()` is `False`
- Verify ROCm PyTorch was installed (not CUDA build): `pip show torch | grep rocm`
- Check `$LD_LIBRARY_PATH` includes `/opt/rocm/lib`
- ROCm user groups: `groups $USER` should include `render` and `video`

### OOM at batch 24
- Drop to `--batch 16`
- The P2 head uses extra VRAM; standard YOLO11s would fit larger batches

## Switching Back to NVIDIA

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```
