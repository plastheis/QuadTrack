# Neural Network Trackers for Edge Hardware: Research Report

**Date:** June 7, 2026
**Context:** Research for QuadTrack — enabling real-time quadrotor tracking on embedded SBCs (Linux, NPU, CSI camera). Target: int8-quantized models deployable via NCNN, ONNX Runtime, TFLite, or CoreML.

---

## 1. Overview of the Landscape

The field of lightweight neural network visual tracking has matured significantly since 2021. The progression has been:

1. **Siamese trackers with handcrafted backbones** (SiamFC 2016 → SiamRPN++ 2018 → SiamCAR 2019 → Ocean 2020)
2. **NAS-designed lightweight trackers** (LightTrack CVPR21, NanoTrack 2021)
3. **Efficient transformer trackers** (E.T.Track 2021, MixFormerV2 2023, DyTrack 2024)
4. **Explicitly quantized edge trackers** (FEAR ECCV22, LightTrack-ncnn, NanoTrack-ncnn)
5. **Token compression / distillation approaches** (ETCTrack 2026, EATrack 2026)

Key trend: the best accuracy used to require heavy backbones (ResNet-50: ~25M params). Now lightweight models (2-8M params) achieve 85-95% of that accuracy while running 10-100x faster.

---

## 2. Detailed Tracker Profiles

### 2.1 NanoTrack (2021)
- **Paper:** Not published on arXiv — GitHub-only release
- **Repo:** https://github.com/HonglinChu/NanoTrack (192 stars)
- **Backbone:** MobileNetV3 (customized)
- **Model Size:** 2.2 MB
- **Framework:** NCNN (Tencent's mobile inference engine)
- **Quantization:** Designed for int8 via NCNN's built-in quantization pipeline
- **Speed:** >120 FPS on Apple M1 CPU
- **Training:** GOT-10k only, 2 hours on RTX 3090
- **Key innovation:** Aggressively pruned Siamese tracker inspired by SiamBAN and LightTrack. Deliberately designed for <3MB footprint.

| Benchmark | Value |
|-----------|-------|
| VOT2018 EAO | 0.311 |
| VOT2019 EAO | 0.247 |
| GOT-10k Val AO | 0.604 |
| GOT-10k Val SR | 0.724 |
| DTB70 Success | 0.532 |
| DTB70 Precision | 0.727 |

**Assessment:** Smallest and fastest of the Siamese-based trackers. The accuracy tradeoff is real — ~20-30% lower than LightTrack on VOT benchmarks. However, at 2.2 MB and >120 FPS on CPU, it represents the extreme efficiency end. Best suited for scenarios where power/perf constraints are absolute.

---

### 2.2 LightTrack (CVPR 2021)
- **Paper:** "LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search"
- **arXiv:** 2104.14545
- **Authors:** Bin Yan, Houwen Peng, Kan Wu, Dong Wang, Jianlong Fu, Huchuan Lu (Microsoft Research)
- **Repo:** https://github.com/researchmm/LightTrack (458 stars)
- **Backbone:** NAS-discovered (MobileNetV3-based search space)
- **Model Size:** 7.7 MB
- **Framework:** PyTorch (research), NCNN port available (https://github.com/Z-Xiong/LightTrack-ncnn, 101 stars)
- **Quantization:** NCNN port supports int8 quantization through Tencent's NCNN toolchain
- **Speed (Snapdragon 845 Adreno GPU):** 12× faster than Ocean, 13× fewer params, 38× fewer FLOPs
- **Training:** SuperNet pre-training on ImageNet + tracking-specific NAS search

| Benchmark | Value |
|-----------|-------|
| VOT2018 EAO | 0.418 |
| VOT2019 EAO | 0.328 |
| GOT-10k Val AO | 0.750 |
| GOT-10k Val SR | 0.877 |
| DTB70 Success | 0.591 |
| DTB70 Precision | 0.766 |

**Assessment:** The gold standard for NAS-designed lightweight trackers. The NCNN port (by Z-Xiong) makes this directly deployable with int8 quantization on ARM/Linux. ~3.5x larger than NanoTrack but with substantially better accuracy (+35% EAO on VOT2018). This is the most battle-tested option for edge deployment — CVPR publication, official Microsoft repo, and a third-party NCNN port with community validation.

---

### 2.3 FEAR (ECCV 2022)
- **Paper:** "FEAR: Fast, Efficient, Accurate and Robust Visual Tracker"
- **arXiv:** 2112.07957
- **Authors:** Vasyl Borsuk, Roman Vei, Orest Kupyn, Tetiana Martyniuk, Igor Krashenyi, Jiři Matas (Ukrainian Catholic University + Czech Technical University)
- **Repo:** https://github.com/PinataFarms/FEARTracker
- **Backbone:** Custom lightweight (three variants: XS, M, L)
- **Variants:**
  - FEAR-XS: 2.4× smaller and 4.3× faster than LightTrack with superior accuracy
  - FEAR-M / FEAR-L: surpass most Siamese trackers
- **Quantization:** FP16 CoreML quantization for iOS demonstrated in repo. Explicitly includes energy consumption benchmarking.
- **Key innovation:** Dual-template representation with single learnable parameter for temporal adaptation. Pixel-wise fusion block.

**Assessment:** The only tracker in this survey that explicitly benchmarks energy consumption on mobile devices. FEAR-XS is the standout for edge deployment — smaller AND faster than LightTrack while being more accurate. The ECCV venue and iOS CoreML demo prove production-readiness. However, no NCNN port exists (CoreML only), so deploying on non-Apple Linux SBCs would require ONNX export + quantization.

---

### 2.4 E.T.Track (2021)
- **Paper:** "Efficient Visual Tracking with Exemplar Transformers"
- **arXiv:** 2112.09686
- **Authors:** Philippe Blatter, Menelaos Kanakis, Martin Danelljan, Luc Van Gool (ETH Zurich)
- **Repo:** https://github.com/pblatter/ettrack
- **Backbone:** Exemplar Transformer (single instance-level attention layer)
- **Speed:** 47 FPS on CPU (8× faster than other transformer trackers)
- **Benchmarks:** Outperforms all lightweight realtime CPU trackers on LaSOT, OTB-100, NFS, TrackingNet, VOT-ST2020

**Assessment:** The first transformer tracker that's actually CPU-viable. At 47 FPS CPU, it bridges the transformer accuracy advantage with practical deployment. Excellent paper quality (ETH Zurich, CVPR Lab). The single-instance attention design is elegant — essentially a "what if we made a transformer that only attends to what matters." No explicit quantization work, but the architecture is simple enough to quantize.

---

### 2.5 MixFormerV2 (2023)
- **Paper:** "MixFormerV2: Efficient Fully Transformer Tracking"
- **arXiv:** 2305.15896
- **Authors:** Yutao Cui, Tianhui Song, Gangshan Wu, Limin Wang (Nanjing University)
- **Backbone:** Fully transformer (no CNN operations)
- **Two variants:**
  - MixFormerV2-B: 70.6% AUC on LaSOT, 165 FPS GPU
  - MixFormerV2-S: surpasses FEAR-L by 2.7% AUC on LaSOT, real-time CPU
- **Key innovation:** Dense-to-sparse + deep-to-shallow distillation. Prediction tokens instead of complex score heads.

**Assessment:** Top accuracy among efficient trackers. The distillation approach (train heavy, deploy light) is the same strategy used by MobileNet and EfficientNet. The "S" variant directly competes with FEAR for the "best CPU tracker" crown. No quantization-specific work in the paper, but the simplicity (pure transformer, no dense convolutions) makes it amenable to int8 quantization.

---

### 2.6 DyTrack (2024)
- **Paper:** "Exploring Dynamic Transformer for Efficient Object Tracking"
- **arXiv:** 2403.17651
- **Authors:** Jiawen Zhu, Xin Chen, Haiwen Diao, Shuai Li, Jun-Yan He, Chenyang Li, Bin Luo, Dong Wang, Huchuan Lu (Dalian University of Technology)
- **Backbone:** Dynamic transformer with early-exit branches
- **Speed:** 64.9% AUC on LaSOT at 256 FPS
- **Key innovation:** Dynamic routing — easy frames exit early, hard frames get more computation. Feature recycling reuses intermediate outputs. Target-aware self-distillation.

**Assessment:** Most innovative approach to the speed-accuracy tradeoff. Instead of making a uniformly small model, DyTrack makes a model that decides how much compute each frame deserves. This is ideal for tracking — most frames are easy (target hasn't changed much), occasional frames are hard (occlusion, fast motion). 256 FPS is fast enough for any drone application.

---

### 2.7 ETCTrack (2026)
- **Paper:** "An Efficient Token Compression Framework for Visual Object Tracking"
- **arXiv:** 2605.08329
- **Authors:** Weijing Wu, Qihua Liang, Bineng Zhong et al.
- **Key result:** Reduces template tokens by 60%, 21.4% MAC reduction, only 0.4% accuracy drop
- **Key innovation:** Adaptive Token Compressor + Hierarchical Interaction Encoder

**Assessment:** State-of-the-art (2026) in the token compression approach. Most relevant for multi-template trackers. If QuadTrack uses temporal templates (multiple historical frames), this approach is directly applicable.

---

### 2.8 EATrack (2026)
- **Paper:** "Dual-branch Distilled Transformer for Efficient Asymmetric UAV Tracking"
- **arXiv:** 2605.28018
- **Authors:** Hongtao Yang, Bineng Zhong et al. (Guangxi Normal University)
- **Key innovation:** Teacher-guided dual-branch distillation for UAV tracking specifically. Spatial + prediction-level distillation.
- **Benchmarks:** Tested on 5 UAV benchmarks

**Assessment:** Most directly relevant to QuadTrack — explicitly designed for UAV/drone tracking. Teacher-student distillation approach means you train a heavy model and deploy a lightweight student. Temporal adaptation module handles the unique challenges of aerial tracking.

---

### 2.9 Lightweight RGB Tracker for AR (2025)
- **Paper:** "Deep Learning-based Lightweight RGB Object Tracking for Augmented Reality Devices"
- **arXiv:** 2511.17508
- **Authors:** Alice Smith, Bob Johnson, Xiaoyu Zhu, Carol Lee
- **Key innovation:** Explicitly combines pruning + quantization + knowledge distillation. Runs at 30 FPS on mobile AR headset.
- **This paper explicitly does what you're asking about — all three compression techniques together.**

---

### 2.10 Other Notable Mentions

| Tracker | Venue | Backbone | Notes |
|---------|-------|----------|-------|
| SiamFC (2016) | ECCVW | AlexNet | Grandparent of all Siamese trackers; ~2.4M params |
| SiamRPN++ (2018) | CVPR | ResNet-50 | Classic, but heavy (~25M params) |
| SiamCAR (2019) | CVPR | ResNet-50 | Anchor-free, simpler architecture |
| Ocean (2020) | ECCV | ResNet-50 | Object-aware anchor-free; baseline for LightTrack comparisons |
| SiamAPN/SiamAPN++ (2021) | ICRA/IROS | AlexNet | UAV-focused but very heavy (118-187 MB!) |
| SiamTPN (2022) | WACV | ShuffleNetV2 | 62 MB, decent GOT-10k (0.728 AO) |
| HiFT (2021) | ICCV | - | Hierarchical feature transformer for UAV |
| TCTrack (2022) | CVPR | - | Temporal contexts for aerial tracking |
| Channel Distillation (2024) | - | ECO/DCF | Distills feature channels; reduces memory |
| T-SiamTPN (2025) | - | - | Temporal Siamese transformer for UAV |

---

## 3. Quantization & Deployment Comparison

This is the core question: which trackers are actually deployable with int8 on edge hardware?

| Tracker | Natively Supports int8? | Framework | Deployment Path | Model Size |
|---------|------------------------|-----------|-----------------|------------|
| **NanoTrack** | Yes (NCNN) | NCNN | NCNN → Android/Linux/ARM | 2.2 MB |
| **LightTrack** | Yes (NCNN port) | PyTorch → NCNN | ONNX → NCNN → int8 | 7.7 MB |
| **FEAR** | FP16 (CoreML) | PyTorch → CoreML | CoreML → iOS (FP16) | ~4 MB (XS) |
| **E.T.Track** | Not yet | PyTorch | ONNX → quantize yourself | ~5-10 MB |
| **MixFormerV2-S** | Not yet | PyTorch | ONNX → TFLite/NCNN | ~8-12 MB |
| **DyTrack** | Not yet | PyTorch | ONNX → quantize | ~6-10 MB |
| **EATrack** | Not yet | PyTorch | ONNX → quantize (UAV) | ~8-15 MB |
| **AR Tracker** | Yes (paper claims) | Not open source | Pruning + QAT + KD | unknown |

### The Quantization Pipeline (for trackers without native int8)

For any PyTorch tracker:
1. Export to ONNX
2. Use ONNX Runtime with INT8 quantization (calibration dataset needed)
3. Or convert ONNX → NCNN → int8 (NCNN has its own quantization tools)
4. Or convert ONNX → TFLite → int8 (for mobile/Android)
5. Or use PyTorch's built-in quantization-aware training (QAT)

Key papers on quantization for trackers specifically:
- **YOLOv4-Tiny INT8 on Raspberry Pi 5** (arXiv:2506.09300) — quantized to INT8 with TFLite, real-time on RPi5. This validates the approach for detection models, and the same pipeline works for trackers.
- **"Quantization with Unified Adaptive Distillation"** (arXiv:2603.29535) — multi-LoRA approach for edge GenAI models; relevant methodology.

---

## 4. Recommended Benchmarks for Head-to-Head Comparison

The standard benchmarks used across all papers:

| Benchmark | What It Measures | Used By |
|-----------|-----------------|---------|
| **VOT2018/2019** | EAO (Expected Average Overlap), Accuracy, Robustness | NanoTrack, LightTrack, FEAR |
| **GOT-10k** | AO (Average Overlap), SR (Success Rate) | NanoTrack, LightTrack, most Siamese |
| **LaSOT** | AUC, Precision, Norm Precision | MixFormerV2, E.T.Track, DyTrack |
| **OTB-100** | Success, Precision | E.T.Track |
| **TrackingNet** | AUC, Precision, Norm Precision | E.T.Track |
| **DTB70** | Success, Precision (drone-specific) | NanoTrack, UAV trackers |
| **UAV123** | AUC, Precision (UAV-specific) | HiFT, TCTrack, EATrack |
| **NFS** | AUC (need-for-speed/high-fps benchmark) | E.T.Track |

---

## 5. Consolidated Benchmark Comparison

Where data exists from papers/repos:

| Tracker | VOT2018 EAO | VOT2019 EAO | GOT-10k AO | LaSOT AUC | Model Size | CPU FPS | GPU FPS |
|---------|-------------|-------------|------------|-----------|------------|---------|---------|
| NanoTrack | 0.311 | 0.247 | 0.604 | - | 2.2 MB | >120 (M1) | - |
| LightTrack-M | 0.418 | 0.328 | 0.750 | - | 7.7 MB | - | - |
| FEAR-XS | >LightTrack | >LightTrack | - | - | ~4 MB | - | - |
| FEAR-L | - | - | - | >SiamRPN++ | - | - | - |
| E.T.Track | - | - | - | higher than lightwt | - | 47 | - |
| MixFormerV2-B | - | - | - | 70.6% | - | - | 165 |
| MixFormerV2-S | - | - | - | FEAR-L+2.7% | - | real-time | - |
| DyTrack | - | - | - | 64.9% | - | - | 256 |
| SiamRPN++ | ~0.414 (VOT18) | - | - | 49.6% | ~25 MB | - | 35 |
| Ocean | ~0.385 (VOT18) | - | - | 56.0% | ~20 MB | - | 25 |

*Note: Direct numerical comparison is tricky because papers use different hardware, different benchmark protocols, and different training data. The patterns are more informative than exact numbers.*

---

## 6. Key Research Groups & Where to Watch

### Active Labs in Efficient Tracking:

1. **Houwen Peng (Microsoft Research)** — LightTrack, Ocean → now at Microsoft, hiring for "visual transformer projects"
2. **Changhong Fu (Tongji University)** — HiFT, TCTrack, SiamAPN++ — the leading UAV tracking lab
3. **Bineng Zhong (Guangxi Normal University)** — EATrack, ETCTrack — active in 2025-2026
4. **Martin Danelljan (ETH Zurich)** — ECO, E.T.Track (co-author Luc Van Gool)
5. **Jiři Matas (Czech Technical University)** — FEAR — strong on deployment/benchmarking

### Conferences to Watch:
- **CVPR** — primary venue for tracking papers
- **ECCV/ICCV** — alternate years, strong tracking presence
- **ICRA/IROS** — robotics-focused; UAV tracking papers here
- **WACV** — winter conference, often has efficiency-focused papers

### arXiv Searches to Set Up:
```
cat:cs.CV AND (all:efficient AND all:tracking) AND all:(lasot OR got10k OR uav)
cat:cs.CV AND (all:quantiz* AND all:tracker)  
cat:cs.CV AND all:lightweight AND all:tracking AND all:(ncnn OR tflite OR onnx)
```

---

## 7. Recommendations for QuadTrack

### Tier 1: Deploy Now (int8-capable, proven)
1. **NanoTrack** — Already NCNN-ready. 2.2 MB, >120 FPS CPU. Drop-in for any Linux SBC with NCNN.
2. **LightTrack-ncnn** — Higher accuracy (35% better EAO), still only 7.7 MB. The Z-Xiong NCNN port is a direct starting point. Same NCNN pipeline as NanoTrack.

### Tier 2: Deploy with Work (export + quantize)
3. **MixFormerV2-S** — Best accuracy among CPU-trackers. Export to ONNX, quantize to int8 via ONNX Runtime. The distillation-based design means the "S" model is already optimized for deployment.
4. **DyTrack** — 256 FPS GPU / fast CPU. The dynamic routing means it adapts to the frame difficulty — ideal for drones where most frames are easy and occasional ones are hard.

### Tier 3: Watch List (UAV-specific, very new)
5. **EATrack** — Explicitly designed for UAV tracking with teacher-student distillation. 2026 paper, very fresh. Code available.
6. **ETCTrack** — 60% token reduction with negligible accuracy loss. If QuadTrack uses multi-template tracking, this is directly applicable.

### Practical Path for QuadTrack:
1. Start with **NanoTrack** for initial integration (simplest, already int8)
2. Benchmark accuracy on your actual drone footage
3. If accuracy is insufficient, upgrade to **LightTrack-ncnn** (same NCNN pipeline)
4. If still insufficient, invest in exporting **MixFormerV2-S** or **DyTrack** to ONNX + quantize

---

## 8. References

All papers with arXiv IDs for direct access:

```
2104.14545  LightTrack (CVPR 2021) — NAS-designed lightweight tracker
2112.07957  FEAR (ECCV 2022) — Fast, efficient, accurate, robust
2112.09686  E.T.Track (2021) — Exemplar transformer, 47 FPS CPU
2305.15896  MixFormerV2 (2023) — Efficient fully transformer tracking
2403.17651  DyTrack (2024) — Dynamic transformer, 256 FPS
2409.11785  Channel Distillation (2024) — Distilling channels for tracking
2511.17508  AR Lightweight RGB Tracker (2025) — Pruning+quantization+KD
2605.08329  ETCTrack (2026) — Token compression, 60% token reduction
2605.28018  EATrack (2026) — Dual-branch distilled transformer for UAV
2506.09300  YOLOv4-Tiny INT8 on RPi5 (2025) — Quantization validation
2006.10721  Ocean (ECCV 2020) — Object-aware anchor-free tracker
1911.07241  SiamCAR (CVPR 2019) — Siamese fully convolutional tracker
2106.08816  SiamAPN++ (ICRA 2021) — UAV tracking
2203.01885  TCTrack (CVPR 2022) — Temporal contexts for aerial tracking
2108.00202  HiFT (ICCV 2021) — Hierarchical feature transformer for UAV
```

GitHub repos:
- NanoTrack: https://github.com/HonglinChu/NanoTrack (192 stars)
- LightTrack: https://github.com/researchmm/LightTrack (458 stars)
- LightTrack-ncnn: https://github.com/Z-Xiong/LightTrack-ncnn (101 stars)
- FEAR: https://github.com/PinataFarms/FEARTracker
- E.T.Track: https://github.com/pblatter/ettrack
- EATrack: https://github.com/GXNU-ZhongLab/EATrack
- ETCTrack: https://github.com/PJD-WJ/ETCTrack
- NCNN: https://github.com/Tencent/ncnn


---

# Appendix A: Rockchip NPU Deployment

## A.1 The Rockchip NPU Landscape

Rockchip's NPU (Neural Processing Unit) is integrated into their ARM SoCs and accessed via the **RKNN SDK** (v2.3.2 as of June 2026):

| Chip | NPU TOPS | Max Frequency | Key Platforms |
|------|----------|---------------|---------------|
| **RK3588** | 6 TOPS | ~1 GHz (single core) | Orange Pi 5, Radxa Rock 5, Firefly ITX-3588J |
| **RK3576** | 6 TOPS | ~1 GHz | Newer mid-range |
| **RK3568/66** | 1 TOPS | ~800 MHz | Orange Pi 3B, Radxa Rock 3 |
| **RK3562** | 1 TOPS | ~800 MHz | Cost-optimized |
| **RV1126** | 2 TOPS | ~800 MHz | Camera-focused, older |

For QuadTrack, **RK3588 is the recommended target** — 6 TOPS gives headroom, widely available on SBCs, and the best-supported in the RKNN Model Zoo.

## A.2 The RKNN Toolchain

Pipeline: **PyTorch/ONNX → RKNN-Toolkit2 → .rknn model → RKNN Runtime (C API or Python Lite)**

Key steps:
1. Export your PyTorch model to ONNX (opset 19)
2. Use `rknn-toolkit2` on an x86 host to convert ONNX → .rknn
3. During conversion, specify `quantized_dtype='int8'` and provide a calibration dataset (~100-500 images)
4. Deploy `.rknn` on the target board using RKNN Runtime C API or Python Lite

Official repos:
- RKNN-Toolkit2: https://github.com/airockchip/rknn-toolkit2 (1,165 stars)
- RKNN Model Zoo: https://github.com/airockchip/rknn_model_zoo (2,523 stars)

## A.3 Critical Operator Support (RKNN v2.3.2)

This is the most important section for deciding which tracker works on NPU.

**Fully supported (tracker-critical ops):**
- Conv, ConvTranspose, BatchNormalization, InstanceNormalization, LayerNormalization
- Relu, LeakyRelu, HardSwish, Sigmoid, Tanh, GELU, Mish
- MatMul (including bmm/mm), Gemm (Linear layers)
- Add, Mul, Sub, Div, Max, Min
- Concat, Split, Reshape, Transpose/Permute, Squeeze, Unsqueeze, Flatten
- AveragePool, MaxPool, GlobalAveragePool, GlobalMaxPool
- Resize (nearest + bilinear — critical for Siamese neck)
- Pad, Softmax (batchsize=1 only), Clip, PRelu
- DepthToSpace, SpaceToDepth

**NOT SUPPORTED (blockers for some architectures):**
- **Einsum** — KILLS standard transformer attention. Must use explicit MatMul+Reshape+Softmax instead
- **GridSample** — KILLS RoIAlign variants, spatial transformer networks
- **GatherND, ScatterElements, ScatterND** — dynamic indexing ops
- **NonMaxSuppression** — must do NMS in post-processing on CPU
- **DeformConv** — deformable convolutions
- **Mean** (ONNX general mean op — ReduceMean IS supported)

**Restricted (workable with care):**
- **Slice: batchsize=1 only** — fine for single-object tracking
- **Softmax: batchsize=1 only** — fine for single-stream tracking
- **Tile: batchsize=1, no broadcast** — must precompute layouts
- **LSTM, GRU: batchsize=1** — only for simple RNN patterns

## A.4 Siamese Tracker Feasibility on RKNN

**Key question: can a Siamese tracker (template branch + search branch → cross-correlation → head) run on RKNN?**

The answer is **yes, with careful architecture choices:**

1. **Backbone (MobileNetV3 / ShuffleNetV2):** ✓ Fully supported. Conv+BN+HardSwish is the sweet spot for RKNN.
2. **Neck (cross-correlation / depthwise cross-correlation):** ✓ Supported via Conv+Reshape+MatMul. The point-wise correlation used in LightTrack/NanoTrack is just Conv + Reshape + MatMul — all supported.
3. **Head (FC layers for bbox regression):** ✓ Supported via Gemm/Linear.
4. **Multi-scale features:** ✓ Concat and Resize are supported.
5. **Dynamic templates:** The dual-template approach in FEAR uses a single learnable parameter — trivially supported.

**What WON'T work on NPU:**
- **E.T.Track-style Exemplar Transformer** — uses instance-level attention with Einsum-like patterns
- **MixFormerV2's full transformer backbone** — uses Einsum for attention. BUT: the S (small) variant uses distillation to a simpler architecture, which may be adaptable
- **DyTrack's dynamic early-exit** — the conditional routing uses Gather/Scatter ops
- **GridSample-based spatial warping** — not supported

## A.5 RKNN Performance Benchmarks (from official Model Zoo on RK3588)

For reference — these are **detection models** at 640×640 input. A tracker with smaller input (256×256 or 288×288) would be proportionally faster:

| Model | Input | Dtype | RK3588 FPS (single NPU core) |
|-------|-------|-------|------------------------------|
| YOLOv5n | 640×640 | INT8 | 82.5 |
| YOLOv5s | 640×640 | INT8 | 48.4 |
| YOLOv8n | 640×640 | INT8 | 73.5 |
| YOLOv8s | 640×640 | INT8 | 38.0 |
| RetinaFace Mobile | 320×320 | INT8 | 227.2 |
| MobileNetV2 | 224×224 | INT8 | 450.7 |
| ResNet-50 | 224×224 | INT8 | 110.1 |

**Estimated tracker performance on RK3588 (single NPU core):**
- A MobileNetV3-based Siamese tracker at 288×288 input, INT8: **~80-120 FPS** on NPU alone
- Add CPU pre/post-processing overhead: **~50-80 FPS** real-world
- With 3 NPU cores (RK3588 has 3): could pipeline template branch on core 0 + search branch on core 1 + neck on core 2, but RKNN single-core API makes this tricky currently

## A.6 NPU vs CPU vs GPU: Is NPU Actually More Efficient?

**Yes, dramatically.** Here's why for a tracking workload:

| Metric | RK3588 CPU (4×A76 + 4×A55) | RK3588 NPU (single core) | Ratio |
|--------|----------------------------|--------------------------|-------|
| YOLOv5s INT8 at 640×640 | ~10 FPS (all 8 cores) | 48.4 FPS | **4.8× NPU** |
| YOLOv8n INT8 at 640×640 | ~8 FPS | 73.5 FPS | **9.2× NPU** |
| MobileNetV2 INT8 at 224×224 | ~60 FPS | 450.7 FPS | **7.5× NPU** |
| Power (approximate) | ~4-6W | ~1-2W | **3× more efficient** |

The NPU is ~5-10× faster AND ~3× more power-efficient than CPU for CNN workloads. For a battery-powered drone, **NPU is essential** — it's the difference between tracking at 30+ FPS for 30 minutes vs tracking at 10 FPS for 10 minutes.

**Why NPU is better than GPU for edge tracking:**
- RK3588 Mali-G610 GPU: good but ~10-15W under load, competes with display
- NPU: dedicated silicon, doesn't compete with GPU for rendering
- NPU INT8 throughput is purpose-built — GPUs are FP16/FP32 optimized
- On a drone, you want the GPU free for any video encoding/decoding

## A.7 One Important Reality Check

**No Siamese tracker exists in the official RKNN Model Zoo.** The zoo has YOLO (detection), classification, segmentation, OCR, speech — but zero tracking models. This means:

1. You will be **the first** to deploy a Siamese tracker on RKNN (or at least among the first)
2. Expect **1-2 weeks of engineering** to get the ONNX→RKNN conversion working
3. The `rknn-toolkit2` conversion may fail on unsupported ops — you'll need to:
   - Trace through the PyTorch model to find which ops fail
   - Replace unsupported ops with supported equivalents
   - Build a calibration dataset for INT8 quantization
   - Test accuracy after quantization (expect 0.5-2% accuracy drop)

This is **doable engineering work**, not research. The operator support is there for the CNN-based trackers.

---

# Appendix B: UAV Tracking Datasets for Fine-Tuning

## B.1 Established UAV Tracking Benchmarks

These are the datasets used by ALL UAV tracking papers. Fine-tune on these to get comparable results:

| Dataset | Sequences | Frames | Resolution | Scenario | Best For |
|---------|-----------|--------|------------|----------|----------|
| **UAV123** | 123 | 110K | 720p | General UAV, various objects | Primary training |
| **UAV20L** | 20 | 58K | 720p | Long-term tracking (avg 2.9K frames/seq) | Long-term robustness |
| **DTB70** | 70 | 15K | 720p/1080p | Drone-to-ground, diverse targets | NanoTrack benchmark |
| **UAVDT** | 100 | 80K | 1080p | Traffic surveillance from drone | Vehicle tracking |
| **VisDrone** | 400 | 265K | various | Large-scale, MOT + SOT | Large-scale pre-training |
| **UAVDark135** | 135 | - | 720p | Nighttime UAV tracking | All-day operation |
| **DarkTrack2021** | 110 | - | - | Nighttime tracking | Low-light robustness |

## B.2 New Datasets (2025-2026)

These are the most exciting — much larger than the classics:

**CosFly-Track (2026)** — The game-changer:
- 12,000 expert + perturbed UAV trajectories from 6,000 pedestrian paths
- 2.4 million timesteps (~334 hours of footage)
- 7 synchronized channels: RGB, depth, semantic segmentation, 6-DOF drone pose, target state, bilingual instructions, trajectory metadata
- Built by **Autel Robotics** (major drone manufacturer!)
- URL: https://huggingface.co/datasets/AutelRobotics/CosFly
- **This is the ideal dataset for QuadTrack** — built by a drone company, multi-modal, huge scale

**UAV-Track VLA (2026):**
- 890K frames, 176 tasks, 85 diverse objects
- Urban scenarios with complex semantic requirements
- Multi-modal: vision + language instructions

**LRDDv3 (2026):**
- Long-range drone detection with range information and thermal data
- Useful for detection pre-training

## B.3 Recommended Training Strategy for QuadTrack

**Phase 1 — Pre-train on general tracking (transfer learning):**
- Use GOT-10k (10,000 sequences, 1.5M frames) for the Siamese backbone pre-training
- Most lightweight trackers (NanoTrack, LightTrack, FEAR-XS) were trained on GOT-10k
- Standard protocol used by all Siamese trackers

**Phase 2 — Fine-tune on UAV-specific data:**
- Primary: **UAV123 + DTB70** (193 sequences combined) — standard UAV tracking benchmarks
- Long sequences: **UAV20L** — for robustness on extended tracking
- If vehicle tracking: add **UAVDT**

**Phase 3 — New capability fine-tuning (optional):**
- **CosFly-Track** for multi-modal and pedestrian tracking in urban environments
- **UAVDark135** for nighttime/all-day operation

**Phase 4 — Domain-specific with your own data:**
- Collect 50-100 sequences of your specific quadrotor tracking scenarios
- Fine-tune the UAV-pre-trained model on your data
- 2-4 hours on a single GPU for this fine-tuning

## B.4 Training Feasibility

All these models are designed to train on consumer GPUs:
- NanoTrack: GOT-10k training in **2 hours on RTX 3090**
- LightTrack: SuperNet pre-training on ImageNet (done for you) + task-specific NAS search
- FEAR-XS: Quick train config uses GOT-10k only
- The SiamTrackers repo (HonglinChu) has PyTorch training code for NanoTrack

---

# Appendix C: Recommended Deployment Path for QuadTrack on Rockchip NPU

## The Three Options, Ranked

### Option 1 (RECOMMENDED — Lowest Risk): NanoTrack + RKNN

**Why:** NanoTrack already runs on NCNN. The NCNN→RKNN path is not direct, but the ONNX export path is: PyTorch → ONNX → RKNN-Toolkit2 → .rknn. NanoTrack's architecture (MobileNetV3 + point-wise correlation + FC heads) uses exclusively supported ops.

**Pipeline:**
1. Export NanoTrack PyTorch model to ONNX (from https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack)
2. Convert ONNX → RKNN with INT8 quantization (calibration on GOT-10k subset)
3. Deploy on RK3588 via RKNN Runtime
4. If RKNN conversion fails on any op, fall back to NCNN on RK3588 CPU (still fast — 120+ FPS on M1, RK3588 big cores are comparable)

**Estimated performance:** ~80-120 FPS NPU inference + CPU pre/post = 50-80 FPS real-world
**Accuracy (DTB70):** 0.532 Success, 0.727 Precision
**Risk:** LOW — if RKNN conversion fails, NCNN CPU fallback works

### Option 2 (Best Accuracy/Size Balance): LightTrack + RKNN

**Why:** LightTrack has 35% better VOT2018 EAO than NanoTrack (0.418 vs 0.311). The NAS-discovered architecture was designed for mobile deployment. The NCNN port (Z-Xiong) proves the ops are simple enough for edge inference.

**Pipeline:**
1. Export LightTrack-Mobile PyTorch to ONNX
2. The architecture splits into two sub-models (template encoder + tracker). Convert each separately.
3. INT8 quantize with UAV calibration data (important: use UAV imagery, not ImageNet)
4. Deploy on RK3588

**Estimated performance:** ~50-80 FPS NPU, 30-60 FPS real-world
**Accuracy (DTB70):** 0.591 Success, 0.766 Precision
**Risk:** MEDIUM — more complex architecture (NAS-discovered, supernet-derived), more ops to validate

### Option 3 (Best Accuracy, Most Engineering): Custom-trained FEAR-style tracker

**Why:** FEAR-XS beats LightTrack on both speed AND accuracy. The architecture is purpose-built for mobile (ECCV 2022, paper explicitly designs for mobile deployment). But there's no pre-trained model for UAV — you'd need to train from scratch.

**Pipeline:**
1. Implement FEAR-XS architecture in PyTorch (code is open source)
2. Replace any unsupported ops with RKNN-compatible equivalents
3. Train on GOT-10k then fine-tune on UAV123 + DTB70
4. Export to ONNX → RKNN → INT8
5. Deploy on RK3588

**Estimated performance:** ~60-100 FPS NPU, 40-70 FPS real-world
**Accuracy:** Expected to exceed LightTrack on UAV benchmarks
**Risk:** HIGH — training a tracker from scratch requires expertise, dataset preparation, and hyperparameter tuning

## Decision Matrix

| Factor | NanoTrack | LightTrack | FEAR (custom) |
|--------|-----------|------------|---------------|
| Engineering effort | 1-2 weeks | 2-4 weeks | 4-8 weeks |
| Deployment risk | Low | Medium | High |
| UAV accuracy | Lowest | Medium-High | Highest (expected) |
| Model size | 2.2 MB | 7.7 MB | ~4 MB |
| NPU FPS (est.) | 80-120 | 50-80 | 60-100 |
| UAV training data | None (general only) | None (general only) | You train it |
| Pre-trained available | Yes (GOT-10k) | Yes (GOT-10k+ImageNet) | Yes (FEAR official, but non-UAV) |

## My Recommendation

**Start with Option 1 (NanoTrack) for system integration, then upgrade to Option 2 (LightTrack) for accuracy.**

The reasoning:
1. NanoTrack lets you validate the entire pipeline (camera → NPU → tracking → flight controller) in weeks, not months
2. You'll discover real-world issues (lighting, motion blur, occlusion) that inform your next model choice
3. Once the pipeline works, swapping NanoTrack for LightTrack is a model replacement, not an architecture change
4. Option 3 (custom training) makes sense AFTER you have a working system and know exactly what accuracy gaps you need to close

For training data: start with **GOT-10k** (general tracking pre-training) + **UAV123** (UAV fine-tuning) + **DTB70** (drone-specific). If CosFly-Track data is accessible, add that as well — it's the best UAV tracking dataset available.


---

# Appendix D: Tracking-by-Detection with YOLO on UAV — Comparison with Siamese Trackers

## D.1 How Tracking-by-Detection Works

The approach is simple and well-established:

```
Frame → YOLO detector → bounding boxes → association algorithm → track IDs
```

The association step is handled by algorithms like:
- **SORT** (2016): Kalman filter + Hungarian algorithm. Dead simple. ~200 lines of code.
- **DeepSORT** (2017): SORT + appearance ReID model. Better at re-identifying after occlusion.
- **ByteTrack** (ECCV 2022): Associates low-confidence detections too — big accuracy gain at no speed cost
- **BoT-SORT** (2022): Camera motion compensation + better Kalman state vector. SOTA on MOT benchmarks.
- **OC-SORT** (2023): Observation-centric — better for non-linear motion (important for drones!)

The pipeline is: **YOLO detects every object of interest in every frame → the tracker assigns consistent IDs across frames using motion prediction and/or appearance matching.**

## D.2 The YOLO + RK3588 Advantage

This is the most important practical finding:

**YOLO already runs on RK3588 NPU. It is literally in the official RKNN Model Zoo with INT8 quantization and validated FPS numbers.** Every YOLO variant from v5 through v11 is supported:

| YOLO Model | RK3588 INT8 FPS | VisDrone-trained? |
|------------|-----------------|-------------------|
| YOLOv5n | 82.5 | Yes (many repos) |
| YOLOv8n | 73.5 | Yes (xuanandsix, 58 stars) |
| YOLOv8s | 38.0 | Yes |
| YOLOv11n | 60.0 | Not verified but likely |

This means you can have a working tracker **today** — literally take a VisDrone-trained YOLOv8n ONNX, convert to RKNN (one command), run SORT on the CPU, and you're tracking.

**The Siamese tracker path requires 1-2 weeks of engineering just to confirm it works on NPU. The YOLO path works right now.**

## D.3 The Critical Difference: Multi-Object vs Single-Object

This is where the architecture choice gets real for QuadTrack:

| Aspect | Siamese Tracker | YOLO + SORT/ByteTrack |
|--------|----------------|----------------------|
| **Tracking paradigm** | Single object — given a template, follow THAT object | Multi-object — detect ALL objects, assign IDs |
| **Needs initialization** | Yes — bounding box in first frame | No — detects everything automatically |
| **Re-acquisition after loss** | Must re-initialize manually | Automatic — detects the object again |
| **Tracks multiple objects** | No (one target at a time) | Yes (all detected objects) |
| **Distinguishes objects** | N/A — only tracks the template | Yes — each object gets a unique ID |
| **Class-agnostic** | Yes — tracks whatever you box | No — only tracks classes it was trained on |

**For QuadTrack specifically:** What are you tracking? A specific quadrotor? Any quadrotor? A person?

- If tracking **one specific quadrotor** (e.g., your own drone in follow mode): Siamese wins — you just box it once, and it follows that specific instance
- If tracking **any/all quadrotors** in the scene: YOLO + tracking wins — it detects all drones automatically
- If tracking **a person on the ground** from the drone: YOLO trained on person class + ByteTrack is the standard solution used by every drone company

## D.4 Real-World Performance Data

**WildLive paper (2025)** — the most directly comparable benchmark because it's onboard UAV tracking:
- YOLO-based detection + sparse optical flow tracking
- On Jetson Orin AGX: 17.81 FPS at 1080p, 7.53 FPS at 4K
- Compared against: OC-SORT, ByteTrack, SORT
- The key insight: "computational resource is focused onto spatio-temporal regions of high uncertainty"

**YOLOv12 + BoT-SORT-ReID (2025)** — multi-UAV tracking:
- Built for Anti-UAV Challenge
- YOLOv12 detector + BoT-SORT tracker
- Works on thermal infrared (harder than RGB)
- Strong baseline without any fancy tricks — just a good detector + good association

**YOLOv8 + SORT on FPGA (2025)** — edge deployment paper:
- Quantized YOLOv8 on FPGA + SORT on ARM
- 0.21 mAP, 38.9 MOTA on COCO/MOT15
- Proves the whole pipeline works on embedded hardware

**Benchmarking Deep Trackers on Aerial Videos (2021)** — the key comparison paper:
- Compared 10 trackers on UAV123, UAV20L, DTB70
- **Finding:** ALL trackers perform significantly worse on aerial data vs ground data
- The gap is due to: smaller targets, camera motion, rotation, out-of-view movement, clutter

## D.5 Head-to-Head: Siamese vs Detection-Based for UAV

| Dimension | Siamese (NanoTrack/LightTrack) | YOLO + ByteTrack/BoT-SORT |
|-----------|-------------------------------|---------------------------|
| **UAV accuracy** | 0.53-0.59 Success (DTB70) | Not directly comparable (different metric) |
| **NPU deployment** | 1-2 weeks of engineering | **Works today** (in RKNN Model Zoo) |
| **FPS on RK3588** | ~50-80 real-world (estimated) | ~35-60 real-world (detection + tracking) |
| **Multi-object** | No | Yes |
| **Re-initialization** | Manual | Automatic |
| **Occlusion handling** | Good (template persists) | Depends on detector consistency |
| **Training data needed** | GOT-10k + UAV fine-tune | VisDrone (already exists!) |
| **Model maturity** | Research-grade | Production-grade (YOLO is industry standard) |
| **Power efficiency** | Very good (small model on NPU) | Good (YOLO-nano on NPU, SORT on CPU) |
| **Off-the-shelf code** | Needs assembly | **BoxMOT library** — pluggable, 8K+ stars |

## D.6 The BoxMOT Ecosystem

This is the secret weapon for the YOLO approach:

**BoxMOT** (https://github.com/mikel-brostrom/boxmot, 8,196 stars) is a pluggable tracking-by-detection library. It means:

```python
from boxmot import BoTSORT

# Your YOLO model on NPU
detections = yolo_model(frame)  # runs on RK3588 NPU

# BoxMOT tracker on CPU (lightweight, ~1ms per frame)
tracks = tracker.update(detections, frame)
```

BoxMOT supports: BoT-SORT, ByteTrack, DeepOCSORT, StrongSORT, OC-SORT, and more. All you need to provide is a detector that outputs bounding boxes — and YOLO on RKNN already does that.

## D.7 The Hybrid Approach (Best of Both Worlds)

This is what I think is actually optimal for QuadTrack:

**YOLO for detection + Siamese-style template matching for specific instance tracking.**

How it works:
1. **YOLO on NPU** detects all objects of interest (all quadrotors, or the specific class you're tracking) — runs at 60-80 FPS
2. **SORT/ByteTrack on CPU** assigns consistent track IDs — runs at <1ms per frame
3. **When the user selects a specific target:** initialize a lightweight Siamese tracker (NanoTrack) to follow that specific instance
4. **YOLO continues running** as fallback — if the Siamese tracker loses the target, detection-based tracking re-acquires it
5. **Fusion:** the Siamese tracker gives fine-grained bounding boxes; YOLO gives coarse detection + ID consistency

This gives you:
- Automatic detection and multi-object awareness (YOLO)
- Fine-grained single-instance tracking (Siamese)
- Automatic recovery from tracking failure (YOLO fallback)
- The best of both architectures

This is actually similar to how the top VOT challenge entries work — they use a detector to initialize/re-initialize and a tracker to follow.

## D.8 Concrete Recommendation

**If you want to fly THIS WEEK:** Use YOLOv8n (VisDrone-trained) on RK3588 NPU + ByteTrack on CPU via BoxMOT. This works today. It's production-grade. It's in the RKNN Model Zoo. You will have a working tracker in an afternoon.

**If you want the best long-term architecture:** Build the hybrid system described above. YOLO as the detection backbone (always running), NanoTrack/LightTrack as the fine-grained instance tracker (initialized on user selection), with fusion logic to combine them.

**Training path for the YOLO approach:**
1. Start with off-the-shelf YOLOv8n pre-trained on COCO
2. Fine-tune on VisDrone (265K frames, aerial imagery) — this already exists, many repos have done it
3. If you need quadrotor-specific detection, add your own annotated quadrotor images to the training set
4. Convert to RKNN INT8 (one command in RKNN-Toolkit2)
5. Deploy + run ByteTrack on CPU

This path is **one day of work** vs 1-2 weeks for the Siamese NPU path.

**Why the Siamese path still matters long-term:**
- Once you need to track a SPECIFIC instance (not just any quadrotor, but THAT quadrotor), Siamese is the right tool
- Siamese is class-agnostic — it'll track anything you box, even objects YOLO wasn't trained on
- For the final production system, the hybrid architecture is the right answer

## D.9 Key Papers and Repos

| Resource | Link | What It Provides |
|----------|------|-----------------|
| YOLOv12 + BoT-SORT for UAV | arXiv:2503.17237 | Production multi-UAV tracking baseline |
| WildLive onboard UAV tracking | arXiv:2504.10165 | Real-world UAV tracking benchmarks |
| YOLOv8 + SORT on FPGA | arXiv:2503.13023 | Edge deployment validation |
| ByteTrack | arXiv:2110.06864 | The association algorithm |
| BoT-SORT | arXiv:2206.14651 | Camera motion compensation for tracking |
| BoxMOT library | github.com/mikel-brostrom/boxmot | Pluggable tracking-by-detection (8K stars) |
| VisDrone YOLOv8 | github.com/xuanandsix/VisDrone-yolov8 | Pre-trained YOLOv8 on VisDrone |
| YOLO following UAV | arXiv:2205.00083 | YOLO + Kalman for drone-to-drone tracking |
| Benchmarking trackers on UAV | arXiv:2103.12924 | Comparison of 10 trackers on aerial data |


---

# Appendix E: Interceptor Tracking — Multi-Stage Architecture for Proportional Navigation Guidance

## E.1 The Problem

An interceptor drone using proportional navigation (PN) to engage a target UAV experiences extreme variation in tracking conditions:

| Phase | Range | Target size | Pixel area | What it looks like |
|-------|-------|-------------|------------|--------------------|
| BVR | 1000m+ | <5×5 px | <25 px² | Point source. Indistinguishable from sensor noise. |
| Detection | 300-1000m | 5×5 to 20×20 px | 25-400 px² | Blob. No structure visible. May wink in/out. |
| Tracking | 50-300m | 20×20 to 100×100 px | 400-10K px² | Recognizable UAV. Arms/body distinguishable. |
| Terminal | <50m | 100×100 to full-frame | 10K+ px² | Full UAV. Rapid scale expansion. Can double in size in <1s. |

**No single algorithm spans all four phases.** The requirements at each phase are fundamentally contradictory: Phase 1 needs temporal-only features (motion consistency), Phase 3-4 needs spatial features (appearance/texture). The tracker that excels at finding a 3×3 moving dot is architecturally different from the tracker that tracks a 100×100 structured object.

## E.2 Why Each Approach Fails at the Extremes

**YOLO-based detectors at BVR:**
Standard YOLO has a minimum detection stride of 8 pixels. A 5×5 target maps to less than 1 cell in the deepest feature map. DroneScan-YOLO and UFO-DETR improve this with modified strides (can detect ~8×8 px), but below that, there are literally not enough pixels for convolution to extract features. The Anti-UAV Challenge winner (arXiv:2505.04917) explicitly added frame differencing as an input channel because "existing trackers depend on cropped template regions and have limited motion modeling capabilities" for tiny targets.

**Siamese trackers at BVR:**
At 5×5 px, the template contains 25 pixels. Cross-correlation with a search region produces a response map where the peak is 1-2 pixels wide. The signal-to-noise ratio is fundamentally too low. These trackers require at minimum ~15×15 px to produce a discriminable correlation peak.

**Classical CV at terminal phase:**
Frame differencing and centroid tracking work for point targets but cannot handle the structured appearance changes of a close-range UAV — rotation, aspect change, component-level motion. A 100×100 px target requires spatial feature matching.

**Correlation filters (KCF/ECO) at both extremes:**
These are the most scale-flexible classical approach, working from ~8×8 px up to large targets. But they fail with rapid scale changes (terminal phase) without explicit scale adaptation, and at very small scales they're outperformed by temporal methods.

## E.3 Phase-by-Phase Scoring

```
                    Phase 1    Phase 2    Phase 3    Phase 4
                    (BVR)      (Detect)   (Track)    (Terminal)
                    ───────    ────────   ───────    ─────────
YOLO + ByteTrack      ✗           △          ✓          ✓
Siamese (LightTrack)   ✗           △          ✓          ✓
Correlation Filter     △           ✓          ✓          △
IRSTD + Trajectory     ✓           ✓          △          ✗
Temporal/Motion-only   ✓           ✓          ✗          ✗
FrameDiff + Flow       ✓           ✓          △          ✗
DroneScan-YOLO         ✗           ✓          ✓          ✓
UFO-DETR               ✗           ✓          ✓          ✓
GenTrack (PSO)         △           ✓          ✓          ✓
```

Key: ✓ = works well, △ = marginal, ✗ = fails

## E.4 The Case for Multi-Stage Architecture

The honest engineering answer is that you need at least two trackers, with a smooth handoff between them. Here's the optimal architecture:

### Stage 1: BVR Acquisition (1000m+ → 200-300m)

**Algorithm:** Frame differencing + adaptive thresholding + trajectory association

This is solved engineering from 40 years of IRST (Infrared Search and Track) systems. No neural network needed. A 3×3 pixel target contains 9 numbers — a CNN has no features to extract. The ONLY discriminating information is:
1. This pixel is slightly different from its neighbors
2. It moves differently than the background
3. Its motion is consistent across frames

Implementation:
```
Input: Raw frames at camera FPS
Process:
  1. Frame differencing (current - previous, thresholded)
  2. Connected components → candidate point detections
  3. Multi-Hypothesis Tracking (MHT) or JPDA for association
  4. Track confirmation after N consistent detections
Output: (x_center, y_center, vx, vy) for each confirmed track

Runtime: ~2ms on RK3588 CPU (no NPU needed)
Transition trigger: Track confidence > 0.9 AND apparent size > ~8×8 px
```

### Stage 2: Detection-Range Tracking (50-300m)

**Algorithm:** Tiny-object YOLO variant + Deep OC-SORT + ReID gallery

This is where neural networks become useful. The target is large enough (~10×10 to 50×50 px) that spatial features exist but small enough that standard YOLO struggles.

Implementation:
```
Detector: UFO-DETR or DroneScan-YOLO with stride=4 (not 8)
  - Handles objects down to ~8×8 px
  - Trained on VisDrone + custom quadrotor dataset
  - Runs on RK3588 NPU, INT8 quantized
  - Input: full frame or ROI around last known position

Tracker: Deep OC-SORT (arXiv:2302.11813)
  - OC-SORT handles non-linear motion (critical for PN intercept trajectories)
  - Adaptive ReID with rolling gallery (100 features)
  - Hard gallery refresh every 30 frames to prevent drift
  
  Why OC-SORT over ByteTrack:
  - ByteTrack uses Kalman with linear motion assumption
  - OC-SORT uses observation-centric updates → handles PN maneuvers
  - Deep OC-SORT adds adaptive appearance matching

Template management:
  - Rolling ReID gallery: 100 features, newest replaces oldest
  - Hard refresh every ~30 frames: clear gallery, re-seed from current detection
  - This adapts to the changing appearance as the UAV transitions from blob → structured

Output: (x, y, w, h, confidence, track_id)
Runtime: ~17ms/frame (14ms NPU + 2ms ReID + 1ms association) ≈ 59 FPS
Transition trigger: Target consistently > ~30×30 px AND detection confidence > 0.8
```

### Stage 3: Terminal Tracking (<50m)

**Algorithm:** Siamese tracker (NanoTrack) + YOLO fallback

At close range, you need fine-grained spatial precision and rapid scale adaptation. The Siamese tracker runs on a cropped ROI for maximum speed, while the YOLO detector continues running on the full frame as a fallback.

Implementation:
```
Primary: NanoTrack (Siamese) on NPU
  - Template extracted from final Stage 2 detection
  - Search region: 2× current target size
  - Scale pyramid: 3 scales (0.95×, 1.0×, 1.05×) to handle rapid scale change
  - Runs at ~80 FPS on NPU

Fallback: YOLO detector still active (Stage 2 pipeline)
  - If Siamese confidence drops below threshold → immediate switch to YOLO track
  - No re-acquisition delay (critical for PN guidance)

Fusion:
  - Both trackers run simultaneously
  - Output = Siamese bbox when confidence > 0.7
  - Output = YOLO bbox when Siamese confidence < 0.7
  - Weighted average in the overlap region

Output: (x, y, w, h, confidence)
Runtime: ~20ms/frame (14ms YOLO + 4ms Siamese + 2ms fusion) ≈ 50 FPS
```

### Handoff Protocol

Critical: the handoff between stages must be smooth. A hard switch causes the PN guidance to see a discontinuity in target position, which produces a spurious acceleration command.

```
Stage 1 → Stage 2 handoff:
  - Stage 1 produces (x, y)_point_track
  - Stage 2 produces (x, y, w, h)_detection
  - Initialize Stage 2 track at Stage 1 position
  - For 5 frames, fuse: output = α * Stage1 + (1-α) * Stage2
    where α = 1.0 → 0.0 over 5 frames (linear ramp)
  - This prevents the PN guidance from seeing a step change

Stage 2 → Stage 3 handoff:
  - Stage 2 final detection bbox → Stage 3 Siamese template
  - Both run for 10 frames of overlap
  - Fuse with weighted average
  - Stage 3 takes over when its confidence exceeds Stage 2's
```

## E.5 Specific Requirements for PN Guidance

Proportional navigation requires:
```
a_cmd = N * V_c * λ_dot

where:
  λ_dot = line-of-sight angular rate (pixels/frame → radians/sec)
  V_c   = closing velocity
  N     = navigation constant (typically 3-5)
```

The tracker must provide:
1. **Target angular position** (x, y in image coordinates → azimuth, elevation angles)
2. **Target angular rate** (dx/dt, dy/dt — the line-of-sight rate)
3. **Low latency** (<20ms end-to-end preferred)
4. **No dropouts** — PN diverges if λ_dot goes to zero or NaN for even a few frames

The multi-stage architecture satisfies all four:
- Angular position: direct from any stage
- Angular rate: from Kalman filter velocity states (all stages maintain a motion model)
- Low latency: NPU inference + CPU tracking = 14-20ms
- No dropouts: redundant stages ensure continuity. If Stage 3 Siamese loses lock, Stage 2 YOLO provides immediate fallback. If Stage 2 YOLO confidence drops, Stage 1 point-track provides a low-precision but continuous estimate.

## E.6 Recommended Implementation Path

**Phase 1: Build Stage 2 first** (detection-range tracking). This is the core of the system and the phase where you'll spend most of the engagement. UFO-DETR-style YOLO on NPU + Deep OC-SORT.

**Phase 2: Add Stage 1** (BVR acquisition). Classical frame differencing + MHT. This is ~500 lines of C++/Python with no dependencies beyond OpenCV. Test with simulated point targets at varying SNRs.

**Phase 3: Add Stage 3** (terminal tracking). NanoTrack Siamese tracker on NPU. The handoff logic from Stage 2 → Stage 3 is the engineering challenge.

**Phase 4: Tune the transitions.** The handoff thresholds and fusion weights are engagement-specific. You'll need to tune them with recorded flight data.

## E.7 Key References for This Architecture

| Component | Reference | Notes |
|-----------|-----------|-------|
| Anti-UAV tiny target tracking | arXiv:2505.04917 | 1st place Anti-UAV Challenge. Frame dynamics + detection. |
| DroneScan-YOLO | arXiv:2604.13278 | Sub-32px detection. Modified stride. |
| UFO-DETR | arXiv:2602.22712 | Frequency-guided tiny object detection. |
| UAV-DETR | arXiv:2603.22841 | DETR-based anti-drone detection. |
| Dim target + trajectory | arXiv:2405.09054 | Temporal energy selective scaling for point targets. |
| GenTrack | arXiv:2510.24399 | Hybrid PSO + deterministic for weak detectors. |
| Deep OC-SORT | arXiv:2302.11813 | Adaptive ReID for non-linear motion. |
| OC-SORT | arXiv:2203.14360 | Observation-centric, handles non-linear motion. |
| LightGBM + trajectory | arXiv:2407.01278 | Small aerial target detection with trajectory constraints. |
| TRADE (UAV tracking-by-detection) | arXiv:2210.03270 | Addresses target switching in tracking-by-detection for UAVs. |
| STMD (bio-inspired) | arXiv:1801.06687 | Small target motion detector — fly visual system. Inspirational for Stage 1.<｜end▁of▁thinking｜>Now let me also save the key long-term facts to memory so this doesn't get lost across sessions.

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="memory">
<｜｜DSML｜｜parameter name="action" string="true">add
