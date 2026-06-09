# Anti-UAV Datasets for Detection and Tracking

Research compiled: 2026-06-07
Purpose: Identify training and evaluation datasets for QuadTrack's interceptor
drone detection pipeline. Every dataset listed here has **UAV as the annotation
target class** — not UAV as the camera platform.

---

## 1. Dataset Summary Table

### Air-to-Air (camera on moving UAV, target is UAV)

| Dataset | Paper | Size | Modality | Platform | Annotations | Task | Availability |
|---------|-------|------|----------|----------|-------------|------|-------------|
| **UAV-Anti-UAV** | arXiv:2512.07385 | 1,810 videos, million-scale frames | RGB, IR, RGB-IR | Moving UAV (pursuer) | Bbox + language prompts + 15 tracking attributes | SOT tracking | GitHub: 983632847/Awesome-Multimodal-Object-Tracking |
| **M²E-UAV** | arXiv:2605.10496 | 87,223 train + 21,395 val samples | Event camera + IMU | Moving UAV (onboard) | Event-level UAV foreground labels + 10 Hz bbox | Detection | GitHub: Wickyan/M2E-UAV |

### Ground-to-Air (camera on ground/static, target is UAV against sky)

| Dataset | Paper | Size | Modality | Platform | Annotations | Task | Availability |
|---------|-------|------|----------|----------|-------------|------|-------------|
| **Anti-UAV Challenge** | arXiv:2305.07290 | 300+ sequences (train + test) | RGB, IR | Static ground cameras | Bbox + presence flag per frame | SOT tracking + Detection | GitHub: ZhaoJ9014/Anti-UAV |
| **Anti-UAV-RGBT** | arXiv:2601.19318 | 226 sequences | RGB + Thermal | Static/moving ground | Bbox + pursuit feasibility | Tracking + trajectory prediction | Via Perception-to-Pursuit paper |
| **CST Anti-UAV** | arXiv:2507.23473 | 220 sequences, 240K+ frames | Thermal IR | Static ground | Bbox + 15 frame-level attributes | SOT tracking | To be released |
| **DroneSOD-30K** | arXiv:2603.25218 | 30,000 images | RGB | Static ground | Bbox (single-frame) | Detection | Via SDD-YOLO paper authors |
| **MMAUD** | arXiv:2402.03706 | Large-scale | RGB + Thermal + Stereo + Lidar + Radar + Audio | Ground + overhead aerial | Bbox + type classification + trajectory | Detection + Classification + Tracking | GitHub: ntu-aris/MMAUD |
| **MAV-VID** | (benchmark only) | Unknown | RGB | Static ground | Bbox | Detection | Used as real-world eval in synthetic papers |

### Synthetic and Mixed

| Dataset | Paper | Size | Modality | Platform | Annotations | Task | Availability |
|---------|-------|------|----------|----------|-------------|------|-------------|
| **SimD3** | arXiv:2601.14742 | Large-scale synthetic | RGB (UE5) | 360° six-camera rig | Bbox + payload types + bird distractors | Detection | Via paper authors |
| **BirDrone** | arXiv:2601.08319 | Large-scale | RGB | Ground cameras | Bbox (drone vs bird classes) | Detection + Classification | Via YOLOBirDrone paper |
| **Cranfield Synthetic** | arXiv:2411.09077 | Synthetic + real eval | RGB | Static ground | Bbox | Detection | HuggingFace: mazqtpopx/cranfield-synthetic-drone-detection |

---

## 2. Detailed Characteristics

### 2.1 UAV-Anti-UAV ★★★★★ (BEST for QuadTrack)

**Why it's the best:** This is the only dataset where both camera and target
are flying UAVs. The pursuer UAV tracks an adversarial target UAV — exactly
the interceptor use case.

- **Dual-dynamic disturbances:** Both platforms are moving independently.
  The camera motion is not compensated — the model must learn to track through
  ego-motion. This is the exact challenge of PN-guided intercept.
- **Multi-modal:** RGB, IR, and RGB-IR fusion available.
- **Annotation quality:** Manual bounding boxes on every frame + language
  prompts + 15 per-sequence tracking attributes (occlusion, fast motion,
  scale variation, etc.).
- **Benchmarked:** Authors tested 50 modern tracking algorithms. All struggled —
  significant room for improvement.
- **Training split:** Provided. Suitable for both training and evaluation.

**Limitations:** Very new (Dec 2025). Availability may be pending — the GitHub
repo is linked in the paper but confirm it's downloadable before committing.

**Priority for QuadTrack: HIGHEST** — this is the domain you're deploying in.

### 2.2 Anti-UAV Challenge ★★★★

**The standard benchmark for anti-UAV tracking.** Three competition iterations
have produced a mature dataset with public training splits.

- **Ground-to-air:** Fixed cameras looking up at UAVs. Sky/clutter backgrounds.
  UAVs are the annotation target. The target appearance transfers well to
  air-to-air (same class of objects), but the static camera vs moving camera
  domain gap remains.
- **IR + RGB:** Both modalities available per sequence. Train on one or both.
- **SOT + Detection tracks:** Competition format provides both single-object
  tracking labels and detection labels.
- **Public training set:** Released for the first time in the 3rd challenge
  (2023). Previous years were test-only.
- **Proven:** Used by 76+ teams. Well-understood evaluation protocol.

**Priority for QuadTrack: HIGH** — best Stage 1 domain adaptation dataset.
Large enough for meaningful training, public, and well-documented.

### 2.3 CST Anti-UAV ★★★★

**Thermal infrared, tiny UAVs, complex scenes.** The first dataset with
complete manual frame-level attribute annotations.

- **Tiny targets:** Specifically designed for small UAV detection. SOTA
  achieves only 35.92% state accuracy — this is a HARD dataset.
- **Thermal only:** No RGB. Useful for night/interceptor thermal sensor.
  Not useful for daytime RGB training.
- **220 sequences, 240K+ frames:** Moderate size. Good for fine-tuning
  after broader domain adaptation.
- **Complex scenes:** Diverse backgrounds, not just clear sky.

**Priority for QuadTrack: MEDIUM** — thermal-only limits applicability for
daytime RGB interceptor. Valuable if you add a thermal sensor.

### 2.4 DroneSOD-30K ★★★★

**30,000 images for detection specifically.** Single-frame, no tracking
continuity — pure detector training data.

- **Detection-only:** Each image is an independent sample. No track
  identities across frames. Perfect for detector pre-training.
- **Diverse weather:** Multiple meteorological conditions. Teaches the
  detector that UAVs look different in rain, fog, sun, cloud.
- **Ground-to-air:** Static camera looking up. Sky + clutter backgrounds.
- **Used by SDD-YOLO:** The paper that introduced this dataset also provides
  baseline results. YOLO11s baseline is what you'd be training.

**Priority for QuadTrack: HIGH** — best dataset for pure detector training
(single-frame mAP optimization). Complements tracking datasets.

### 2.5 MMAUD ★★★★

**The most comprehensive multi-modal anti-UAV dataset.** Includes modalities
no other dataset has: stereo vision, Lidar, radar, audio.

- **Overhead aerial views:** Unique — provides UAV-looking-down-at-UAV angles.
  Transfers to air-to-air better than pure ground-to-air.
- **UAV type classification:** Not just detection — identifies UAV model.
  Useful if you need to distinguish friendly vs hostile.
- **Multi-modal:** If you ever add Lidar/radar to your interceptor, this is
  the training data.
- **Leica ground truth:** High-precision annotations.
- **Heavy machinery noise:** Realistic acoustic interference.

**Priority for QuadTrack: MEDIUM** — the overhead views are valuable for
air-to-air transfer, but the dataset is very large and multi-modal. Start
with RGB-only subsets.

### 2.6 M²E-UAV ★★★

**Event camera, motion-on-motion.** The only dataset where both camera and
target are moving AND the sensor is event-based.

- **Onboard UAV-view:** Camera is ON a flying UAV looking at another UAV.
  Dual-motion. Exactly your scenario — but with an event camera, not RGB.
- **IMU data:** Synchronized IMU measurements. Potentially useful for
  ego-motion compensation or sensor fusion.
- **87K train + 21K val:** Good size for event-based models.
- **Four scene families:** Sunny building-forest, sunny farm-village,
  sunset variants.

**Priority for QuadTrack: LOW** — event cameras are not your current sensor.
Valuable if you switch to event-based for high-speed intercept.

### 2.7 SimD3 ★★★

**High-fidelity synthetic with explicit payload and bird distractors.**

- **UE5 engine:** Photorealistic rendering with controlled weather, lighting,
  and flight trajectories.
- **Payload modeling:** Drones carry heterogeneous payloads — changes
  appearance significantly. Good for robustness.
- **Bird distractors:** The #1 false positive source for drone detection.
  Training with explicit bird negatives improves precision.
- **360° six-camera rig:** Multi-view. Can generate air-to-air views.
- **Synthetic → real transfer validated:** In-domain and cross-dataset
  evaluation proves the synthetic data helps on real benchmarks.

**Priority for QuadTrack: MEDIUM** — excellent supplementary data for
handling edge cases (payloads, birds, weather). Not a primary training source.

### 2.8 BirDrone / YOLOBirDrone ★★★

**Drone vs bird classification dataset.** The key differentiator: it
explicitly labels BOTH drones and birds.

- **Two-class:** Drone and bird. Binary classification within detection.
- **Challenging small objects:** Designed for distant, small targets where
  birds and drones are visually similar.
- **Large-scale:** Sufficient for training from scratch or fine-tuning.
- **Ground-to-air:** Static camera perspective.

**Priority for QuadTrack: MEDIUM** — the bird-vs-drone distinction is
critical for reducing false positives. Use as supplementary data.

### 2.9 Cranfield Synthetic ★★

**Pure synthetic, validated on real MAV-VID.** Proves synthetic data works
for drone detection.

- **Faster-RCNN trained purely on synthetic** achieves 97.0% AP50 on MAV-VID,
  vs 97.8% for real-data-trained equivalent. Synthetic is nearly as good.
- **HuggingFace hosted:** Easy download. Already in dataset format.
- **Single-scene:** Limited diversity compared to SimD3.

**Priority for QuadTrack: LOW** — SimD3 is better for synthetic data.
Cranfield is a proof-of-concept, not a training resource.

---

## 3. Recommendation for QuadTrack Training

### Stage 1 — Detector pre-training (highest priority)

| Dataset | Role | Why |
|---------|------|-----|
| **DroneSOD-30K** | Primary detector training | 30K images, detection-optimized, diverse weather |
| **Anti-UAV Challenge** | Supplementary | Tracking annotations, proven benchmark, public |

Train YOLO11s on DroneSOD-30K first (pure detection mAP optimization), then
fine-tune on Anti-UAV Challenge (adds temporal awareness via tracking labels).

### Stage 2 — Air-to-air domain adaptation (highest priority)

| Dataset | Role | Why |
|---------|------|-----|
| **UAV-Anti-UAV** | Primary air-to-air | THE domain match. Moving platform, dual-motion, sky backgrounds. |

This is the most important fine-tuning step. The model from Stage 1 knows what
UAVs look like. UAV-Anti-UAV teaches it the dual-motion dynamics and sky
backgrounds of actual air-to-air engagement.

### Stage 3 — Robustness and edge cases (medium priority)

| Dataset | Role | Why |
|---------|------|-----|
| **SimD3** | Bird distractors, payloads, weather | Reduces false positives, improves robustness |
| **BirDrone** | Bird vs drone discrimination | Supplementary negatives |

### Stage 4 — Instance specialization (optional)

| Dataset | Role | Why |
|---------|------|-----|
| **Custom footage** | Your specific UAV type | 500-1000 annotated frames of YOUR target drone |

### Datasets NOT recommended for QuadTrack

| Dataset | Reason |
|---------|--------|
| VisDrone | UAV-as-platform, targets are people/vehicles on ground |
| UAVDT | UAV-as-platform, targets are vehicles on road |
| UAV123 / DTB70 | Generic object tracking from drone, not UAV-as-target |
| DOTA | Aerial imagery of buildings/vehicles, not UAV detection |

---

## 4. Dataset Format Notes

### Annotation formats vary across datasets:

| Dataset | Format | Bbox convention |
|---------|--------|-----------------|
| Anti-UAV Challenge | JSON: `{"gt_rect": [[x,y,w,h],...], "exist": [0,1,...]}` | Top-left + size |
| UAV-Anti-UAV | TBD (paper says "bounding boxes, language prompt, 15 attributes") | TBD |
| DroneSOD-30K | TBD (likely YOLO or COCO format, as it's used with YOLO training) | TBD |
| CST Anti-UAV | TBD ("frame-level attribute annotations") | TBD |
| SimD3 | YOLO format (used with YOLOv5 training in paper) | Center + size normalized |
| BirDrone | TBD | TBD |

**Conversion needed:** All datasets must be converted to YOLO format
(`class x_center y_center width height`, normalized 0-1) for Ultralytics
training. Conversion scripts live in `training/scripts/`.

### Training format (YOLO):

```
dataset_root/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt    # one row per object: 0 x_center y_center w h
    labels/val/*.txt
```

Class 0 = UAV (all datasets are single-class or we collapse to single-class).

---

## 5. Key Verification Notes

Every dataset listed in Section 1 was verified from its paper abstract to
ensure **UAV is the annotation target class**, not merely the camera platform.

Excluded datasets specifically checked:
- **VisDrone:** "VisDrone2019-DET dataset" — targets are pedestrians, vehicles, bicycles. UAV is the camera platform.
- **UAVDT:** "Unmanned Aerial Vehicle Detection and Tracking" — UAV-mounted camera, ground vehicle targets.
- **UAV123 / DTB70:** Single-object tracking benchmarks with generic objects (cars, people, animals) filmed FROM a drone. Not UAV-as-target.
- **DOTA:** "Dataset for Object deTection in Aerial images" — buildings, vehicles, ships. Not UAV-specific.
- **GOT-10k:** Generic object tracking. 563 object classes, UAV is not specifically one of them.

---

## 6. References

| Dataset | Paper | Code/Download |
|---------|-------|---------------|
| UAV-Anti-UAV | [arXiv:2512.07385](https://arxiv.org/abs/2512.07385) | [GitHub](https://github.com/983632847/Awesome-Multimodal-Object-Tracking) |
| Anti-UAV Challenge | [arXiv:2305.07290](https://arxiv.org/abs/2305.07290) | [GitHub](https://github.com/ZhaoJ9014/Anti-UAV) |
| CST Anti-UAV | [arXiv:2507.23473](https://arxiv.org/abs/2507.23473) | To be released |
| DroneSOD-30K | [arXiv:2603.25218](https://arxiv.org/abs/2603.25218) | Via SDD-YOLO paper authors |
| MMAUD | [arXiv:2402.03706](https://arxiv.org/abs/2402.03706) | [GitHub](https://github.com/ntu-aris/MMAUD) |
| M²E-UAV | [arXiv:2605.10496](https://arxiv.org/abs/2605.10496) | [GitHub](https://github.com/Wickyan/M2E-UAV) |
| SimD3 | [arXiv:2601.14742](https://arxiv.org/abs/2601.14742) | Via paper authors |
| BirDrone | [arXiv:2601.08319](https://arxiv.org/abs/2601.08319) | Via YOLOBirDrone paper |
| Cranfield Synthetic | [arXiv:2411.09077](https://arxiv.org/abs/2411.09077) | [HuggingFace](https://huggingface.co/datasets/mazqtpopx/cranfield-synthetic-drone-detection) |
| Perception-to-Pursuit | [arXiv:2601.19318](https://arxiv.org/abs/2601.19318) | Uses Anti-UAV-RGBT (226 sequences) |
