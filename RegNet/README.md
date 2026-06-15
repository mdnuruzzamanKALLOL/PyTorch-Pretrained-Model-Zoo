# RegNet — Designing Network Design Spaces

**Paper:** Designing Network Design Spaces
**Authors:** Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollar
**Conference:** CVPR 2020

---

## Overview

RegNet is not a single network — it is a **design space** of networks found by searching over quantized linear parameterizations of block widths. The key finding: the best networks follow a simple regularity — widths grow linearly with stage index, and the group width is constant across all stages.

**Two series:**
- **RegNet X**: Standard XBlock with group convolution (no attention)
- **RegNet Y**: RegNet X + **Squeeze-and-Excitation** (SE) blocks (se_ratio=0.25)

---

## XBlock (building block)

```
Input
  |
  +-- Conv1x1(in -> w) -> BN -> ReLU          (1x1 project)
      Conv3x3(w -> w, groups=w/gw) -> BN -> ReLU  (group conv)
      [SE(w, 0.25)  if RegNet Y]              (optional SE)
      Conv1x1(w -> out) -> BN                 (1x1 expand)
  +-- shortcut: Identity or Conv1x1/stride BN
  |
  ReLU -> output
```

- `w` = stage width, `gw` = group_width, groups = w / gw
- `bottleneck_multiplier = 1.0` for all standard configs (no compression)

---

## Variants

### RegNet X (se_ratio = 0.0)

| Variant    | depths       | widths               | group_widths         | Params   | Top-1 | in_features |
|------------|--------------|----------------------|----------------------|----------|-------|-------------|
| X_400MF    | [1,2,7,12]   | [32,64,160,384]      | [16,16,16,16]        | ~5.6M    | 72.8% | 384         |
| X_800MF    | [1,3,7,5]    | [64,128,288,672]     | [16,16,16,16]        | ~7.3M    | 75.2% | 672         |
| X_1.6GF    | [2,4,10,2]   | [72,168,408,912]     | [24,24,24,24]        | ~9.2M    | 77.0% | 912         |
| X_3.2GF    | [2,6,15,2]   | [96,192,432,1008]    | [48,48,48,48]        | ~15.3M   | 78.4% | 1008        |
| X_8GF      | [2,5,15,1]   | [80,240,720,1920]    | [120,120,120,120]    | ~39.6M   | 79.3% | 1920        |
| X_16GF     | [2,6,13,1]   | [256,512,896,2048]   | [128,128,128,128]    | ~54.3M   | 80.1% | 2048        |
| X_32GF     | [2,7,13,1]   | [336,672,1344,2520]  | [168,168,168,168]    | ~107.8M  | 80.6% | 2520        |

### RegNet Y (se_ratio = 0.25)

| Variant    | depths       | widths               | group_widths         | Params   | Top-1 | in_features |
|------------|--------------|----------------------|----------------------|----------|-------|-------------|
| Y_400MF    | [1,3,6,6]    | [48,104,208,440]     | [8,8,8,8]            | ~4.3M    | 74.0% | 440         |
| Y_800MF    | [1,3,8,2]    | [64,128,320,768]     | [16,16,16,16]        | ~6.4M    | 76.4% | 768         |
| Y_1.6GF    | [2,6,17,2]   | [48,120,336,888]     | [24,24,24,24]        | ~11.2M   | 77.9% | 888         |
| Y_3.2GF    | [2,5,13,1]   | [72,216,576,1512]    | [24,24,24,24]        | ~19.4M   | 78.9% | 1512        |
| Y_8GF      | [2,17,5,1]   | [168,448,896,2016]   | [56,56,56,56]        | ~39.4M   | 81.7% | 2016        |
| Y_16GF     | [2,4,11,1]   | [224,448,1232,3024]  | [112,112,112,112]    | ~83.6M   | 82.0% | 3024        |
| Y_32GF     | [2,5,12,1]   | [232,696,1392,3712]  | [232,232,232,232]    | ~145.0M  | 82.2% | 3712        |
| Y_128GF    | [2,7,13,1]   | [528,1056,2904,7392] | [264,264,264,264]    | ~644.8M  | 83.4% | 7392        |

Y_128GF uses `IMAGENET1K_SWAG_LINEAR_V1` weights (224x224 compatible).

---

## Architecture Pipeline

```
Input (3x224x224)
    |
Stem: Conv3x3/2 (3->32) -> BN -> ReLU          [32 x 112x112]
    |
Stage 1: depth[0] XBlocks, stride-2 first      [w1 x 56x56]
Stage 2: depth[1] XBlocks, stride-2 first      [w2 x 28x28]
Stage 3: depth[2] XBlocks, stride-2 first      [w3 x 14x14]
Stage 4: depth[3] XBlocks, stride-2 first      [w4 x  7x 7]
    |
AdaptiveAvgPool(1,1) -> Flatten -> FC(w4 -> classes)
```

---

## Training Configuration

| Setting    | MF variants | 1.6-3.2GF | 8-16GF | 32GF+ |
|------------|-------------|-----------|--------|-------|
| Batch Size | 64          | 32        | 16     | 8 (Y_128GF: 4) |
| Optimizer  | Adam (lr=1e-3) for all                        |
| Scheduler  | StepLR (step_size=7, gamma=0.1) for all       |
| Epochs     | 20 (all)                                      |

---

## Transfer Learning

```python
from torchvision import models
import torch.nn as nn

model    = models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Y_128GF — SWAG weights
model    = models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
```

Replace `model.fc` for all variants. `in_features = widths[-1]` of each config.

---

## Folder Structure

```
RegNet/
+-- README.md
+-- RegNet X_400MF/   NoteBook/  Python Scripts/  Using Weight File/
+-- RegNet X_800MF/   ...
+-- RegNet X_1.6GF/   ...
+-- RegNet X_3.2GF/   ...
+-- RegNet X_8GF/     ...
+-- RegNet X_16GF/    ...
+-- RegNet X_32GF/    ...
+-- RegNet Y_400MF/   ...
+-- RegNet Y_800MF/   ...
+-- RegNet Y_1.6GF/   ...
+-- RegNet Y_3.2GF/   ...
+-- RegNet Y_8GF/     ...
+-- RegNet Y_16GF/    ...
+-- RegNet Y_32GF/    ...
+-- RegNet Y_128GF/   ...
```

---

## Citation

```bibtex
@inproceedings{radosavovic2020designing,
  title     = {Designing Network Design Spaces},
  author    = {Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Dollar, Piotr},
  booktitle = {CVPR},
  year      = {2020}
}
```
