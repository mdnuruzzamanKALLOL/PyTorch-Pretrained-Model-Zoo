# MobileNet V3

**Paper:** Searching for MobileNetV3 (Howard et al., ICCV 2019)
**Authors:** Andrew Howard, Ruoming Pang, Hartwig Adam, Quoc V. Le, Mark Sandler,
             Bo Chen, Weijun Wang, Liang-Chieh Chen, Mingxing Tan, Grace Chu, Vijay Vasudevan, Yukun Zhu
**Institution:** Google AI / Google Brain

---

## Overview

MobileNetV3 combines three complementary techniques to push the accuracy–efficiency frontier:

1. **Neural Architecture Search (NAS)** — same platform-aware NAS as MNASNet, used for the coarse macro-architecture.
2. **NetAdapt algorithm** — fine-grained per-layer tuning to meet a latency budget after the NAS stage.
3. **Novel design improvements** — HardSwish activation (approximates Swish without exp()), HardSigmoid in SE modules, and a redesigned classifier head that moves the expensive average-pooling before the projection.

Two model families are released targeting different latency regimes: **Large** and **Small**.

---

## Variants

| Variant          | Blocks | Parameters | Top-1 (ImageNet) | Pretrained       | Batch Size |
|------------------|--------|-----------|-----------------|------------------|------------|
| MobileNet V3 Large | 15  | ~5.5M     | ~74.0%          | Yes (V1)         | 32         |
| MobileNet V3 Small | 11  | ~2.5M     | ~67.7%          | Yes (V1)         | 64         |

---

## Architecture Pipeline

```
Input (3×224×224)
    │
    ▼
Stem: Conv3×3/stride=2 → BN → HardSwish      [→ 16 × 112×112]
    │
    ▼
InvertedResidual blocks (with optional SE + HardSwish per block)
    │
    ▼
Head Conv: Conv1×1 → last_conv_ch → BN → HardSwish
    │
    ▼
AdaptiveAvgPool(1×1) → Flatten
    │
    ▼
Classifier:
  Linear(last_conv_ch → last_channel) → HardSwish → Dropout(0.2) → Linear(last_channel → classes)
       ↑ [0]                            ↑ [1]        ↑ [2]           ↑ [3] ← replace for transfer
```

---

## Block Configuration — Large

Input channels start at 16 (after stem).

| # | Kernel | Expand Ch | Out Ch | SE    | Act | Stride |
|---|--------|-----------|--------|-------|-----|--------|
| 1 | 3×3    | 16        | 16     | No    | RE  | 1      |
| 2 | 3×3    | 64        | 24     | No    | RE  | 2      |
| 3 | 3×3    | 72        | 24     | No    | RE  | 1      |
| 4 | 5×5    | 72        | 40     | **Yes** | RE | 2     |
| 5 | 5×5    | 120       | 40     | **Yes** | RE | 1     |
| 6 | 5×5    | 120       | 40     | **Yes** | RE | 1     |
| 7 | 3×3    | 240       | 80     | No    | **HS** | 2  |
| 8 | 3×3    | 200       | 80     | No    | **HS** | 1  |
| 9 | 3×3    | 184       | 80     | No    | **HS** | 1  |
|10 | 3×3    | 184       | 80     | No    | **HS** | 1  |
|11 | 3×3    | 480       | 112    | **Yes** | **HS** | 1 |
|12 | 3×3    | 672       | 112    | **Yes** | **HS** | 1 |
|13 | 5×5    | 672       | 160    | **Yes** | **HS** | 2 |
|14 | 5×5    | 960       | 160    | **Yes** | **HS** | 1 |
|15 | 5×5    | 960       | 160    | **Yes** | **HS** | 1 |

Head Conv: 160 → **960** → BN → HS | Classifier: 960 → **1280** → HS → Dropout → FC

---

## Block Configuration — Small

| # | Kernel | Expand Ch | Out Ch | SE    | Act | Stride |
|---|--------|-----------|--------|-------|-----|--------|
| 1 | 3×3    | 16        | 16     | **Yes** | RE | 2     |
| 2 | 3×3    | 72        | 24     | No    | RE  | 2      |
| 3 | 3×3    | 88        | 24     | No    | RE  | 1      |
| 4 | 5×5    | 96        | 40     | **Yes** | **HS** | 2 |
| 5 | 5×5    | 240       | 40     | **Yes** | **HS** | 1 |
| 6 | 5×5    | 240       | 40     | **Yes** | **HS** | 1 |
| 7 | 5×5    | 120       | 48     | **Yes** | **HS** | 1 |
| 8 | 5×5    | 144       | 48     | **Yes** | **HS** | 1 |
| 9 | 5×5    | 288       | 96     | **Yes** | **HS** | 2 |
|10 | 5×5    | 576       | 96     | **Yes** | **HS** | 1 |
|11 | 5×5    | 576       | 96     | **Yes** | **HS** | 1 |

Head Conv: 96 → **576** → BN → HS | Classifier: 576 → **1024** → HS → Dropout → FC

RE = ReLU, HS = HardSwish

---

## InvertedResidual Block

```
[if in_ch != exp_ch]  Conv1×1(expand) → BN → Act
                      DWConv(k×k, stride) → BN → Act
[if use_se]           SqueezeExcitation
                      Conv1×1(project) → BN
[+ residual if stride==1 and in_ch==out_ch]
```

### Squeeze-and-Excitation Module

```python
SE: AdaptiveAvgPool(1×1) → Conv1×1(ch → ch//4) → ReLU → Conv1×1(ch//4 → ch) → HardSigmoid
output = input × SE(input)
```

---

## Activation Functions

### HardSwish (replaces Swish)
```python
HardSwish(x) = x × hardtanh(x + 3, 0, 6) / 6
```
Approximates `x × sigmoid(x)` but avoids the expensive `exp()` operation. Used in the second half of both models and in all classifier layers.

### HardSigmoid (in SE modules)
```python
HardSigmoid(x) = hardtanh(x + 3, 0, 6) / 6
```
Approximates `sigmoid(x)`. Always used inside SE modules.

---

## Classifier Head Structure

| Layer      | Large                  | Small                  |
|------------|------------------------|------------------------|
| `[0]`      | Linear(960 → 1280)     | Linear(576 → 1024)     |
| `[1]`      | HardSwish()            | HardSwish()            |
| `[2]`      | Dropout(0.2)           | Dropout(0.2)           |
| `[3]`      | Linear(1280 → classes) | Linear(1024 → classes) |

**Replace `classifier[3]` for transfer learning.**

`in_features`: **1280** (Large) / **1024** (Small)

---

## Training Configuration (From Scratch)

| Setting       | Large   | Small   |
|---------------|---------|---------|
| Input Size    | 224×224 | 224×224 |
| Batch Size    | 32      | 64      |
| Optimizer     | Adam (lr=1e-3) | Adam (lr=1e-3) |
| Scheduler     | CosineAnnealingLR (T_max=20) | CosineAnnealingLR (T_max=20) |
| Loss          | CrossEntropyLoss | CrossEntropyLoss |
| Epochs        | 20      | 20      |
| Normalization | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | same |

---

## Transfer Learning Quick Reference

### Load Pretrained Weights

```python
from torchvision import models
import torch.nn as nn

# Large
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(1280, NUM_CLASSES)

# Small
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(1024, NUM_CLASSES)
```

### Feature Extraction (freeze backbone)

```python
for param in model.parameters():
    param.requires_grad = False

model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=1e-3)
```

### Fine-Tuning (dual learning rates)

```python
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier.3' not in n],
     'lr': 1e-5},
    {'params': model.classifier[3].parameters(),
     'lr': 1e-3},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
```

---

## Pretrained Weights (torchvision)

| Variant | Function                      | Weights Enum                             | in_features |
|---------|------------------------------|------------------------------------------|-------------|
| Large   | `models.mobilenet_v3_large()` | `MobileNet_V3_Large_Weights.IMAGENET1K_V1` | 1280      |
| Small   | `models.mobilenet_v3_small()` | `MobileNet_V3_Small_Weights.IMAGENET1K_V1` | 1024      |

---

## Folder Structure

```
MobileNet V3/
├── README.md                            ← this file
├── MobileNet V3 Large/
│   ├── NoteBook/
│   │   └── mobilenet_v3_large.ipynb     — full notebook (arch + train + ROC AUC)
│   ├── Python Scripts/
│   │   ├── mobilenet_v3_large.py        — model architecture
│   │   ├── train.py                     — training loop
│   │   ├── inference.py                 — single-image prediction
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py           — load torchvision weights
│       ├── feature_extraction.py        — frozen backbone training
│       ├── fine_tuning.py               — dual-LR fine-tuning
│       └── How to run.txt
└── MobileNet V3 Small/  (same structure)
```

---

## Comparison with MobileNet Family

| Model            | Params  | Top-1  | Key Additions over V2            |
|------------------|---------|--------|----------------------------------|
| MobileNetV2      | ~3.4M   | 72.0%  | Baseline                         |
| MNASNet 1.0      | ~4.4M   | 73.5%  | Hardware-aware NAS               |
| MobileNetV3 Small| ~2.5M   | 67.7%  | NAS + NetAdapt + HardSwish + SE  |
| MobileNetV3 Large| ~5.5M   | 74.0%  | NAS + NetAdapt + HardSwish + SE  |

---

## Citation

```bibtex
@inproceedings{howard2019searching,
  title     = {Searching for MobileNetV3},
  author    = {Howard, Andrew and Pang, Ruoming and Adam, Hartwig and Le, Quoc V
               and Sandler, Mark and Chen, Bo and Wang, Weijun and Chen, Liang-Chieh
               and Tan, Mingxing and Chu, Grace and Vasudevan, Vijay and Zhu, Yukun},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV)},
  year      = {2019}
}
```
