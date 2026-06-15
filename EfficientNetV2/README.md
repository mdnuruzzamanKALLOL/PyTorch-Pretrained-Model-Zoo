# EfficientNetV2

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
![Paper](https://img.shields.io/badge/Paper-ICML%202021-blue)

PyTorch implementations of **EfficientNetV2-S/M/L** from scratch, plus pretrained-weight workflows.

> Tan, M., & Le, Q. V. (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML 2021.

---

## Architecture Overview

```
Input (3 × H × W)
       │
  ┌────▼────┐
  │  Stem   │  Conv2d 3×3, stride 2  →  BN → SiLU
  └────┬────┘
       │
  ┌────▼─────────────────────────────────────────────┐
  │  Early Stages: Fused-MBConv blocks                │
  │  ┌────────────────────────────────────────────┐  │
  │  │ Conv(k×k, expand) → BN → SiLU              │  │
  │  │ Conv(1×1, project) → BN                    │  │
  │  │ + residual (if same shape)                  │  │
  │  └────────────────────────────────────────────┘  │
  ├──────────────────────────────────────────────────┤
  │  Later Stages: MBConv blocks (with SE)            │
  │  ┌────────────────────────────────────────────┐  │
  │  │ [Expand Conv1×1 → BN → SiLU]              │  │
  │  │  DWConv k×k → BN → SiLU                   │  │
  │  │  SE: AvgPool → FC → SiLU → FC → Sigmoid   │  │
  │  │  Project Conv1×1 → BN                      │  │
  │  │  + residual (if same shape)                 │  │
  │  └────────────────────────────────────────────┘  │
  └────┬─────────────────────────────────────────────┘
       │
  ┌────▼────┐
  │  Head   │  Conv1×1(→1280) → BN → SiLU → AvgPool → Dropout → Linear
  └────┬────┘
       │
  Output (num_classes)
```

---

## Model Variants

| Model | Stages (F/M) | Resolution | Dropout | Params   | Batch |
|-------|-------------|-----------|---------|----------|-------|
| S     | 3F + 3M     | 300×300   | 0.2     | ~21.5M   | 16    |
| M     | 3F + 4M     | 384×384   | 0.3     | ~54.1M   | 8     |
| L     | 3F + 4M     | 480×480   | 0.4     | ~118.5M  | 4     |

F = Fused-MBConv  |  M = MBConv with Squeeze-and-Excitation

---

## Key Improvements over EfficientNetV1

| Feature | V1 | V2 |
|---------|----|----|
| Block type | MBConv only | Fused-MBConv (early) + MBConv (late) |
| SE in early stages | Yes | No (Fused-MBConv has no SE) |
| Training | Fixed resolution | Progressive learning (paper) |
| Speed | Baseline | 5–11× faster training |
| Scaling | Compound (width/depth) | Pre-defined stage configs |

---

## Folder Structure

```
EfficientNetV2/
├── EfficientNetV2 S/
│   ├── NoteBook/
│   │   └── efficientnetv2_s.ipynb
│   ├── Python Scripts/
│   │   ├── efficientnetv2_s.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py
│       ├── feature_extraction.py
│       ├── fine_tuning.py
│       └── How to run.txt
├── EfficientNetV2 M/   (same structure)
├── EfficientNetV2 L/   (same structure)
└── README.md
```

---

## Quick Start (From Scratch)

```python
import torch
from efficientnetv2_s import efficientnetv2_s

model  = efficientnetv2_s(num_classes=10)
dummy  = torch.randn(1, 3, 300, 300)
output = model(dummy)
print(output.shape)  # torch.Size([1, 10])
```

---

## Quick Start (Pretrained Weights)

```python
import torch.nn as nn
from torchvision import models

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

in_features = model.classifier[1].in_features  # 1280
model.classifier[1] = nn.Linear(in_features, 10)
```

---

## Transfer Learning Approaches

| Approach | When to Use | Backbone | Head LR | Backbone LR |
|----------|------------|---------|---------|------------|
| Feature Extraction | Small dataset / fast training | Frozen | 1e-3 | — |
| Fine-Tuning | Larger dataset / best accuracy | Unfrozen | 1e-3 | 1e-5 |

---

## Pretrained Classifier Head (all variants)

| Model | `classifier[1].in_features` |
|-------|----------------------------|
| S     | 1280 |
| M     | 1280 |
| L     | 1280 |

---

## Citation

```bibtex
@inproceedings{tan2021efficientnetv2,
  title     = {EfficientNetV2: Smaller Models and Faster Training},
  author    = {Tan, Mingxing and Le, Quoc V},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2021}
}
```
