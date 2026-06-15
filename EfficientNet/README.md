# EfficientNet

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
![Paper](https://img.shields.io/badge/Paper-ICML%202019-blue)

PyTorch implementations of **EfficientNet B0–B7** from scratch, plus pretrained-weight workflows.

> Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.

---

## Architecture Overview

```
Input (3 × H × W)
       │
  ┌────▼────┐
  │  Stem   │  Conv2d 3×3, stride 2  →  BN → SiLU
  └────┬────┘
       │
  ┌────▼────────────────────────────────────────┐
  │  MBConv Blocks  (7 stages, stochastic depth) │
  │                                             │
  │  Each MBConv block:                         │
  │  ┌──────────────────────────────────────┐   │
  │  │ [Expand Conv1×1 → BN → SiLU]         │   │
  │  │  DWConv k×k → BN → SiLU             │   │
  │  │  SE: AvgPool → FC → SiLU → FC → σ   │   │
  │  │  Project Conv1×1 → BN               │   │
  │  │  + residual (if same shape)          │   │
  │  └──────────────────────────────────────┘   │
  └────┬────────────────────────────────────────┘
       │
  ┌────▼────┐
  │  Head   │  Conv1×1 → BN → SiLU → AvgPool → Dropout → Linear
  └────┬────┘
       │
  Output (num_classes)
```

---

## Model Variants

| Model | Width | Depth | Resolution | Dropout | Params | Batch Size |
|-------|-------|-------|-----------|---------|--------|------------|
| B0    | 1.0   | 1.0   | 224×224   | 0.2     | ~5.3M  | 64         |
| B1    | 1.0   | 1.1   | 240×240   | 0.2     | ~7.8M  | 32         |
| B2    | 1.1   | 1.2   | 260×260   | 0.3     | ~9.2M  | 32         |
| B3    | 1.2   | 1.4   | 300×300   | 0.3     | ~12M   | 32         |
| B4    | 1.4   | 1.8   | 380×380   | 0.4     | ~19M   | 16         |
| B5    | 1.6   | 2.2   | 456×456   | 0.4     | ~30M   | 16         |
| B6    | 1.8   | 2.6   | 528×528   | 0.5     | ~43M   | 8          |
| B7    | 2.0   | 3.1   | 600×600   | 0.5     | ~66M   | 8          |

---

## Folder Structure

```
EfficientNet/
├── EfficientNet B0/
│   ├── NoteBook/
│   │   └── efficientnet_b0.ipynb
│   ├── Python Scripts/
│   │   ├── efficientnet_b0.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py
│       ├── feature_extraction.py
│       ├── fine_tuning.py
│       └── How to run.txt
├── EfficientNet B1/ ... B7/   (same structure)
└── README.md
```

---

## Quick Start (From Scratch)

```python
import torch
from efficientnet_b0 import efficientnet_b0

model  = efficientnet_b0(num_classes=10)
dummy  = torch.randn(1, 3, 224, 224)
output = model(dummy)
print(output.shape)  # torch.Size([1, 10])
```

---

## Quick Start (Pretrained Weights)

```python
import torch.nn as nn
from torchvision import models

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

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

## Pretrained Classifier Head

| Model | `classifier[1].in_features` |
|-------|----------------------------|
| B0    | 1280 |
| B1    | 1280 |
| B2    | 1408 |
| B3    | 1536 |
| B4    | 1792 |
| B5    | 2048 |
| B6    | 2304 |
| B7    | 2560 |

---

## Citation

```bibtex
@inproceedings{tan2019efficientnet,
  title     = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author    = {Tan, Mingxing and Le, Quoc V},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2019}
}
```
