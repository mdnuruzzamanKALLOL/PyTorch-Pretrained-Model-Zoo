# EfficientNetV2 (S / M / L) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNetV2 S M L PyTorch pretrained ImageNet 2021 ICML Fused-MBConv progressive learning fast training classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNetV2 redesigns the scaling strategy and adds Fused-MBConv blocks in early stages, training 5–11× faster than EfficientNet-B7 while achieving higher accuracy. PyTorch torchvision provides S/M/L variants pretrained on ImageNet-1K.

---

<div align="center">

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `efficientnet_v2_s` | 21.5 M | 384² | 84.2% | 96.9% |
| `efficientnet_v2_m` | 54.1 M | 480² | 85.1% | 97.4% |
| `efficientnet_v2_l` | 119 M | 480² | 85.8% | 97.8% |

</div>

---

## Architecture Highlights

- Fused-MBConv in stages 1–3: 3×3 fused conv instead of expand+depthwise+project
- Progressive learning: image size and augmentation scale together during training
- NAS optimized for training speed (seconds/step) not just inference FLOPs
- V2-S trains 6.8× faster than EfficientNet-B7 at comparable accuracy

---

## When to Use EfficientNetV2 (S / M / L)

Always prefer V2-S/M/L over EfficientNet-B6/B7 in new projects — faster training, better accuracy, modern architecture.

---

## Real-World Use Cases

- High-accuracy server-side classification (84–86% range)
- Medical imaging and satellite imagery requiring large input resolution
- EfficientDetV2 backbone for object detection

---

## Folder Structure

```
EfficientNetV2/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
# Replace _s with _m or _l for other variants
model.eval()
```

---

## Transfer Learning

```python
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 10  # replace with your class count

# Load pretrained backbone
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")

# Replace the classifier head
if hasattr(model, "fc"):
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
elif hasattr(model, "classifier"):
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

# Freeze backbone for initial training
for param in list(model.parameters())[:-4]:
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
criterion = nn.CrossEntropyLoss()
```

---

## Citation

```bibtex
@inproceedings{tan2021efficientnetv2,
  title={{EfficientNetV2}: Smaller Models and Faster Training},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={ICML},
  pages={10096--10106},
  year={2021}
}
```

**Paper:** EfficientNetV2: Smaller Models and Faster Training
**Authors:** Mingxing Tan, Quoc V. Le
**Venue:** ICML 2021  **arXiv:** https://arxiv.org/abs/2104.00298

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.EfficientNetV2&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
