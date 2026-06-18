# MobileNetV3 (Large / Small) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** MobileNetV3 Large Small PyTorch pretrained 2019 ICCV NAS Hard-Swish SE mobile classification edge

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

MobileNetV3 combines NAS architecture search with NetAdapt to co-optimize accuracy and latency, adding Hard-Swish activation and SE modules. Two variants cover different speed targets: Large (75.3% top-1) for standard mobile, Small (67.7%) for ultra-constrained devices.

---

## Variants & ImageNet Performance

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `mobilenet_v3_large` | 5.4 M | 224² | 75.3% | 92.6% |
| `mobilenet_v3_small` | 2.5 M | 224² | 67.7% | 87.4% |

</div>

---

## Architecture Highlights

- NAS + NetAdapt for co-optimizing accuracy and latency jointly
- Hard-Swish activation: approximation of Swish with integer-friendly hardware ops
- SE blocks on the expensive bottleneck layers only (cost-efficient placement)
- Efficient last stage: replaces expensive stage with 1×1 conv after pooling

---

## When to Use MobileNetV3 (Large / Small)

Use V3-Large when MobileNetV2 accuracy (71.9%) is insufficient for mobile. V3-Small targets ultra-constrained < 3 M parameter budgets.

---

## Real-World Use Cases

- Production mobile apps on Android (Large) and wearables/IoT (Small)
- Quantization-friendly deployment to ARM NPU hardware
- Real-time video classification at > 30 FPS on mobile

---

## Folder Structure

```
MobileNet V3/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
large.eval()
```

---

## Transfer Learning

```python
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 10  # replace with your class count

# Load pretrained backbone
model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

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
@inproceedings{howard2019searching,
  title={Searching for {MobileNetV3}},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V and Adam, Hartwig},
  booktitle={ICCV},
  pages={1314--1324},
  year={2019}
}
```

**Paper:** Searching for MobileNetV3
**Authors:** Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
**Venue:** ICCV 2019  **arXiv:** https://arxiv.org/abs/1905.02244

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.MobileNet-V3&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
