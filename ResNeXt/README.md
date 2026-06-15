# ResNeXt (50_32x4d / 101_32x8d / 101_64x4d) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** ResNeXt 50 101 32x4d 64x4d PyTorch pretrained CVPR 2017 cardinality grouped convolution ImageNet classification detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ResNeXt introduces grouped convolution ('cardinality') as a new dimension to scale, alongside width and depth. Using 32 groups of 4-channel paths (32×4d), ResNeXt-50 outperforms ResNet-50 by 1–2% on ImageNet at the same FLOPs, and ResNeXt-101-32×8d achieves 79.3% top-1 as the backbone of many detection and segmentation systems.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `resnext50_32x4d` | 25 M | 224² | 77.6% | 93.7% |
| `resnext101_32x8d` | 88.8 M | 224² | 79.3% | 94.5% |
| `resnext101_64x4d` | 83.5 M | 224² | 83.2% | 96.4% |

---

## Architecture Highlights

- Cardinality C: number of independent transformation paths (32 or 64)
- Each path has d=4 or d=8 channels — grouped convolution in bottleneck
- Same FLOPs/parameters as ResNet but consistently higher accuracy
- 101-32×8d is Detectron2's high-accuracy detection backbone

---

## When to Use ResNeXt (50_32x4d / 101_32x8d / 101_64x4d)

Use ResNeXt-50-32×4d over ResNet-50 when you want 1–2% better accuracy at identical cost. Use ResNeXt-101-32×8d for high-accuracy detection/segmentation.

---

## Real-World Use Cases

- High-accuracy detection backbone (COCO benchmark)
- Instance segmentation with Mask R-CNN at 79%+ accuracy level
- Panoptic segmentation research requiring the best ResNet-family backbone

---

## Folder Structure

```
ResNeXt/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
# or resnext101_32x8d / resnext101_64x4d
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
model = models.resnext50_32x4d(weights="IMAGENET1K_V1")

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
@inproceedings{xie2017aggregated,
  title={Aggregated Residual Transformations for Deep Neural Networks},
  author={Xie, Saining and Girshick, Ross and Doll\'ar, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={CVPR},
  pages={1492--1500},
  year={2017}
}
```

**Paper:** Aggregated Residual Transformations for Deep Neural Networks
**Authors:** Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
**Venue:** CVPR 2017  **arXiv:** https://arxiv.org/abs/1611.05431

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.ResNeXt&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
