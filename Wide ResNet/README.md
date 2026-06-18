# Wide ResNet (WRN-50-2 / WRN-101-2) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** Wide ResNet WRN-50-2 WRN-101-2 PyTorch pretrained 2016 BMVC width multiplier 81.6% ImageNet classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

Wide ResNets increase the width (channel multiplier ×2) of ResNet bottleneck blocks instead of increasing depth, achieving better accuracy at similar or fewer parameters because wider networks are more efficiently parallelizable on modern GPUs. WRN-50-2 achieves 81.6% ImageNet top-1 vs ResNet-50's 76.1%.

---

## Variants & ImageNet Performance

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `wide_resnet50_2` | 68.9 M | 224² | 81.6% | 95.8% |
| `wide_resnet101_2` | 126.9 M | 224² | 82.5% | 96.2% |

</div>

---

## Architecture Highlights

- Width multiplier k=2: all ResNet bottleneck channels doubled
- Fewer layers but wider: better GPU utilization than deep thin networks
- WRN-50-2: 68.9 M params vs ResNet-50's 25.6 M — 5.5% accuracy gain
- WRN-101-2: 126.9 M params achieves 82.5% — matches ConvNeXt-Small performance
- Dropout in bottleneck residual blocks for regularization

---

## When to Use Wide ResNet (WRN-50-2 / WRN-101-2)

Use Wide ResNet when you want the ResNet family's simplicity but need 80%+ accuracy. WRN-50-2 (81.6%) outperforms ResNet-152 (78.3%) with simpler architecture. WRN-101-2 competes with ConvNeXt-Small at 126 M parameters.

---

## Real-World Use Cases

- Improved ResNet baseline for classification benchmarks
- Detection and segmentation backbone where width adds more than depth
- Knowledge distillation with wider teacher networks
- Semi-supervised and self-supervised learning teacher models

---

## Folder Structure

```
Wide ResNet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
# or wide_resnet101_2
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
model = models.wide_resnet50_2(weights="IMAGENET1K_V1")

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
@inproceedings{zagoruyko2016wide,
  title={Wide Residual Networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle={BMVC},
  year={2016}
}
```

**Paper:** Wide Residual Networks
**Authors:** Sergey Zagoruyko, Nikos Komodakis
**Venue:** BMVC 2016  **arXiv:** https://arxiv.org/abs/1605.07146

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.Wide-ResNet&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
