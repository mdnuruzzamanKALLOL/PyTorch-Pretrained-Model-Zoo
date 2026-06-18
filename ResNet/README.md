# ResNet (18 / 34 / 50 / 101 / 152) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** ResNet 18 34 50 101 152 PyTorch pretrained ImageNet 2016 CVPR residual learning backbone transfer learning classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ResNet introduced identity shortcut connections that allow gradients to flow through very deep networks, winning ILSVRC 2015 with a 3.57% top-5 error. PyTorch torchvision provides five variants (18/34/50/101/152) plus V2 pre-activation versions, making ResNet the most universally-used backbone in computer vision.

---

## Variants & ImageNet Performance

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `resnet18` | 11.7 M | 224² | 69.8% | 89.1% |
| `resnet34` | 21.8 M | 224² | 73.3% | 91.4% |
| `resnet50` | 25.6 M | 224² | 76.1% | 92.9% |
| `resnet101` | 44.5 M | 224² | 77.4% | 93.5% |
| `resnet152` | 60.2 M | 224² | 78.3% | 94.1% |

</div>

---

## Architecture Highlights

- Identity shortcuts: output = F(x) + x bypasses stacked layers
- Projection shortcuts (1×1 conv) when channel dimensions differ
- ResNet-18/34 use basic blocks (3×3 + 3×3); 50/101/152 use bottleneck (1×1+3×3+1×1)
- Global average pooling before classifier — no fully-connected layers in backbone
- The single most referenced CNN architecture in all of deep learning research

---

## When to Use ResNet (18 / 34 / 50 / 101 / 152)

ResNet-50 is the universal baseline — use it for any new project's initial experiment. ResNet-18 for speed-constrained systems; ResNet-101/152 for dense prediction tasks.

---

## Real-World Use Cases

- Universal classification, detection, and segmentation backbone
- Faster R-CNN, Mask R-CNN, and Detectron2 default backbone
- ResNet-50 is the de facto baseline for fair comparison in papers
- Knowledge distillation: ResNet-18 student / ResNet-50 teacher

---

## Folder Structure

```
ResNet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# Replace resnet50 with resnet18 / resnet34 / resnet101 / resnet152
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
model = models.resnet18(weights="IMAGENET1K_V1")

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
@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  pages={770--778},
  year={2016}
}
```

**Paper:** Deep Residual Learning for Image Recognition
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Venue:** CVPR 2016  **arXiv:** https://arxiv.org/abs/1512.03385

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.ResNet&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
