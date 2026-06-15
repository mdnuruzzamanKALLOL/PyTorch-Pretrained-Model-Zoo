# Swin Transformer (T / S / B / V2-T / V2-S / V2-B) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** Swin Transformer T S B V2 PyTorch pretrained 2021 ICCV shifted window hierarchical ViT ImageNet classification detection segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

Swin Transformer introduces hierarchical Vision Transformer features via shifted window self-attention, achieving O(N) complexity and linear-scaled feature maps suitable for dense prediction. PyTorch torchvision provides six variants: Swin-T/S/B and their SwinV2 counterparts with improved training stability at higher resolutions.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `swin_t` | 28 M | 224² | 81.5% | 95.5% |
| `swin_s` | 50 M | 224² | 83.2% | 96.2% |
| `swin_b` | 88 M | 224² | 83.6% | 96.5% |
| `swin_v2_t` | 28 M | 256² | 82.1% | 96.0% |
| `swin_v2_s` | 50 M | 256² | 83.7% | 96.6% |
| `swin_v2_b` | 88 M | 256² | 84.1% | 96.9% |

---

## Architecture Highlights

- Hierarchical patch embedding: 4× patches with 2× merge at each stage (like ResNet stages)
- Window self-attention: local W×W windows reduce quadratic complexity to linear
- Shifted windows (SW-MSA): cross-window connections for global context
- Relative position bias for generalization to different window sizes at inference
- V2: log-scale continuous relative position bias + cosine attention for high-resolution

---

## When to Use Swin Transformer (T / S / B / V2-T / V2-S / V2-B)

Use Swin-T/S for most tasks — it matches or beats ResNet-50/101 with better transfer. Use SwinV2 when training at higher resolutions (256²+) or larger scale.

---

## Real-World Use Cases

- Object detection (COCO): Swin-T/S/B as Cascade Mask R-CNN backbone
- Semantic segmentation (ADE20K): Swin + UperNet state-of-the-art
- Panoptic segmentation and instance segmentation at high mAP
- Image restoration: Swin architecture used in SwinIR (super-resolution, denoising)
- SwinV2-B/L for high-resolution fine-tuning tasks (up to 1024²)

---

## Folder Structure

```
SwinTransformer/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

# Swin-T
model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

# Swin V2-B
model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
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
model = models.swin_t(weights="IMAGENET1K_V1")

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
@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={ICCV},
  pages={10012--10022},
  year={2021}
}
```

**Paper:** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
**Authors:** Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
**Venue:** ICCV 2021  **arXiv:** https://arxiv.org/abs/2103.14030

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.SwinTransformer&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
