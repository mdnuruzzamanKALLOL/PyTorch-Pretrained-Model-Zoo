# VGG (11 / 13 / 16 / 19, with/without BatchNorm) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** VGG 11 13 16 19 BatchNorm PyTorch pretrained ImageNet 2014 ICLR style transfer perceptual loss GAN classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

The VGG family uses exclusively 3×3 convolutional filters in networks from 11 to 19 layers deep, proving that depth with small filters is the key driver of accuracy. PyTorch torchvision provides 8 variants including BatchNorm versions (vgg11_bn etc.) that improve accuracy by 1–2% with faster training convergence.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `vgg11` | 132.9 M | 224² | 69.0% | 88.6% |
| `vgg11_bn` | 132.9 M | 224² | 70.4% | 89.8% |
| `vgg13` | 133.0 M | 224² | 69.9% | 89.2% |
| `vgg13_bn` | 133.0 M | 224² | 71.6% | 90.4% |
| `vgg16` | 138.4 M | 224² | 71.6% | 90.4% |
| `vgg16_bn` | 138.4 M | 224² | 73.4% | 91.5% |
| `vgg19` | 143.7 M | 224² | 72.4% | 90.9% |
| `vgg19_bn` | 143.7 M | 224² | 74.2% | 91.8% |

---

## Architecture Highlights

- Uniform 3×3, stride-1 convolutions throughout all stages
- 5 max-pooling layers for 32× total spatial downsampling
- 3 large fully-connected layers: 4096→4096→1000
- BN variants add BatchNorm after each conv: 1–2% accuracy gain, faster convergence
- Simple architecture ideal for educational understanding and feature visualization

---

## When to Use VGG (11 / 13 / 16 / 19, with/without BatchNorm)

Use VGG-16 or VGG-19 only for neural style transfer and perceptual loss computation — the field defines perceptual loss using VGG features. Always use BatchNorm variants for better training stability. For classification, ResNet50 is far more efficient.

---

## Real-World Use Cases

- Neural style transfer: VGG-16/19 feature maps define content and style losses
- Perceptual loss for SRGAN, pix2pix, CycleGAN, and EnlightenGAN
- Image feature extraction for classic computer vision pipelines
- Educational teaching tool for understanding depth vs accuracy

---

## Folder Structure

```
VGG/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

# VGG-16 with BatchNorm (recommended)
model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)

# VGG-19 for style transfer
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
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
model = models.vgg11(weights="IMAGENET1K_V1")

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
@article{simonyan2014very,
  title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```

**Paper:** Very Deep Convolutional Networks for Large-Scale Image Recognition
**Authors:** Karen Simonyan, Andrew Zisserman
**Venue:** ICLR 2015  **arXiv:** https://arxiv.org/abs/1409.1556

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.VGG&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
