# MobileNetV2 — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** MobileNetV2 PyTorch pretrained 3.4M 71.9% ImageNet mobile CVPR 2018 inverted residual SSDLite edge classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

MobileNetV2 introduces Inverted Residual blocks with Linear Bottlenecks: channel expansion inside each block prevents information loss, while residual connections across narrow bottleneck layers improve gradient flow. With 3.4 M parameters it achieves 71.9% ImageNet top-1 and is the backbone for SSDLite mobile object detection and DeepLab V3+ mobile.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `mobilenet_v2` | 3.4 M | 224² | 71.9% | 90.3% |

---

## Architecture Highlights

- Inverted bottleneck: narrow input → expand (t=6) → depthwise → project back narrow
- Linear (no ReLU) output of bottleneck preserves low-dimensional manifold
- Residual connection across bottleneck when stride=1
- SSDLite and DeepLab V3+ compatible mobile backbone

---

## When to Use MobileNetV2

The default mobile backbone for production apps. Use EfficientNet-B0 when accuracy > 77% is needed at similar hardware budget.

---

## Real-World Use Cases

- Real-time mobile classification and detection on Android/iOS
- SSDLite object detection for on-device inference
- Semantic segmentation with DeepLabV3+ for mobile
- Coral Edge TPU and Jetson Nano deployment via TFLite/ONNX

---

## Folder Structure

```
MobileNet V2/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
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
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

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
@inproceedings{sandler2018mobilenetv2,
  title={{MobileNetV2}: Inverted Residuals and Linear Bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={CVPR},
  pages={4510--4520},
  year={2018}
}
```

**Paper:** MobileNetV2: Inverted Residuals and Linear Bottlenecks
**Authors:** Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
**Venue:** CVPR 2018  **arXiv:** https://arxiv.org/abs/1801.04381

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://komarev.com/ghpvc/?username=mdnuruzzamanKALLOL&label=Profile%20Views&color=EE4C2C&style=flat-square)

</div>
