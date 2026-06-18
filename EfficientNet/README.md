# EfficientNet (B0–B7) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** EfficientNet B0-B7 PyTorch pretrained ImageNet 2019 ICML compound scaling MBConv SE transfer learning classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

EfficientNet B0–B7 applies compound scaling — jointly scaling network depth, width, and input resolution with a fixed ratio — to create a family of models spanning 77.1% (B0) to 84.3% (B7) ImageNet top-1 accuracy. PyTorch torchvision added all eight variants in 2022 with pretrained ImageNet weights.

---

## Variants & ImageNet Performance

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `efficientnet_b0` | 5.3 M | 224² | 77.7% | 93.6% |
| `efficientnet_b1` | 7.8 M | 240² | 78.6% | 94.3% |
| `efficientnet_b2` | 9.1 M | 288² | 80.6% | 95.3% |
| `efficientnet_b3` | 12 M | 300² | 82.0% | 96.1% |
| `efficientnet_b4` | 19 M | 380² | 83.4% | 96.6% |
| `efficientnet_b5` | 30 M | 456² | 83.4% | 96.7% |
| `efficientnet_b6` | 43 M | 528² | 84.0% | 96.9% |
| `efficientnet_b7` | 66 M | 600² | 84.1% | 97.0% |

</div>

---

## Architecture Highlights

- Compound scaling: depth d, width w, resolution r scaled by φ with fixed ratio
- MBConv blocks with Squeeze-and-Excitation (SE) attention at ratio 0.25
- Swish activation (β=1) discovered by automated search
- NAS baseline (B0) scaled uniformly to create B1–B7
- B0 is 8.4× smaller and 6.1× faster than ResNet-50 at better accuracy

---

## When to Use EfficientNet (B0–B7)

Use B0 or B1 for edge/mobile. Use B3 as the versatile production sweet spot. For new projects consider EfficientNetV2 variants which train faster at similar accuracy.

---

## Real-World Use Cases

- B0–B2: mobile and edge inference via TorchScript / ONNX export
- B3–B4: server-side API classification (82–83% accuracy range)
- B5–B7: research and high-accuracy ensemble systems
- EfficientDet backbone for object detection pipelines

---

## Folder Structure

```
EfficientNet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# Replace b0 with b1..b7 for other variants
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
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

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
@inproceedings{tan2019efficientnet,
  title={{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={ICML},
  pages={6105--6114},
  year={2019}
}
```

**Paper:** EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
**Authors:** Mingxing Tan, Quoc V. Le
**Venue:** ICML 2019  **arXiv:** https://arxiv.org/abs/1905.11946

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.EfficientNet&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
