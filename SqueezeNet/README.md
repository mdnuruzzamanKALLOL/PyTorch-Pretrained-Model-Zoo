# SqueezeNet (1.0 / 1.1) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** SqueezeNet PyTorch pretrained 1.2M ImageNet 2016 Fire module ultra-lightweight < 0.5MB edge IoT classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

SqueezeNet achieves AlexNet-level accuracy (57.5% top-1) with 50× fewer parameters by using Fire modules: a squeeze layer (1×1 conv) followed by an expand layer (1×1 + 3×3 convs concatenated). The 1.1 variant further reduces computation by 2.4× with identical accuracy by reducing spatial downsampling frequency.

---

<div align="center">

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `squeezenet1_0` | 1.2 M | 224² | 58.1% | 80.4% |
| `squeezenet1_1` | 1.2 M | 224² | 58.2% | 80.6% |

</div>

---

## Architecture Highlights

- Fire module: squeeze 1×1 conv → expand with 1×1 + 3×3 in parallel
- < 0.5 MB model size — fits in embedded flash memory
- 50× fewer parameters than AlexNet at equivalent accuracy
- V1.1 reduces computation 2.4× by moving max-pool layers earlier

---

## When to Use SqueezeNet (1.0 / 1.1)

Use SqueezeNet only when < 2 M parameters is a hard constraint. For any accuracy requirement > 60%, MobileNetV2 or ShuffleNet V2 are better.

---

## Real-World Use Cases

- Extreme edge deployment: microcontrollers, wearables, IoT sensors
- OTA model updates where < 0.5 MB is a hard constraint
- Historical reference for fire module design pattern

---

## Folder Structure

```
SqueezeNet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
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
model = models.squeezenet1_0(weights="IMAGENET1K_V1")

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
@article{iandola2016squeezenet,
  title={{SqueezeNet}: {AlexNet}-level Accuracy with 50x Fewer Parameters and Less than 0.5{MB} Model Size},
  author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
  journal={arXiv preprint arXiv:1602.07360},
  year={2016}
}
```

**Paper:** SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and Less than 0.5MB Model Size
**Authors:** Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
**Venue:** arXiv 2016  **arXiv:** https://arxiv.org/abs/1602.07360

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.SqueezeNet&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
