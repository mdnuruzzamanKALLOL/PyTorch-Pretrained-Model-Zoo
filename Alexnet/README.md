# AlexNet — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** AlexNet PyTorch pretrained ImageNet 2012 NeurIPS classification ReLU dropout transfer learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

AlexNet sparked the modern deep learning era by winning ILSVRC 2012 with a 10.9% top-5 error rate — 10.8 percentage points better than the runner-up. It introduced ReLU activations, dropout regularization, data augmentation, and multi-GPU training to the field. Available in torchvision.models with ImageNet pretrained weights.

---

<div align="center">

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `alexnet` | 61 M | 224² | 56.5% | 79.1% |

</div>

---

## Architecture Highlights

- First CNN to win ImageNet LSVRC — launched the deep learning revolution
- ReLU activation instead of tanh/sigmoid: 6× faster convergence
- Dropout (p=0.5) in fully-connected layers to prevent overfitting
- Local Response Normalization (LRN) for lateral inhibition
- Original multi-GPU split across two GTX 580 GPUs

---

## When to Use AlexNet

Use AlexNet only for educational purposes or historical baselines. For any practical task, ResNet-50, MobileNetV2, or EfficientNet are far superior.

---

## Real-World Use Cases

- Educational baseline: understanding how deep learning began
- Simple feature extraction for small custom datasets
- Historical benchmark reproduction for NeurIPS 2012 results

---

## Folder Structure

```
Alexnet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
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
model = models.alexnet(weights="IMAGENET1K_V1")

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
@inproceedings{krizhevsky2012imagenet,
  title={ImageNet Classification with Deep Convolutional Neural Networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={NeurIPS},
  pages={1097--1105},
  year={2012}
}
```

**Paper:** ImageNet Classification with Deep Convolutional Neural Networks
**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
**Venue:** NeurIPS 2012  

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.Alexnet&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
