# GoogLeNet (Inception V1) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** GoogLeNet Inception V1 PyTorch pretrained ImageNet 2014 CVPR inception module 6.6M classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

GoogLeNet introduced the Inception module — parallel branches with 1×1, 3×3, 5×5 convolutions and max-pooling — achieving 93.3% top-5 accuracy on ImageNet with only 6.6 M parameters (12× fewer than AlexNet). Its depth (22 layers), auxiliary classifiers, and factorized convolutions set the template for the entire Inception family.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `googlenet` | 6.6 M | 224² | 69.8% | 89.5% |

---

## Architecture Highlights

- Inception modules: parallel 1×1, 3×3, 5×5 branches + 3×3 max-pool concatenated
- 1×1 convolutions for dimensionality reduction before expensive 3×3/5×5 conv
- Auxiliary classifiers at two intermediate points for gradient injection
- 22-layer network with 6.6 M parameters (vs AlexNet's 61 M)

---

## When to Use GoogLeNet (Inception V1)

Use GoogLeNet only for historical reference or education. MobileNetV2 or EfficientNet-B0 provide better accuracy at similar parameter counts for modern work.

---

## Real-World Use Cases

- Educational reference for Inception architecture design
- Lightweight feature extraction where < 7 M parameters is needed
- Multi-scale feature learning research baselines

---

## Folder Structure

```
GoogLeNet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
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
model = models.googlenet(weights="IMAGENET1K_V1")

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
@inproceedings{szegedy2015going,
  title={Going Deeper with Convolutions},
  author={Szegedy, Christian and Liu, Wei and Jia, Yangqing and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew},
  booktitle={CVPR},
  pages={1--9},
  year={2015}
}
```

**Paper:** Going Deeper with Convolutions
**Authors:** Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
**Venue:** CVPR 2015  **arXiv:** https://arxiv.org/abs/1409.4842

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FmdnuruzzamanKALLOL%2FPyTorch-Pretrained-Model-Zoo%2Ftree%2Fmaster%2FGoogLeNet&count_bg=%23EE4C2C&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Profile%20Views&edge_flat=false)

</div>
