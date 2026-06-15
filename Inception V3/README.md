# Inception V3 — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** Inception V3 PyTorch pretrained 299px ImageNet 2016 CVPR FID medical imaging retinal classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

Inception V3 modernizes GoogLeNet with factorized convolutions (n×n → 1×n + n×1), grid reduction modules, and label smoothing, achieving 77.3% ImageNet top-1 at 27.2 M parameters with 299² input. It is the standard Inception backbone for medical imaging benchmarks and was used in landmark clinical AI papers on retinal disease and dermatology.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `inception_v3` | 27.2 M | 299² | 77.3% | 93.5% |

---

## Architecture Highlights

- Factorized 7×7 → 1×7 + 7×1 in the stem for efficient large-filter approximation
- Inception modules A/B/C with different factorization strategies per stage
- Label smoothing ε=0.1 — first major model to adopt this technique
- Auxiliary classifier at stage 4 for intermediate gradient injection
- 299×299 input resolution vs GoogLeNet's 224×224

---

## When to Use Inception V3

Use Inception V3 when reproducing medical AI benchmarks or computing FID scores. The FID metric is defined using Inception V3 features — this cannot be substituted.

---

## Real-World Use Cases

- Medical imaging clinical AI papers (diabetic retinopathy, dermatology benchmarks)
- FID computation for generative models (Inception V3 is the standard FID feature extractor)
- Multi-scale scene understanding with inception feature diversity

---

## Folder Structure

```
Inception V3/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models
model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
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
model = models.inception_v3(weights="IMAGENET1K_V1")

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
@inproceedings{szegedy2016rethinking,
  title={Rethinking the Inception Architecture for Computer Vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle={CVPR},
  pages={2818--2826},
  year={2016}
}
```

**Paper:** Rethinking the Inception Architecture for Computer Vision
**Authors:** Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathan Shlens, Zbigniew Wojna
**Venue:** CVPR 2016  **arXiv:** https://arxiv.org/abs/1512.00567

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>
