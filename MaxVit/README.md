# MaxViT — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** MaxViT PyTorch pretrained 83.7% ImageNet 2022 ECCV multi-axis attention vision transformer linear complexity classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

MaxViT (Multi-Axis Vision Transformer) interleaves local window attention and global grid attention within each block, achieving a theoretically linear complexity while maintaining global context. The MaxViT-T variant achieves 83.7% ImageNet top-1 — competitive with Swin-B — and is available in torchvision 0.15+.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `maxvit_t` | 31 M | 224² | 83.7% | 96.6% |

---

## Architecture Highlights

- Multi-axis attention: alternating local (window) and global (grid) self-attention
- Linear complexity O(N) vs standard attention O(N²) for high-resolution inputs
- MBConv block before attention for local feature pre-processing
- Relative position bias shared across window/grid partitions
- Hybrid: benefits from both CNN inductive bias and Transformer global context

---

## When to Use MaxViT

Use MaxViT-T when you want Transformer-level accuracy (83.7%) without quadratic attention cost. A strong modern alternative to Swin-T.

---

## Real-World Use Cases

- High-accuracy classification competing with Swin Transformer
- High-resolution tasks where linear complexity matters
- Research comparing local vs global attention mechanisms
- Transfer learning onto satellite and medical datasets

---

## Folder Structure

```
MaxVit/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models
model = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
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
model = models.maxvit_t(weights="IMAGENET1K_V1")

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
@inproceedings{tu2022maxvit,
  title={{MaxViT}: Multi-Axis Vision Transformer},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  booktitle={ECCV},
  pages={459--479},
  year={2022}
}
```

**Paper:** MaxViT: Multi-Axis Vision Transformer
**Authors:** Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, Yinxiao Li
**Venue:** ECCV 2022  **arXiv:** https://arxiv.org/abs/2204.01697

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://komarev.com/ghpvc/?username=mdnuruzzamanKALLOL&label=Profile%20Views&color=EE4C2C&style=flat-square)

</div>
