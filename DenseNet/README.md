# DenseNet — Densely Connected Convolutional Networks

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Paper](https://img.shields.io/badge/Paper-CVPR%202017-blue?style=flat-square)
![Variants](https://img.shields.io/badge/Variants-121%20%7C%20161%20%7C%20169%20%7C%20201-orange?style=flat-square)

> **Paper:** [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) — Huang et al., CVPR 2017

---

## Overview

DenseNet connects **every layer to every other layer** in a feed-forward fashion. Where traditional CNNs have `L` connections for `L` layers, DenseNet has `L(L+1)/2` connections. Each layer receives the feature maps of **all preceding layers** as input and passes its own feature maps to all subsequent layers.

This design eliminates the vanishing-gradient problem, strengthens feature propagation, encourages feature reuse, and substantially reduces the number of parameters.

---

## Architecture

```
Input (3 × 224 × 224)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Stem                                           │
│  Conv 7×7, stride 2, pad 3  →  BN → ReLU       │
│  MaxPool 3×3, stride 2                          │
└──────────────────────┬──────────────────────────┘
                       │  56 × 56
        ┌──────────────┴──────────────────────┐
        │          Dense Block 1              │
        │  ┌───┐   ┌───┐         ┌───┐       │
        │  │L₁ │◄──│L₂ │◄── ··· │Lₙ │       │
        │  └─┬─┘   └─┬─┘         └─┬─┘       │
        │    └────────┴─────────────┘          │
        │    (each layer sees all prev layers) │
        └──────────────────────────────────────┘
                       │  BN → ReLU → Conv 1×1 → AvgPool 2×2
                       ▼  (Transition layer)
                  28 × 28
              Dense Block 2
                       │  Transition
                  14 × 14
              Dense Block 3
                       │  Transition
                   7 × 7
              Dense Block 4
                       │
                       ▼
        BN → ReLU → GlobalAvgPool → Flatten
                       │
               Linear → Softmax
                       │
             Output (num_classes)
```

### Dense Connection Detail

```
         ┌──────────────────────────────────────┐
         │   Dense Block (num_layers = n)        │
         │                                       │
x₀ ──────┼──► L₁ ──► L₂ ──► L₃ ──► ··· ──► Lₙ │
         │     ↑        ↑      ↑                │
         │     x₀      x₀,x₁  x₀,x₁,x₂         │
         │         (concatenated along channel) │
         └──────────────────────────────────────┘

Each DenseLayer:
  BN → ReLU → Conv 1×1 (bottleneck: 4k channels)
           → BN → ReLU → Conv 3×3 (k channels)
  output  = concat([x_input, new_features])
```

---

## Model Variants

| Variant | Growth Rate | Block Config | Init Features | Parameters | Top-1 (ImageNet) |
|---------|:-----------:|:------------:|:-------------:|:----------:|:----------------:|
| DenseNet-121 | 32 | (6, 12, 24, 16) | 64 | ~7.9 M | 74.43% |
| DenseNet-161 | 48 | (6, 12, 36, 24) | 96 | ~28.7 M | 77.14% |
| DenseNet-169 | 32 | (6, 12, 32, 32) | 64 | ~14.1 M | 75.90% |
| DenseNet-201 | 32 | (6, 12, 48, 32) | 64 | ~20.0 M | 77.15% |

> **Growth rate (k):** number of feature maps each DenseLayer adds to the collective knowledge.

### Classifier Head Channels

| Variant | Final Feature Channels | torchvision head |
|---------|:---------------------:|:----------------:|
| DenseNet-121 | 1024 | `model.classifier` → Linear(1024, C) |
| DenseNet-161 | 2208 | `model.classifier` → Linear(2208, C) |
| DenseNet-169 | 1664 | `model.classifier` → Linear(1664, C) |
| DenseNet-201 | 1920 | `model.classifier` → Linear(1920, C) |

---

## Folder Structure

```
DenseNet/
├── README.md
│
├── DenseNet 121/
│   ├── NoteBook/
│   │   └── densenet121.ipynb          (14-cell notebook)
│   ├── Python Scripts/
│   │   ├── densenet121.py             (from-scratch model)
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py
│       ├── feature_extraction.py
│       ├── fine_tuning.py
│       └── How to run.txt
│
├── DenseNet 161/
│   ├── NoteBook/
│   │   └── densenet161.ipynb
│   ├── Python Scripts/
│   │   ├── densenet161.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py
│       ├── feature_extraction.py
│       ├── fine_tuning.py
│       └── How to run.txt
│
├── DenseNet 169/
│   ├── NoteBook/
│   │   └── densenet169.ipynb
│   ├── Python Scripts/
│   │   ├── densenet169.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py
│       ├── feature_extraction.py
│       ├── fine_tuning.py
│       └── How to run.txt
│
└── DenseNet 201/
    ├── NoteBook/
    │   └── densenet201.ipynb
    ├── Python Scripts/
    │   ├── densenet201.py
    │   ├── train.py
    │   ├── inference.py
    │   └── How to run.txt
    └── Using Weight File/
        ├── load_pretrained.py
        ├── feature_extraction.py
        ├── fine_tuning.py
        └── How to run.txt
```

---

## Jupyter Notebook — 14-Cell Structure

Each variant's notebook follows the same cell-by-cell layout:

| Cell | Content |
|:----:|---------|
| 0 | Markdown header with config table |
| 1 | Imports |
| 2 | DenseLayer class |
| 3 | DenseBlock + Transition classes |
| 4 | Full DenseNet model class |
| 5 | Instantiate model + device setup |
| 6 | Forward pass shape test |
| 7 | Total parameter count |
| 8 | Layer-wise parameter breakdown |
| — | Markdown: Training section |
| 9 | Dataset & DataLoader setup |
| 10 | Full training loop (AdamW + CosineAnnealingLR) |
| — | Markdown: Training Curves |
| 11 | Loss & accuracy curves (matplotlib) |
| — | Markdown: Inference |
| 12 | Single-image inference with Top-K bar chart |
| — | Markdown: ROC AUC Curve |
| 13 | Collect validation predictions |
| 14 | Per-class + Macro ROC AUC curves |

---

## Quick Start

### From Scratch (Python Scripts)

```bash
# 1. Prepare data in ImageFolder layout
data/
├── train/  class_A/  class_B/ ...
└── val/    class_A/  class_B/ ...

# 2. Edit DATA_DIR and NUM_CLASSES in train.py, then:
cd "DenseNet 121/Python Scripts"
python train.py

# 3. Single-image inference
python inference.py image.jpg class_names.json
```

### Using Pretrained Weights (torchvision)

```python
from torchvision import models

# DenseNet-121
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

# DenseNet-161
model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)

# DenseNet-169
model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)

# DenseNet-201
model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)

# Replace classifier head
import torch.nn as nn
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
```

### Feature Extraction vs Fine-Tuning

```bash
cd "DenseNet 121/Using Weight File"

# Feature extraction  (backbone frozen, only head trained)
python feature_extraction.py

# Fine-tuning  (all layers, dual LR: backbone 1e-5, head 1e-3)
python fine_tuning.py
```

---

## Key Design Principles

| Concept | Description |
|---------|-------------|
| **Dense Connectivity** | Layer `l` receives feature maps from all layers `0 … l-1` via concatenation |
| **Growth Rate (k)** | Each layer produces exactly `k` feature maps; controls model width |
| **Bottleneck (BN)** | 1×1 conv reduces channels to `4k` before the 3×3 conv |
| **Compression (θ)** | Transition layer reduces channels by factor `θ=0.5` (DenseNet-C) |
| **Composite Function** | BN → ReLU → Conv (pre-activation, unlike ResNet) |
| **Global Average Pooling** | Final 7×7 → 1×1 before classifier; no large FC layers |

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=30) |
| Batch Size | 32 (DenseNet-121/169/201), 16 (DenseNet-161) |
| Epochs | 30 |
| Input Size | 224 × 224 |
| Augmentation | RandomCrop, RandomHorizontalFlip, ColorJitter |

---

## DenseNet vs ResNet vs VGG

| Model | Skip Connections | Feature Reuse | Params (approx.) | Memory |
|-------|:----------------:|:-------------:|:-----------------:|:------:|
| VGG-16 | None | No | 138 M | High |
| ResNet-50 | Add (element-wise) | Partial | 25 M | Medium |
| **DenseNet-121** | **Concat (all prev)** | **Full** | **7.9 M** | **Medium** |
| DenseNet-161 | Concat (all prev) | Full | 28.7 M | Medium-High |

---

## Historical Context

| Year | Event |
|------|-------|
| 2014 | VGG shows depth matters |
| 2015 | ResNet introduces skip connections |
| 2016 | DenseNet paper submitted (arxiv Aug 2016) |
| 2017 | DenseNet wins **Best Paper Award at CVPR 2017** |
| 2017–present | Widely used as backbone in detection, segmentation, medical imaging |

---

## Requirements

```bash
pip install torch torchvision pillow scikit-learn matplotlib
```

| Package | Version |
|---------|---------|
| torch | ≥ 2.0 |
| torchvision | ≥ 0.15 |
| pillow | ≥ 9.0 |
| scikit-learn | ≥ 1.0 |
| matplotlib | ≥ 3.5 |

---

## Citation

```bibtex
@inproceedings{huang2017densely,
  title     = {Densely Connected Convolutional Networks},
  author    = {Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017},
  pages     = {4700--4708}
}
```

---

*Implementations from scratch — no torchvision model classes used in the Python Scripts or NoteBook folders.*
