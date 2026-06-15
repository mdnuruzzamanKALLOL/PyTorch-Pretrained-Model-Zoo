# ConvNeXt — A ConvNet for the 2020s

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **Paper:** [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) — Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie (CVPR 2022)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Model Variants](#model-variants)
- [Folder Structure](#folder-structure)
- [Quick Start](#quick-start)
- [Notebook Guide](#notebook-guide)
- [Training](#training)
- [Inference](#inference)
- [Using Pretrained Weights](#using-pretrained-weights)
- [Results](#results)
- [Requirements](#requirements)

---

## Overview

ConvNeXt is a pure convolutional network that was redesigned by taking inspiration from Vision Transformers (ViT). The key idea was: *"What if we modernize a standard ResNet step by step, borrowing ideas from Swin Transformer?"*

The result is a family of models that:
- **Outperforms** Swin Transformers on ImageNet classification
- **Matches** Swin Transformers on COCO detection and ADE20K segmentation
- **Maintains** the simplicity and efficiency of standard convnets
- Requires **no special attention mechanisms** or positional encodings

### Key Design Decisions

| Design Choice | Details |
|---------------|---------|
| **Stem** | Non-overlapping 4×4 conv, stride 4 (like ViT patchify) |
| **Depthwise Conv** | 7×7 kernel — larger receptive field per block |
| **Inverted Bottleneck** | Expand ratio 4× in MLP (like Transformers) |
| **Activation** | GELU instead of ReLU |
| **Normalization** | LayerNorm instead of BatchNorm |
| **Downsampling** | Separate LayerNorm + 2×2 conv stride 2 |
| **Layer Scale** | Learnable per-channel scale initialized at 1e-6 |

---

## Architecture

```
Input (3 × 224 × 224)
        │
        ▼
┌─────────────────────────────────────┐
│  Stem                               │
│  Conv2d(3→C₀, 4×4, stride=4)       │
│  LayerNorm                          │
└──────────────┬──────────────────────┘
               │  56×56×C₀
        ┌──────▼──────┐
        │   Stage 1   │  ×d₀ ConvNeXt Blocks
        └──────┬──────┘
               │  Downsample (LayerNorm + 2×2 conv, stride=2)
               │  28×28×C₁
        ┌──────▼──────┐
        │   Stage 2   │  ×d₁ ConvNeXt Blocks
        └──────┬──────┘
               │  Downsample
               │  14×14×C₂
        ┌──────▼──────┐
        │   Stage 3   │  ×d₂ ConvNeXt Blocks
        └──────┬──────┘
               │  Downsample
               │  7×7×C₃
        ┌──────▼──────┐
        │   Stage 4   │  ×d₃ ConvNeXt Blocks
        └──────┬──────┘
               │
               ▼
        Global Avg Pool
        LayerNorm
        Linear → num_classes
```

### ConvNeXt Block

```
Input (N, C, H, W)
        │
        ├─────────────────── Residual ──────────────────┐
        │                                                │
        ▼                                                │
  DWConv2d (7×7, groups=C)   ← large kernel             │
        │                                                │
        ▼                                                │
  permute → (N, H, W, C)                                │
        │                                                │
        ▼                                                │
  LayerNorm (channels_last)                              │
        │                                                │
        ▼                                                │
  Linear (C → 4C)            ← inverted bottleneck       │
        │                                                │
        ▼                                                │
  GELU                        ← smoother activation      │
        │                                                │
        ▼                                                │
  Linear (4C → C)                                        │
        │                                                │
        ▼                                                │
  Layer Scale (γ × x)         ← per-channel learnable    │
        │                                                │
        ▼                                                │
  permute → (N, C, H, W)                                │
        │                                                │
        └─────────────────── + ─────────────────────────┘
        │
        ▼
     Output
```

---

## Model Variants

| Variant | Depths | Dims | Parameters | ImageNet Top-1 |
|---------|--------|------|:----------:|:--------------:|
| **ConvNeXt-Tiny** | (3, 3, 9, 3) | (96, 192, 384, 768) | 28M | 82.1% |
| **ConvNeXt-Small** | (3, 3, 27, 3) | (96, 192, 384, 768) | 50M | 83.1% |
| **ConvNeXt-Base** | (3, 3, 27, 3) | (128, 256, 512, 1024) | 89M | 83.8% |
| **ConvNeXt-Large** | (3, 3, 27, 3) | (192, 384, 768, 1536) | 198M | 84.3% |

> All variants use input resolution **224×224** and are pretrained on **ImageNet-1K**.

---

## Folder Structure

```
ConvNeXt/
│
├── README.md                          ← You are here
│
├── ConvNeXt Tiny/
│   ├── NoteBook/
│   │   └── convnext_tiny.ipynb        ← Full notebook (model + train + ROC AUC)
│   ├── Python Scripts/
│   │   ├── convnext_tiny.py           ← Model definition + param count
│   │   ├── train.py                   ← Training script
│   │   ├── inference.py               ← Inference on single image
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py         ← ImageNet inference
│       ├── feature_extraction.py      ← Frozen backbone
│       ├── fine_tuning.py             ← All layers unfrozen
│       └── How to run.txt
│
├── ConvNeXt Small/
│   ├── NoteBook/
│   │   └── convnext_small.ipynb
│   ├── Python Scripts/
│   │   ├── convnext_small.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py
│       ├── feature_extraction.py
│       ├── fine_tuning.py
│       └── How to run.txt
│
├── ConvNeXt Base/
│   ├── NoteBook/
│   │   └── convnext_base.ipynb
│   ├── Python Scripts/
│   │   ├── convnext_base.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py
│       ├── feature_extraction.py
│       ├── fine_tuning.py
│       └── How to run.txt
│
└── ConvNeXt Large/
    ├── NoteBook/
    │   └── convnext_large.ipynb
    ├── Python Scripts/
    │   ├── convnext_large.py
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

## Quick Start

### Installation

```bash
pip install torch torchvision pillow matplotlib scikit-learn
```

### Run Model + Parameter Count

```bash
# Tiny (~28M params)
cd "ConvNeXt Tiny/Python Scripts"
python convnext_tiny.py

# Small (~50M params)
cd "ConvNeXt Small/Python Scripts"
python convnext_small.py

# Base (~89M params)
cd "ConvNeXt Base/Python Scripts"
python convnext_base.py

# Large (~198M params)
cd "ConvNeXt Large/Python Scripts"
python convnext_large.py
```

### ImageNet Pretrained Inference (no training needed)

```bash
cd "ConvNeXt Tiny/Using Weight File"
python load_pretrained.py your_image.jpg
```

---

## Notebook Guide

Each variant's notebook (`convnext_[variant].ipynb`) contains **14 cells**:

| Cell | Section | Description |
|------|---------|-------------|
| 1 | Imports | torch, torchvision, matplotlib, sklearn |
| 2 | LayerNorm | Custom LayerNorm supporting `channels_first` |
| 3 | ConvNeXt Block | DWConv + LayerNorm + inverted MLP + LayerScale |
| 4 | ConvNeXt Model | Full model with stem, 4 stages, head |
| 5 | Instantiate | Create model with variant-specific config |
| 6 | Forward Pass | Test with random `(2, 3, 224, 224)` input |
| 7 | Param Count | Total / trainable / non-trainable breakdown |
| 8 | Layer Breakdown | Per-layer name, shape, parameter count |
| 9 | DataLoader | Train/val transforms + ImageFolder setup |
| 10 | Training Loop | AdamW + CosineAnnealingLR, epoch table |
| 11 | Training Curves | Loss & accuracy plots → `training_curves.png` |
| 12 | Inference | Single image prediction with bar chart → `inference_result.png` |
| 13 | Collect Preds | Run val set, collect probs + labels |
| 14 | ROC AUC Curve | Per-class + Macro AUC → `roc_auc_curve.png` |

---

## Training

### Dataset Structure

```
data/
├── train/
│   ├── class_1/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    └── ...
```

### Training Command

```bash
# Edit NUM_CLASSES and DATA_DIR in train.py, then:
python train.py
```

### Recommended Batch Sizes

| Variant | BATCH_SIZE | VRAM Needed |
|---------|:----------:|:-----------:|
| Tiny    | 32         | ~4 GB       |
| Small   | 32         | ~6 GB       |
| Base    | 16         | ~8 GB       |
| Large   | 8          | ~12 GB      |

### Optimizer & Scheduler

```python
optimizer = AdamW(lr=5e-4, weight_decay=0.05)
scheduler = CosineAnnealingLR(T_max=EPOCHS)
```

---

## Inference

```bash
# Basic (shows class index)
python inference.py image.jpg

# With class name labels
python inference.py image.jpg class_names.json
```

**`class_names.json` format:**
```json
["cat", "dog", "bird", "fish", "horse"]
```

---

## Using Pretrained Weights

Three transfer learning approaches are provided:

| Script | Frozen Layers | Best For | Saved As |
|--------|:-------------:|----------|----------|
| `load_pretrained.py` | All (inference only) | Quick test on any image | — |
| `feature_extraction.py` | All except last FC | Small datasets (< 5K/class) | `convnext_[v]_feature_extract.pth` |
| `fine_tuning.py` | None (all trainable) | Medium / large datasets | `convnext_[v]_finetuned.pth` |

### Fine-tuning Learning Rates

```python
# Dual LR strategy — backbone gets smaller LR to preserve pretrained features
optimizer = AdamW([
    {"params": model.features.parameters(),   "lr": 1e-5},  # backbone
    {"params": model.classifier.parameters(), "lr": 1e-3},  # head
])
```

---

## Results

### Expected Parameter Counts (from scratch)

```
========================================
  Model           : ConvNeXt-Tiny
  Total params    : 27,818,592
  Trainable       : 27,818,592
  Non-trainable   : 0
========================================

========================================
  Model           : ConvNeXt-Small
  Total params    : 49,453,152
  Trainable       : 49,453,152
  Non-trainable   : 0
========================================

========================================
  Model           : ConvNeXt-Base
  Total params    : 87,564,416
  Trainable       : 87,564,416
  Non-trainable   : 0
========================================

========================================
  Model           : ConvNeXt-Large
  Total params    : 196,227,264
  Trainable       : 196,227,264
  Non-trainable   : 0
========================================
```

### ImageNet-1K Top-1 Accuracy (pretrained)

```
ConvNeXt-Tiny   ████████████████████████████████░░  82.1%
ConvNeXt-Small  ██████████████████████████████████░  83.1%
ConvNeXt-Base   ███████████████████████████████████░  83.8%
ConvNeXt-Large  ████████████████████████████████████  84.3%
```

---

## Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

Install all:
```bash
pip install torch torchvision pillow matplotlib scikit-learn numpy
```

---

## Reference

```bibtex
@article{liu2022convnet,
  title   = {A ConvNet for the 2020s},
  author  = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and
             Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal = {CVPR},
  year    = {2022}
}
```

---

<p align="center">
  Built from scratch with PyTorch — no torchvision model dependency for the raw implementations
</p>
