# AlexNet — ImageNet Classification with Deep CNNs

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **Paper:** [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (NeurIPS 2012)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Layer Details](#layer-details)
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

AlexNet was the **breakthrough model** that launched the deep learning era in computer vision. It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 by a large margin — reducing the top-5 error from 26.2% to 15.3%.

### Why AlexNet Matters

| Contribution | Details |
|-------------|---------|
| **ReLU Activation** | First large-scale use of ReLU — faster training than tanh/sigmoid |
| **GPU Training** | Trained on 2× GTX 580 GPUs — pioneered multi-GPU deep learning |
| **Dropout** | Used dropout (p=0.5) in FC layers to prevent overfitting |
| **Data Augmentation** | Random crops, horizontal flips, PCA color jitter |
| **Local Response Norm** | LRN after Conv1 and Conv2 — lateral inhibition |
| **Overlapping Pooling** | stride < kernel_size in MaxPool — reduces overfitting |

---

## Architecture

```
Input (3 × 227 × 227)
        │
        ▼
┌─────────────────────────────────────────┐
│  Conv1: 96 filters, 11×11, stride=4     │  → 55×55×96
│  ReLU                                   │
│  LocalResponseNorm                      │
│  MaxPool: 3×3, stride=2                 │  → 27×27×96
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  Conv2: 256, 5×5    │  → 27×27×256
        │  ReLU + LRN         │
        │  MaxPool 3×3 s=2    │  → 13×13×256
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Conv3: 384, 3×3    │  → 13×13×384
        │  ReLU               │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Conv4: 384, 3×3    │  → 13×13×384
        │  ReLU               │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Conv5: 256, 3×3    │  → 13×13×256
        │  ReLU               │
        │  MaxPool 3×3 s=2    │  → 6×6×256
        └──────────┬──────────┘
                   │
          AdaptiveAvgPool2d(6,6)
          Flatten → 9216
                   │
        ┌──────────▼──────────┐
        │  Dropout(0.5)       │
        │  FC1: 9216 → 4096   │
        │  ReLU               │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Dropout(0.5)       │
        │  FC2: 4096 → 4096   │
        │  ReLU               │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  FC3: 4096 → N      │  ← num_classes
        └──────────┬──────────┘
                   │
                Output
```

---

## Layer Details

| Layer | Type | Filters | Kernel | Stride | Padding | Output Size |
|-------|------|:-------:|:------:|:------:|:-------:|:-----------:|
| Conv1 | Conv2d + ReLU + LRN + MaxPool | 96 | 11×11 | 4 | 0 | 27×27×96 |
| Conv2 | Conv2d + ReLU + LRN + MaxPool | 256 | 5×5 | 1 | 2 | 13×13×256 |
| Conv3 | Conv2d + ReLU | 384 | 3×3 | 1 | 1 | 13×13×384 |
| Conv4 | Conv2d + ReLU | 384 | 3×3 | 1 | 1 | 13×13×384 |
| Conv5 | Conv2d + ReLU + MaxPool | 256 | 3×3 | 1 | 1 | 6×6×256 |
| FC1 | Linear + Dropout + ReLU | — | — | — | — | 4096 |
| FC2 | Linear + Dropout + ReLU | — | — | — | — | 4096 |
| FC3 | Linear | — | — | — | — | num_classes |

### Parameter Distribution

```
Conv Layers  (features)  :   3,746,160  params  (~6%)
FC Layers    (classifier) :  58,632,184  params  (~94%)
─────────────────────────────────────────────────
Total                     :  62,378,344  params
```

> **Note:** ~94% of AlexNet's parameters are in the 3 fully-connected layers — a key motivation for later architectures (VGG, ResNet) that reduced FC layer size.

---

## Folder Structure

```
Alexnet/
│
├── README.md                          ← You are here
│
├── NoteBook/
│   └── alexnet.ipynb                  ← Full notebook (model + train + curves + ROC AUC)
│
├── Python Scripts/
│   ├── alexnet.py                     ← Model definition + parameter count
│   ├── train.py                       ← Training script
│   ├── inference.py                   ← Inference on single image
│   └── How to run.txt                 ← Step-by-step instructions
│
└── Using Weight File/
    ├── load_pretrained.py             ← ImageNet inference (torchvision weights)
    ├── feature_extraction.py          ← Frozen backbone, train FC head only
    ├── fine_tuning.py                 ← All layers unfrozen, dual LR
    └── How to run.txt                 ← Step-by-step instructions
```

---

## Quick Start

### Installation

```bash
pip install torch torchvision pillow matplotlib scikit-learn
```

### Run Model + Parameter Count

```bash
cd "Python Scripts"
python alexnet.py
```

### ImageNet Pretrained Inference

```bash
cd "Using Weight File"
python load_pretrained.py your_image.jpg
```

---

## Notebook Guide

`alexnet.ipynb` contains **12 cells**:

| Cell | Section | Description |
|------|---------|-------------|
| 1 | Imports | torch, torchvision, matplotlib, sklearn |
| 2 | AlexNet Class | Full raw architecture — 5 conv + 3 FC layers |
| 3 | Instantiate | Create model, print architecture |
| 4 | Forward Pass | Test with `(2, 3, 227, 227)` random input |
| 5 | Param Count | Total / trainable / non-trainable |
| 6 | Layer Breakdown | Per-layer name, shape, param count |
| 7 | DataLoader | Train/val transforms + ImageFolder |
| 8 | Training Loop | Adam + StepLR, epoch table, saves `.pth` |
| 9 | Training Curves | Loss & accuracy plots → `training_curves.png` |
| 10 | Inference | Single image prediction + bar chart → `inference_result.png` |
| 11 | Collect Preds | Run val set, collect probs + labels |
| 12 | ROC AUC Curve | Per-class + Macro AUC → `roc_auc_curve.png` |

---

## Training

### Dataset Structure

```
data/
├── train/
│   ├── class_1/
│   │   ├── img001.jpg
│   │   └── ...
│   └── class_2/
└── val/
    ├── class_1/
    └── class_2/
```

### Training Command

```bash
# Edit NUM_CLASSES and DATA_DIR in train.py, then:
python train.py
```

### Optimizer & Scheduler

```python
optimizer = Adam(lr=0.001)
scheduler = StepLR(step_size=5, gamma=0.1)
batch_size = 64
```

---

## Inference

```bash
# Basic (class index output)
python inference.py image.jpg

# With class labels
python inference.py image.jpg class_names.json
```

**`class_names.json` format:**
```json
["cat", "dog", "bird", "fish", "horse"]
```

---

## Using Pretrained Weights

| Script | Layers Trained | Best For | Saves |
|--------|:--------------:|----------|-------|
| `load_pretrained.py` | None (inference) | Quick test on any image | — |
| `feature_extraction.py` | FC head only | Small datasets | `alexnet_feature_extract.pth` |
| `fine_tuning.py` | All layers | Medium / large datasets | `alexnet_finetuned.pth` |

### Transfer Learning Tips

```python
# Feature Extraction — freeze all conv layers
for param in model.features.parameters():
    param.requires_grad = False

# Fine-Tuning — separate LR for backbone vs head
optimizer = Adam([
    {"params": model.features.parameters(),    "lr": 0.00001},
    {"params": model.classifier.parameters(),  "lr": 0.001},
])
```

---

## Results

### Parameter Count

```
========================================
  Total parameters      : 62,378,344
  Trainable parameters  : 62,378,344
  Non-trainable params  : 0
========================================
```

### Layer-wise Breakdown

```
Layer                          Shape              Params
──────────────────────────────────────────────────────
features.0.weight              [96, 3, 11, 11]     34,848
features.0.bias                [96]                    96
features.4.weight              [256, 96, 5, 5]    614,400
features.4.bias                [256]                  256
features.8.weight              [384, 256, 3, 3]   884,736
features.8.bias                [384]                  384
features.10.weight             [384, 384, 3, 3] 1,327,104
features.10.bias               [384]                  384
features.12.weight             [256, 384, 3, 3]   884,736
features.12.bias               [256]                  256
classifier.1.weight            [4096, 9216]    37,748,736
classifier.1.bias              [4096]               4,096
classifier.4.weight            [4096, 4096]    16,777,216
classifier.4.bias              [4096]               4,096
classifier.6.weight            [1000, 4096]     4,096,000
classifier.6.bias              [1000]               1,000
──────────────────────────────────────────────────────
TOTAL                                          62,378,344
```

### ImageNet Performance (original paper)

```
Top-1 Error  :  37.5%   (62.5% accuracy)
Top-5 Error  :  17.0%   (83.0% accuracy)
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

```bash
pip install torch torchvision pillow matplotlib scikit-learn numpy
```

---

## Historical Context

```
Year  Model        Top-5 Error   Params
────────────────────────────────────────
2012  AlexNet         17.0%       61M    ← Breakthrough
2014  VGG-16           7.3%      138M
2015  ResNet-50        5.2%       26M
2016  DenseNet-121     5.4%        8M
2022  ConvNeXt-T       ...        28M
```

> AlexNet's victory in 2012 started the **deep learning revolution** in computer vision. Every modern CNN architecture can be traced back to the ideas introduced here.

---

## Reference

```bibtex
@inproceedings{krizhevsky2012imagenet,
  title     = {ImageNet Classification with Deep Convolutional Neural Networks},
  author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {25},
  year      = {2012}
}
```

---

<p align="center">
  Built from scratch with PyTorch — implementing the model that started the deep learning revolution
</p>
