# ResNeXt — Aggregated Residual Transformations

**Paper:** Aggregated Residual Transformations for Deep Neural Networks (Xie et al., CVPR 2017)
**Authors:** Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
**Institution:** UC San Diego / Facebook AI Research
**Award:** Best Paper Finalist — CVPR 2017

---

## Overview

ResNeXt extends ResNet by introducing **cardinality** as a new dimension of model design alongside depth and width. Instead of wider or deeper residual blocks, ResNeXt aggregates a set of **C parallel transformations** (where C is the cardinality), implemented efficiently as a single **grouped convolution**.

The core insight: increasing cardinality is more parameter-efficient than increasing width or depth for achieving higher accuracy. With the same parameter budget as ResNet-50, ResNeXt-50_32x4d achieves noticeably better top-1 accuracy.

```
Aggregated Transform:  F(x) = sum_{i=1}^{C}  T_i(x)
```

where each `T_i` is an independent path (group). All C paths share the same topology, implemented as `Conv3x3(groups=C)`.

---

## Variants

| Variant            | Layers        | Cardinality | Base Width | Parameters | Top-1     | Pretrained |
|--------------------|---------------|-------------|------------|-----------|-----------|------------|
| ResNeXt-50 32x4d   | [3, 4,  6, 3] | 32          | 4          | ~25M      | ~79.8%    | Yes (V2)   |
| ResNeXt-101 32x8d  | [3, 4, 23, 3] | 32          | 8          | ~88M      | ~82.8%    | Yes (V2)   |
| ResNeXt-101 64x4d  | [3, 4, 23, 3] | 64          | 4          | ~83M      | ~83.2%    | Yes (V1)   |

**Naming convention:** `CxDd` means cardinality=C, base_width=D per group.

---

## Architecture Pipeline

```
Input (3x224x224)
    |
    v
Stem: Conv7x7/stride=2 -> BN -> ReLU -> MaxPool3x3/stride=2  [-> 64 x 56x56]
    |
    v
Layer 1: N x Bottleneck, planes= 64, stride=1  [-> 256 x 56x56]
Layer 2: N x Bottleneck, planes=128, stride=2  [-> 512 x 28x28]
Layer 3: N x Bottleneck, planes=256, stride=2  [->1024 x 14x14]
Layer 4: N x Bottleneck, planes=512, stride=2  [->2048 x  7x 7]
    |
    v
AdaptiveAvgPool(1x1) -> Flatten
    |
    v
FC: Linear(2048 -> classes)
```

---

## Bottleneck Block with Grouped Convolution

```
Input -------------------------------------------+
  |                                               |
  v                                               | (identity or Conv1x1+BN
Conv1x1(in_ch -> width)     -> BN -> ReLU        |  if channel/stride mismatch)
Conv3x3(width -> width,                          |
        stride=s, groups=C) -> BN -> ReLU        |
Conv1x1(width -> planes*4)  -> BN               |
  |                                               |
  +---------- + ----------------------------------+
              |
             ReLU
             Output
```

**width** is computed as:
```python
width = int(planes * (base_width / 64.0)) * cardinality
```

The grouped 3x3 convolution with `groups=cardinality` partitions the channels into C independent groups, each with `width // C` channels. This is equivalent to C parallel 3x3 convolutions concatenated.

---

## Width Per Layer Across Variants

| Layer (planes) | ResNeXt-50 32x4d | ResNeXt-101 32x8d | ResNeXt-101 64x4d |
|----------------|------------------|-------------------|-------------------|
| 64             | 128              | 256               | 256               |
| 128            | 256              | 512               | 512               |
| 256            | 512              | 1024              | 1024              |
| 512            | 1024             | 2048              | 2048              |

Note: 32x8d and 64x4d produce identical widths at every layer despite different cardinality and base_width configurations.

---

## Layer Configurations

### ResNeXt-50 (32x4d) — Bottleneck

| Layer | Blocks | Planes | Output Shape  | Stride |
|-------|--------|--------|---------------|--------|
| Stem  | —      | 64     | 64 x 56x56    | 2+2    |
| 1     | 3      | 64     | 256 x 56x56   | 1      |
| 2     | 4      | 128    | 512 x 28x28   | 2      |
| 3     | 6      | 256    | 1024 x 14x14  | 2      |
| 4     | 3      | 512    | 2048 x 7x7    | 2      |
| FC    | —      | 2048   | 2048          | —      |

### ResNeXt-101 (32x8d / 64x4d) — Bottleneck

| Layer | Blocks | Planes | Output Shape  | Stride |
|-------|--------|--------|---------------|--------|
| Stem  | —      | 64     | 64 x 56x56    | 2+2    |
| 1     | 3      | 64     | 256 x 56x56   | 1      |
| 2     | 4      | 128    | 512 x 28x28   | 2      |
| 3     | **23** | 256    | 1024 x 14x14  | 2      |
| 4     | 3      | 512    | 2048 x 7x7    | 2      |
| FC    | —      | 2048   | 2048          | —      |

---

## Classifier Head

```python
self.fc = nn.Linear(512 * 4, num_classes)  # 2048 -> classes
```

**Replace `model.fc` for all variants.** `in_features = 2048` for all three.

---

## Training Configuration (From Scratch)

| Setting       | 50_32x4d    | 101_32x8d   | 101_64x4d   |
|---------------|-------------|-------------|-------------|
| Input Size    | 224x224     | 224x224     | 224x224     |
| Batch Size    | 32          | 16          | 8           |
| Optimizer     | Adam        | Adam        | Adam        |
| LR            | 1e-3        | 1e-3        | 1e-3        |
| Scheduler     | StepLR      | StepLR      | StepLR      |
|               | step=7, g=0.1 | step=7, g=0.1 | step=7, g=0.1 |
| Loss          | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| Epochs        | 20          | 20          | 20          |

---

## Transfer Learning Quick Reference

### Load Pretrained Weights

```python
from torchvision import models
import torch.nn as nn

# ResNeXt-50 32x4d
model    = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(2048, NUM_CLASSES)

# ResNeXt-101 32x8d
model    = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(2048, NUM_CLASSES)

# ResNeXt-101 64x4d
model    = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(2048, NUM_CLASSES)
```

### Feature Extraction (freeze backbone)

```python
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

### Fine-Tuning (dual learning rates)

```python
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'fc' not in n],
     'lr': 1e-5},
    {'params': model.fc.parameters(),
     'lr': 1e-3},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
```

---

## Pretrained Weights (torchvision)

| Variant          | Function                    | Weights Enum                               | Top-1  |
|------------------|-----------------------------|---------------------------------------------|--------|
| 50_32x4d         | `models.resnext50_32x4d()`  | `ResNeXt50_32X4D_Weights.IMAGENET1K_V2`    | ~79.8% |
| 101_32x8d        | `models.resnext101_32x8d()` | `ResNeXt101_32X8D_Weights.IMAGENET1K_V2`   | ~82.8% |
| 101_64x4d        | `models.resnext101_64x4d()` | `ResNeXt101_64X4D_Weights.IMAGENET1K_V1`   | ~83.2% |

V2 weights use improved training recipes. All variants have `in_features=2048` for `model.fc`.

---

## Folder Structure

```
ResNeXt/
+-- README.md                         <- this file
+-- ResNeXt 50_32x4D/
|   +-- NoteBook/
|   |   +-- resnext_50_32x4d.ipynb    - full notebook (arch + train + ROC AUC)
|   +-- Python Scripts/
|   |   +-- resnext_50_32x4d.py       - model architecture (Bottleneck + ResNeXt)
|   |   +-- train.py                  - training loop with StepLR
|   |   +-- inference.py              - single-image top-K prediction
|   |   +-- How to run.txt
|   +-- Using Weight File/
|       +-- load_pretrained.py        - load torchvision weights
|       +-- feature_extraction.py     - frozen backbone training
|       +-- fine_tuning.py            - dual-LR fine-tuning
|       +-- How to run.txt
+-- ResNeXt 101_32x8D/  (same structure)
+-- ResNeXt 101_64x4D/  (same structure)
```

---

## Comparison with Related Architectures

| Model              | Params | Top-1  | Key Difference vs ResNeXt            |
|--------------------|--------|--------|--------------------------------------|
| ResNet-50          | ~25.6M | 80.9%  | Same params, no grouped conv         |
| ResNeXt-50 32x4d   | ~25M   | 79.8%  | Baseline (cardinality=32)            |
| ResNeXt-101 32x8d  | ~88M   | 82.8%  | Deeper + wider groups                |
| ResNeXt-101 64x4d  | ~83M   | 83.2%  | Higher cardinality (64 groups)       |
| ResNet-101         | ~44.5M | 81.9%  | Deeper without grouped conv          |
| Wide ResNet-50-2   | ~68.9M | 81.6%  | Width scaling instead of cardinality |

---

## Citation

```bibtex
@inproceedings{xie2017aggregated,
  title     = {Aggregated Residual Transformations for Deep Neural Networks},
  author    = {Xie, Saining and Girshick, Ross and Dollar, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  pages     = {1492--1500},
  year      = {2017}
}
```
