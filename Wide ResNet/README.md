# Wide ResNet — Wide Residual Networks

**Paper:** Wide Residual Networks (Zagoruyko & Komodakis, BMVC 2016)
**Authors:** Sergey Zagoruyko, Nikos Komodakis
**Institution:** Universite Paris Est, Ecole des Ponts ParisTech

---

## Overview

Wide ResNet challenges the trend of making residual networks deeper by showing that **making them wider** is more effective and parameter-efficient. The key argument: the residual block in standard ResNet is too thin, and depth alone leads to diminishing returns due to gradient flow issues even with skip connections.

The architecture introduces a **width multiplier k** that scales the number of channels in every residual block. Standard ResNet corresponds to k=1; the torchvision models use **k=2** (indicated by the `_2` suffix), doubling the feature map width at every layer.

```
Wide Bottleneck width = planes * k   (k=2 => width_per_group = 128)
```

Compared to standard ResNet-50, Wide ResNet-50-2 achieves higher accuracy with fewer layers, and is faster per epoch on GPU because wide operations parallelize better than deep ones.

---

## Variants

| Variant           | Layers        | Width Mult | width_per_group | Parameters | Top-1  | Pretrained |
|-------------------|---------------|------------|-----------------|-----------|--------|------------|
| Wide ResNet-50-2  | [3, 4,  6, 3] | 2x         | 128             | ~68.9M    | ~81.6% | Yes (V2)   |
| Wide ResNet-101-2 | [3, 4, 23, 3] | 2x         | 128             | ~126.9M   | ~82.5% | Yes (V2)   |

Both variants share the same `width_per_group=128`. The `_2` in the name denotes the 2x width multiplier.

---

## Architecture Pipeline

```
Input (3x224x224)
    |
    v
Stem: Conv7x7/stride=2 -> BN -> ReLU -> MaxPool3x3/stride=2   [-> 64 x 56x56]
    |
    v
Layer 1: N x WideBottleneck, planes= 64, stride=1   [->  256 x 56x56]
Layer 2: N x WideBottleneck, planes=128, stride=2   [->  512 x 28x28]
Layer 3: N x WideBottleneck, planes=256, stride=2   [-> 1024 x 14x14]
Layer 4: N x WideBottleneck, planes=512, stride=2   [-> 2048 x  7x 7]
    |
    v
AdaptiveAvgPool(1x1) -> Flatten
    |
    v
FC: Linear(2048 -> classes)
```

---

## Wide Bottleneck Block

The only structural difference from standard ResNet is the width of the 3x3 convolution:

```
Input -------------------------------------------+
  |                                               |
  v                                               | (identity or Conv1x1+BN
Conv1x1(in_ch -> width)    -> BN -> ReLU          |  if stride!=1 or ch mismatch)
Conv3x3(width -> width,                           |
        stride=s)          -> BN -> ReLU          |
Conv1x1(width -> planes*4) -> BN                 |
  |                                               |
  +---------- + ----------------------------------+
              |
             ReLU
             Output
```

**width** = `int(planes * (width_per_group / 64.0))`

| Layer (planes) | Standard ResNet | Wide ResNet (k=2) | Ratio  |
|----------------|-----------------|-------------------|--------|
| 64             | 64              | 128               | 2x     |
| 128            | 128             | 256               | 2x     |
| 256            | 256             | 512               | 2x     |
| 512            | 512             | 1024              | 2x     |

---

## Layer Configurations

### Wide ResNet-50-2 — Bottleneck (k=2)

| Layer | Blocks | Planes | Output Shape  | Stride |
|-------|--------|--------|---------------|--------|
| Stem  | —      | 64     | 64 x 56x56    | 2+2    |
| 1     | 3      | 64     | 256 x 56x56   | 1      |
| 2     | 4      | 128    | 512 x 28x28   | 2      |
| 3     | 6      | 256    | 1024 x 14x14  | 2      |
| 4     | 3      | 512    | 2048 x 7x7    | 2      |
| FC    | —      | 2048   | 2048          | —      |

### Wide ResNet-101-2 — Bottleneck (k=2)

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

**Replace `model.fc` for all variants.** `in_features = 2048` for both.

---

## Training Configuration (From Scratch)

| Setting       | Wide ResNet-50-2 | Wide ResNet-101-2 |
|---------------|-----------------|-------------------|
| Input Size    | 224x224         | 224x224           |
| Batch Size    | 16              | 8                 |
| Optimizer     | Adam (lr=1e-3)  | Adam (lr=1e-3)    |
| Scheduler     | StepLR step=7, gamma=0.1 | StepLR step=7, gamma=0.1 |
| Loss          | CrossEntropyLoss | CrossEntropyLoss |
| Epochs        | 20              | 20                |

---

## Transfer Learning Quick Reference

### Load Pretrained Weights

```python
from torchvision import models
import torch.nn as nn

# Wide ResNet-50-2
model    = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(2048, NUM_CLASSES)

# Wide ResNet-101-2
model    = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
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

| Variant           | Function                     | Weights Enum                              | Top-1  |
|-------------------|------------------------------|-------------------------------------------|--------|
| Wide ResNet-50-2  | `models.wide_resnet50_2()`   | `Wide_ResNet50_2_Weights.IMAGENET1K_V2`   | ~81.6% |
| Wide ResNet-101-2 | `models.wide_resnet101_2()`  | `Wide_ResNet101_2_Weights.IMAGENET1K_V2`  | ~82.5% |

Both use V2 weights (improved training recipes). `in_features = 2048` for both variants.

---

## Folder Structure

```
Wide ResNet/
+-- README.md                            <- this file
+-- Wide ResNet 50_2/
|   +-- NoteBook/
|   |   +-- wide_resnet_50_2.ipynb       - full notebook (arch + train + ROC AUC)
|   +-- Python Scripts/
|   |   +-- wide_resnet_50_2.py          - model architecture (WideBottleneck + WideResNet)
|   |   +-- train.py                     - training loop with StepLR
|   |   +-- inference.py                 - single-image top-K prediction
|   |   +-- How to run.txt
|   +-- Using Weight File/
|       +-- load_pretrained.py           - load torchvision V2 weights
|       +-- feature_extraction.py        - frozen backbone training
|       +-- fine_tuning.py               - dual-LR fine-tuning
|       +-- How to run.txt
+-- Wide ResNet 101_2/  (same structure)
```

---

## Comparison with Related Architectures

| Model               | Params   | Top-1  | Key Difference                          |
|---------------------|----------|--------|-----------------------------------------|
| ResNet-50           | ~25.6M   | 80.9%  | Standard depth, no width scaling        |
| ResNet-101          | ~44.5M   | 81.9%  | Deeper, not wider                       |
| Wide ResNet-50-2    | ~68.9M   | 81.6%  | 2x wider than ResNet-50                 |
| Wide ResNet-101-2   | ~126.9M  | 82.5%  | 2x wider than ResNet-101                |
| ResNeXt-50 32x4d    | ~25M     | 79.8%  | Cardinality (grouped conv), not width   |
| ResNeXt-101 32x8d   | ~88M     | 82.8%  | Similar params to WRN-50-2, grouped     |

---

## Citation

```bibtex
@inproceedings{zagoruyko2016wide,
  title     = {Wide Residual Networks},
  author    = {Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year      = {2016}
}
```
