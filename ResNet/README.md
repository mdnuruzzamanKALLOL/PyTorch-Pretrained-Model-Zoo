# ResNet — Deep Residual Learning

**Paper:** Deep Residual Learning for Image Recognition (He et al., CVPR 2016)
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
**Institution:** Microsoft Research
**Award:** Best Paper — CVPR 2016

---

## Overview

ResNet introduced **residual (skip) connections** that allow gradients to flow directly through layers during backpropagation, solving the **vanishing/exploding gradient problem** that prevented training of very deep networks. The core insight is simple:

Instead of learning `H(x)`, learn the **residual** `F(x) = H(x) − x`, which is easier to optimize because if the identity mapping is optimal, the network just drives `F(x)` toward zero.

```
output = F(x, {Wᵢ}) + x
```

This allowed ResNet to train networks with 50, 101, and 152 layers — far deeper than was previously possible — and achieve state-of-the-art on ImageNet, COCO, and other benchmarks.

---

## Variants

| Variant    | Block Type  | Layers       | Parameters | Top-1 (ImageNet) | FC in_features | Pretrained |
|------------|-------------|--------------|-----------|-----------------|----------------|------------|
| ResNet-18  | BasicBlock  | [2, 2, 2, 2] | ~11.7M    | ~69.8%          | 512            | Yes (V1)   |
| ResNet-34  | BasicBlock  | [3, 4, 6, 3] | ~21.8M    | ~73.3%          | 512            | Yes (V1)   |
| ResNet-50  | Bottleneck  | [3, 4, 6, 3] | ~25.6M    | ~80.9%          | 2048           | Yes (V2)   |
| ResNet-101 | Bottleneck  | [3, 4, 23, 3]| ~44.5M    | ~81.9%          | 2048           | Yes (V2)   |
| ResNet-152 | Bottleneck  | [3, 8, 36, 3]| ~60.2M    | ~82.0%          | 2048           | Yes (V2)   |

---

## Architecture Pipeline

```
Input (3×224×224)
    │
    ▼
Stem: Conv7×7 / stride=2 → BN → ReLU → MaxPool3×3 / stride=2   [→ 64×56×56]
    │
    ▼
Layer 1: N × Block, planes= 64, stride=1   [→  64×exp × 56×56]
Layer 2: N × Block, planes=128, stride=2   [→ 128×exp × 28×28]
Layer 3: N × Block, planes=256, stride=2   [→ 256×exp × 14×14]
Layer 4: N × Block, planes=512, stride=2   [→ 512×exp ×  7× 7]
    │
    ▼
AdaptiveAvgPool(1×1) → Flatten
    │
    ▼
FC: Linear(512×exp → classes)
```

`exp = 1` for BasicBlock (18, 34) → final features = **512**
`exp = 4` for Bottleneck (50, 101, 152) → final features = **2048**

---

## Block Types

### BasicBlock  (ResNet-18, ResNet-34)

Used for shallower networks. Two 3×3 convolutions per block.

```
Input ──────────────────────────────────────────────┐
  │                                                  │
  ▼                                                  │ (identity or 1×1 conv
Conv3×3(in_ch, planes, stride) → BN → ReLU          │  if channel/stride mismatch)
Conv3×3(planes, planes)        → BN                 │
  │                                                  │
  └──────────── + ──────────────────────────────────┘
                │
               ReLU
               Output
```

- **expansion = 1**: output channels = planes
- Residual shortcut: identity if dimensions match; `Conv1×1 + BN` (projection) if not

### Bottleneck  (ResNet-50, ResNet-101, ResNet-152)

Used for deeper networks. Three convolutions forming a bottleneck: compress → process → expand.

```
Input ──────────────────────────────────────────────┐
  │                                                  │
  ▼                                                  │ (identity or 1×1 conv
Conv1×1(in_ch, planes)          → BN → ReLU         │  if channel/stride mismatch)
Conv3×3(planes, planes, stride) → BN → ReLU         │
Conv1×1(planes, planes×4)       → BN                │
  │                                                  │
  └──────────── + ──────────────────────────────────┘
                │
               ReLU
               Output
```

- **expansion = 4**: output channels = planes × 4
- The 1×1 convolutions reduce computational cost while the 3×3 conv handles spatial features
- Residual shortcut: identity if dimensions match; `Conv1×1 + BN` (projection) if not

---

## Layer Configurations Per Variant

### ResNet-18 — BasicBlock, expansion=1

| Layer | Blocks | Planes | Output Shape   | Stride |
|-------|--------|--------|----------------|--------|
| Stem  | —      | 64     | 64 × 56×56     | 2+2    |
| 1     | 2      | 64     | 64 × 56×56     | 1      |
| 2     | 2      | 128    | 128 × 28×28    | 2      |
| 3     | 2      | 256    | 256 × 14×14    | 2      |
| 4     | 2      | 512    | 512 × 7×7      | 2      |
| FC    | —      | 512    | 512            | —      |

### ResNet-34 — BasicBlock, expansion=1

| Layer | Blocks | Planes | Output Shape   | Stride |
|-------|--------|--------|----------------|--------|
| Stem  | —      | 64     | 64 × 56×56     | 2+2    |
| 1     | 3      | 64     | 64 × 56×56     | 1      |
| 2     | 4      | 128    | 128 × 28×28    | 2      |
| 3     | 6      | 256    | 256 × 14×14    | 2      |
| 4     | 3      | 512    | 512 × 7×7      | 2      |
| FC    | —      | 512    | 512            | —      |

### ResNet-50 — Bottleneck, expansion=4

| Layer | Blocks | Planes | Output Shape   | Stride |
|-------|--------|--------|----------------|--------|
| Stem  | —      | 64     | 64 × 56×56     | 2+2    |
| 1     | 3      | 64     | 256 × 56×56    | 1      |
| 2     | 4      | 128    | 512 × 28×28    | 2      |
| 3     | 6      | 256    | 1024 × 14×14   | 2      |
| 4     | 3      | 512    | 2048 × 7×7     | 2      |
| FC    | —      | 2048   | 2048           | —      |

### ResNet-101 — Bottleneck, expansion=4

| Layer | Blocks | Planes | Output Shape   | Stride |
|-------|--------|--------|----------------|--------|
| 1     | 3      | 64     | 256 × 56×56    | 1      |
| 2     | 4      | 128    | 512 × 28×28    | 2      |
| 3     | **23** | 256    | 1024 × 14×14   | 2      |
| 4     | 3      | 512    | 2048 × 7×7     | 2      |
| FC    | —      | 2048   | 2048           | —      |

### ResNet-152 — Bottleneck, expansion=4

| Layer | Blocks | Planes | Output Shape   | Stride |
|-------|--------|--------|----------------|--------|
| 1     | 3      | 64     | 256 × 56×56    | 1      |
| 2     | **8**  | 128    | 512 × 28×28    | 2      |
| 3     | **36** | 256    | 1024 × 14×14   | 2      |
| 4     | 3      | 512    | 2048 × 7×7     | 2      |
| FC    | —      | 2048   | 2048           | —      |

---

## Classifier Head

The classifier is a **single fully-connected layer** for all variants:

```python
self.fc = nn.Linear(512 * block.expansion, num_classes)
# ResNet-18/34 → Linear(512, classes)
# ResNet-50/101/152 → Linear(2048, classes)
```

**Replace `model.fc` for transfer learning.**

---

## Training Configuration (From Scratch)

| Setting       | ResNet-18/34 | ResNet-50  | ResNet-101 | ResNet-152 |
|---------------|-------------|------------|------------|------------|
| Input Size    | 224×224     | 224×224    | 224×224    | 224×224    |
| Batch Size    | 64          | 32         | 16         | 8          |
| Optimizer     | Adam        | Adam       | Adam       | Adam       |
| LR            | 1e-3        | 1e-3       | 1e-3       | 1e-3       |
| Scheduler     | StepLR      | StepLR     | StepLR     | StepLR     |
|               | step=7, γ=0.1 | step=7, γ=0.1 | step=7, γ=0.1 | step=7, γ=0.1 |
| Loss          | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| Epochs        | 20          | 20         | 20         | 20         |

---

## Transfer Learning Quick Reference

### Load Pretrained Weights

```python
from torchvision import models
import torch.nn as nn

# ResNet-18 / 34  (in_features = 512)
model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, NUM_CLASSES)

# ResNet-50 / 101 / 152  (in_features = 2048)
model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
```

---

## Pretrained Weights (torchvision)

| Variant    | Function            | Weights Enum                         | in_features |
|------------|---------------------|--------------------------------------|-------------|
| ResNet-18  | `models.resnet18()` | `ResNet18_Weights.IMAGENET1K_V1`     | 512         |
| ResNet-34  | `models.resnet34()` | `ResNet34_Weights.IMAGENET1K_V1`     | 512         |
| ResNet-50  | `models.resnet50()` | `ResNet50_Weights.IMAGENET1K_V2`     | 2048        |
| ResNet-101 | `models.resnet101()`| `ResNet101_Weights.IMAGENET1K_V2`    | 2048        |
| ResNet-152 | `models.resnet152()`| `ResNet152_Weights.IMAGENET1K_V2`    | 2048        |

V2 weights use improved training recipes (better augmentation + longer schedules) and achieve ~1–2% higher top-1 than V1.

---

## Folder Structure

```
ResNet/
├── README.md                      ← this file
├── ResNet 18/
│   ├── NoteBook/
│   │   └── resnet18.ipynb         — full notebook (arch + train + ROC AUC)
│   ├── Python Scripts/
│   │   ├── resnet18.py            — model architecture (BasicBlock + ResNet)
│   │   ├── train.py               — training loop
│   │   ├── inference.py           — single-image top-K prediction
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py     — load torchvision weights
│       ├── feature_extraction.py  — frozen backbone training
│       ├── fine_tuning.py         — dual-LR fine-tuning
│       └── How to run.txt
├── ResNet 34/   (same structure, BasicBlock)
├── ResNet 50/   (same structure, Bottleneck)
├── ResNet 101/  (same structure, Bottleneck)
└── ResNet 152/  (same structure, Bottleneck)
```

---

## Comparison with Related Architectures

| Model      | Params  | Top-1  | Key Design Feature                         |
|------------|---------|--------|--------------------------------------------|
| VGG-16     | ~138M   | 74.2%  | Deep plain network, no skip connections    |
| ResNet-18  | ~11.7M  | 69.8%  | Residual connections, BasicBlock           |
| ResNet-50  | ~25.6M  | 80.9%  | Bottleneck blocks, much deeper             |
| ResNet-101 | ~44.5M  | 81.9%  | 101 layers, strong feature extractor       |
| ResNeXt-50 | ~25M    | 81.2%  | Grouped convolutions (cardinality)         |
| EfficientNet-B4 | ~19M | 83.4% | Compound scaling of width/depth/resolution |

---

## Citation

```bibtex
@inproceedings{he2016deep,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}
```
