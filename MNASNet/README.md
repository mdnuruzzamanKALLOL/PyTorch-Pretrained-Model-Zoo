# MNASNet — Mobile Neural Architecture Search Network

**Paper:** MnasNet: Platform-Aware Neural Architecture Search for Mobile (Tan et al., CVPR 2019)
**Authors:** Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le
**Institution:** Google Brain / Google AI

---

## Overview

MNASNet introduces **hardware-aware neural architecture search** that directly incorporates real-device latency into the search objective. Unlike prior NAS methods that optimize for FLOPs or parameter count, MNASNet uses a multi-objective reward:

```
Maximize  ACC(m) × [LAT(m) / T]^w
```

where `ACC` is top-1 accuracy, `LAT` is measured latency on a Pixel 1 phone, `T` is the target latency, and `w` controls the accuracy-latency trade-off.

The resulting architecture uses **Inverted Residual blocks (MBConv)** with mixed kernel sizes (3×3 and 5×5) and mixed expand ratios (3× and 6×) across different stages — unlike MobileNetV2 which uses a fixed 6× expand ratio everywhere.

The **alpha (width multiplier)** scales all channel widths uniformly to produce a family of models at different accuracy–efficiency operating points.

---

## Variants

| Variant      | Alpha | Parameters | Top-1 (ImageNet) | Latency (ms) | Pretrained       | Batch Size |
|--------------|-------|-----------|-----------------|--------------|------------------|------------|
| MNASNet 0.5  | 0.5   | ~2.2M     | ~67.7%          | ~20ms        | Yes (V1)         | 64         |
| MNASNet 0.75 | 0.75  | ~3.2M     | ~71.2%          | ~28ms        | Not available    | 64         |
| MNASNet 1.0  | 1.0   | ~4.4M     | ~73.5%          | ~38ms        | Yes (V1)         | 64         |
| MNASNet 1.3  | 1.3   | ~6.3M     | ~75.2%          | ~52ms        | Not available    | 32         |

---

## Architecture Pipeline

```
Input (3×224×224)
    │
    ▼
Stem: Conv3×3/stride=2 → BN → ReLU         [→ stem_ch × 112×112]
    │
    ▼
Stage 1: 1× SepConv 3×3/stride=1            [expand=1, out=16α]
Stage 2: 3× MBConv 3×3/stride=2             [expand=3, out=24α]
Stage 3: 3× MBConv 5×5/stride=2             [expand=3, out=40α]
Stage 4: 3× MBConv 3×3/stride=2             [expand=6, out=80α]
Stage 5: 2× MBConv 3×3/stride=1             [expand=6, out=96α]
Stage 6: 4× MBConv 5×5/stride=2             [expand=6, out=192α]
Stage 7: 1× MBConv 3×3/stride=1             [expand=6, out=320α]
    │
    ▼
Head: Conv1×1 → 1280 → BN → ReLU
    │
    ▼
AdaptiveAvgPool(1×1) → Flatten
    │
    ▼
Classifier: Dropout(0.2) → Linear(1280 → classes)
```

---

## Stage Configuration

| Stage | Block   | Kernel | Expand | Out (α=1.0) | Blocks | Stride |
|-------|---------|--------|--------|-------------|--------|--------|
| 1     | SepConv | 3×3    | 1×     | 16          | 1      | 1      |
| 2     | MBConv  | 3×3    | 3×     | 24          | 3      | 2      |
| 3     | MBConv  | 5×5    | 3×     | 40          | 3      | 2      |
| 4     | MBConv  | 3×3    | 6×     | 80          | 3      | 2      |
| 5     | MBConv  | 3×3    | 6×     | 96          | 2      | 1      |
| 6     | MBConv  | 5×5    | 6×     | 192         | 4      | 2      |
| 7     | MBConv  | 3×3    | 6×     | 320         | 1      | 1      |

All channel counts are rounded to the nearest multiple of 8 via `_make_divisible`.

---

## InvertedResidual Block (MBConv)

### expand_ratio = 1 (SepConv / Stage 1)
```
DWConv(k×k, stride) → BN → ReLU → Conv1×1(project) → BN
```

### expand_ratio > 1 (standard MBConv)
```
Conv1×1(expand) → BN → ReLU → DWConv(k×k, stride) → BN → ReLU → Conv1×1(project) → BN
         [+ residual skip if stride==1 and in_ch==out_ch]
```

---

## Channel Widths Across Variants

| Layer          | α=0.5 | α=0.75 | α=1.0 | α=1.3 |
|----------------|-------|--------|-------|-------|
| Stem (32×α)    | 16    | 24     | 32    | 40    |
| Stage 1 (16×α) | 8     | 16     | 16    | 24    |
| Stage 2 (24×α) | 16    | 16     | 24    | 32    |
| Stage 3 (40×α) | 24    | 32     | 40    | 56    |
| Stage 4 (80×α) | 40    | 64     | 80    | 104   |
| Stage 5 (96×α) | 48    | 72     | 96    | 128   |
| Stage 6 (192×α)| 96    | 144    | 192   | 248   |
| Stage 7 (320×α)| 160   | 240    | 320   | 416   |
| Head           | 1280  | 1280   | 1280  | 1280  |

---

## Classifier Head

The head is identical for all alpha variants:

```python
self.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, num_classes),  # ← replace [1] for transfer learning
)
```

**`in_features = 1280` for all alpha variants.**

---

## Training Configuration (From Scratch)

| Setting       | Value                        |
|---------------|------------------------------|
| Input Size    | 224×224                      |
| Normalization | mean=[0.485,0.456,0.406]     |
|               | std=[0.229,0.224,0.225]      |
| Augmentation  | RandomHorizontalFlip + ColorJitter |
| Optimizer     | Adam (lr=1e-3)               |
| Scheduler     | CosineAnnealingLR (T_max=20) |
| Loss          | CrossEntropyLoss             |
| Epochs        | 20                           |

---

## Transfer Learning Quick Reference

### Load Pretrained Weights

```python
from torchvision import models
import torch.nn as nn

# Alpha 0.5
model = models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1)

# Alpha 1.0
model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)

# Replace head (same for all variants)
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
```

### Feature Extraction (freeze backbone)

```python
for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=1e-3)
```

### Fine-Tuning (dual learning rates)

```python
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier.1' not in n],
     'lr': 1e-5},
    {'params': model.classifier[1].parameters(),
     'lr': 1e-3},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
```

---

## Pretrained Weights (torchvision)

| Variant | Function             | Weights Enum                         |
|---------|---------------------|--------------------------------------|
| 0.5     | `models.mnasnet0_5()` | `MNASNet0_5_Weights.IMAGENET1K_V1` |
| 0.75    | `models.mnasnet0_75()` | Not available (use `weights=None`) |
| 1.0     | `models.mnasnet1_0()` | `MNASNet1_0_Weights.IMAGENET1K_V1` |
| 1.3     | `models.mnasnet1_3()` | Not available (use `weights=None`) |

---

## Folder Structure

```
MNASNet/
├── README.md                      ← this file
├── MNASNet 0.5/
│   ├── NoteBook/
│   │   └── mnasnet_0_5.ipynb      — full notebook (arch + train + ROC AUC)
│   ├── Python Scripts/
│   │   ├── mnasnet_0_5.py         — model architecture
│   │   ├── train.py               — training loop
│   │   ├── inference.py           — single-image prediction
│   │   └── How to run.txt
│   └── Using Weight File/
│       ├── load_pretrained.py     — load torchvision weights
│       ├── feature_extraction.py  — frozen backbone training
│       ├── fine_tuning.py         — dual-LR fine-tuning
│       └── How to run.txt
├── MNASNet 0.75/  (same structure)
├── MNASNet 1.0/   (same structure)
└── MNASNet 1.3/   (same structure)
```

---

## Comparison with Related Architectures

| Model         | Params | Top-1  | Key Difference                        |
|---------------|--------|--------|---------------------------------------|
| MNASNet 1.0   | ~4.4M  | 73.5%  | Hardware-aware NAS, mixed kernels     |
| MobileNetV2   | ~3.4M  | 72.0%  | Fixed 6× expand, all 3×3 kernels      |
| MobileNetV3-S | ~2.5M  | 67.5%  | NAS + NetAdapt + HardSwish + SE       |
| ShuffleNetV2  | ~2.3M  | 72.6%  | Channel split and shuffle             |

---

## Citation

```bibtex
@inproceedings{tan2019mnasnet,
  title     = {MnasNet: Platform-Aware Neural Architecture Search for Mobile},
  author    = {Tan, Mingxing and Chen, Bo and Pang, Ruoming and Vasudevan, Vijay
               and Sandler, Mark and Howard, Andrew and Le, Quoc V},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2019}
}
```
