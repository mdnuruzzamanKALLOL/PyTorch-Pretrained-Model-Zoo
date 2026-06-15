# ShuffleNet V2

**Paper:** ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
**Authors:** Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
**Conference:** ECCV 2018

---

## Overview

ShuffleNet V2 proposes four practical guidelines for efficient network design based on direct measurement of memory access cost (MAC) and hardware utilization — not just FLOPs. The key insight is that FLOPs alone are a poor proxy for inference speed.

**Four guidelines:**
1. Equal channel widths minimize MAC → balance channel counts across branches
2. Excessive group convolution increases MAC → use standard convolutions after split
3. Network fragmentation reduces parallelism → limit branches
4. Element-wise operations matter → reduce Adds, ReLU, depthwise convs

**Key mechanism — Channel Split + Channel Shuffle:**
- Split input channels into two halves at each block
- One half passes through identity (left branch)
- Other half passes through 3 convolutions (right branch)
- Concatenate → Channel Shuffle (enables cross-branch communication)

---

## Channel Split Block (stride=1)

```
Input (C channels)
    |
    +------- identity --------+
    |                         |
  split                       |
    |                         |
  C/2 channels                |
  Conv1x1 -> BN -> ReLU       |
  DWConv3x3 -> BN             |
  Conv1x1 -> BN -> ReLU       |
    |                         |
    +-------- Cat ---------> (C channels)
                   |
              Channel Shuffle
```

## Downsampling Block (stride=2)

```
Input (C channels) — NO split, both branches get full C
    |
    +-- DWConv3x3/2 -> BN -----------+
    |   Conv1x1 -> BN -> ReLU        |
    |                                |
    +-- Conv1x1 -> BN -> ReLU -------+
        DWConv3x3/2 -> BN            |
        Conv1x1 -> BN -> ReLU        |
                                     |
    +--------------------------------+
    Cat (2C channels) -> Channel Shuffle
```

---

## Variants

| Variant | stages_out_channels      | Params | Top-1  | FC in_features |
|---------|--------------------------|--------|--------|----------------|
| x0.5    | [24,48,96,192,1024]      | ~1.4M  | ~60.6% | 1024           |
| x1.0    | [24,116,232,464,1024]    | ~2.3M  | ~69.4% | 1024           |
| x1.5    | [24,176,352,704,1024]    | ~3.5M  | ~73.0% | 1024           |
| x2.0    | [24,244,488,976,2048]    | ~7.4M  | ~76.2% | 2048           |

All variants: stages_repeats = [4, 8, 4]

---

## Architecture Pipeline

```
Input (3x224x224)
    |
Conv3x3/2 (3->24) + BN + ReLU -> MaxPool3x3/2     [-> 24 x 56x56]
    |
Stage2: 4 blocks (first stride=2)                  [-> ch2 x 28x28]
Stage3: 8 blocks (first stride=2)                  [-> ch3 x 14x14]
Stage4: 4 blocks (first stride=2)                  [-> ch4 x  7x 7]
    |
Conv1x1 (ch4 -> conv5_out) + BN + ReLU
    |
GlobalAvgPool -> FC(conv5_out -> classes)
```

---

## Training Configuration (From Scratch)

| Setting    | x0.5 / x1.0 | x1.5 | x2.0 |
|------------|-------------|------|------|
| Batch Size | 128         | 64   | 32   |
| Optimizer  | Adam (lr=1e-3, all variants)            |
| Scheduler  | StepLR (step_size=7, gamma=0.1, all)    |
| Epochs     | 20          | 20   | 20   |
| Input Size | 224x224 (all variants)                  |

---

## Transfer Learning Quick Reference

```python
from torchvision import models
import torch.nn as nn

model    = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
```

**Feature Extraction:**
```python
for param in model.parameters():
    param.requires_grad = False
model.fc  = nn.Linear(model.fc.in_features, NUM_CLASSES)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

**Fine-Tuning:**
```python
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters()
                if n not in ('fc.weight', 'fc.bias')], 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

---

## Pretrained Weights (torchvision)

| Variant | Function                      | Weights Enum                           | in_features |
|---------|-------------------------------|----------------------------------------|-------------|
| x0.5    | `models.shufflenet_v2_x0_5()` | `ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1` | 1024    |
| x1.0    | `models.shufflenet_v2_x1_0()` | `ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1` | 1024    |
| x1.5    | `models.shufflenet_v2_x1_5()` | `ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1` | 1024    |
| x2.0    | `models.shufflenet_v2_x2_0()` | `ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1` | 2048    |

---

## Comparison with Related Lightweight Models

| Model              | Params | Top-1  | Notes                                  |
|--------------------|--------|--------|----------------------------------------|
| SqueezeNet 1.1     | ~1.2M  | ~58.2% | No residual, Fire modules              |
| ShuffleNet V2 x0.5 | ~1.4M  | ~60.6% | Channel split, very lightweight        |
| MobileNet V2       | ~3.4M  | ~71.9% | Inverted residuals, linear bottleneck  |
| ShuffleNet V2 x1.0 | ~2.3M  | ~69.4% | Better accuracy/efficiency tradeoff    |
| ShuffleNet V2 x2.0 | ~7.4M  | ~76.2% | Close to ResNet-50 at 3x fewer params  |

---

## Folder Structure

```
ShuffleNet V2/
+-- README.md
+-- ShuffleNet V2 x0.5/
|   +-- NoteBook/          shufflenet_v2_x0_5.ipynb
|   +-- Python Scripts/    shufflenet_v2_x0_5.py  train.py  inference.py  How to run.txt
|   +-- Using Weight File/ load_pretrained.py  feature_extraction.py  fine_tuning.py  How to run.txt
+-- ShuffleNet V2 x1.0/   (same)
+-- ShuffleNet V2 x1.5/   (same)
+-- ShuffleNet V2 x2.0/   (same)
```

---

## Citation

```bibtex
@inproceedings{ma2018shufflenet,
  title     = {ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design},
  author    = {Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle = {ECCV},
  year      = {2018}
}
```
