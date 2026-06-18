# ShuffleNet V2 (×0.5 / ×1.0 / ×1.5 / ×2.0) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** ShuffleNet V2 PyTorch pretrained 2018 ECCV efficient mobile channel split shuffle ultra-lightweight classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

ShuffleNet V2 proposes four practical efficiency guidelines that optimize actual runtime speed (not just theoretical FLOPs), including equal channel splitting, fewer group convolutions, and element-wise operations avoidance. Four width multipliers (×0.5–×2.0) provide a range from 60.6% (0.5×) to 74.6% (2.0×) top-1 accuracy.

---

## Variants & ImageNet Performance

<div align="center">

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `shufflenet_v2_x0_5` | 1.4 M | 224² | 60.6% | 81.7% |
| `shufflenet_v2_x1_0` | 2.3 M | 224² | 69.4% | 88.3% |
| `shufflenet_v2_x1_5` | 3.5 M | 224² | 72.6% | 90.6% |
| `shufflenet_v2_x2_0` | 7.4 M | 224² | 74.6% | 92.0% |

</div>

---

## Architecture Highlights

- Channel split: block splits input channels into two branches (G1=G2=C/2)
- Channel shuffle after concatenation mixes information across branches
- Four efficiency guidelines: equal channel widths, no excessive group convolution, minimal fragment operations, minimal element-wise operations
- ×0.5 is the smallest torchvision model at 1.4 M parameters

---

## When to Use ShuffleNet V2 (×0.5 / ×1.0 / ×1.5 / ×2.0)

Use ShuffleNet V2 ×0.5 for extreme size constraints (< 2 M parameters). ×1.0 competes with MobileNetV2 on real hardware. ×2.0 approaches MobileNetV3-Large accuracy.

---

## Real-World Use Cases

- Ultra-constrained inference on MCUs and embedded systems (×0.5: 1.4 M params)
- Real-time mobile apps requiring < 2 ms inference on ARM CPU
- Hardware-aware architecture research studying runtime vs FLOPs trade-off

---

## Folder Structure

```
ShuffleNet V2/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
# Replace x1_0 with x0_5 / x1_5 / x2_0
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
model = models.shufflenet_v2_x0_5(weights="IMAGENET1K_V1")

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
@inproceedings{ma2018shufflenet,
  title={{ShuffleNet V2}: Practical Guidelines for Efficient {CNN} Architecture Design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={ECCV},
  pages={116--131},
  year={2018}
}
```

**Paper:** ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
**Authors:** Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
**Venue:** ECCV 2018  **arXiv:** https://arxiv.org/abs/1807.11164

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.ShuffleNet-V2&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
