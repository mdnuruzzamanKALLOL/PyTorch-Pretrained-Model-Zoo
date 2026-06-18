# MNASNet (0.5 / 0.75 / 1.0 / 1.3) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** MNASNet PyTorch pretrained mobile NAS 2019 CVPR platform-aware latency ImageNet classification edge

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

MNASNet applies multi-objective NAS to optimize for both ImageNet accuracy and real-world mobile latency on Pixel phones, using a factorized hierarchical search space. Four width-multiplier variants (0.5–1.3) are available in torchvision. MNASNet-1.0 achieves 73.5% top-1 with 3.9 M parameters.

---

<div align="center">

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `mnasnet0_5` | 2.2 M | 224² | 67.7% | 87.5% |
| `mnasnet0_75` | 3.2 M | 224² | 71.2% | 90.5% |
| `mnasnet1_0` | 3.9 M | 224² | 73.5% | 91.5% |
| `mnasnet1_3` | 6.3 M | 224² | 76.5% | 93.2% |

</div>

---

## Architecture Highlights

- Multi-objective NAS optimizing accuracy × latency on Pixel phone hardware
- Factorized hierarchical search space: cell type, kernel size, SE, skip ops
- SE blocks (ratio 0.25) and various kernel sizes (3×3 to 5×5) mixed per layer
- Real mobile latency objective — not proxy FLOPs — in the NAS reward

---

## When to Use MNASNet (0.5 / 0.75 / 1.0 / 1.3)

Use MNASNet when targeting specific mobile latency budgets with NAS-designed architecture. MNASNet-1.0 (73.5%) outperforms MobileNetV2 (71.9%) at similar size.

---

## Real-World Use Cases

- Ultra-low-power mobile inference targeting specific device latency budgets
- IoT and edge devices where MNASNet-0.5 (2.2 M) fits in 1 MB model size
- NAS research baseline for platform-aware architecture design

---

## Folder Structure

```
MNASNet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
# Replace 1_0 with 0_5 / 0_75 / 1_3 for other widths
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
model = models.mnasnet0_5(weights="IMAGENET1K_V1")

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
@inproceedings{tan2019mnasnet,
  title={{MnasNet}: Platform-Aware Neural Architecture Search for Mobile},
  author={Tan, Mingxing and Chen, Bo and Pang, Ruoming and Vasudevan, Vijay and Sandler, Mark and Howard, Andrew and Le, Quoc V},
  booktitle={CVPR},
  pages={2820--2828},
  year={2019}
}
```

**Paper:** MnasNet: Platform-Aware Neural Architecture Search for Mobile
**Authors:** Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le
**Venue:** CVPR 2019  **arXiv:** https://arxiv.org/abs/1807.11626

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>

---

<div align="center">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.PyTorch-Pretrained-Model-Zoo.MNASNet&left_color=%23EE4C2C&right_color=%230e75b6&left_text=Page%20Views)

</div>
