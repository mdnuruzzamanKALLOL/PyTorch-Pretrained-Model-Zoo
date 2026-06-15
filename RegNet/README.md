# RegNet (Y and X series) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** RegNet Y X PyTorch pretrained 2020 CVPR design space FLOPs scalable classification backbone

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

RegNet uses a design space analysis to find a population of good architectures — 'Regular Networks' — parameterized by width, depth, group width, and bottleneck ratio. PyTorch torchvision provides 15 variants across Y (with SE) and X (without SE) series, from RegNetY-400MF (21 M FLOPs) to RegNetY-128GF (644 M parameters).

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `regnet_y_400mf` | 4 M | 224² | 74.0% | 91.6% |
| `regnet_y_800mf` | 6.4 M | 224² | 76.4% | 93.0% |
| `regnet_y_1_6gf` | 11.2 M | 224² | 77.9% | 93.7% |
| `regnet_y_3_2gf` | 19.4 M | 224² | 78.9% | 94.4% |
| `regnet_y_8gf` | 39.4 M | 224² | 80.0% | 95.0% |
| `regnet_y_16gf` | 83.6 M | 224² | 80.4% | 95.2% |
| `regnet_y_32gf` | 145.0 M | 224² | 80.9% | 95.2% |
| `regnet_x_400mf` | 5.5 M | 224² | 72.8% | 90.9% |
| `regnet_x_800mf` | 7.3 M | 224² | 75.2% | 92.3% |
| `regnet_x_1_6gf` | 9.2 M | 224² | 77.0% | 93.4% |
| `regnet_x_3_2gf` | 15.3 M | 224² | 78.4% | 94.0% |
| `regnet_x_8gf` | 39.6 M | 224² | 79.3% | 94.6% |
| `regnet_x_16gf` | 54.3 M | 224² | 80.1% | 94.9% |
| `regnet_x_32gf` | 107.8 M | 224² | 80.6% | 95.0% |

---

## Architecture Highlights

- Design space analysis: study populations of networks rather than single architectures
- Parameterized by {depth, initial width, slope, quantization, bottleneck ratio, group width}
- Y series adds SE blocks to X series for additional accuracy
- Smooth FLOPs scaling from 400 MF to 128 GF across the family
- Strong linear scaling relationship between FLOPs and accuracy

---

## When to Use RegNet (Y and X series)

Use RegNet when you need a well-studied FLOPs-parameterized model family. RegNetY variants (with SE) consistently outperform X variants at the same FLOPs.

---

## Real-World Use Cases

- FLOPs-constrained deployment at various efficiency points
- Neural architecture search baselines and design space analysis research
- Scalable backbone for object detection (Detectron2 compatible)
- Research on network design principles and compute-accuracy trade-offs

---

## Folder Structure

```
RegNet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

# RegNetY-400MF (smallest Y variant)
model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)

# RegNetY-8GF (80% top-1)
model = models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1)
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
model = models.regnet_y_400mf(weights="IMAGENET1K_V1")

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
@inproceedings{radosavovic2020designing,
  title={Designing Network Design Spaces},
  author={Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll\'ar, Piotr},
  booktitle={CVPR},
  pages={10428--10436},
  year={2020}
}
```

**Paper:** Designing Network Design Spaces
**Authors:** Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár
**Venue:** CVPR 2020  **arXiv:** https://arxiv.org/abs/2003.13678

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FmdnuruzzamanKALLOL%2FPyTorch-Pretrained-Model-Zoo%2Ftree%2Fmaster%2FRegNet&count_bg=%23EE4C2C&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Profile%20Views&edge_flat=false)

</div>
