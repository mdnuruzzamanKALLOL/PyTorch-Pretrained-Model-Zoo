# ConvNeXt — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** ConvNeXt PyTorch pretrained ImageNet 2022 CVPR CNN transfer learning backbone classification fine-tuning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

The ConvNeXt family modernizes ResNet with design ideas from Vision Transformers: patchify stem, inverted bottleneck with 7×7 depthwise conv, Layer Norm, GELU, and fewer normalization layers. PyTorch torchvision provides four variants matching Swin Transformer accuracy while keeping the simplicity of pure CNNs.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `convnext_tiny` | 28 M | 224² | 82.5% | 96.1% |
| `convnext_small` | 50 M | 224² | 83.6% | 96.6% |
| `convnext_base` | 89 M | 224² | 84.1% | 96.8% |
| `convnext_large` | 198 M | 224² | 84.4% | 96.9% |

---

## Architecture Highlights

- Patchify stem: 4×4 stride-4 non-overlapping patches as first layer
- 7×7 depthwise conv in each block mimics self-attention's large receptive field
- Layer Normalization for training stability without batch-size dependency
- GELU activation and one BN/activation per block (fewer than ResNet)
- Stochastic depth + EMA training recipe from DeiT/Swin

---

## When to Use ConvNeXt

Prefer ConvNeXt-Tiny/Small over ResNet for new classification and detection tasks. Matches ViT accuracy with CNN deployment advantages (no attention, standard ops).

---

## Real-World Use Cases

- Object detection backbone (COCO) — exceeds Swin Transformer mAP
- Semantic segmentation (ADE20K) with UperNet head
- Transfer learning for satellite, medical, and fine-grained domains
- Pure-CNN alternative to Vision Transformers for deployment simplicity

---

## Folder Structure

```
ConvNeXt/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

# ConvNeXt Tiny
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

# ConvNeXt Small
model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)

# ConvNeXt Base
model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

# ConvNeXt Large
model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
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
model = models.convnext_tiny(weights="IMAGENET1K_V1")

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
@inproceedings{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={CVPR},
  pages={11976--11986},
  year={2022}
}
```

**Paper:** A ConvNet for the 2020s
**Authors:** Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie
**Venue:** CVPR 2022  **arXiv:** https://arxiv.org/abs/2201.03545

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://komarev.com/ghpvc/?username=mdnuruzzamanKALLOL&label=Profile%20Views&color=EE4C2C&style=flat-square)

</div>
