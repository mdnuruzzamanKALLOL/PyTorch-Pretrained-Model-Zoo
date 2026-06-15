# Vision Transformer ViT (B/16, B/32, L/16, L/32, H/14) — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** ViT Vision Transformer B16 B32 L16 H14 PyTorch pretrained 2021 ICLR patch tokens self-attention ImageNet classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

Vision Transformer (ViT) applies pure Transformer encoder to non-overlapping image patches (16×16 or 32×32 tokens), demonstrating that CNNs are not necessary for high-accuracy vision when pre-trained on large datasets. PyTorch torchvision provides five ViT variants pretrained with improved recipes (ViT-B/16 reaches 81.1% top-1).

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `vit_b_16` | 86.6 M | 224² | 81.1% | 95.3% |
| `vit_b_32` | 88.2 M | 224² | 75.9% | 92.5% |
| `vit_l_16` | 307.4 M | 224² | 79.7% | 94.6% |
| `vit_l_32` | 306.5 M | 224² | 76.9% | 93.1% |
| `vit_h_14` | 633.5 M | 518² | 88.6% | 98.7% |

---

## Architecture Highlights

- Image split into N fixed-size patches (16×16 or 32×32) → linear embedding as token sequence
- Learnable [CLS] token aggregates global representation for classification
- Standard Transformer encoder: multi-head self-attention + MLP + LayerNorm
- Position embeddings (learnable 1D) encode spatial order of patches
- ViT-H/14: 633 M parameters at 518² achieves 88.6% — among highest on ImageNet

---

## When to Use Vision Transformer ViT (B/16, B/32, L/16, L/32, H/14)

Use ViT-B/16 (81.1%) as the standard Transformer baseline. Prefer ConvNeXt or Swin for detection/segmentation (ViT lacks feature hierarchy). Use ViT-H/14 only for maximum accuracy research tasks.

---

## Real-World Use Cases

- Large-scale pretraining and fine-tuning on domain-specific datasets
- CLIP-style vision encoder in vision-language models
- Few-shot learning via prompt-tuning and adapter layers
- Research on attention mechanisms and patch-based representations
- ViT-H/14 for maximum accuracy when compute is unconstrained

---

## Folder Structure

```
VisionTransformer/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

# ViT-B/16 (recommended default)
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# ViT-H/14 (highest accuracy, 518² input)
model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
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
model = models.vit_b_16(weights="IMAGENET1K_V1")

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
@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={ICLR},
  year={2021}
}
```

**Paper:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
**Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
**Venue:** ICLR 2021  **arXiv:** https://arxiv.org/abs/2010.11929

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://komarev.com/ghpvc/?username=mdnuruzzamanKALLOL&label=Profile%20Views&color=EE4C2C&style=flat-square)

</div>
