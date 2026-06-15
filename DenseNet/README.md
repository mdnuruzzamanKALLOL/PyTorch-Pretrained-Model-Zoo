# DenseNet — PyTorch / Torchvision Pretrained Model | ImageNet Classification

> **Keywords:** DenseNet PyTorch pretrained ImageNet 2017 CVPR medical imaging Keras dense connections CheXNet classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-pretrained-3776AB?style=flat-square)](https://pytorch.org/vision/)
[![ImageNet](https://img.shields.io/badge/Pretrained-ImageNet-4ecdc4?style=flat-square)](https://www.image-net.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)](../../LICENSE)

---

## Overview

DenseNet connects each layer to every subsequent layer, enabling maximum feature reuse with minimum parameters. PyTorch torchvision provides four variants (121/161/169/201), all pretrained on ImageNet. DenseNet-121 achieved 71.5% top-1 at just 8 M parameters and remains the gold-standard backbone for medical imaging AI.

---

## Variants & ImageNet Performance

| Model | Params | Input | Top-1 | Top-5 |
|-------|:------:|:-----:|:-----:|:-----:|
| `densenet121` | 8 M | 224² | 74.4% | 91.9% |
| `densenet161` | 29 M | 224² | 77.1% | 93.6% |
| `densenet169` | 14 M | 224² | 75.6% | 92.8% |
| `densenet201` | 20 M | 224² | 76.9% | 93.5% |

---

## Architecture Highlights

- Dense connectivity: L(L+1)/2 direct connections in an L-layer dense block
- Growth rate k (32 or 48) controls how many new feature maps each layer adds
- Bottleneck layers (1×1→3×3) reduce computation within dense blocks
- Transition layers with compression θ=0.5 compact feature maps between blocks
- Implicit deep supervision from dense gradient flow to all layers

---

## When to Use DenseNet

Use DenseNet-121 for medical imaging and low-data tasks. DenseNet-161 (k=48, wider) is the highest-accuracy variant. For general use, EfficientNet or ConvNeXt provide better accuracy/cost ratio.

---

## Real-World Use Cases

- Medical imaging: CheXNet chest X-ray uses DenseNet-121 as backbone
- Skin lesion classification: top-performing ISIC challenge solutions
- Low-data regimes: strong performance with < 10 k training images
- Multi-label classification with sigmoid output head

---

## Folder Structure

```
DenseNet/
├── NoteBook/                 # Jupyter notebook: architecture walkthrough, training, evaluation
├── Python Scripts/           # Standalone .py: build from scratch, training loop, inference
└── Using Weight File/        # Load pretrained weights, feature extraction, fine-tuning
```

---

## Quick Start

```python
import torchvision.models as models

model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
# or densenet161 / densenet169 / densenet201
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
model = models.densenet121(weights="IMAGENET1K_V1")

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
@inproceedings{huang2017densely,
  title={Densely Connected Convolutional Networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={CVPR},
  pages={4700--4708},
  year={2017}
}
```

**Paper:** Densely Connected Convolutional Networks
**Authors:** Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
**Venue:** CVPR 2017  **arXiv:** https://arxiv.org/abs/1608.06993

---

<div align="center">
<sub>Part of the <a href="../README.md">PyTorch Pretrained Model Zoo</a> — 80 models, 20 families, ready-to-run notebooks and scripts</sub>
</div>


---

<div align="center">

![Profile Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FmdnuruzzamanKALLOL%2FPyTorch-Pretrained-Model-Zoo%2Ftree%2Fmaster%2FDenseNet&count_bg=%23EE4C2C&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Profile%20Views&edge_flat=false)

</div>
