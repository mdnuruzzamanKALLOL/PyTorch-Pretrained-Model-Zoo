# Swin Transformer — Hierarchical Vision Transformer using Shifted Windows

**Paper (V1):** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
**Authors:** Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
**Conference:** ICCV 2021 (Best Paper)

**Paper (V2):** Swin Transformer V2: Scaling Up Capacity and Resolution
**Conference:** CVPR 2022

---

## Overview

Swin Transformer introduces a **hierarchical feature map** (like CNNs) via **Patch Merging** and limits self-attention to **local windows**, making it scale linearly with image size. Key contributions:

- **Shifted Window Attention**: alternates between regular (W-MSA) and shifted (SW-MSA) windows to enable cross-window connections
- **Hierarchical design**: 4 stages with 4×, 8×, 16×, 32× downsampling — compatible with dense prediction tasks
- **Relative Position Bias**: adds learned bias table to attention logits for spatial awareness

Swin V2 improvements:
- **Scaled cosine attention**: replaces dot-product with cosine similarity to stabilize large-model training
- **Continuous log-spaced position bias (CPB)**: MLP-generated bias from log-spaced relative coordinates, enabling resolution transfer
- **Post-normalization**: LayerNorm after attention/MLP instead of before

---

## Variants

| Variant    | Version | img  | embed | depths      | heads           | Params | Top-1 | window |
|------------|---------|------|-------|-------------|-----------------|--------|-------|--------|
| Swin-T     | V1      | 224  | 96    | [2,2,6,2]   | [3,6,12,24]     | ~28M   | 81.3% | 7      |
| Swin-S     | V1      | 224  | 96    | [2,2,18,2]  | [3,6,12,24]     | ~50M   | 83.0% | 7      |
| Swin-B     | V1      | 224  | 128   | [2,2,18,2]  | [4,8,16,32]     | ~88M   | 83.5% | 7      |
| Swin-V2-T  | V2      | 256  | 96    | [2,2,6,2]   | [3,6,12,24]     | ~28M   | 82.0% | 8      |
| Swin-V2-S  | V2      | 256  | 96    | [2,2,18,2]  | [3,6,12,24]     | ~50M   | 83.7% | 8      |
| Swin-V2-B  | V2      | 256  | 128   | [2,2,18,2]  | [4,8,16,32]     | ~88M   | 84.6% | 8      |

---

## Architecture Pipeline

```
Input (3 x H x W)
    |
    v
PatchEmbed: Conv2d(3, C, 4, stride=4) + LayerNorm   -> (B, H/4*W/4, C)
    |
    v
Stage 1: depth[0] x SwinBlock  (W-MSA / SW-MSA)     -> (B, H/4*W/4,   C)
    |  PatchMerging: 4C -> 2C, halves spatial res
Stage 2: depth[1] x SwinBlock                        -> (B, H/8*W/8,   2C)
    |  PatchMerging
Stage 3: depth[2] x SwinBlock                        -> (B, H/16*W/16, 4C)
    |  PatchMerging
Stage 4: depth[3] x SwinBlock (no downsampling)      -> (B, H/32*W/32, 8C)
    |
    v
LayerNorm -> GlobalAvgPool -> Linear(8C, num_classes)
```

Final feature dim: `8 * embed_dim` = 768 (T/S) or 1024 (B)

---

## Shifted Window Attention

Each stage alternates between:
- **W-MSA**: attention within fixed windows (no shift)
- **SW-MSA**: windows shifted by `(window_size//2, window_size//2)`, with cyclic padding + masking

Masking ensures shifted tokens from different semantic regions do not attend to each other.

---

## V1 vs V2 Key Differences

| Aspect | Swin V1 | Swin V2 |
|--------|---------|---------|
| Attention | Dot-product + scale | Scaled cosine attention |
| Position bias | Learned table | Log-spaced CPB via MLP |
| Normalization | Pre-LN | Post-LN |
| Input size | 224x224 | 256x256 |
| Window size | 7 | 8 |

---

## Classifier Head

All variants: replace `model.head`

```python
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
```

| Variant   | in_features |
|-----------|-------------|
| Swin-T/S  | 768         |
| Swin-B    | 1024        |
| Swin-V2-T/S | 768       |
| Swin-V2-B | 1024        |

---

## Training Configuration (From Scratch)

| Setting    | Swin-T/V2-T | Swin-S/V2-S | Swin-B/V2-B |
|------------|-------------|-------------|-------------|
| Batch Size | 32          | 16          | 8           |
| Optimizer  | AdamW (lr=1e-4, wd=0.01) for all      |
| Scheduler  | CosineAnnealingLR (T_max=20) for all  |
| Epochs     | 20          | 20          | 20          |

---

## Transfer Learning Quick Reference

```python
from torchvision import models
import torch.nn as nn

# Swin-T
model      = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

# Swin-V2-B
model      = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
```

**Feature Extraction** — freeze backbone, train head only:
```python
for param in model.parameters():
    param.requires_grad = False
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
optimizer  = optim.Adam(model.head.parameters(), lr=1e-3)
```

**Fine-Tuning** — dual learning rates:
```python
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'head' not in n],
     'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.head.parameters(), 'lr': 1e-3},
])
```

---

## Pretrained Weights (torchvision)

| Variant    | Function              | Weights Enum                      | in_features |
|------------|-----------------------|-----------------------------------|-------------|
| Swin-T     | `models.swin_t()`     | `Swin_T_Weights.IMAGENET1K_V1`    | 768         |
| Swin-S     | `models.swin_s()`     | `Swin_S_Weights.IMAGENET1K_V1`    | 768         |
| Swin-B     | `models.swin_b()`     | `Swin_B_Weights.IMAGENET1K_V1`    | 1024        |
| Swin-V2-T  | `models.swin_v2_t()`  | `Swin_V2_T_Weights.IMAGENET1K_V1` | 768         |
| Swin-V2-S  | `models.swin_v2_s()`  | `Swin_V2_S_Weights.IMAGENET1K_V1` | 768         |
| Swin-V2-B  | `models.swin_v2_b()`  | `Swin_V2_B_Weights.IMAGENET1K_V1` | 1024        |

---

## Folder Structure

```
SwinTransformer/
+-- README.md
+-- Swin_T/
|   +-- NoteBook/          swin_t.ipynb
|   +-- Python Scripts/    swin_t.py  train.py  inference.py  How to run.txt
|   +-- Using Weight File/ load_pretrained.py  feature_extraction.py  fine_tuning.py  How to run.txt
+-- Swin_S/   (same)
+-- Swin_B/   (same)
+-- Swin_V2_T/ (same, V2 arch, 256x256)
+-- Swin_V2_S/ (same)
+-- Swin_V2_B/ (same)
```

---

## Citation

```bibtex
@inproceedings{liu2021swin,
  title     = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author    = {Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle = {ICCV},
  year      = {2021}
}

@inproceedings{liu2022swinv2,
  title     = {Swin Transformer V2: Scaling Up Capacity and Resolution},
  author    = {Liu, Ze and Hu, Han and Lin, Yutong and Yao, Zhuliang and Xie, Zhenda and Wei, Yixuan and Ning, Jia and Cao, Yue and Zhang, Zheng and Dong, Li and others},
  booktitle = {CVPR},
  year      = {2022}
}
```
