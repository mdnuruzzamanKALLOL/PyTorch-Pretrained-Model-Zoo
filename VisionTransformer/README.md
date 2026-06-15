# Vision Transformer (ViT)

**Paper:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
**Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
             Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
             Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
**Institution:** Google Research / Google Brain
**Conference:** ICLR 2021

---

## Overview

Vision Transformer (ViT) applies the standard Transformer architecture — originally designed for NLP — **directly to sequences of image patches** with minimal modifications. The key insight: a pure transformer operating on patch sequences can match or exceed CNN performance on image classification, especially at scale.

Core idea: split a 224x224 image into fixed-size non-overlapping patches, linearly embed each patch, prepend a learnable **[CLS] token**, add **positional embeddings**, and pass the sequence through a standard Transformer encoder. The CLS token representation at the final layer is used for classification.

```
Patch sequence length = (image_size / patch_size)^2
e.g.  224x224 / 16x16 = 196 patches  (+1 CLS = 197 tokens)
```

ViT requires large-scale pretraining (JFT-300M or ImageNet-21k) to outperform CNNs. On ImageNet-1k alone, CNNs still held an edge in the original paper.

---

## Variants

| Variant   | Patch | Embed Dim | Depth | Heads | MLP Dim | Patches | Params  | Top-1  | Pretrained  |
|-----------|-------|-----------|-------|-------|---------|---------|---------|--------|-------------|
| ViT-B/16  | 16    | 768       | 12    | 12    | 3072    | 196     | ~86M    | ~81.1% | Yes (V1)    |
| ViT-B/32  | 32    | 768       | 12    | 12    | 3072    | 49      | ~88M    | ~75.9% | Yes (V1)    |
| ViT-L/16  | 16    | 1024      | 24    | 16    | 4096    | 196     | ~307M   | ~79.7% | Yes (V1)    |
| ViT-L/32  | 32    | 1024      | 24    | 16    | 4096    | 49      | ~307M   | ~76.9% | Yes (V1)    |
| ViT-H/14  | 14    | 1280      | 32    | 16    | 5120    | 256     | ~632M   | ~85.7% | Yes (SWAG)  |

---

## Architecture Pipeline

```
Input (3x224x224)
    |
    v
Patch Embedding: Conv2d(3, embed_dim, kernel=patch, stride=patch)
                 -> flatten -> transpose    [B, num_patches, embed_dim]
    |
    v
Prepend CLS token                          [B, num_patches+1, embed_dim]
    |
    v
Add Positional Embedding (learned 1D)      [B, num_patches+1, embed_dim]
    |
    v
Dropout
    |
    v
x depth TransformerBlocks:
  LayerNorm -> MultiHeadSelfAttention -> residual
  LayerNorm -> MLP(GELU)              -> residual
    |
    v
LayerNorm
    |
    v
Extract CLS token: x[:, 0]             [B, embed_dim]
    |
    v
Linear(embed_dim -> classes)
```

---

## Transformer Block

```
Input
  |
  +-- LayerNorm -> MultiHeadSelfAttention -> + (residual)
  |                                          |
  +-- LayerNorm -> MLP(GELU)             -> + (residual)
  |
Output
```

### Multi-Head Self-Attention

```python
qkv   = Linear(embed_dim -> embed_dim * 3)
q,k,v = split into [B, heads, N, head_dim]  where head_dim = embed_dim // heads
attn  = softmax(Q @ K^T / sqrt(head_dim))   # scaled dot-product
out   = Dropout(attn) @ V
out   = proj(reshape(out))                  # Linear(embed_dim -> embed_dim)
```

### MLP Block

```
Linear(embed_dim -> mlp_dim) -> GELU -> Dropout -> Linear(mlp_dim -> embed_dim) -> Dropout
```

`mlp_dim = embed_dim * 4` for all ViT variants.

---

## Sequence Length Per Variant

| Variant  | Patch | Grid   | Patches | + CLS | Seq Len |
|----------|-------|--------|---------|-------|---------|
| ViT-B/16 | 16    | 14x14  | 196     | +1    | 197     |
| ViT-B/32 | 32    | 7x7    | 49      | +1    | 50      |
| ViT-L/16 | 16    | 14x14  | 196     | +1    | 197     |
| ViT-L/32 | 32    | 7x7    | 49      | +1    | 50      |
| ViT-H/14 | 14    | 16x16  | 256     | +1    | 257     |

Smaller patch size = longer sequence = more tokens = higher accuracy but more compute.

---

## Classifier Head

The torchvision ViT uses `model.heads.head` as the final linear layer:

```python
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
```

| Variant       | in_features |
|---------------|-------------|
| ViT-B/16, B/32 | 768        |
| ViT-L/16, L/32 | 1024       |
| ViT-H/14      | 1280        |

---

## Training Configuration (From Scratch)

| Setting       | B/16  | B/32  | L/16  | L/32  | H/14  |
|---------------|-------|-------|-------|-------|-------|
| Batch Size    | 32    | 32    | 8     | 16    | 4     |
| Optimizer     | AdamW | AdamW | AdamW | AdamW | AdamW |
| LR            | 1e-4  | 1e-4  | 1e-4  | 1e-4  | 1e-4  |
| Weight Decay  | 0.01  | 0.01  | 0.01  | 0.01  | 0.01  |
| Scheduler     | CosineAnnealingLR (T_max=20) for all |
| Loss          | CrossEntropyLoss for all |
| Dropout       | 0.1   | 0.1   | 0.1   | 0.1   | 0.1   |
| Epochs        | 20    | 20    | 20    | 20    | 20    |

ViT uses AdamW + Cosine schedule instead of SGD + StepLR (standard for transformers).

---

## Transfer Learning Quick Reference

### Load Pretrained Weights

```python
from torchvision import models
import torch.nn as nn

# ViT-B/16
model            = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model.heads.head = nn.Linear(768, NUM_CLASSES)

# ViT-L/16
model            = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
model.heads.head = nn.Linear(1024, NUM_CLASSES)

# ViT-H/14 (SWAG linear probing weights, 224x224)
model            = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
model.heads.head = nn.Linear(1280, NUM_CLASSES)
```

### Feature Extraction (freeze backbone)

```python
for param in model.parameters():
    param.requires_grad = False

model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
optimizer = torch.optim.Adam(model.heads.head.parameters(), lr=1e-3)
```

### Fine-Tuning (dual learning rates)

```python
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'heads' not in n],
     'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.heads.head.parameters(),
     'lr': 1e-3},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
```

---

## Pretrained Weights (torchvision)

| Variant   | Function              | Weights Enum                                  | Top-1  |
|-----------|-----------------------|-----------------------------------------------|--------|
| ViT-B/16  | `models.vit_b_16()`   | `ViT_B_16_Weights.IMAGENET1K_V1`              | ~81.1% |
| ViT-B/32  | `models.vit_b_32()`   | `ViT_B_32_Weights.IMAGENET1K_V1`              | ~75.9% |
| ViT-L/16  | `models.vit_l_16()`   | `ViT_L_16_Weights.IMAGENET1K_V1`              | ~79.7% |
| ViT-L/32  | `models.vit_l_32()`   | `ViT_L_32_Weights.IMAGENET1K_V1`              | ~76.9% |
| ViT-H/14  | `models.vit_h_14()`   | `ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1`  | ~85.7% |

ViT-H/14 also has `IMAGENET1K_SWAG_E2E_V1` (518x518, top-1 ~88.6%) for higher accuracy.

---

## Folder Structure

```
VisionTransformer/
+-- README.md                      <- this file
+-- ViT b_16/
|   +-- NoteBook/
|   |   +-- vit_b_16.ipynb         - full notebook (arch + train + ROC AUC)
|   +-- Python Scripts/
|   |   +-- vit_b_16.py            - model architecture (all classes + factory fn)
|   |   +-- train.py               - AdamW + CosineAnnealingLR training loop
|   |   +-- inference.py           - single-image top-K prediction
|   |   +-- How to run.txt
|   +-- Using Weight File/
|       +-- load_pretrained.py     - load torchvision weights
|       +-- feature_extraction.py  - frozen backbone training
|       +-- fine_tuning.py         - dual-LR fine-tuning
|       +-- How to run.txt
+-- ViT b_32/  (same structure)
+-- ViT l_16/  (same structure)
+-- ViT l_32/  (same structure)
+-- ViT h_14/  (same structure)
```

---

## Comparison with CNN Architectures

| Model           | Params  | Top-1  | Architecture Type          |
|-----------------|---------|--------|----------------------------|
| ResNet-50       | ~25.6M  | 80.9%  | CNN (residual)             |
| ResNeXt-50 32x4d| ~25M    | 79.8%  | CNN (grouped conv)         |
| ViT-B/16        | ~86M    | 81.1%  | Pure Transformer           |
| ViT-L/16        | ~307M   | 79.7%  | Pure Transformer (larger)  |
| ViT-H/14        | ~632M   | 85.7%  | Pure Transformer (SWAG)    |
| Swin-T          | ~28M    | 81.3%  | Hierarchical Transformer   |

ViT performs best when pretrained on large datasets. For small datasets, CNN-based models are generally more sample-efficient.

---

## Citation

```bibtex
@inproceedings{dosovitskiy2021image,
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author    = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and
               Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and
               Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and
               Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}
```
