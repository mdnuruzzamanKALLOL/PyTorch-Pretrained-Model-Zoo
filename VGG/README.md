# VGG — Very Deep Convolutional Networks

**Paper:** Very Deep Convolutional Networks for Large-Scale Image Recognition
**Authors:** Karen Simonyan, Andrew Zisserman
**Institution:** Visual Geometry Group (VGG), University of Oxford
**Conference:** ICLR 2015

---

## Overview

VGGNet demonstrated that **network depth** using very small (3x3) convolution filters is a critical component for achieving strong performance in image recognition. By replacing large-kernel convolutions (7x7, 11x11 used in AlexNet) with stacks of 3x3 filters, VGG:

- Achieves the same effective receptive field with fewer parameters
- Introduces more non-linearities (more ReLU activations)
- Enables a simple, uniform architecture easy to extend

Two 3x3 convolutions have the same receptive field as one 5x5, and three 3x3 convolutions match one 7x7 — but with fewer parameters and more activations.

The BN variants add **Batch Normalization** after each Conv, which stabilizes training and allows higher learning rates.

---

## Variants

| Variant    | Conv Layers | BN  | Parameters | Top-1  | Pretrained |
|------------|-------------|-----|-----------|--------|------------|
| VGG-11     | 8           | No  | ~132M     | ~69.0% | Yes (V1)   |
| VGG-11-BN  | 8           | Yes | ~132M     | ~70.4% | Yes (V1)   |
| VGG-13     | 10          | No  | ~133M     | ~69.9% | Yes (V1)   |
| VGG-13-BN  | 10          | Yes | ~133M     | ~71.6% | Yes (V1)   |
| VGG-16     | 13          | No  | ~138M     | ~71.6% | Yes (V1)   |
| VGG-16-BN  | 13          | Yes | ~138M     | ~73.4% | Yes (V1)   |
| VGG-19     | 16          | No  | ~144M     | ~72.4% | Yes (V1)   |
| VGG-19-BN  | 16          | Yes | ~144M     | ~74.2% | Yes (V1)   |

---

## Architecture Pipeline

```
Input (3x224x224)
    |
    v
Block 1: N x [Conv3x3 -> (BN ->) ReLU]  -> MaxPool2x2   [-> 64  x 112x112]
Block 2: N x [Conv3x3 -> (BN ->) ReLU]  -> MaxPool2x2   [-> 128 x  56x 56]
Block 3: N x [Conv3x3 -> (BN ->) ReLU]  -> MaxPool2x2   [-> 256 x  28x 28]
Block 4: N x [Conv3x3 -> (BN ->) ReLU]  -> MaxPool2x2   [-> 512 x  14x 14]
Block 5: N x [Conv3x3 -> (BN ->) ReLU]  -> MaxPool2x2   [-> 512 x   7x  7]
    |
    v
AdaptiveAvgPool(7x7) -> Flatten          [-> 512*7*7 = 25088]
    |
    v
Classifier:
  Linear(25088 -> 4096) -> ReLU -> Dropout(0.5)   [0,1,2]
  Linear(4096  -> 4096) -> ReLU -> Dropout(0.5)   [3,4,5]
  Linear(4096  -> classes)                         [6] <- replace for transfer
```

---

## Conv Layers Per Block

| Variant  | Block1 (64ch) | Block2 (128ch) | Block3 (256ch) | Block4 (512ch) | Block5 (512ch) | Total |
|----------|---------------|----------------|----------------|----------------|----------------|-------|
| VGG-11   | 1             | 1              | 2              | 2              | 2              | 8     |
| VGG-13   | 2             | 2              | 2              | 2              | 2              | 10    |
| VGG-16   | 2             | 2              | 3              | 3              | 3              | 13    |
| VGG-19   | 2             | 2              | 4              | 4              | 4              | 16    |

The BN variants (-BN) have the same conv structure but with BatchNorm inserted after each Conv, before ReLU.

---

## Config List (make_layers format)

```python
VGG_CONFIGS = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# int = Conv3x3 out_channels,  'M' = MaxPool2x2
```

---

## Classifier Head Structure

| Index  | Layer               | Notes                          |
|--------|---------------------|--------------------------------|
| [0]    | Linear(25088, 4096) | 512 * 7 * 7 = 25088            |
| [1]    | ReLU                |                                |
| [2]    | Dropout(0.5)        |                                |
| [3]    | Linear(4096, 4096)  |                                |
| [4]    | ReLU                |                                |
| [5]    | Dropout(0.5)        |                                |
| [6]    | Linear(4096, 1000)  | **Replace this for transfer**  |

`in_features = 4096` for all VGG variants.

---

## Training Configuration (From Scratch)

| Setting       | VGG-11/13 | VGG-16   | VGG-19   |
|---------------|-----------|----------|----------|
| Input Size    | 224x224   | 224x224  | 224x224  |
| Batch Size    | 64        | 32       | 16       |
| Optimizer     | Adam      | Adam     | Adam     |
| LR            | 1e-3      | 1e-3     | 1e-3     |
| Scheduler     | StepLR step=7, gamma=0.1 (all) |
| Loss          | CrossEntropyLoss (all)         |
| Epochs        | 20        | 20       | 20       |

---

## Transfer Learning Quick Reference

### Load Pretrained Weights

```python
from torchvision import models
import torch.nn as nn

# VGG-16
model               = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

# VGG-16-BN
model               = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
```

### Feature Extraction (freeze backbone)

```python
for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=1e-3)
```

### Fine-Tuning (dual learning rates)

```python
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier.6' not in n],
     'lr': 1e-5},
    {'params': model.classifier[6].parameters(),
     'lr': 1e-3},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
```

---

## Pretrained Weights (torchvision)

| Variant   | Function              | Weights Enum                    | in_features |
|-----------|-----------------------|---------------------------------|-------------|
| VGG-11    | `models.vgg11()`      | `VGG11_Weights.IMAGENET1K_V1`   | 4096        |
| VGG-11-BN | `models.vgg11_bn()`   | `VGG11_BN_Weights.IMAGENET1K_V1`| 4096        |
| VGG-13    | `models.vgg13()`      | `VGG13_Weights.IMAGENET1K_V1`   | 4096        |
| VGG-13-BN | `models.vgg13_bn()`   | `VGG13_BN_Weights.IMAGENET1K_V1`| 4096        |
| VGG-16    | `models.vgg16()`      | `VGG16_Weights.IMAGENET1K_V1`   | 4096        |
| VGG-16-BN | `models.vgg16_bn()`   | `VGG16_BN_Weights.IMAGENET1K_V1`| 4096        |
| VGG-19    | `models.vgg19()`      | `VGG19_Weights.IMAGENET1K_V1`   | 4096        |
| VGG-19-BN | `models.vgg19_bn()`   | `VGG19_BN_Weights.IMAGENET1K_V1`| 4096        |

---

## Folder Structure

```
VGG/
+-- README.md                      <- this file
+-- VGG 11/
|   +-- NoteBook/
|   |   +-- vgg_11.ipynb           - full notebook (arch + train + ROC AUC)
|   +-- Python Scripts/
|   |   +-- vgg_11.py              - model architecture (make_layers + VGG)
|   |   +-- train.py               - training loop with StepLR
|   |   +-- inference.py           - single-image top-K prediction
|   |   +-- How to run.txt
|   +-- Using Weight File/
|       +-- load_pretrained.py     - load torchvision weights
|       +-- feature_extraction.py  - frozen backbone training
|       +-- fine_tuning.py         - dual-LR fine-tuning
|       +-- How to run.txt
+-- VGG 11_BN/  (same structure, with BatchNorm)
+-- VGG 13/     (same structure)
+-- VGG 13_BN/  (same structure)
+-- VGG 16/     (same structure)
+-- VGG 16_BN/  (same structure)
+-- VGG 19/     (same structure)
+-- VGG 19_BN/  (same structure)
```

---

## Comparison with Related Architectures

| Model      | Params  | Top-1  | Key Difference                           |
|------------|---------|--------|------------------------------------------|
| AlexNet    | ~61M    | ~56.5% | Large kernels (11x11, 5x5), LRN          |
| VGG-11     | ~132M   | ~69.0% | All 3x3 kernels, no BN                   |
| VGG-16-BN  | ~138M   | ~73.4% | 13 conv layers + BN                      |
| VGG-19-BN  | ~144M   | ~74.2% | Deepest VGG variant                      |
| ResNet-50  | ~25.6M  | ~80.9% | Residual connections, 5x fewer params    |

VGG's main limitation is its large parameter count (~132-144M) concentrated in the FC layers (the 3 FC layers alone hold ~119M params). Modern architectures like ResNet achieve better accuracy with far fewer parameters by removing large FC layers.

---

## Citation

```bibtex
@inproceedings{simonyan2015very,
  title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author    = {Simonyan, Karen and Zisserman, Andrew},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2015}
}
```
