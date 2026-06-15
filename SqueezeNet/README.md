# SqueezeNet

**Paper:** SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and less than 0.5MB model size
**Authors:** Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
**Year:** 2016 (arXiv)

---

## Overview

SqueezeNet achieves AlexNet-level accuracy (~58%) on ImageNet with **50x fewer parameters** (~1.24M vs ~61M) and a model size under 0.5MB. This makes it ideal for deployment on embedded systems and mobile devices.

### Design Strategies

1. **Replace 3×3 filters with 1×1 filters** — 9× fewer parameters per filter
2. **Decrease input channels to 3×3 filters** — via squeeze layers
3. **Downsample late** — keep large activation maps for better accuracy

### Fire Module

The core building block: a squeeze layer (1×1 Conv) feeding two parallel expand layers (1×1 and 3×3 Conv).

```
input
  |
  v
Squeeze: Conv1x1(in, s1x1) -> ReLU          # compresses channels
  |
  +---> Expand1x1: Conv1x1(s1x1, e1x1) -> ReLU    +---> Expand3x3: Conv3x3(s1x1, e3x3) -> ReLU  / cat -> output
```

Output channels = `e1x1 + e3x3`. Typical ratio: `s1x1 < (e1x1 + e3x3) / 4`.

---

## Variants

| Variant | First Conv | MaxPool positions | Params | Top-1  |
|---------|-----------|-------------------|--------|--------|
| 1.0     | Conv7x7/2  | after F2, F5, F9  | ~1.25M | ~58.1% |
| 1.1     | Conv3x3/2  | after F2, F4, F7  | ~1.24M | ~58.2% |

**1.1 uses 2.4x less computation** than 1.0 with comparable accuracy, by repositioning the MaxPool layers earlier (reduces spatial size before expensive Fire modules).

---

## Architecture (SqueezeNet 1.0)

```
Input (3x224x224)
Conv(3, 96, 7, /2) -> ReLU -> MaxPool(3, /2)
Fire(96,  s=16, e=64+64=128)
Fire(128, s=16, e=64+64=128)
Fire(128, s=32, e=128+128=256) -> MaxPool(3, /2)
Fire(256, s=32, e=128+128=256)
Fire(256, s=48, e=192+192=384)
Fire(384, s=48, e=192+192=384)
Fire(384, s=64, e=256+256=512) -> MaxPool(3, /2)
Fire(512, s=64, e=256+256=512)
Dropout(0.5)
Conv(512, num_classes, 1) -> ReLU -> AdaptiveAvgPool(1x1)
Flatten -> logits
```

## Architecture (SqueezeNet 1.1)

```
Input (3x224x224)
Conv(3, 64, 3, /2) -> ReLU -> MaxPool(3, /2)
Fire(64,  s=16, e=64+64=128)
Fire(128, s=16, e=64+64=128) -> MaxPool(3, /2)
Fire(128, s=32, e=128+128=256)
Fire(256, s=32, e=128+128=256) -> MaxPool(3, /2)
Fire(256, s=48, e=192+192=384)
Fire(384, s=48, e=192+192=384)
Fire(384, s=64, e=256+256=512)
Fire(512, s=64, e=256+256=512)
Dropout(0.5)
Conv(512, num_classes, 1) -> ReLU -> AdaptiveAvgPool(1x1)
Flatten -> logits
```

---

## Classifier Head — IMPORTANT

Unlike most architectures, SqueezeNet uses **Conv2d as the final classification layer**, not Linear.

```python
# structure
model.classifier = nn.Sequential(
    nn.Dropout(0.5),                          # [0]
    nn.Conv2d(512, num_classes, kernel_size=1), # [1] <-- replace this
    nn.ReLU(inplace=True),                    # [2]
    nn.AdaptiveAvgPool2d((1, 1)),             # [3]
)
```

**Replace `model.classifier[1]` for transfer learning:**

```python
model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
# in_channels = 512  (same for both variants)
```

---

## Training Configuration (From Scratch)

| Setting    | SqueezeNet 1.0 | SqueezeNet 1.1 |
|------------|----------------|----------------|
| Input Size | 224x224        | 224x224        |
| Batch Size | 128            | 128            |
| Optimizer  | Adam (lr=1e-3) | Adam (lr=1e-3) |
| Scheduler  | StepLR step=7, gamma=0.1 (both) |
| Loss       | CrossEntropyLoss (both)          |
| Epochs     | 20             | 20             |

---

## Transfer Learning Quick Reference

```python
from torchvision import models
import torch.nn as nn

# SqueezeNet 1.0
model               = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)

# SqueezeNet 1.1
model               = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
```

**Feature Extraction:**
```python
for param in model.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=1e-3)
```

**Fine-Tuning:**
```python
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier.1' not in n],
     'lr': 1e-5},
    {'params': model.classifier[1].parameters(), 'lr': 1e-3},
])
```

---

## Pretrained Weights (torchvision)

| Variant        | Function                  | Weights Enum                        | in_channels |
|----------------|---------------------------|-------------------------------------|-------------|
| SqueezeNet 1.0 | `models.squeezenet1_0()`  | `SqueezeNet1_0_Weights.IMAGENET1K_V1` | 512         |
| SqueezeNet 1.1 | `models.squeezenet1_1()`  | `SqueezeNet1_1_Weights.IMAGENET1K_V1` | 512         |

---

## Comparison

| Model          | Params  | Top-1  | Notes                              |
|----------------|---------|--------|------------------------------------|
| AlexNet        | ~61M    | ~56.5% | Reference baseline                 |
| SqueezeNet 1.0 | ~1.25M  | ~58.1% | 50x smaller, better accuracy       |
| SqueezeNet 1.1 | ~1.24M  | ~58.2% | 2.4x less compute than 1.0        |
| MobileNetV2    | ~3.4M   | ~71.9% | More accurate, more params         |
| ResNet-18      | ~11.7M  | ~69.8% | Much larger, significantly better  |

---

## Folder Structure

```
SqueezeNet/
+-- README.md
+-- SqueezeNet 1.0/
|   +-- NoteBook/          squeezenet1_0.ipynb
|   +-- Python Scripts/    squeezenet1_0.py  train.py  inference.py  How to run.txt
|   +-- Using Weight File/ load_pretrained.py  feature_extraction.py  fine_tuning.py  How to run.txt
+-- SqueezeNet 1.1/        (same structure)
```

---

## Citation

```bibtex
@article{iandola2016squeezenet,
  title   = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and less than 0.5MB model size},
  author  = {Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
  journal = {arXiv preprint arXiv:1602.07360},
  year    = {2016}
}
```
