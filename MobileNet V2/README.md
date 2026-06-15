# MobileNet V2

**Paper:** MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., CVPR 2018)

## Architecture Overview

| Component | Details |
|-----------|---------|
| Stem | Conv3×3/2(3→32) → BN → ReLU6 |
| Stage 1 | 1× InvRes, expand=1, out=16, stride=1 |
| Stage 2 | 2× InvRes, expand=6, out=24, stride=2 |
| Stage 3 | 3× InvRes, expand=6, out=32, stride=2 |
| Stage 4 | 4× InvRes, expand=6, out=64, stride=2 |
| Stage 5 | 3× InvRes, expand=6, out=96, stride=1 |
| Stage 6 | 3× InvRes, expand=6, out=160, stride=2 |
| Stage 7 | 1× InvRes, expand=6, out=320, stride=1 |
| Head | Conv1×1(320→1280) → AvgPool → Dropout → FC |
| Input Size | 224×224 |
| Parameters | ~3.4M |

### Key Design: Linear Bottleneck
The last pointwise conv in each block has no activation (linear bottleneck)
to prevent destroying information in low-dimensional manifolds.

## Quick Start

### From Scratch
```
cd "Python Scripts"
python train.py
python inference.py your_image.jpg
```

### Using Pretrained Weights
```
cd "Using Weight File"
python feature_extraction.py
python fine_tuning.py
```

## Training Config

| Setting | Value |
|---------|-------|
| Input | 224×224 |
| Batch Size | 64 |
| Epochs | 20 |
| Optimizer | Adam, LR=1e-3 |
| Scheduler | CosineAnnealingLR |

## Pretrained Weights (torchvision)

```python
from torchvision import models
import torch.nn as nn

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
# in_features = 1280
```
