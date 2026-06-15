# MaxViT (Tiny)

**Paper:** MaxViT: Multi-Axis Vision Transformer (Tu et al., ECCV 2022)

## Architecture Overview

| Component | Details |
|-----------|---------|
| Stem | Conv3×3/2 → BN → GELU → Conv3×3 → BN → GELU |
| Stages | 4 stages, depths (2, 2, 5, 2), channels (64, 128, 256, 512) |
| Per Block | MBConv → Block Attention (local 8×8 windows) → Grid Attention (global dilated) |
| MBConv | PreNorm → expand 4× → DWConv3×3 → SE → project |
| Attention | Relative position bias for window; standard for grid |
| Head | GlobalAvgPool → Flatten → LayerNorm → Linear → Tanh → Linear |
| Input Size | 224×224 |
| Parameters | ~31M |

### Multi-Axis Attention
- **Block attention**: partitions feature map into non-overlapping 8×8 windows → local context
- **Grid attention**: picks every 8th token in a dilated grid → global context
- Together they achieve O(n) complexity while capturing both local and global patterns

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
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| LR | 1e-3 |
| Scheduler | CosineAnnealingLR |

## Pretrained Weights (torchvision)

```python
from torchvision import models
import torch.nn as nn

model = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
# in_features = 512 for MaxViT-T
```
