# Inception V3

**Paper:** Rethinking the Inception Architecture for Computer Vision (Szegedy et al., CVPR 2016)

## Architecture Overview

| Component | Details |
|-----------|---------|
| Stem | Conv3×3/2 → Conv3×3 → Conv3×3/pad → MaxPool → Conv1×1 → Conv3×3 → MaxPool |
| InceptionA | 3× blocks, factorized 5×5 → two 3×3, pool_proj varies |
| InceptionB | Grid reduction (stride-2) |
| InceptionC | 4× blocks, factorized 7×7 as 1×7 + 7×1 |
| InceptionD | Second grid reduction |
| InceptionE | 2× wide blocks with parallel 1×3 + 3×1 branches |
| Auxiliary | 1 classifier after InceptionC4 (weight=0.3) |
| Head | GlobalAvgPool → Dropout(0.5) → FC(2048→classes) |
| Input Size | 299×299 |
| Parameters | ~27M |

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
| Input | 299×299 |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| LR | 1e-3 |
| Auxiliary Loss Weight | 0.3 |

## Pretrained Weights (torchvision)

```python
from torchvision import models
import torch.nn as nn

model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)         # in_features=2048
model.AuxLogits.fc = nn.Linear(768, NUM_CLASSES)
```
