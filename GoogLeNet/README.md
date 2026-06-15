# GoogLeNet (Inception V1)

**Paper:** Going Deeper with Convolutions (Szegedy et al., CVPR 2015)

## Architecture Overview

| Component | Details |
|-----------|---------|
| Stem | Conv7×7/2 → MaxPool → Conv1×1 → Conv3×3 → MaxPool |
| Inception Modules | 9 modules (3a–3b, 4a–4e, 5a–5b) |
| Auxiliary Classifiers | 2 (after 4a and 4d, only during training) |
| Head | GlobalAvgPool → Dropout(0.4) → FC(1024→classes) |
| Input Size | 224×224 |
| Parameters | ~6.8M |

### Inception Module
Each module has 4 parallel branches concatenated along channel dim:
- Branch 1: 1×1 Conv
- Branch 2: 1×1 Conv → 3×3 Conv
- Branch 3: 1×1 Conv → 5×5 Conv
- Branch 4: 3×3 MaxPool → 1×1 Conv

### Auxiliary Loss
During training, two auxiliary classifiers inject gradient at intermediate layers.
Total loss = main_loss + 0.3 × aux1_loss + 0.3 × aux2_loss

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
python feature_extraction.py   # frozen backbone
python fine_tuning.py          # dual-LR fine-tuning
```

## Training Config

| Setting | Value |
|---------|-------|
| Input | 224×224 |
| Batch Size | 64 |
| Epochs | 20 |
| Optimizer | Adam |
| LR | 1e-3 |
| Scheduler | StepLR (step=5, γ=0.1) |

## Pretrained Weights (torchvision)

```python
from torchvision import models
import torch.nn as nn

model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
```
