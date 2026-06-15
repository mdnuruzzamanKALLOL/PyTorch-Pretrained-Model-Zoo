import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = models.Swin_V2_S_Weights.IMAGENET1K_V1
model   = models.swin_v2_s(weights=weights)

model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model      = model.to(DEVICE)

print(f'Loaded Swin-V2-S with IMAGENET1K_V1 weights')
print(f'head : {model.head}')

dummy = torch.randn(1, 3, 256, 256).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f'Output shape : {out.shape}')
