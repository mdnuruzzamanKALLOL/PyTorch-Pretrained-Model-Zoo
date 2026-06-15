import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model    = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model    = model.to(DEVICE)

print(f'Loaded RegNet-X-800MF with IMAGENET1K_V1 weights')
print(f'FC : {model.fc}')

dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f'Output shape : {out.shape}')
