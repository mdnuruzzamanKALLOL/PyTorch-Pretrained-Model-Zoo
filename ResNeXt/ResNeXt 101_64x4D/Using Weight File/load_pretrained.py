import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
model   = models.resnext101_64x4d(weights=weights)

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model    = model.to(DEVICE)

print(f'Loaded {dis} with {wv} weights')
print(f'Classifier fc : {model.fc}')

dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f'Output shape  : {out.shape}')
