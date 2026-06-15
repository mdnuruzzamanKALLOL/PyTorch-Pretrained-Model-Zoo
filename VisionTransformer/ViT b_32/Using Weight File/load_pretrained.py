import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = models.ViT_B_32_Weights.IMAGENET1K_V1
model   = models.vit_b_32(weights=weights)

model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model            = model.to(DEVICE)

print(f'Loaded ViT-B/32 with IMAGENET1K_V1 weights')
print(f'Classifier head : {model.heads.head}')

dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f'Output shape    : {out.shape}')
