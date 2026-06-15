import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = models.VGG19_Weights.IMAGENET1K_V1
model   = models.vgg19(weights=weights)

model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
model               = model.to(DEVICE)

print(f'Loaded VGG-19 with IMAGENET1K_V1 weights')
print(f'Classifier[6] : {model.classifier[6]}')

dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f'Output shape  : {out.shape}')
