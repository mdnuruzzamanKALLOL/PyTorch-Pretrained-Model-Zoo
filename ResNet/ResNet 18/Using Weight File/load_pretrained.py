import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model           = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features     = model.fc.in_features
model.fc        = nn.Linear(in_features, NUM_CLASSES)
model           = model.to(DEVICE)
print(f'ResNet-18 loaded. FC: {in_features} -> {NUM_CLASSES}')
print(model)
