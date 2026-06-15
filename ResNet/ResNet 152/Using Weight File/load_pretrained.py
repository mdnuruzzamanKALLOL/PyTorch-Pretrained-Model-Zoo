import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model           = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
in_features     = model.fc.in_features
model.fc        = nn.Linear(in_features, NUM_CLASSES)
model           = model.to(DEVICE)
print(f'ResNet-152 loaded. FC: {in_features} -> {NUM_CLASSES}')
print(model)
