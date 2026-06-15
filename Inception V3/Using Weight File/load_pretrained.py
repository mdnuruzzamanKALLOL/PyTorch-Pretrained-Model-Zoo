import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc    = nn.Linear(in_features, NUM_CLASSES)
model       = model.to(DEVICE)
print(f'Inception V3 loaded. FC: {in_features} -> {NUM_CLASSES}')
print(model)
