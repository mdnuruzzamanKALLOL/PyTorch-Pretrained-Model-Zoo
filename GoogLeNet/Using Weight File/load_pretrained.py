import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
in_features    = model.fc.in_features
model.fc       = nn.Linear(in_features, NUM_CLASSES)
model          = model.to(DEVICE)
print(f'GoogLeNet loaded. FC: {in_features} → {NUM_CLASSES}')
print(model)
