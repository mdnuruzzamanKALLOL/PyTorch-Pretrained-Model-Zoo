import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
in_features           = model.classifier[3].in_features
model.classifier[3]   = nn.Linear(in_features, NUM_CLASSES)
model                 = model.to(DEVICE)
print(f'MaxViT-T loaded. Classifier head: {in_features} -> {NUM_CLASSES}')
print(model)
