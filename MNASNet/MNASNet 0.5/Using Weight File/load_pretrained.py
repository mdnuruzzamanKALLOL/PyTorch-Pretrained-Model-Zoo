import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model           = models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1)
in_features     = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model           = model.to(DEVICE)
print(f'MNASNet 0.5 loaded. Classifier: {in_features} -> {NUM_CLASSES}')
print(model)
