import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model           = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
in_features     = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model           = model.to(DEVICE)
print(f'MNASNet 1.0 loaded. Classifier: {in_features} -> {NUM_CLASSES}')
print(model)
