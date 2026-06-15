import torch
import torch.nn as nn
from torchvision import models

# Note: MNASNet 0.75 does not have official ImageNet pretrained weights in torchvision.
# The model is instantiated with random weights below.
NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model           = models.mnasnet0_75(weights=None)
in_features     = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model           = model.to(DEVICE)
print(f'MNASNet 0.75 instantiated (no pretrained weights). Classifier: {in_features} -> {NUM_CLASSES}')
print(model)
