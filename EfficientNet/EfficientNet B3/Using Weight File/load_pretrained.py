import torch
import torch.nn as nn
from torchvision import models

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10

# Load EfficientNet-B3 with ImageNet pretrained weights
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

print('Pretrained model loaded.')
print(f'Original classifier : {model.classifier}')

# Replace the final linear layer for your number of classes
in_features = model.classifier[1].in_features   # 1536
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)
print(f'\nModified classifier : {model.classifier}')
print(f'in_features         : {in_features}')
print(f'num_classes         : {NUM_CLASSES}')

# Quick forward-pass check
dummy  = torch.randn(2, 3, 300, 300).to(DEVICE)
output = model(dummy)
print(f'\nOutput shape : {output.shape}')
