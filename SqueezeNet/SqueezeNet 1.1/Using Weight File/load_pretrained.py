import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NOTE: SqueezeNet classifier is Conv2d, not Linear!
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
model = model.to(DEVICE)

print(f'Loaded SqueezeNet 1.1 with IMAGENET1K_V1 weights')
print(f'classifier[1] : {model.classifier[1]}')

dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f'Output shape : {out.shape}')
