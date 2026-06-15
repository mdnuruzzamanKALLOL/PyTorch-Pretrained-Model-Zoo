import sys
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from efficientnet_b3 import efficientnet_b3

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = int(sys.argv[4]) if len(sys.argv) > 4 else 10
WEIGHTS     = sys.argv[3] if len(sys.argv) > 3 else 'efficientnet_b3_best.pth'
CLASS_FILE  = sys.argv[2] if len(sys.argv) > 2 else None
IMAGE_PATH  = sys.argv[1]
TOP_K       = 5

transform = transforms.Compose([
    transforms.Resize(332),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

if CLASS_FILE:
    with open(CLASS_FILE) as f:
        class_names = json.load(f)
else:
    class_names = [str(i) for i in range(NUM_CLASSES)]

model = efficientnet_b3(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

img    = Image.open(IMAGE_PATH).convert('RGB')
tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs = F.softmax(model(tensor), dim=1)

top_probs, top_idx = torch.topk(probs, TOP_K, dim=1)
top_probs = top_probs[0].cpu().tolist()
top_idx   = top_idx[0].cpu().tolist()

print(f'\nEfficientNet-B3 — Inference Results')
print('=' * 45)
BAR_LEN = 30
for rank, (idx, prob) in enumerate(zip(top_idx, top_probs), 1):
    name    = class_names[idx]
    filled  = int(prob * BAR_LEN)
    bar     = '█' * filled + '░' * (BAR_LEN - filled)
    print(f'  {rank}. {name:<20} [{bar}] {prob*100:5.1f}%')
print('=' * 45)
print(f'  Predicted : {class_names[top_idx[0]]}  ({top_probs[0]*100:.2f}%)')
