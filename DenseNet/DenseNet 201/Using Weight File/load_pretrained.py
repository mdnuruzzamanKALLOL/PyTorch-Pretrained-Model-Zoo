import torch, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import urllib.request, json

# ── Load torchvision pretrained DenseNet-201 ─────────────────────────────────
model  = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

total = sum(p.numel() for p in model.parameters())
print(f"DenseNet-201 pretrained loaded  |  Parameters : {total:,}")
print(f"Classifier : {model.classifier}\n")

# ── ImageNet class labels ────────────────────────────────────────────────────
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
try:
    with urllib.request.urlopen(IMAGENET_LABELS_URL) as r:
        imagenet_labels = json.loads(r.read().decode())
except Exception:
    imagenet_labels = [f"class_{i}" for i in range(1000)]

# ── Inference helper ─────────────────────────────────────────────────────────
def predict_imagenet(image_path, top_k=5):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img    = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    top_probs, top_idx = torch.topk(probs, top_k, dim=1)
    print(f"Image : {image_path}")
    print("=" * 50)
    for i, (idx, prob) in enumerate(zip(top_idx[0].tolist(), top_probs[0].tolist()), 1):
        print(f"  Top-{i}  {imagenet_labels[idx]:<30} {prob*100:6.2f}%")
    print("=" * 50)

# ── Quick smoke test with a random tensor ────────────────────────────────────
dummy = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    out = model(dummy)
print(f"Smoke test — output shape : {out.shape}")
print("\nTo run on a real image call:")
print('  predict_imagenet("your_image.jpg")')
