import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import sys


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imagenet_model():
    # Load official pretrained AlexNet (ImageNet weights)
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.to(DEVICE)
    model.eval()
    return model


def load_imagenet_labels():
    # 1000 ImageNet class names
    from torchvision.models import AlexNet_Weights
    return AlexNet_Weights.IMAGENET1K_V1.meta["categories"]


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)


def predict_top5(model, tensor, labels):
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)

    top_probs, top_indices = torch.topk(probs, 5, dim=1)
    top_probs   = top_probs[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    print(f"\n{'Rank':<6} {'Class':<35} {'Confidence':>10}")
    print("-" * 54)
    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        print(f"{rank:<6} {labels[idx]:<35} {prob*100:>9.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_pretrained.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Device  : {DEVICE}")
    print(f"Image   : {image_path}")
    print("Loading pretrained AlexNet (ImageNet weights)...")

    model  = load_imagenet_model()
    labels = load_imagenet_labels()
    tensor = preprocess(image_path)

    predict_top5(model, tensor, labels)
