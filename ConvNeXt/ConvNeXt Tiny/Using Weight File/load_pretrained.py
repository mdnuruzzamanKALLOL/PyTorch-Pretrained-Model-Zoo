import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import sys


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    return model.to(DEVICE).eval()


def get_labels():
    return models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.meta["categories"]


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)


def predict_top5(model, tensor, labels):
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    top_probs, top_idx = torch.topk(probs, 5, dim=1)
    top_probs = top_probs[0].cpu().tolist()
    top_idx   = top_idx[0].cpu().tolist()

    print(f"\n{'Rank':<6} {'Class':<35} {'Confidence':>10}")
    print("-" * 54)
    for rank, (idx, prob) in enumerate(zip(top_idx, top_probs), 1):
        print(f"{rank:<6} {labels[idx]:<35} {prob*100:>9.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_pretrained.py <image_path>")
        sys.exit(1)

    print(f"Device : {DEVICE}  |  Model : ConvNeXt-Tiny (ImageNet weights)")
    model  = load_model()
    labels = get_labels()
    tensor = preprocess(sys.argv[1])
    predict_top5(model, tensor, labels)
