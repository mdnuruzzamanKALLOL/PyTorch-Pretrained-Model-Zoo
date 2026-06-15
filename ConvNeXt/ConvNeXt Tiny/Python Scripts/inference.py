import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json, sys

from convnext_tiny import convnext_tiny


WEIGHTS_PATH = "convnext_tiny_best.pth"
NUM_CLASSES  = 10
TOP_K        = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = convnext_tiny(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)


def predict(model, tensor, class_names=None):
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    top_probs, top_idx = torch.topk(probs, TOP_K, dim=1)
    top_probs = top_probs[0].cpu().tolist()
    top_idx   = top_idx[0].cpu().tolist()

    print(f"\n{'Rank':<6} {'Class':<30} {'Confidence':>10}")
    print("-" * 49)
    for rank, (idx, prob) in enumerate(zip(top_idx, top_probs), 1):
        label = class_names[idx] if class_names else str(idx)
        print(f"{rank:<6} {label:<30} {prob*100:>9.2f}%")
    return top_idx[0], top_probs[0]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [class_names.json]")
        sys.exit(1)

    class_names = json.load(open(sys.argv[2])) if len(sys.argv) == 3 else None

    print(f"Device  : {DEVICE}")
    print(f"Model   : ConvNeXt-Tiny")
    print(f"Weights : {WEIGHTS_PATH}")

    model  = load_model()
    tensor = preprocess(sys.argv[1])
    pred_class, conf = predict(model, tensor, class_names)
    print(f"\nPredicted class : {pred_class}  ({conf*100:.2f}% confidence)")
