import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import sys

from alexnet import AlexNet


# ── Config ────────────────────────────────────────────────
WEIGHTS_PATH = "alexnet_best.pth"
NUM_CLASSES  = 10
TOP_K        = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────


def load_model(weights_path, num_classes):
    model = AlexNet(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)   # add batch dim


def predict(model, tensor, class_names=None, top_k=5):
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)

    top_probs, top_indices = torch.topk(probs, top_k, dim=1)
    top_probs   = top_probs[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    print(f"\n{'Rank':<6} {'Class':<25} {'Confidence':>10}")
    print("-" * 44)
    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        label = class_names[idx] if class_names else str(idx)
        print(f"{rank:<6} {label:<25} {prob*100:>9.2f}%")

    return top_indices[0], top_probs[0]


def load_class_names(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # Usage: python inference.py <image_path> [class_names.json]
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [class_names.json]")
        sys.exit(1)

    image_path  = sys.argv[1]
    class_names = None

    if len(sys.argv) == 3:
        class_names = load_class_names(sys.argv[2])

    print(f"Device  : {DEVICE}")
    print(f"Image   : {image_path}")
    print(f"Weights : {WEIGHTS_PATH}")

    model  = load_model(WEIGHTS_PATH, NUM_CLASSES)
    tensor = preprocess(image_path)

    pred_class, confidence = predict(model, tensor, class_names, top_k=TOP_K)
    print(f"\nPredicted class index : {pred_class}  ({confidence*100:.2f}% confidence)")
