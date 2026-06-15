import sys, json, torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from densenet121 import densenet121

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path, num_classes=10):
    model = densenet121(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess(image_path):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0).to(DEVICE), img

def predict(model, image_path, class_names=None, top_k=5):
    tensor, img = preprocess(image_path)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)
    top_probs   = top_probs[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()
    print(f"\nDenseNet-121 Inference — {image_path}")
    print("=" * 50)
    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        label = class_names[idx] if class_names else f"class_{idx}"
        bar   = "#" * int(prob * 40)
        print(f"  Top-{rank}  {label:<20} {prob*100:6.2f}%  |{bar}")
    print("=" * 50)
    best_label = class_names[top_indices[0]] if class_names else f"class_{top_indices[0]}"
    print(f"\nPrediction : {best_label}  ({top_probs[0]*100:.2f}%)\n")
    return top_indices, top_probs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [class_names.json] [weights.pth] [num_classes]")
        sys.exit(1)
    image_path   = sys.argv[1]
    class_file   = sys.argv[2] if len(sys.argv) > 2 else None
    weights_path = sys.argv[3] if len(sys.argv) > 3 else "densenet121_best.pth"
    num_classes  = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    class_names  = json.load(open(class_file)) if class_file else None
    model        = load_model(weights_path, num_classes)
    predict(model, image_path, class_names)
