import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from vit_l_16 import vit_l_16

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10
CLASS_NAMES = [f'class_{i}' for i in range(NUM_CLASSES)]

model = vit_l_16(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('vit_l_16_best.pth', map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image_path, top_k=5):
    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    top_probs, top_idx = torch.topk(probs, top_k, dim=1)
    print(f'\nTop-{top_k} predictions for: {image_path}')
    for rank, (idx, prob) in enumerate(zip(top_idx[0], top_probs[0]), 1):
        print(f'  {rank}. {CLASS_NAMES[idx.item()]:<20} {prob.item()*100:.2f}%')


if __name__ == '__main__':
    import sys
    predict(sys.argv[1] if len(sys.argv) > 1 else 'test.jpg')
