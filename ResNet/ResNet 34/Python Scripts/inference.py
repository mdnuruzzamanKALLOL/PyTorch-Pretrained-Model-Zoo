import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from resnet34 import resnet34

CLASS_NAMES = ['class0', 'class1', 'class2', 'class3', 'class4',
               'class5', 'class6', 'class7', 'class8', 'class9']
NUM_CLASSES = len(CLASS_NAMES)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet34(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('resnet34_best.pth', map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_image(image_path, top_k=5):
    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)

    top_probs, top_idx = torch.topk(probs, top_k, dim=1)
    top_probs = top_probs[0].cpu().tolist()
    top_idx   = top_idx[0].cpu().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img); axes[0].axis('off'); axes[0].set_title('Input Image')
    labels = [CLASS_NAMES[i] for i in top_idx]
    colors = ['#F44336' if i == 0 else '#2196F3' for i in range(top_k)]
    bars   = axes[1].barh(labels[::-1], [p*100 for p in top_probs[::-1]], color=colors[::-1])
    axes[1].set_xlabel('Confidence (%)'); axes[1].set_title(f'Top-{top_k} Predictions')
    for bar, prob in zip(bars, top_probs[::-1]):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{prob*100:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig('inference_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Predicted: {CLASS_NAMES[top_idx[0]]}  ({top_probs[0]*100:.2f}%)')


if __name__ == '__main__':
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
    predict_image(img_path)
