import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

NUM_CLASSES = 10;  BATCH_SIZE = 16;  EPOCHS = 5;  LR = 1e-3
DATA_DIR    = "./data"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    for param in model.parameters(): param.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    print("Trainable layers:")
    for name, p in model.named_parameters():
        if p.requires_grad: print(f"  {name}")
    return model.to(DEVICE)

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return (DataLoader(datasets.ImageFolder(f"{DATA_DIR}/train", transform), batch_size=BATCH_SIZE, shuffle=True,  num_workers=4),
            DataLoader(datasets.ImageFolder(f"{DATA_DIR}/val",   transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4))

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if training: optimizer.zero_grad()
            outputs = model(images); loss = criterion(outputs, labels)
            if training: loss.backward(); optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item(); total += labels.size(0)
    return total_loss / total, 100.0 * correct / total

if __name__ == "__main__":
    print(f"Device : {DEVICE}  |  Mode : Feature Extraction (ConvNeXt-Base)\n")
    model = build_model()
    train_loader, val_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion)
        print(f"Epoch [{epoch}/{EPOCHS}]  Train Acc: {tr_acc:.2f}%  |  Val Acc: {vl_acc:.2f}%")
        if vl_acc > best_acc:
            best_acc = vl_acc; torch.save(model.state_dict(), "convnext_base_feature_extract.pth")
            print(f"  --> Saved ({best_acc:.2f}%)")
    print(f"\nDone. Best Val Acc: {best_acc:.2f}%")
