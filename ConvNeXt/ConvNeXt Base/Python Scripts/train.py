import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from convnext_base import convnext_base


# ── Config ────────────────────────────────────────────────
NUM_CLASSES = 10
BATCH_SIZE  = 16
EPOCHS      = 10
LR          = 5e-4
DATA_DIR    = "./data"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(datasets.ImageFolder(f"{DATA_DIR}/train", train_transform),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(datasets.ImageFolder(f"{DATA_DIR}/val",   val_transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += outputs.max(1)[1].eq(labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    print(f"Device : {DEVICE}  |  Model : ConvNeXt-Base\n")

    train_loader, val_loader = get_dataloaders()

    model     = convnext_base(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion)
        scheduler.step()

        print(f"Epoch [{epoch:>2}/{EPOCHS}]  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  |  "
              f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.2f}%")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), "convnext_base_best.pth")
            print(f"  --> Best model saved ({best_acc:.2f}%)")

    print(f"\nDone. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
