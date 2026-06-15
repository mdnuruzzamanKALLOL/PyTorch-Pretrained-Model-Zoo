import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets


# ── Config ────────────────────────────────────────────────
NUM_CLASSES    = 10
BATCH_SIZE     = 32
EPOCHS         = 10
LR_BACKBONE    = 0.0001    # small lr for pretrained layers
LR_CLASSIFIER  = 0.001     # larger lr for new head
DATA_DIR       = "./data"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────


def build_model(num_classes):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

    # Unfreeze ALL layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Replace final classifier layer
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model.to(DEVICE)


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader


def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()

    # Separate lr for backbone vs new head
    optimizer = optim.Adam([
        {"params": model.features.parameters(),    "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(),  "lr": LR_CLASSIFIER},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(f"Epoch [{epoch:>2}/{EPOCHS}]  "
              f"Train Acc: {train_acc:.2f}%  |  Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "alexnet_finetuned.pth")
            print(f"  --> Best model saved ({best_acc:.2f}%)")

    print(f"\nDone. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print("Mode   : Fine-Tuning (all layers unfrozen)\n")

    model                    = build_model(NUM_CLASSES)
    train_loader, val_loader = get_dataloaders()
    train(model, train_loader, val_loader)
