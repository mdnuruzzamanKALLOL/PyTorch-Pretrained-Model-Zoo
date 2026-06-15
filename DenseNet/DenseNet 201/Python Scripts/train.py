import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from densenet201 import densenet201

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "./data"
NUM_CLASSES = 10
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
WEIGHT_DECAY= 1e-4
SAVE_PATH   = "densenet201_best.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ───────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Data ─────────────────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transforms)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"Classes : {train_dataset.classes}")
print(f"Train   : {len(train_dataset)} samples | Val : {len(val_dataset)} samples")

# ── Model ─────────────────────────────────────────────────────────────────────
model     = densenet201(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters : {total_params:,} | Device : {DEVICE}\n")

# ── Train/Eval helpers ────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct      += outputs.max(1)[1].eq(labels).sum().item()
            total        += labels.size(0)
    return running_loss / total, 100.0 * correct / total

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0
print(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<10}{'LR':<10}")
print("-" * 64)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"{epoch:<8}{train_loss:<12.4f}{train_acc:<12.2f}{val_loss:<12.4f}{val_acc:<10.2f}{current_lr:<10.6f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  --> Saved best model  (val_acc={best_val_acc:.2f}%)")

print(f"\nTraining complete. Best Val Acc : {best_val_acc:.2f}%")
print(f"Model saved : {SAVE_PATH}")
