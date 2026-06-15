import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "./data"
NUM_CLASSES   = 10
BATCH_SIZE    = 32
EPOCHS        = 20
LR_BACKBONE   = 1e-5
LR_CLASSIFIER = 1e-3
WEIGHT_DECAY  = 1e-4
SAVE_PATH     = "densenet201_finetuned.pth"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load pretrained & replace head ───────────────────────────────────────────
model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
in_features     = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Fine-Tuning mode  (all layers unfrozen)")
print(f"  Backbone LR   : {LR_BACKBONE}   |  Classifier LR : {LR_CLASSIFIER}")
print(f"  Trainable     : {trainable:,}  /  Total : {total:,}\n")

# ── Data ──────────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=train_transforms), batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR,"val"),   transform=val_transforms),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ── Dual learning-rate optimizer ─────────────────────────────────────────────
backbone_params   = [p for n, p in model.named_parameters() if "classifier" not in n]
classifier_params = list(model.classifier.parameters())
optimizer = optim.AdamW([
    {"params": backbone_params,   "lr": LR_BACKBONE},
    {"params": classifier_params, "lr": LR_CLASSIFIER},
], weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tl, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            out  = model(imgs); loss = criterion(out, labels)
            if train: loss.backward(); optimizer.step()
            tl      += loss.item() * imgs.size(0)
            correct += out.max(1)[1].eq(labels).sum().item()
            total   += labels.size(0)
    return tl / total, 100.0 * correct / total

best_val_acc = 0.0
print(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<10}")
print("-" * 54)
for epoch in range(1, EPOCHS + 1):
    trl, tra = run_epoch(train_loader, True)
    vll, vla = run_epoch(val_loader,   False)
    scheduler.step()
    print(f"{epoch:<8}{trl:<12.4f}{tra:<12.2f}{vll:<12.4f}{vla:<10.2f}")
    if vla > best_val_acc:
        best_val_acc = vla
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  --> Saved  (val_acc={best_val_acc:.2f}%)")

print(f"\nBest Val Acc : {best_val_acc:.2f}%  |  Saved : {SAVE_PATH}")
