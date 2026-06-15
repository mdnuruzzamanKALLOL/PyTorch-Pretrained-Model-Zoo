import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR       = './data'
NUM_CLASSES    = 10
BATCH_SIZE     = 8
EPOCHS         = 20
LR_BACKBONE    = 1e-5
LR_CLASSIFIER  = 1e-3
WEIGHT_DECAY   = 1e-4

# ── Load & Modify Head ────────────────────────────────────────────
model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)

in_features = model.classifier[1].in_features   # 2560
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ── Transforms ───────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(632),
    transforms.RandomCrop(600),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(632),
    transforms.CenterCrop(600),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(f'{DATA_DIR}/train', transform=train_transform)
val_dataset   = datasets.ImageFolder(f'{DATA_DIR}/val',   transform=val_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ── Dual-LR Optimizer ─────────────────────────────────────────────
backbone_params   = [p for n, p in model.named_parameters() if 'classifier' not in n]
classifier_params = list(model.classifier.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': backbone_params,   'lr': LR_BACKBONE},
    {'params': classifier_params, 'lr': LR_CLASSIFIER},
], weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training Loop ────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if train: loss.backward(); optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += outputs.max(1)[1].eq(labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, 100.0 * correct / total

best_val_acc = 0.0
print(f"{'Epoch':<8} {'Tr Loss':<10} {'Tr Acc':<10} {'Val Loss':<10} {'Val Acc':<10}")
print('-' * 50)
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step()
    print(f'{epoch:<8} {tr_loss:<10.4f} {tr_acc:<10.2f} {vl_loss:<10.4f} {vl_acc:<10.2f}')
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), 'efficientnet_b7_ft.pth')

print(f'\nBest Val Acc : {best_val_acc:.2f}%')
print('Saved        : efficientnet_b7_ft.pth')
