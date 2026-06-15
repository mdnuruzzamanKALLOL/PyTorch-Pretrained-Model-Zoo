import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR    = './data'
BATCH_SIZE  = 32
EPOCHS      = 20
LR_BACKBONE = 1e-5
LR_HEAD     = 1e-3
NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
in_features         = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model               = model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(f'{DATA_DIR}/train', transform=transform)
val_dataset   = datasets.ImageFolder(f'{DATA_DIR}/val',   transform=val_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier.1' not in n],
      'lr': LR_BACKBONE},
    {'params': model.classifier[1].parameters(), 'lr': LR_HEAD},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += outputs.max(1)[1].eq(labels).sum().item()
        total      += labels.size(0)
    scheduler.step()

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels   = images.to(DEVICE), labels.to(DEVICE)
            outputs          = model(images)
            val_correct     += outputs.max(1)[1].eq(labels).sum().item()
            val_total       += labels.size(0)
    val_acc = 100. * val_correct / val_total
    print(f'Epoch {epoch:02d} | Loss: {total_loss/total:.4f} | Val Acc: {val_acc:.2f}%')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'mnasnet_1_0_fine_tuned.pth')

print(f'Best Val Acc: {best_val_acc:.2f}%')
print('Saved: mnasnet_1_0_fine_tuned.pth')
