import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR    = './data'
BATCH_SIZE  = 16
EPOCHS      = 10
NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model    = model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_loader = DataLoader(
    datasets.ImageFolder(f'{DATA_DIR}/train', transform=transform),
    batch_size=16, shuffle=True,  num_workers=4)
val_loader   = DataLoader(
    datasets.ImageFolder(f'{DATA_DIR}/val',   transform=transform),
    batch_size=16, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            if train: loss.backward(); optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += out.max(1)[1].eq(labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, 100.0 * correct / total


print('Feature Extraction — backbone frozen, fc only')
print(f"{'Epoch':<8} {'Tr Loss':<10} {'Tr Acc':<10} {'Val Loss':<10} {'Val Acc':<10}")
print('-' * 50)
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step()
    print(f'{epoch:<8} {tr_loss:<10.4f} {tr_acc:<10.2f} {vl_loss:<10.4f} {vl_acc:<10.2f}')

torch.save(model.state_dict(), 'regnet_y_8gf_fe.pth')
print('Saved: regnet_y_8gf_fe.pth')
