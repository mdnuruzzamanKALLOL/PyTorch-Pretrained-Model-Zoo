import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from googlenet import googlenet

DATA_DIR   = './data'
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 0.001
NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
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

train_dataset = datasets.ImageFolder(f'{DATA_DIR}/train', transform=train_transform)
val_dataset   = datasets.ImageFolder(f'{DATA_DIR}/val',   transform=val_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model     = googlenet(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out = model(images)
            if train and isinstance(out, tuple):
                main_out, aux1, aux2 = out
                loss = criterion(main_out, labels)
                loss = loss + 0.3 * criterion(aux1, labels) + 0.3 * criterion(aux2, labels)
            else:
                main_out = out if not isinstance(out, tuple) else out[0]
                loss = criterion(main_out, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += main_out.max(1)[1].eq(labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, 100.0 * correct / total


print(f"{'Epoch':<8} {'Tr Loss':<10} {'Tr Acc':<10} {'Val Loss':<10} {'Val Acc':<10}")
print('-' * 50)
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step()
    print(f'{epoch:<8} {tr_loss:<10.4f} {tr_acc:<10.2f} {vl_loss:<10.4f} {vl_acc:<10.2f}')
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), 'googlenet_best.pth')

print(f'\nBest Val Acc: {best_val_acc:.2f}%')
print('Model saved: googlenet_best.pth')
