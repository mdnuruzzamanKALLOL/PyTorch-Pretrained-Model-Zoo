import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR    = './data'
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3
NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
in_features  = model.fc.in_features
model.fc     = nn.Linear(in_features, NUM_CLASSES)
model.AuxLogits.fc = nn.Linear(768, NUM_CLASSES)
model        = model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(f'{DATA_DIR}/train', transform=transform)
val_dataset   = datasets.ImageFolder(f'{DATA_DIR}/val',   transform=val_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
head_params = list(model.fc.parameters()) + list(model.AuxLogits.fc.parameters())
optimizer   = optim.Adam(head_params, lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(images)
        outputs = out.logits if hasattr(out, 'logits') else out
        loss = criterion(outputs, labels)
        if hasattr(out, 'aux_logits') and out.aux_logits is not None:
            loss = loss + 0.3 * criterion(out.aux_logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += outputs.max(1)[1].eq(labels).sum().item()
        total      += labels.size(0)
    print(f'Epoch {epoch:02d} | Loss: {total_loss/total:.4f} | Acc: {100.*correct/total:.2f}%')

torch.save(model.state_dict(), 'inceptionv3_feature_extraction.pth')
print('Saved: inceptionv3_feature_extraction.pth')
