import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from net import MyAlexNet

from tqdm import tqdm

ROOT_TRAIN = 'data/train'
ROOT_VAL = 'data/val'
SAVE_DIR = 'checkpoints'

# Transform
normalize = transforms.Normalize([0.5]*3, [0.5]*3)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# Datasets & Dataloaders
train_loader = DataLoader(ImageFolder(ROOT_TRAIN, transform=train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(ImageFolder(ROOT_VAL, transform=val_transform), batch_size=32)

# Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyAlexNet().to(device)

# Loss, Optimizer, LR Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training Function
def train_one_epoch(loader):
    model.train()
    total_loss, total_correct = 0, 0
    loop = tqdm(loader, desc="Training", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

        # 动态更新进度条
        loop.set_postfix(loss=loss.item(), acc=total_correct / ((loop.n + 1) * loader.batch_size))

    return total_loss / len(loader), total_correct / len(loader.dataset)

# Validation Function
def validate(loader):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)

# Train Loop
best_acc = 0
epochs = 20
train_losses, val_losses = [], []
train_accs, val_accs = [], []

os.makedirs(SAVE_DIR, exist_ok=True)

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    train_loss, train_acc = train_one_epoch(train_loader)
    val_loss, val_acc = validate(val_loader)

    print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Save best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print("Saved best model.")

    # Save last
    if epoch == epochs - 1:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_model.pth"))

    scheduler.step()
# Plot
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend(); plt.title("Loss Curve"); plt.show()

plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend(); plt.title("Accuracy Curve"); plt.show()

print("Done!")
