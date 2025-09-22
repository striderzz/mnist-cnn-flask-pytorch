import os, json, math
from pathlib import Path
from typing import Tuple
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------- Model ----------------
class FashionMNISTModelV2(nn.Module):
    """TinyVGG-style CNN for FashionMNIST (28x28 grayscale)."""
    def __init__(self, input_shape: int = 1, hidden_units: int = 32, output_shape: int = 10):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 28->14
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 14->7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        return self.classifier(x)

# ---------------- Data ----------------
def get_dataloaders(batch_size: int = 128, pin_mem: bool = False) -> Tuple[DataLoader, DataLoader]:
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=tf_train)
    test_ds  = datasets.FashionMNIST(root="data", train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_mem)
    return train_loader, test_loader

# ---------------- Eval ----------------
@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, total_correct / total

# ---------------- Train ----------------
def train(epochs: int = 20, lr: float = 1e-2, batch_size: int = 128,
          save_path: str = "models/03_pytorch_computer_vision_model_2.pth",
          hidden_units: int = 32) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = torch.cuda.is_available()  # avoid pin_memory warning on CPU
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, pin_mem=pin_mem)

    model = FashionMNISTModelV2(input_shape=1, hidden_units=hidden_units, output_shape=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # to populate meta at the end
    last_train_loss = None
    last_test_loss = None
    last_test_acc = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()

        # epoch metrics
        train_loss = running / max(1, len(train_loader))
        test_loss, test_acc = evaluate(model, test_loader, device)
        last_train_loss, last_test_loss, last_test_acc = train_loss, test_loss, test_acc

        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved better weights to {save_path} (acc={best_acc*100:.2f}%)")

    # -------- write sidecar meta for the UI --------
    meta = {
        "epochs": epochs,
        "final_train_loss": round(float(last_train_loss), 4) if last_train_loss is not None else None,
        "final_val_loss": round(float(last_test_loss), 4) if last_test_loss is not None else None,  # using test as "val" here
        "optimizer": type(optimizer).__name__,
        "loss_fn": type(loss_fn).__name__,
        "lr": optimizer.param_groups[0]["lr"],
        "best_test_acc": round(best_acc * 100.0, 2),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = save_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta written: {meta_path}")

    return save_path

if __name__ == "__main__":
    path = train(epochs=int(os.environ.get("EPOCHS", 50)),
                 lr=float(os.environ.get("LR", 1e-2)),
                 batch_size=int(os.environ.get("BATCH_SIZE", 128)),
                 save_path=os.environ.get("SAVE_PATH", "models/03_pytorch_computer_vision_model_2.pth"),
                 hidden_units=int(os.environ.get("HIDDEN_UNITS", 32)))
    print(f"Done. Best model saved to: {path}")
