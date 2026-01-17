import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dataset import MouthFramesDataset
from models.lipreading_model import LipReadingResNetBiLSTM

def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    frames_root = "data/processed/mouth_frames"
    train_csv = "data/splits/train.csv"
    val_csv = "data/splits/val.csv"

    train_ds = MouthFramesDataset(train_csv, frames_root)
    val_ds = MouthFramesDataset(val_csv, frames_root)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = LipReadingResNetBiLSTM(num_classes=4, lstm_hidden=128, lstm_layers=1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, 11):  # start with 10 epochs
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits.detach(), y)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x, y = x.to(device), torch.tensor(y).to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_acc += accuracy(logits, y)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = f"checkpoints/best.pt"
            torch.save({"model": model.state_dict()}, ckpt_path)
            print("Saved best checkpoint:", ckpt_path)

    print("Training done. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
