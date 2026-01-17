import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

from models.dataset import MouthFramesDataset, LABEL_MAP
from models.lipreading_model import LipReadingResNetBiLSTM

IDX2LABEL = {v: k for k, v in LABEL_MAP.items()}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    frames_root = "data/processed/mouth_frames"
    test_csv = "data/splits/test.csv"
    ckpt_path = "checkpoints/best.pt"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first to create checkpoints/best.pt"
        )

    # Dataset / loader
    test_ds = MouthFramesDataset(test_csv, frames_root)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # Model
    model = LipReadingResNetBiLSTM(num_classes=4, lstm_hidden=256, lstm_layers=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(y.cpu().numpy().tolist())

    acc = accuracy_score(all_true, all_preds)
    print("\nTest Accuracy:", acc)

    # Confusion matrix
    labels = [LABEL_MAP["red"], LABEL_MAP["green"], LABEL_MAP["blue"], LABEL_MAP["white"]]
    cm = confusion_matrix(all_true, all_preds, labels=labels)
    print("\nConfusion Matrix (rows=true, cols=pred) [red, green, blue, white]:\n", cm)

    # Classification report
    target_names = ["red", "green", "blue", "white"]
    print("\nClassification Report:\n")
    print(classification_report(all_true, all_preds, target_names=target_names, digits=4))

    # --- Demo: show 5 sample predictions ---
    print("\nSample predictions (first 5 test samples):")
    for i in range(min(5, len(all_true))):
        print(f"  true={IDX2LABEL[all_true[i]]:>5s} | pred={IDX2LABEL[all_preds[i]]:>5s}")

if __name__ == "__main__":
    main()
