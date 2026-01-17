import pandas as pd
from sklearn.model_selection import train_test_split
import os

IN_CSV = "data/labels_filtered.csv"
OUT_DIR = "data/splits"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
)

train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
print("Train label dist:\n", train_df["label"].value_counts())
