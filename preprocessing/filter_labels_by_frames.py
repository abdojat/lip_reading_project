import os
import pandas as pd

LABELS_CSV = "data/labels.csv"
FRAMES_DIR = "data/processed/mouth_frames"
OUT_CSV = "data/labels_filtered.csv"

df = pd.read_csv(LABELS_CSV)

existing = set(os.listdir(FRAMES_DIR))  # folders like s1_bbaf2n
keep_rows = []

for _, r in df.iterrows():
    sample_id = f"{r['speaker']}_{os.path.splitext(r['video'])[0]}"
    if sample_id in existing:
        keep_rows.append(r)

df2 = pd.DataFrame(keep_rows)
df2.to_csv(OUT_CSV, index=False)
print("Original:", len(df), "Filtered:", len(df2), "Saved:", OUT_CSV)
print("Labels distribution:\n", df2["label"].value_counts())
