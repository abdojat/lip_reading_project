import os
import csv

RAW_DIR = "data/raw"
OUTPUT_CSV = "data/labels.csv"

COLOR_WORDS = ["red", "green", "blue", "white"]

rows = []

for speaker in os.listdir(RAW_DIR):
    speaker_path = os.path.join(RAW_DIR, speaker)
    align_path = os.path.join(speaker_path, "align")

    if not os.path.isdir(align_path):
        continue

    for file in os.listdir(align_path):
        if not file.endswith(".align"):
            continue

        filepath = os.path.join(align_path, file)
        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                word = parts[2]
                if word in COLOR_WORDS:
                    rows.append({
                        "speaker": speaker,
                        "video": file.replace(".align", ".mpg"),
                        "label": word
                    })
                    break

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["speaker", "video", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} samples to {OUTPUT_CSV}")
