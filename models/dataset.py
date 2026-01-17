import os
from typing import Tuple, List
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

LABEL_MAP = {"red": 0, "green": 1, "blue": 2, "white": 3}

class MouthFramesDataset(Dataset):
    """
    Each sample:
      - loads 29 mouth frames from data/processed/mouth_frames/{speaker}_{video_id}/000.png..028.png
      - returns tensor: (T, C, H, W) float32 in [0,1]
      - returns label int
    """
    def __init__(self, csv_path: str, frames_root: str, t_frames: int = 29, out_size: int = 96):
        self.df = pd.read_csv(csv_path)
        self.frames_root = frames_root
        self.t_frames = t_frames
        self.out_size = out_size

    def __len__(self) -> int:
        return len(self.df)

    def _sample_id(self, speaker: str, video: str) -> str:
        video_id = os.path.splitext(video)[0]
        return f"{speaker}_{video_id}"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        speaker = row["speaker"]
        video = row["video"]
        label_str = row["label"]

        sample_id = self._sample_id(speaker, video)
        sample_dir = os.path.join(self.frames_root, sample_id)

        frames: List[np.ndarray] = []
        for i in range(self.t_frames):
            fp = os.path.join(sample_dir, f"{i:03d}.png")
            img = cv2.imread(fp)  # BGR uint8
            if img is None:
                raise FileNotFoundError(f"Missing frame: {fp}")
            img = cv2.resize(img, (self.out_size, self.out_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        # (T, H, W, C) -> float32 -> (T, C, H, W)
        x = np.stack(frames, axis=0).astype(np.float32) / 255.0
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.from_numpy(x)

        y = LABEL_MAP[label_str]
        return x, torch.tensor(y, dtype=torch.long)
