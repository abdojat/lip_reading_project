import os
import cv2
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

RAW_DIR = "data/raw"
LABELS_CSV = "data/labels.csv"
OUT_DIR = "data/processed/mouth_frames"

# Number of frames per sample (LRW style)
T_FRAMES = 29
# Output size for each mouth crop
OUT_SIZE = 96  # 96x96 mouth ROI

mp_face_mesh = mp.solutions.face_mesh

# Mouth landmark indices in MediaPipe FaceMesh (lips region)
MOUTH_LANDMARKS = list(set([
    # Outer lips
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # Inner lips
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]))

def sample_frame_indices(num_frames: int, t: int):
    """Uniformly sample t indices from [0, num_frames-1]."""
    if num_frames <= 0:
        return []
    if num_frames >= t:
        return [int(round(i)) for i in list(
            (num_frames - 1) * (j / (t - 1)) for j in range(t)
        )]
    # If video shorter than t: repeat last frame
    idxs = list(range(num_frames))
    while len(idxs) < t:
        idxs.append(num_frames - 1)
    return idxs

def get_mouth_bbox(landmarks, w, h, pad=0.2):
    xs, ys = [], []
    for i in MOUTH_LANDMARKS:
        lm = landmarks[i]
        xs.append(lm.x * w)
        ys.append(lm.y * h)

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Padding
    bw = x2 - x1
    bh = y2 - y1
    x1 -= pad * bw
    x2 += pad * bw
    y1 -= pad * bh
    y2 += pad * bh

    # Clip
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2))
    y2 = min(h - 1, int(y2))

    return x1, y1, x2, y2

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(LABELS_CSV)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for _, row in tqdm(df.iterrows(), total=len(df)):
            speaker = row["speaker"]
            video = row["video"]
            sample_id = f"{speaker}_{os.path.splitext(video)[0]}"

            out_sample_dir = os.path.join(OUT_DIR, sample_id)
            if os.path.isdir(out_sample_dir) and len(os.listdir(out_sample_dir)) >= T_FRAMES:
                continue  # already processed

            video_path = os.path.join(RAW_DIR, speaker, "video", video)
            if not os.path.exists(video_path):
                continue

            frames = read_video_frames(video_path)
            if len(frames) == 0:
                continue

            idxs = sample_frame_indices(len(frames), T_FRAMES)

            os.makedirs(out_sample_dir, exist_ok=True)

            last_bbox = None
            saved = 0

            for j, fi in enumerate(idxs):
                frame = frames[fi]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)

                h, w = frame.shape[:2]

                if res.multi_face_landmarks:
                    landmarks = res.multi_face_landmarks[0].landmark
                    bbox = get_mouth_bbox(landmarks, w, h, pad=0.3)
                    last_bbox = bbox
                else:
                    # Fallback: reuse last bbox if tracking fails
                    if last_bbox is None:
                        continue
                    bbox = last_bbox

                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE))
                out_path = os.path.join(out_sample_dir, f"{j:03d}.png")
                cv2.imwrite(out_path, crop)
                saved += 1

            # If failed to save enough frames, clean the sample folder
            if saved < T_FRAMES:
                for f in os.listdir(out_sample_dir):
                    os.remove(os.path.join(out_sample_dir, f))
                os.rmdir(out_sample_dir)

    print("Done. Mouth frame sequences saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
