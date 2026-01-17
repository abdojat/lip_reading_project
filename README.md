# Lip Reading Project

This repository trains a lip-reading model based on a ResNet backbone followed by a BiLSTM decoder using frame-level mouth crops.

## Requirements
- Python 3.9+
- CUDA-enabled GPU (optional but recommended for training)
- Install dependencies with `pip install -r requirements.txt`

## Repository layout
- `train.py` / `evaluate.py`: training and evaluation entry points
- `models/`: dataset loader and LipReading model implementation
- `preprocessing/`: utilities for building splits and filtering labels
- `checkpoints/`: model weights created during training (ignored by git)
- `data/`: raw, processed, and split data (ignored by git)

## Setup
1. Create a virtual environment, e.g. `python -m venv .venv` and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Prepare the dataset:
   * place mouth frame tensors under `data/processed/mouth_frames/`
   * place CSV splits in `data/splits/` (`train.csv`, `val.csv`, etc.)
   * use `preprocessing/make_splits.py` to build splits or `preprocessing/filter_labels_by_frames.py` to prune noisy labels before training.

## Training
Run `python train.py` after the dataset is ready. The script trains for 10 epochs by default, writes checkpoints to `checkpoints/`, and prints per-epoch train/validation accuracy.

## Evaluation
Run `python evaluate.py` to load saved model weights and compute metrics (open `evaluate.py` to customize evaluation datasets and checkpoints).

## GitHub Push Checklist
1. Review `git status` and ensure only tracked changes that should be committed remain.
2. Stage files: `git add README.md .gitignore train.py evaluate.py models preprocessing requirements.txt`.
3. Create the initial commit: `git commit -m "Initial project setup"`.
4. Create a GitHub repository (via `gh repo create` or the website) and add it as the `origin` remote: `git remote add origin git@github.com:<username>/<repo>.git`.
5. Push the main branch: `git push -u origin main`.

## Next steps
- Add dataset download scripts or instructions if you plan to share the project.
- Consider adding automated tests or CI to validate model training/evaluation.
