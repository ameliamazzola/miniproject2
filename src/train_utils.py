#!pip install ultralytics

import os
import sys
from pathlib import Path
from ultralytics import YOLO

# so notebooks can import from src/
sys.path.append(str(Path.cwd().parent))
from src.dataset_utils import build_sequences, convert_to_yolo_format, write_yaml

# ── change these to match where you downloaded UA-DETRAC ──
DATA_ROOT = "/path/to/UA-DETRAC"        # folder containing MVI_XXXXX subfolders
YOLO_DATA_DIR = "../outputs/yolo_data"  # where converted data will be written
YAML_PATH = "../outputs/detrac.yaml"
MODEL_DIR = "../model"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(YOLO_DATA_DIR, exist_ok=True)

import random

all_sequences = build_sequences(DATA_ROOT)
print(f"Total sequences found: {len(all_sequences)}")

# 80/20 train/val split at the sequence level
random.seed(42)
random.shuffle(all_sequences)
split_idx = int(0.8 * len(all_sequences))
train_sequences = all_sequences[:split_idx]
val_sequences = all_sequences[split_idx:]

print(f"Train sequences: {len(train_sequences)}")
print(f"Val sequences:   {len(val_sequences)}")

# Convert annotations to YOLO format
print("\nConverting training data...")
convert_to_yolo_format(train_sequences, YOLO_DATA_DIR, split="train")

print("Converting validation data...")
convert_to_yolo_format(val_sequences, YOLO_DATA_DIR, split="val")

print("Done.")
