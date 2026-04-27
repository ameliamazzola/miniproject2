import shutil
import random
from pathlib import Path
from datetime import datetime

from src.dataset_utils import build_sequences, convert_to_yolo_format, write_yaml

sequences = build_sequences("data")

random.seed(42)
random.shuffle(sequences)

split_idx = int(0.8 * len(sequences))

train_sequences = sequences[:split_idx]
val_sequences = sequences[split_idx:]

output_dir = Path("yolo_dataset")

# rename old converted dataset instead of deleting it
if output_dir.exists():
    backup_name = Path(f"yolo_dataset_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.rename(backup_name)
    print(f"Old yolo_dataset renamed to {backup_name}")

print("Train sequences:", train_sequences)
print("Val sequences:", val_sequences)

convert_to_yolo_format(train_sequences, output_dir, split="train")
convert_to_yolo_format(val_sequences, output_dir, split="val")

write_yaml(output_dir, "yolo_dataset.yaml")

print("Conversion complete")