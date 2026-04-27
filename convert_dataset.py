from src.dataset_utils import build_sequences, convert_to_yolo_format, write_yaml

sequences = build_sequences("data")

# simple split
train_sequences = sequences[:1]
val_sequences = sequences[1:2] if len(sequences) > 1 else sequences[:1]

output_dir = "yolo_dataset"

convert_to_yolo_format(train_sequences, output_dir, split="train")
convert_to_yolo_format(val_sequences, output_dir, split="val")

write_yaml(output_dir, "yolo_dataset.yaml")

print("Conversion complete")