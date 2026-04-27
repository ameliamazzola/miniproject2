import xml.etree.ElementTree as ET
import torch
import cv2
import os
import shutil
import yaml
from pathlib import Path
from torch.utils.data import Dataset

VEHICLE_CLASSES = {"car": 0, "bus": 1, "van": 2, "others": 3}


# ─────────────────────────────────────────
# PyTorch Dataset (used for eval/testing)
# ─────────────────────────────────────────

class DETRACDataset(Dataset):
    def __init__(self, sequences, transform=None):
        """
        sequences: list of (xml_file, img_dir) tuples, one per MVI sequence
        transform: optional albumentations transform
        """
        self.transform = transform
        self.samples = []  # list of (img_path, boxes, labels)

        for xml_file, img_dir in sequences:
            self._parse_sequence(xml_file, img_dir)

    def _parse_sequence(self, xml_file, img_dir):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for frame in root.findall(".//frame"):
            fid = int(frame.get("num"))
            img_path = os.path.join(img_dir, f"img{fid:05d}.jpg")
            if not os.path.exists(img_path):
                continue

            boxes, labels = [], []
            target_list = frame.find("target_list")
            if target_list is not None:
                for target in target_list.findall("target"):
                    box = target.find("box")
                    attr = target.find("attribute")
                    if box is None:
                        continue

                    x = float(box.get("left"))
                    y = float(box.get("top"))
                    w = float(box.get("width"))
                    h = float(box.get("height"))

                    vtype = "others"
                    if attr is not None:
                        vtype = attr.get("vehicle_type", "others").lower()
                        if vtype not in VEHICLE_CLASSES:
                            vtype = "others"

                    boxes.append([x, y, x + w, y + h])
                    labels.append(VEHICLE_CLASSES[vtype])

            if boxes:
                self.samples.append((img_path, boxes, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "img_size": torch.tensor([h, w])
        }

        return img_tensor, target


# ─────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────

def build_sequences(data_root):
    """
    data_root should contain folders like MVI_20011/, MVI_20012/, etc.
    each with a matching XML file alongside it.
    """
    sequences = []
    for entry in sorted(os.listdir(data_root)):
        img_dir = os.path.join(data_root, entry)
        xml_file = os.path.join(data_root, f"{entry}.xml")
        if os.path.isdir(img_dir) and os.path.exists(xml_file):
            sequences.append((xml_file, img_dir))
    return sequences


# ─────────────────────────────────────────
# YOLO format conversion (for Ultralytics)
# ─────────────────────────────────────────

def convert_to_yolo_format(sequences, output_dir, split="train"):
    """
    Converts UA-DETRAC XML annotations into YOLO txt format.

    Output structure:
        output_dir/
            images/train/   images/val/
            labels/train/   labels/val/

    YOLO label format per line: class cx cy w h (all normalized 0-1)
    """
    images_out = Path(output_dir) / "images" / split
    labels_out = Path(output_dir) / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for xml_file, img_dir in sequences:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for frame in root.findall(".//frame"):
            fid = int(frame.get("num"))
            img_path = Path(img_dir) / f"img{fid:05d}.jpg"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            lines = []
            target_list = frame.find("target_list")
            if target_list is not None:
                for target in target_list.findall("target"):
                    box = target.find("box")
                    attr = target.find("attribute")
                    if box is None:
                        continue

                    x = float(box.get("left"))
                    y = float(box.get("top"))
                    w = float(box.get("width"))
                    h = float(box.get("height"))

                    # normalize to 0-1 and convert to center format
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h

                    vtype = "others"
                    if attr is not None:
                        vtype = attr.get("vehicle_type", "others").lower()
                        if vtype not in VEHICLE_CLASSES:
                            vtype = "others"

                    cls_id = VEHICLE_CLASSES[vtype]
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if lines:
                shutil.copy(img_path, images_out / img_path.name)
                label_path = labels_out / img_path.with_suffix(".txt").name
                label_path.write_text("\n".join(lines))


def write_yaml(output_dir, yaml_path):
    """
    Writes the dataset config YAML that Ultralytics needs for training.
    """
    config = {
        "path": str(Path(output_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": ["car", "bus", "van", "others"]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"YAML written to {yaml_path}")
