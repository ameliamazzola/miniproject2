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


import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import yaml

VEHICLE_CLASSES = {
    "car": 0,
    "bus": 1,
    "van": 2,
    "others": 3
}

def build_yolo_dataset(all_sequences, output_dir):
    output_dir = Path(output_dir)

    # 🔴 DELETE OLD DATA (IMPORTANT)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    (output_dir / "images/train").mkdir(parents=True)
    (output_dir / "images/val").mkdir(parents=True)
    (output_dir / "labels/train").mkdir(parents=True)
    (output_dir / "labels/val").mkdir(parents=True)

    # 🔀 80/20 split
    random.seed(42)
    random.shuffle(all_sequences)
    split_idx = int(0.8 * len(all_sequences))

    splits = {
        "train": all_sequences[:split_idx],
        "val": all_sequences[split_idx:]
    }

    for split, sequences in splits.items():
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
                    seq_name = Path(img_dir).name

                    new_img = f"{seq_name}_{img_path.name}"
                    new_lbl = f"{seq_name}_{img_path.stem}.txt"

                    shutil.copy(img_path, output_dir / f"images/{split}/{new_img}")
                    (output_dir / f"labels/{split}/{new_lbl}").write_text("\n".join(lines))

    # 🧾 Write YAML
    yaml_path = output_dir / "data.yaml"
    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": ["car", "bus", "van", "others"]
    }

    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    print("Dataset built at:", output_dir)