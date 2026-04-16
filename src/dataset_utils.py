import xml.etree.ElementTree as ET
import torch
import cv2
import os
from torch.utils.data import Dataset

class DETRACDataset(Dataset):
    def __init__(self, xml_file, img_dir, transform=None):
        self.xml_file = xml_file
        self.img_dir = img_dir
        self.transform = transform
        
        self.frames = self.parse_xml()
        self.frame_ids = list(self.frames.keys())

    def parse_xml(self):
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        frames = {}

        for frame in root.findall(".//frame"):
            fid = int(frame.get("num"))
            boxes = []

            target_list = frame.find("target_list")
            if target_list is not None:
                for target in target_list.findall("target"):
                    box = target.find("box")
                    if box is not None:
                        x = float(box.get("left"))
                        y = float(box.get("top"))
                        w = float(box.get("width"))
                        h = float(box.get("height"))

                        # convert to x1, y1, x2, y2
                        boxes.append([x, y, x + w, y + h])

            if boxes:
                frames[fid] = boxes

        return frames

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]

        # load image
        img_path = os.path.join(self.img_dir, f"img{fid:05d}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # convert to tensor
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # get boxes
        boxes = torch.tensor(self.frames[fid], dtype=torch.float32)

        # labels (1 = vehicle)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target
