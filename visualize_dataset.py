import cv2
import matplotlib.pyplot as plt
from src.dataset_utils import build_sequences, DETRACDataset

sequences = build_sequences("data")
dataset = DETRACDataset(sequences)

img, target = dataset[0]

# tensor CHW -> numpy HWC
img_np = img.permute(1, 2, 0).numpy()
img_np = (img_np * 255).astype("uint8").copy()

for box in target["boxes"]:
    x1, y1, x2, y2 = box.int().tolist()
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.figure(figsize=(12, 7))
plt.imshow(img_np)
plt.axis("off")
plt.title("UA-DETRAC Bounding Box Check")
plt.show()