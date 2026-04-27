from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ameliamazzola/miniproject2
%cd miniproject2

import sys
sys.path.insert(0, '/content/miniproject2')

!pip install ultralytics

import sys
from ultralytics import YOLO
from src.inference_utils import run_inference

MODEL_PATH  = "/content/drive/MyDrive/model/model.pt"
VIDEO_IN    = "/content/drive/MyDrive/your_test_video.mp4"
VIDEO_OUT   = "/content/drive/MyDrive/outputs/annotated_output.mp4"

model = YOLO(MODEL_PATH)
print("Model loaded successfully")

results = run_inference(
    model=model,
    video_path=VIDEO_IN,
    output_path=VIDEO_OUT,
    device="cuda",
    score_thresh=0.5,
    fps_window_seconds=5,
    model_type="yolo"
)
print("=" * 40)
print("TRAFFIC ANALYSIS RESULTS")
print("=" * 40)
print(f"Total frames processed: {results['total_frames']}")
print(f"Max road load:          {results['max_load']} vehicles in a single frame")
print(f"Max traffic flow:       {results['max_flow']} vehicles in {results['window_seconds']} seconds")

import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(VIDEO_OUT)
frames_to_show = [0, 50, 100, 200]

for fnum in frames_to_show:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.imshow(frame)
    plt.title(f"Frame {fnum}")
    plt.axis("off")
    plt.show()

cap.release()

from google.colab import files
files.download(VIDEO_OUT)
