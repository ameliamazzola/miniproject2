import os

print("Contents of data folder:")
print(os.listdir("data"))

from src.dataset_utils import build_sequences, DETRACDataset

sequences = build_sequences("data")
print("Number of sequences:", len(sequences))
print("Sequences found:", sequences)

dataset = DETRACDataset(sequences)
print("Dataset size:", len(dataset))

img, target = dataset[0]
print("Image shape:", img.shape)
print("Target:", target)

