from pathlib import Path

from Dataset import Dataset

DATASET_PATH = Path("../../datasets/mnist.npz")

# Dataset
print("Loading datasets...")
dataset = Dataset(path=DATASET_PATH)

one_batch = dataset.get_train_batch(0, 1)
print(one_batch)
print("Done.")