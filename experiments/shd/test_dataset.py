import h5py
import gzip
import shutil
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from Dataset import Dataset

DATASET_PATH = Path("../../datasets/")

# Initialize the Dataset object
dataset = Dataset(DATASET_PATH)
dataset.shuffle()
spikes, n_spikes, labels = dataset.get_train_batch(0, 8156)

# Print shapes for verification

# Plot the histogram of processed spike times
plt.hist(spikes[spikes != np.inf].flatten(), bins=100)
plt.xlabel('Spike Time (s)')
plt.ylabel('Frequency')
plt.title('Distribution of Processed Spike Times in SHD Dataset')
plt.show()

