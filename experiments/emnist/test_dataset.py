from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from Dataset import Dataset

DATASET_PATH = Path("../../datasets/emnist-balanced.mat")

dataset = Dataset(DATASET_PATH)
dataset.shuffle()
spikes, n_spikes, labels = dataset.get_train_batch(0, 112800)


#print(type(spikes))
#print(spikes.shape)

# print(type(n_spikes))
# print(n_spikes)
# print(n_spikes.shape)

#print(labels.shape)

plt.hist(n_spikes.flatten(), bins=100)
plt.show()
print()