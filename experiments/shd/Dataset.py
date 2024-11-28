from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import h5py
import gzip
import shutil

from matplotlib import pyplot as plt

TIME_WINDOW = 100e-3
MAX_VALUE = 1.4
N_NEURONS = 700

class Dataset:
    def __init__(self, path: Path, n_train_samples: int = None, n_test_samples: int = None):

        self.__train_spike_times, self.__train_n_spikes_per_neuron, self.__train_labels = self.load_shd_data(path / "shd_train.h5.gz")
        self.__test_spike_times, self.__test_n_spikes_per_neuron, self.__test_labels = self.load_shd_data(path / "shd_test.h5.gz")

        if n_train_samples is not None:
            self.__train_spike_times, self.__train_n_spikes_per_neuron, self.__train_labels = \
                self.__reduce_data(self.__train_spike_times, self.__train_n_spikes_per_neuron, self.__train_labels, n_train_samples)

        if n_test_samples is not None:
            self.__test_spike_times, self.__test_n_spikes_per_neuron, self.__test_labels = \
                self.__reduce_data(self.__test_spike_times, self.__test_n_spikes_per_neuron, self.__test_labels, n_test_samples)

        #if bias: we will not be playing around with robustness so no need


    def decompress_gz(self, gz_path: Path) -> Path:
        h5_path = gz_path.with_suffix('')  # Remove the .gz suffix
        if not h5_path.exists():
            with gzip.open(gz_path, 'rb') as f_in:
                with open(h5_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return h5_path

    def load_shd_data(self, gz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h5_path = self.decompress_gz(gz_path)
        with h5py.File(h5_path, 'r') as f:
            spiketimes = f['spikes']['times'][:]
            neuron_ids = f['spikes']['units'][:]
            labels = f['labels'][:]

        num_samples = spiketimes.shape[0]
        max_len = 1
        # Initialize arrays with TIME_WINDOW (or another large number for timing purposes)
        for i in range(num_samples):
            unique, counts = np.unique(neuron_ids[i], return_counts=True)
            count_max = np.max(counts)
            max_len = max(max_len, count_max)

        fixed_spiketimes = np.full((num_samples, N_NEURONS, max_len), np.inf, dtype=float)
        fixed_neuron_ids = np.zeros((num_samples, N_NEURONS, max_len), dtype=int)

        # Vectorized operation to mark active neurons
        for i in range(num_samples):
            fixed_neuron_ids[i, neuron_ids[i]] = 1  # Mark active neurons for each sample

        # Vectorized operation to fill spike times
        for n in range(1, N_NEURONS):
            for i in range(num_samples):
                indxs = np.where(neuron_ids[i] == n)[0]
                if len(indxs) > 0:
                    fixed_spiketimes[i, n, :len(indxs)] = spiketimes[i][indxs]
            #print(n)

        # Reshape fixed_spiketimes to (num_samples, N_NEURONS, 1)
        # fixed_spiketimes = (fixed_spiketimes / MAX_VALUE) * TIME_WINDOW # scaling
        # fixed_spiketimes[fixed_spiketimes >= TIME_WINDOW] = np.inf
        # fixed_spiketimes = fixed_spiketimes.reshape(num_samples, N_NEURONS, 1)
        #print(fixed_spiketimes.shape)
        return fixed_spiketimes, fixed_neuron_ids, labels

    def __reduce_data(self, spike_times, n_spikes_per_neuron, labels, n):
        shuffled_indices = np.arange(len(labels))
        np.random.shuffle(shuffled_indices)
        shuffled_indices = shuffled_indices[:n]
        return spike_times[shuffled_indices], n_spikes_per_neuron[shuffled_indices], labels[shuffled_indices]

    def shuffle(self) -> None:
        shuffled_indices = np.arange(len(self.__train_labels))
        np.random.shuffle(shuffled_indices)
        self.__train_spike_times = self.__train_spike_times[shuffled_indices]
        self.__train_n_spikes_per_neuron = self.__train_n_spikes_per_neuron[shuffled_indices]
        self.__train_labels = self.__train_labels[shuffled_indices]

    def __get_batch(self, spike_times, n_spikes_per_neuron, labels, batch_index, batch_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = batch_index * batch_size
        end = start + batch_size
        return spike_times[start:end], n_spikes_per_neuron[start:end], labels[start:end]

    def get_train_batch(self, batch_index: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__train_spike_times, self.__train_n_spikes_per_neuron, self.__train_labels, batch_index, batch_size)

    def get_test_batch(self, batch_index: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__test_spike_times, self.__test_n_spikes_per_neuron, self.__test_labels, batch_index, batch_size)

    @property
    def train_labels(self) -> np.ndarray:
        return self.__train_labels

    @property
    def test_labels(self) -> np.ndarray:
        return self.__test_labels
