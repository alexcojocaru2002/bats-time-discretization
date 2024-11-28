from pathlib import Path
from typing import Tuple
from elasticdeform import deform_random_grid
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

warnings.filterwarnings("ignore")

from tensorflow import keras
import numpy as np

TIME_WINDOW = 100e-3
MAX_VALUE = 255
RESOLUTION = 28
N_NEURONS = RESOLUTION * RESOLUTION
ELASTIC_ALPHA_RANGE = [8, 10]
ELASTIC_SIGMA = 3
WIDTH_SHIFT = 0
HEIGHT_SHIFT = 0
ZOOM_RANGE = 12 / 100
ROTATION_RANGE = 12


def elastic_transform(image, alpha_range, sigma):
    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=0, mode='reflect').reshape(shape)


class Dataset:
    def __init__(self, path: Path):
        loaded_data = np.load(path, allow_pickle=True)
        self.__train_X = loaded_data['x_train']
        self.__train_labels = loaded_data['y_train']
        self.__test_X = loaded_data['x_test']
        self.__test_labels = loaded_data['y_test']
        self.__datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=WIDTH_SHIFT,
                                                                      height_shift_range=HEIGHT_SHIFT,
                                                                      zoom_range=ZOOM_RANGE,
                                                                      rotation_range=ROTATION_RANGE,
                                                                      fill_mode="nearest",
                                                                      preprocessing_function=lambda x:
                                                                      elastic_transform(x,
                                                                                        alpha_range=ELASTIC_ALPHA_RANGE,
                                                                                        sigma=ELASTIC_SIGMA))

    @property
    def train_labels(self) -> np.ndarray:
        return self.__train_labels

    @property
    def test_labels(self) -> np.ndarray:
        return self.__test_labels

    def __to_sequential_spikes(self, samples):
        """
        Convert samples into a sequential spike format.
        Each pixel is assigned to a unique time step in the sequence.
        """
        batch_size = samples.shape[0]
        sequential_spikes = []

        # Reshape the image (28x28) into a sequential format (784 pixels in sequence)
        samples = samples.reshape((batch_size, N_NEURONS))

        # Create spike times for each pixel, assigning each pixel to a unique time step.
        for t in range(N_NEURONS):
            # Each time step t corresponds to pixel index t
            pixel_values = samples[:, t]  # pixel t for all samples in the batch
            spike_time = TIME_WINDOW * (1 - (pixel_values / MAX_VALUE))  # scale pixel values to spike times
            spike_time[spike_time == TIME_WINDOW] = np.inf  # Inf means no spike
            sequential_spikes.append(spike_time.reshape(batch_size, 1))

        # Stack spikes over time: output shape will be (batch_size, N_NEURONS, 1) where each time step corresponds to a pixel
        sequential_spikes = np.concatenate(sequential_spikes, axis=1)  # (batch_size, N_NEURONS)

        # Number of spikes per neuron (1 spike for each neuron, either spike or no spike)
        n_spikes_per_neuron = np.isfinite(sequential_spikes).astype('int')

        # print(sequential_spikes)

        return sequential_spikes, n_spikes_per_neuron

    def shuffle(self) -> None:
        shuffled_indices = np.arange(len(self.__train_labels))
        np.random.shuffle(shuffled_indices)
        self.__train_X = self.__train_X[shuffled_indices]
        self.__train_labels = self.__train_labels[shuffled_indices]

    def __get_batch(self, samples, labels, batch_index, batch_size, augment) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = batch_index * batch_size
        end = start + batch_size

        samples = samples[start:end]
        labels = labels[start:end]

        if augment:
            samples = deform_random_grid(list(samples), sigma=1.0, points=2, order=0)
            samples = np.array(samples)

        sequential_spikes, n_spikes_per_neuron = self.__to_sequential_spikes(samples)
        return sequential_spikes, n_spikes_per_neuron, labels

    def get_train_batch(self, batch_index: int, batch_size: int, augment: bool = False) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__train_X, self.__train_labels,
                                batch_index, batch_size, augment)

    def get_test_batch(self, batch_index: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__test_X, self.__test_labels,
                                batch_index, batch_size, False)

    def get_test_image_at_index(self, index):
        return self.__test_X[index]

