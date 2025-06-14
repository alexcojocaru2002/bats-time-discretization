import csv
import math
import os
from pathlib import Path
import cupy as cp
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import sys

from matplotlib.lines import Line2D
from tqdm import tqdm

import utils

sys.path.insert(0, "../../")  # Add repository root to python path

from bats.Monitors import *
from bats.Layers import InputLayer, LIFLayer
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *

from Dataset import Dataset as Dataset_Mnist

# Dataset
DATASET_PATH_FMNIST = Path("../../datasets/")
DATASET_PATH_EMNIST = Path("../../datasets/emnist-balanced.mat")
DATASET_PATH_MNIST = Path("../../datasets/mnist.npz")

N_INPUTS = 28 * 28
SIMULATION_TIME = 0.2

# Hidden layer
N_NEURONS_1 = 400
TAU_S_1 = 0.130
THRESHOLD_HAT_1 = 0.13
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 5

# Hidden layer
N_NEURONS_2 = 400
TAU_S_2 = 0.130
THRESHOLD_HAT_2 = 0.45
DELTA_THRESHOLD_2 = 1 * THRESHOLD_HAT_2
SPIKE_BUFFER_SIZE_2 = 5

# Output_layer
N_OUTPUTS = 10
TAU_S_OUTPUT = 0.130
THRESHOLD_HAT_OUTPUT = 0.7
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 20

# Training parameters
N_TRAINING_EPOCHS = 30
N_TRAIN_SAMPLES = 60000
N_TEST_SAMPLES = 10000
TRAIN_BATCH_SIZE = 5
TEST_BATCH_SIZE = 100
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = 0.0005
LR_DECAY_EPOCH = 10  # Perform decay very n epochs
LR_DECAY_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-4
TARGET_FALSE = 3
TARGET_TRUE = 15

DT = 0.0003922

DT_LIST = utils.generate_dt_list_from_bounds(min_dt=0.0, max_dt=0.01, step=0.00001)

# Plot parameters
EXPORT_METRICS = True
# EXPORT_DIR = Path("./output_metrics/" + "experiment_" + str(i) + "_ " + str(str(np_seed) + "_" + str(cp_seed)))

EXPORT_DIR = Path("./output_metrics/continuous_vs_discrete")

SAVE_DIR = Path("./best_model")

def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)



def add_monitors(network):
    # Metrics
    train_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_train")
    train_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_train")
    train_silent_label_monitor = SilentLabelsMonitor()
    train_time_monitor = TimeMonitor()
    # train_prediction_confidence_monitor = ValueMonitor(name='Average Prediction Confidence', decimal=5)
    train_monitors_manager = MonitorsManager([train_loss_monitor,
                                              train_accuracy_monitor,
                                              train_silent_label_monitor,
                                              train_time_monitor],
                                             print_prefix="Train | ")

    test_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_test")
    test_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_test")
    test_learning_rate_monitor = ValueMonitor(name="Learning rate", decimal=5)
    # Only monitor LIF layers
    test_spike_counts_monitors = {
        l: SpikeCountMonitor(l.name, export_path=EXPORT_DIR / ("spike_count_" + str(l.name))) for l in
        network.layers if isinstance(l, LIFLayer)}
    test_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_norm_monitors = {l: WeightsNormMonitor(l.name, export_path=EXPORT_DIR / ("weight_norm_" + l.name))
                          for l in network.layers if isinstance(l, LIFLayer)}
    test_time_monitor = TimeMonitor()
    all_test_monitors = [test_loss_monitor, test_accuracy_monitor, test_learning_rate_monitor]
    all_test_monitors.extend(test_spike_counts_monitors.values())
    all_test_monitors.extend(test_silent_monitors.values())
    all_test_monitors.extend(test_norm_monitors.values())
    all_test_monitors.append(test_time_monitor)
    test_monitors_manager = MonitorsManager(all_test_monitors,
                                            print_prefix="Test | ")
    return (train_monitors_manager, train_accuracy_monitor, train_loss_monitor, train_time_monitor, train_silent_label_monitor,
            test_monitors_manager, test_accuracy_monitor, test_loss_monitor, test_silent_monitors, test_time_monitor, test_norm_monitors, test_learning_rate_monitor, test_spike_counts_monitors)


def create_network():
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)
    previous_layer = input_layer
    print(DT)
    curr_layer = LIFLayer(previous_layer=previous_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                        theta=THRESHOLD_HAT_1,
                        delta_theta=DELTA_THRESHOLD_1,
                        time_delta=DT,
                        weight_initializer=weight_initializer,
                        max_n_spike=SPIKE_BUFFER_SIZE_1,
                        name="Hidden layer " + '1')
    network.add_layer(curr_layer)
    previous_layer = curr_layer
    curr_layer = LIFLayer(previous_layer=previous_layer, n_neurons=N_NEURONS_2, tau_s=TAU_S_2,
                        theta=THRESHOLD_HAT_2,
                        delta_theta=DELTA_THRESHOLD_2,
                        time_delta=DT,
                        weight_initializer=weight_initializer,
                        max_n_spike=SPIKE_BUFFER_SIZE_2,
                        name="Hidden layer " + '2')
    network.add_layer(curr_layer)
    previous_layer = curr_layer
    output_layer = LIFLayer(previous_layer=previous_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            time_delta=DT,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)
    return network



def test(network, loss_fct, dataset):
    dataset.shuffle()

    test_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_test")
    test_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_test")
    # Only monitor LIF layers
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name, export_path=EXPORT_DIR / ("spike_count_" + str(l.name)))
                                  for l in network.layers if isinstance(l, LIFLayer)}
    all_test_monitors = [test_loss_monitor, test_accuracy_monitor]
    all_test_monitors.extend(test_spike_counts_monitors.values())
    test_monitors_manager = MonitorsManager(all_test_monitors,
                                            print_prefix="Test | ")
    for batch_idx in range(N_TEST_BATCH):
        spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
        if DT != 0.0:
            discrete_spikes = utils.discrete(spikes, DT)
        else:
            discrete_spikes = spikes
        network.reset()
        network.forward(spikes, n_spikes, discrete_spikes, max_simulation=SIMULATION_TIME)
        out_spikes, n_out_spikes, discrete_out_spikes = network.output_spike_trains

        pred = loss_fct.predict(discrete_out_spikes, n_out_spikes)
        loss = loss_fct.compute_loss(discrete_out_spikes, n_out_spikes, labels)

        pred_cpu = pred.get()
        loss_cpu = loss.get()
        test_loss_monitor.add(loss_cpu)
        test_accuracy_monitor.add(pred_cpu, labels)
        for l, mon in test_spike_counts_monitors.items():
            mon.add(l.spike_trains[1])
    test_monitors_manager.record(1)
    # test_monitors_manager.export()
    return test_monitors_manager

NR_EXPERIMENTS = 1

def plot(dt_list, base_folder='output_metrics/continuous_vs_discrete/'):
    fig = plt.figure(figsize=(10, 6))

    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(min(dt_list), max(dt_list))

    dt_handles = []

    for dt in dt_list:
        color = cmap(norm(dt))
        dt_handles.append(Line2D([0], [0], marker='o', color='w', label=f'Δt = {dt}',
                                  markerfacecolor=color, markersize=10))

        files = {
            "Discrete-Train / Discrete-Test": EXPORT_DIR / f"test_discrete_train_discrete_{dt}.csv",
            "Continuous-Train / Discrete-Test": EXPORT_DIR / f"test_continuous_train_discrete_{dt}.csv",
        }

        for label, file_path in files.items():
            df = pd.read_csv(file_path)

            spike_counts = df["Hidden layer 1 spike counts"]
            accuracy = df["Accuracy (%)"]

            marker = 'x' if "Discrete-Train" in label else 'o'

            plt.scatter(spike_counts, accuracy, color=color, marker=marker, s=200)

    plt.xlabel("Hidden layer 1 spike counts", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    # plt.title("EMNIST Spike Count vs Accuracy across Δt", fontsize=16)
    plt.grid(True)

    # First legend (Δt values)
    first_legend = plt.legend(handles=dt_handles, fontsize=15,
                               bbox_to_anchor=(1.23, 0.9), loc='upper left', borderaxespad=0.)
    plt.gca().add_artist(first_legend)

    # Second legend (training regimes)
    marker_handles = [
        Line2D([0], [0], marker='x', color='k', label='Discrete Train / Discrete Test', linestyle='None', markersize=10),
        Line2D([0], [0], marker='o', color='k', label='Continuous Train / Discrete Test', linestyle='None', markersize=10)
    ]

    plt.legend(handles=marker_handles, fontsize=15,
               bbox_to_anchor=(1.02, 0.2), loc='center left', borderaxespad=0.)

    # Instead of tight_layout, we control axis area manually
    plt.subplots_adjust(left=0.1, right=0.575, top=0.9, bottom=0.1)
    save_path = os.path.join(base_folder, 'continous_vs_discrete_fmnist.pdf')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show()


def run_continuous_vs_discrete(dt_list):

    if not EXPORT_DIR.exists():
        EXPORT_DIR.mkdir()

    print("Loading datasets...")
    global DT

    # dataset_test_hypothesis = Dataset(path=DATASET_PATH)
    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)

    dataset = Dataset_Mnist(path=DATASET_PATH_FMNIST)

    for dt in dt_list:
        print(f"Running experiments for Δt = {dt}")

        # Discrete-Train / Continuous-Test
        DT = dt
        network_discrete_continuous = create_network()
        network_discrete_continuous.restore(SAVE_DIR / ("DT = " + str(0.0)))

        test_monitor_discrete_continuous = test(network_discrete_continuous, loss_fct, dataset)
        df_discrete_discrete_continuous = test_monitor_discrete_continuous.return_vals()
        df_discrete_discrete_continuous.to_csv(EXPORT_DIR / ("test_discrete_train_continuous_" + str(dt) + ".csv"), index=False)

        # Continuous-Train / Continuous-Test
        DT = 0.0
        network_continuous_continuous = create_network()
        network_continuous_continuous.restore(SAVE_DIR / ("DT = " + str(0.0)))

        test_monitor_continuous_continuous = test(network_continuous_continuous, loss_fct, dataset)
        df_continuous_continuous = test_monitor_continuous_continuous.return_vals()
        df_continuous_continuous.to_csv(EXPORT_DIR / ("test_continuous_train_continuous_" + str(dt) + ".csv"), index=False)

        # Discrete-Train / Discrete-Test
        DT = dt
        network_discrete_discrete = create_network()
        network_discrete_discrete.restore(SAVE_DIR / ("DT = " + str(dt)))

        test_monitor_discrete_discrete = test(network_discrete_discrete, loss_fct, dataset)
        df_discrete_discrete = test_monitor_discrete_discrete.return_vals()
        df_discrete_discrete.to_csv(EXPORT_DIR / ("test_discrete_train_discrete_" + str(dt) + ".csv"), index=False)

        # Continuous-Train / Discrete-Test
        DT = 0.0
        network_continuous_discrete = create_network()
        network_continuous_discrete.restore(SAVE_DIR / ("DT = " + str(dt)))
        dataset.shuffle()

        test_monitor_continuous_discrete = test(network_continuous_discrete, loss_fct, dataset)
        df_continuous_discrete = test_monitor_continuous_discrete.return_vals()
        df_continuous_discrete.to_csv(EXPORT_DIR / ("test_continuous_train_discrete_" + str(dt) + ".csv"), index=False)

    print("All experiments completed.")

# run_continuous_vs_discrete(0.0001)
# run_continuous_vs_discrete(0.000768)
DT_LIST = [0.00087, 0.00205, 0.00321, 0.0039, 0.0053, 0.006]
# run_continuous_vs_discrete(DT_LIST)
plot(DT_LIST)