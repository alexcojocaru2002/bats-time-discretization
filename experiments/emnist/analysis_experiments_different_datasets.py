import csv
import math
from pathlib import Path
import cupy as cp
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import sys

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
N_NEURONS_1 = 800
TAU_S_1 = 0.130
THRESHOLD_HAT_1 = 0.2
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 30

# Output_layer
N_OUTPUTS = 10
TAU_S_OUTPUT = 0.130
THRESHOLD_HAT_OUTPUT = 1.3
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 30

# Training parameters
N_TRAINING_EPOCHS = 10
N_TRAIN_SAMPLES = 60000
N_TEST_SAMPLES = 1
TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE = 1
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = 0.003
LR_DECAY_EPOCH = 10  # Perform decay very n epochs
LR_DECAY_FACTOR = 1.0
MIN_LEARNING_RATE = 0
TARGET_FALSE = 3
TARGET_TRUE = 15

DT = 0.001
DT_LIST = [0.001, 0.008]

def show_spike_differences(input_spike_times_continuous, input_spike_times_discrete, hidden_spike_times_continuous, hidden_spike_times_discrete, output_spikes_times, discrete_output_spike_times, label, DT, timestep, PLOT_DIR):
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 15))

    for neuron_index in range(input_spike_times_continuous.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax0.scatter(input_spike_times_continuous[neuron_index], [neuron_index] * input_spike_times_continuous.shape[1],
                    label=f'Neuron {neuron_index}', color='b')

    for neuron_index in range(input_spike_times_discrete.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax0.scatter(input_spike_times_discrete[neuron_index], [neuron_index] * input_spike_times_discrete.shape[1],
                    label=f'Neuron {neuron_index}', color='r')

    ax0.set_xlabel('Spike Times (s)')
    ax0.set_ylabel('Neuron Index')
    ax0.set_title('Scatter Plot of Input Layer Spike Times for Each Neuron')

    plt.subplots_adjust(hspace=0.4)

    for neuron_index in range(hidden_spike_times_continuous.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax1.scatter(hidden_spike_times_continuous[neuron_index], [neuron_index] * hidden_spike_times_continuous.shape[1],
                    label=f'Neuron {neuron_index}', color='b')

    for neuron_index in range(hidden_spike_times_discrete.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax1.scatter(hidden_spike_times_discrete[neuron_index], [neuron_index] * hidden_spike_times_discrete.shape[1],
                    label=f'Neuron {neuron_index}', color='r')

    ax1.set_xlabel('Spike Times (s)')
    ax1.set_ylabel('Neuron Index')
    ax1.set_title('Scatter Plot of Hidden Layer Spike Times for Each Neuron')

    for neuron_index in range(discrete_output_spike_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax2.scatter(discrete_output_spike_times[neuron_index], [neuron_index] * discrete_output_spike_times.shape[1],
                    label=f'Neuron {neuron_index}', color='b')

    for neuron_index in range(output_spikes_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax2.scatter(output_spikes_times[neuron_index], [neuron_index] * output_spikes_times.shape[1],
                    label=f'Neuron {neuron_index}', color='r')

    ax2.set_xlabel('Spike Times (s)')
    ax2.set_ylabel('Neuron Index')
    ax2.set_title('Scatter Plot of Output Layer Spike Times for Each Neuron')
    ax2.hlines(label, 0, 0.2, color='r', linestyles='dashed', label=f'True Label')
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle("Spike times for a forward pass of a digit " + str(label) + " using a DT of " + str(DT))

    plt.savefig( PLOT_DIR / ('spike_times_plot_' + str(timestep) + ".png"))
    plt.show()

# Plot parameters
EXPORT_METRICS = True
# EXPORT_DIR = Path("./output_metrics/" + "experiment_" + str(i) + "_ " + str(str(np_seed) + "_" + str(cp_seed)))
PLOT_DIR = Path('./scatter_plots/')
EXPORT_DIR = Path("./output_metrics/")

SAVE_DIR = Path("./best_model")

def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


def discrete(spikes: cp.ndarray, DT: float):
    # Create a mask for finite values
    finite_mask = (spikes != cp.inf)

    # Initialize the output array with infinities
    discrete_spikes = cp.full(spikes.shape, cp.inf, dtype=spikes.dtype)

    # Compute the discrete spikes for finite values using vectorized operations
    finite_spikes = spikes[finite_mask]
    discrete_spikes_finite = finite_spikes + (DT - np.fmod(finite_spikes, DT))

    # Assign the computed values to the appropriate locations
    discrete_spikes[finite_mask] = discrete_spikes_finite
    return discrete_spikes

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


def create_network(network_config):
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)
    previous_layer = input_layer
    for i, neurons in enumerate(network_config):
        curr_layer = LIFLayer(previous_layer=previous_layer, n_neurons=neurons, tau_s=TAU_S_1,
                            theta=THRESHOLD_HAT_1,
                            delta_theta=DELTA_THRESHOLD_1,
                            time_delta=DT,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_1,
                            name="Hidden layer " + str(i))
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



def test(network, dataset, loss_fct, epoch_metrics, optimizer, test_time_monitor, test_accuracy_monitor, test_loss_monitor, test_silent_monitors, test_monitors_manager, test_spike_counts_monitors, test_norm_monitors, test_learning_rate_monitor):
    test_time_monitor.start()
    for batch_idx in range(N_TEST_BATCH):
        spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
        if DT != 0.0:
            discrete_spikes = discrete(spikes, DT)
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

        for l, mon in test_silent_monitors.items():
            mon.add(l.spike_trains[1])
    # Here we want to test how the network performs on the same dataset throughout training
    # Testing hypothesis if spikes get pushed earlier and discretization does not affect so much later epochs
    # spikes, n_spikes, labels = dataset.get_test_batch(0,TEST_BATCH_SIZE)  # Using input 0
    # if DT != 0.0:
    #     discrete_spikes = discrete(spikes, DT)
    # else:
    #     discrete_spikes = spikes
    # network.reset()
    # network.forward(spikes, n_spikes, discrete_spikes, max_simulation=SIMULATION_TIME)
    # out_spikes, n_out_spikes, discrete_out_spikes = network.output_spike_trains

    for l, mon in test_norm_monitors.items():
        mon.add(l.weights)
    test_learning_rate_monitor.add(optimizer.learning_rate)

    records = test_monitors_manager.record(epoch_metrics)
    test_monitors_manager.print(epoch_metrics)
    # test_monitors_manager.export()

    acc = records[test_accuracy_monitor]
    return acc

def mse_loss(spike_train, discrete_spike_train):
    mse = 0.0
    total_count = 0
    for neuron_index in range(spike_train.shape[0]):
        i = 0
        while i < spike_train.shape[1] and spike_train[neuron_index][i] != np.inf:
            mse = mse + (discrete_spike_train[neuron_index][i] - spike_train[neuron_index][i]) ** 2
            i = i + 1
            total_count += 1

    if total_count > 0:
        # print("MSE IS " + str(mse) + "and total count is " + str(total_count))
        return mse / total_count
    else:
        return 0

NR_EXPERIMENTS = 10


print("Loading datasets...")

# dataset_test_hypothesis = Dataset(path=DATASET_PATH)
loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

network_configs=[[800, 800], [800, 800, 800, 800, 800, 800, 800]] # Problem, make sure the biggest network is last!
dataset = Dataset_Mnist(path=DATASET_PATH_EMNIST)
# dataset_emnist = Dataset_Emnist(path=DATASET_PATH_EMNIST)
# dataset_fmnist = Dataset_Fmnist(path=DATASET_PATH_FMNIST)

dataset_name = 'ENIST'

network = create_network(network_configs[-1]) # Because of the DF initialization

mse = pd.DataFrame()
mse['Layers'] = np.zeros((len(network_configs), ))
mse['Spike Count'] = np.empty((len(network_configs), ))
mse['Dataset'] = np.empty((len(network_configs), ))

# Add columns to track both average and standard deviation
for layer in network.layers:
    mse[layer.name] = np.zeros((len(network_configs), ))  # For average MSE loss
    mse[layer.name + ' StdDev'] = np.zeros((len(network_configs), ))  # For standard deviation

for layer in network.layers:
    mse['Spike Count ' + layer.name] = np.zeros((len(network_configs), ))

# Temporary storage for per-experiment losses to calculate standard deviation
layer_losses = {layer.name: [[] for _ in range(len(network_configs))] for layer in network.layers}

for i, config in enumerate(network_configs):
        mse['Layers'][i] = len(config)
        mse['Dataset'][i] = dataset_name
        print("Creating network for network with " + str(len(config)) + " layers")

        for experiment in range(NR_EXPERIMENTS):
            print(f"Experiment {experiment + 1}/{NR_EXPERIMENTS} for network with {len(config)} layers")
            network = create_network(config)
            (train_monitors_manager, train_accuracy_monitor, train_loss_monitor,
             train_time_monitor, train_silent_label_monitor, test_monitors_manager,
             test_accuracy_monitor, test_loss_monitor, test_silent_monitors, test_time_monitor,
             test_norm_monitors, test_learning_rate_monitor, test_spike_counts_monitors) = add_monitors(
                network)

            test(network, dataset, loss_fct, 1.0, optimizer, test_time_monitor, test_accuracy_monitor,
                 test_loss_monitor, test_silent_monitors, test_monitors_manager,
                 test_spike_counts_monitors, test_norm_monitors, test_learning_rate_monitor)

            spikes, n_spikes, labels = dataset.get_test_batch(0, TEST_BATCH_SIZE)
            out_spikes, n_out_spikes, discrete_out_spikes = network.output_spike_trains


            for layer in network.layers:
                loss = mse_loss(layer.spike_trains[0].get()[0], layer.spike_trains[2].get()[0])
                mse[layer.name][i] += loss  # Accumulate the loss directly in the DataFrame
                layer_losses[layer.name][i].append(loss)  # Store the loss for variance calculation
                mse['Spike Count'][i] += len(
                    layer.spike_trains[0].get()[0][(layer.spike_trains[0].get()[0] != np.inf)]
                )
                mse['Spike Count ' + layer.name][i] += len(
                    layer.spike_trains[0].get()[0][(layer.spike_trains[0].get()[0] != np.inf)]
                )

        # Average the results after all experiments
        mse['Spike Count'][i] /= NR_EXPERIMENTS
        for layer in network.layers:
            mse[layer.name][i] /= NR_EXPERIMENTS  # Average the loss for each layer

            # Compute standard deviation
            mse[layer.name + ' StdDev'][i] = np.std(layer_losses[layer.name][i], ddof=1)  # ddof=1 for sample stddev

mse.to_csv("mnist_experiments", index=False)

def plot_mse_stdev(mse_df):
    # Iterate over each layer configuration (row) in the DataFrame
    for i, row in mse_df.iterrows():
        layer_configuration = int(row["Layers"])

        # Extract MSE and StdDev columns for this configuration, ignoring Output Layer
        mse_values = {}
        stddev_values = {}

        for column in mse_df.columns:
            if "StdDev" in column and "Output layer" not in column:  # Ignore Output Layer
                layer_name = column.replace(" StdDev", "")
                if layer_name in mse_df.columns:
                    mse_values[layer_name] = row[layer_name]
                    stddev_values[layer_name] = row[column]

        # Prepare plot data
        layers = list(mse_values.keys())
        mse = list(mse_values.values())
        stddev = list(stddev_values.values())

        # Create a plot for the current configuration
        plt.figure(figsize=(10, 6))
        plt.plot(layers, mse, marker="o", label="MSE", linestyle="-")
        plt.fill_between(layers,
                         [m - s for m, s in zip(mse, stddev)],
                         [m + s for m, s in zip(mse, stddev)],
                         color="blue", alpha=0.2, label="MSE ± StdDev")

        # Customize the plot
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel("MSE", fontsize=12)
        plt.title(f"Layer Configuration {layer_configuration}: MSE and StdDev (for EMNIST)", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.legend()

        # Show the plot for the current layer configuration
        plt.show()

plot_mse_stdev(mse)