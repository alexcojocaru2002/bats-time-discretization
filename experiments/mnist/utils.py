import csv
import math
from pathlib import Path
import cupy as cp
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import sys
import utils

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


def mse_loss(spike_train, discrete_spike_train):
    mse = 0.0
    total_count = 0
    print(spike_train.shape)

    for batch_idx in range(spike_train.shape[0]):
        for neuron_index in range(spike_train.shape[1]):
            i = 0
            while i < spike_train.shape[2] and spike_train[batch_idx][neuron_index][i] != np.inf:
                mse = mse + np.sqrt(((discrete_spike_train[batch_idx][neuron_index][i] - spike_train[batch_idx][neuron_index][i]) ** 2))
                i = i + 1
                total_count += 1
    if total_count > 0:
        # print("MSE IS " + str(mse / total_count) + "and total count is " + str(total_count))
        return mse / total_count
    else:
        return 0


def plot_mse_stdev(mse_df, DT):
    # Iterate over each layer configuration (row) in the DataFrame
    for i, row in mse_df.iterrows():
        layer_configuration = int(row["Layers"])

        # Extract MSE and StdDev columns for this configuration, ignoring Output Layer
        mse_values = {}
        stddev_values = {}

        for column in mse_df.columns:
            if "StdDev" in column and "Output layer" not in column :   # Ignore Output Layer
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
        theoretical_value = ( DT / np.sqrt(3)) ** 2

        plt.hlines(color='red', xmin = 0, xmax = len(mse_values.keys()), y=theoretical_value, label="Theoretical Value")
        plt.hlines(color='green', xmin = 0, xmax = len(mse_values.keys()), y=np.mean(mse), label="Current Average")
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel("MSE", fontsize=12)
        plt.title(f"Layer Configuration {layer_configuration}: MSE and StdDev (For MNIST) for DT = {DT}", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.legend()

        # Show the plot for the current layer configuration
        plt.show()



def plot_single_row_dt(csv_path):
    df = pd.read_csv(csv_path)

    # Drop the 'DTs' column header and get the only row of data
    row = df.iloc[0, 1:]  # skip first column (DTs)

    # Extract DT values from column headers like "DT=0.001"
    x = np.array([float(col.split('=')[1]) for col in row.index])
    y = row.values.astype(float)

    # Theoretical values: (dt / sqrt(3))
    theoretical = (x / np.sqrt(3))

    # Plot
    plt.plot(x, y, marker='o', linestyle='-', label='Measured Input Layer MSE')
    plt.plot(x, theoretical, marker='x', linestyle='--', label='Theoretical: (dt/√3)')

    plt.xlabel('Delta Time (DT)')
    plt.ylabel('Input Layer MSE')
    plt.title('MSE vs Delta Time (DT)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()