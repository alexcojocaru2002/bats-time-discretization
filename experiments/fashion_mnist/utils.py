from pathlib import Path
import cupy as cp
import numpy as np
import sys
import pandas as pd
from matplotlib import pyplot as plt
#
# def calculate_average_across_experiments(results):
#     average_results = {}
#
#     for dt_index, dt_value in enumerate(DT_list):
#         dt_frames = [experiment[dt_index] for experiment in results]
#
#         # Concatenate DataFrames along the rows
#         concatenated_df = pd.concat(dt_frames, axis=0)
#
#         # Calculate the mean for each group of epochs
#         average_df = concatenated_df.groupby('Epochs').mean().reset_index()
#
#         average_results[dt_value] = average_df
#
#     return average_results
#
# def plot_metrics_across_dt(average_results):
#     # Get the metrics from the DataFrame columns, excluding 'Epochs'
#     example_df = next(iter(average_results.values()))
#     metrics = [col for col in example_df.columns if col != 'Epochs']
#
#     for metric in metrics:
#         plt.figure(figsize=(10, 6))
#         for dt_value, avg_df in average_results.items():
#             plt.plot(avg_df['Epochs'], avg_df[metric], label=f'DT = {dt_value}')
#
#         plt.grid(True)
#         plt.xlabel('Epochs')
#         plt.ylabel(metric)
#         plt.title(f'{metric} Across Epochs for Different DT values')
#         plt.legend()
#         plt.show()
#
# # average_results = calculate_average_across_experiments(all_results)
# average_results_test = calculate_average_across_experiments(all_results_test)
# # plot_metrics_across_dt(average_results)
# plot_metrics_across_dt(average_results_test)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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



def plot_single_row_dt(csv_path, dataset_name='MNIST'):
    df = pd.read_csv(csv_path)

    # Drop the 'DTs' column header and get the only row of data
    row = df.iloc[0, 1:]  # skip first column (DTs)

    # Extract DT values from column headers like "DT=0.001"
    x = np.array([float(col.split('=')[1]) for col in row.index])
    y = row.values.astype(float)

    # Theoretical values: (dt / sqrt(3))
    theoretical = (x / np.sqrt(3))

    # Plot
    plt.plot(x, y, linestyle='-', label='Measured Input Layer MSE')
    plt.plot(x, theoretical, linestyle='--', label='Theoretical: (dt/√3)')

    plt.xlabel('Delta Time (DT)')
    plt.ylabel('Input Layer MSE')
    plt.title('MSE vs Delta Time (DT) for dataset ' + dataset_name)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_dt_list(central_dt: float, n: int, step: float) -> list:
    """
    Generate a list of `n` DT values evenly spaced around `central_dt`.

    Parameters:
    - central_dt: float, the central DT value
    - n: int, number of DT values (preferably odd for symmetry)
    - step: float, step size between DT values

    Returns:
    - List of DT values (floats)
    """
    if n % 2 == 0:
        raise ValueError("Please provide an odd number for n to ensure symmetry.")

    half = n // 2
    return [central_dt + i * step for i in range(-half, half + 1)]

def generate_dt_list_from_bounds(min_dt: float, max_dt: float, step: float) -> list:
    """
    Generate a list of DT values from min_dt to max_dt with the given step size.

    Parameters:
    - min_dt: float, minimum DT value (must be > 0)
    - max_dt: float, maximum DT value (must be > min_dt)
    - step: float, spacing between values (must be > 0)

    Returns:
    - List of DT values (floats), in ascending order
    """
    if min_dt < 0 or max_dt < 0:
        raise ValueError("Both min_dt and max_dt must be positive.")
    if max_dt <= min_dt:
        raise ValueError("max_dt must be greater than min_dt.")
    if step <= 0:
        raise ValueError("Step size must be positive.")

    num_steps = int((max_dt - min_dt) / step) + 1
    dt_list = [min_dt + i * step for i in range(num_steps) if min_dt + i * step <= max_dt]
    return dt_list


def plot_accuracy_over_epoch(file_list):
    """
    Plots accuracy over epochs for multiple CSV files.

    Each CSV file is expected to contain columns named "Epochs" and "Accuracy (%)".

    Parameters:
        file_list (list): A list of file paths to the CSV files.
    """
    plt.figure(figsize=(10, 6))

    # Iterate through each file
    for file in file_list:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Plot Accuracy over Epochs
        plt.plot(df["Epochs"], df["Accuracy (%)"], marker='o', label=file)

    # Label the axes and add a title
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs for Each DT File")
    plt.legend(title="File")
    plt.grid(True)
    plt.show()


# Example usage:
# files = ["output_metrics/experiment_0_ 619594659_1708786812/Test DT = 0.0001957", "output_metrics/experiment_0_ 619594659_1708786812/Test DT = 0.0001961"]  # Replace with your actual file paths
# plot_accuracy_over_epoch(files)
