import os
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Path to folder containing CSV files
folder_path = "output_metrics/new_experiments/experiment_0_ 332571382_660420983"

# Data structure: {dt_value: [DataFrames]}
dt_to_dfs = {}

# Step 1: Read and group CSVs by DT value
# Regular expression to extract DT value from filename
dt_pattern = re.compile(r'DT\s*=\s*(\d*\.?\d+(?:[eE][-+]?\d+)?)')

for filename in sorted(os.listdir(folder_path)):
    if "Test DT =" not in filename:
        continue

    filepath = os.path.join(folder_path, filename)

    match = dt_pattern.search(filename)
    if not match:
        print(f"Skipping: {filename} (no DT match)")
        continue

    try:
        dt = float(match.group(1))
    except ValueError:
        print(f"Invalid DT in filename: {filename}")
        continue

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        continue

    if dt not in dt_to_dfs:
        dt_to_dfs[dt] = []
    dt_to_dfs[dt].append(df)

# Step 2: Average across experiments per DT
def calculate_average_across_experiments(grouped_results):
    average_results = {}

    for dt_value, df_list in grouped_results.items():
        concatenated_df = pd.concat(df_list, axis=0)
        averaged_df = concatenated_df.groupby('Epochs').mean().reset_index()
        average_results[dt_value] = averaged_df

    return average_results


# Step 3: Plot metrics over epochs for each DT
def plot_metrics_across_dt(average_results, selected_metrics=None, start_epoch=1):
    """
    Plot selected metrics over epochs for each DT value.

    Parameters:
    - average_results: dict of {DT: DataFrame}
    - selected_metrics: list of column names to plot
    - start_epoch: int, epoch number from which to start plotting (inclusive)
    """
    example_df = next(iter(average_results.values()))
    metrics = [col for col in example_df.columns if col != 'Epochs']

    if selected_metrics:
        metrics = [m for m in metrics if m in selected_metrics]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for dt_value, avg_df in sorted(average_results.items()):
            # Filter from start_epoch onward
            filtered_df = avg_df[avg_df['Epochs'] >= start_epoch]
            print(filtered_df)
            plt.plot(filtered_df['Epochs'], filtered_df[metric], label=f'DT = {dt_value:.4g}')
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.title(f'{metric} Across Epochs for Different DT values (from epoch {start_epoch})')
        plt.legend()
        plt.tight_layout()
        plt.show()


def load_csv_metrics(base_folder, phase="Test"):
    """
    Loads CSVs like 'Test DT = 0.001' and groups them by DT value.

    Returns:
        {
            dt_str: list of DataFrames (one per experiment)
        }
    """
    dt_data = defaultdict(list)
    for exp_folder in os.listdir(base_folder):
        exp_path = os.path.join(base_folder, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        for file in os.listdir(exp_path):
            if file.startswith(f"{phase} DT ="):
                dt_str = file.split('=')[1].strip()
                file_path = os.path.join(exp_path, file)
                try:
                    df = pd.read_csv(file_path)
                    dt_data[dt_str].append(df)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
    return dt_data


def plot_metric_mean_std(base_folder, metric="Loss", phase="Test", start_epoch=0):
    """
    Plots the average ± std for a selected metric over epochs, grouped by DT.
    """
    dt_data = load_csv_metrics(base_folder, phase)

    plt.figure(figsize=(10, 6))
    for dt_str in sorted(dt_data.keys(), key=lambda x: float(x)):
        dfs = dt_data[dt_str]
        dfs_trimmed = []

        for df in dfs:
            # Filter from start_epoch onwards
            df = df[df['Epochs'] >= start_epoch]
            df = df[['Epochs', metric]].copy()
            dfs_trimmed.append(df.set_index('Epochs'))

        # Align all dataframes by Epoch index
        aligned = pd.concat(dfs_trimmed, axis=1)
        mean_series = aligned.mean(axis=1)
        std_series = aligned.std(axis=1)

        epochs = mean_series.index.to_numpy()
        mean = mean_series.to_numpy()
        std = std_series.to_numpy()

        plt.plot(epochs, mean, label=f"DT = {dt_str}")
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title(f"{metric} over Epochs ({phase}, mean ± std)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Step 4: Execute
# average_results = calculate_average_across_experiments(dt_to_dfs)
#
# plot_metrics_across_dt(
#     average_results,
#     selected_metrics=['Accuracy (%)', 'Hidden layer 1 spike counts'],
#     start_epoch=5
# )


base_folder = "output_metrics"

plot_metric_mean_std("output_metrics", metric="Accuracy (%)", phase="Test", start_epoch=5)
plot_metric_mean_std("output_metrics", metric="Hidden layer 1 spike counts", phase="Test", start_epoch=5)
plot_metric_mean_std("output_metrics", metric="Loss", phase="Train", start_epoch=5)
plot_metric_mean_std("output_metrics", metric="Accuracy (%)", phase="Train", start_epoch=5)

