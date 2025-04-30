# from pathlib import Path
# import cupy as cp
# import numpy as np
# import sys
# import pandas as pd
# from matplotlib import pyplot as plt
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


import pandas as pd
import matplotlib.pyplot as plt


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
files = ["output_metrics/experiment_0_ 619594659_1708786812/Test DT = 0.0001957", "output_metrics/experiment_0_ 619594659_1708786812/Test DT = 0.0001961"]  # Replace with your actual file paths
plot_accuracy_over_epoch(files)
