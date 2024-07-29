import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load the CSV files and extract the necessary data
def load_data(experiments_path, dt_values):
    accuracy_data_train = {dt: [] for dt in dt_values}
    prediction_confidence_test = {dt: [] for dt in dt_values}
    accuracy_data_test = {dt: [] for dt in dt_values}
    hidden_spike_data = {dt: [] for dt in dt_values}
    output_spike_data = {dt: [] for dt in dt_values}
    time_data_train = {dt: [] for dt in dt_values}

    experiment_folders = [d for d in os.listdir(experiments_path) if d.startswith('experiment_')]
    for exp_folder in experiment_folders:
        full_exp_path = os.path.join(experiments_path, exp_folder)
        for dt in dt_values:
            file_path_test = os.path.join(full_exp_path, f'Test DT = {dt}')
            file_path_train = os.path.join(full_exp_path, f'Train DT = {dt}')
            file_path_prediction = os.path.join(full_exp_path, f'Prediction confidence for DT = {dt}')

            df0 = pd.read_csv(file_path_train)
            accuracy_data_train[dt].append(df0['Accuracy (%)'])
            # Accumulate time across epochs
            cumulative_time = df0['Time (s)']
            time_data_train[dt].append(cumulative_time)
            df = pd.read_csv(file_path_test)
            accuracy_data_test[dt].append(df['Accuracy (%)'])
            hidden_spike_data[dt].append(df['Hidden layer 1 spike counts'])
            output_spike_data[dt].append(df['Output layer spike counts'])
            dfpred = pd.read_csv(file_path_prediction)
            prediction_confidence_test[dt].append(dfpred)
            #TO DO add to prediction_confidence_test what i need to plot

    return prediction_confidence_test, accuracy_data_train, accuracy_data_test, hidden_spike_data, output_spike_data, time_data_train

# Function to compute the average and standard deviation
def compute_statistics(data):
    stats = {}
    for dt, values in data.items():
        values_matrix = np.array(values)
        avg = np.mean(values_matrix, axis=0)
        std = np.std(values_matrix, axis=0)
        stats[dt] = {'average': avg, 'std': std}
    return stats

# Function to compute the average and standard deviation
def compute_statistics2(data):
    stats = {}
    for dt, values in data.items():
        values_matrix = np.array(values[-1])
        avg = np.mean(values_matrix, axis=0)
        std = np.std(values_matrix, axis=0)
        stats[dt] = {'average': avg, 'std': std}
    return stats

# Function to compute cumulative time
def compute_cumulative_time(time_data):
    cumulative_time_data = {}
    for dt, times in time_data.items():
        accumulated_times = []
        for run_times in times:
            cumulative_times = run_times.copy()
            for i in range(10, len(run_times), 10):
                cumulative_times[i:i+10] += cumulative_times[i-1]
            accumulated_times.append(cumulative_times)
        cumulative_time_data[dt] = np.mean(accumulated_times, axis=0)
        #print(dt)
        #print(cumulative_time_data[dt])
    return cumulative_time_data


# Function to plot the results with epochs as x-axis
def plot_statistics(stats, ylabel, title):
    plt.figure(figsize=(12, 8))
    for dt, stat in stats.items():
        epochs = range(1, len(stat['average']) + 1)
        avg = stat['average']
        std = stat['std']
        plt.plot(epochs, avg, marker='o', label=f'DT = {dt}')
        plt.fill_between(epochs, avg - std, avg + std, alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_statistics_prediction(stats, dt_value, ylabel, title):
    plt.figure(figsize=(12, 8))
    for dt, stat in stats.items():
        epochs = range(1, len(stat['average']) + 1)
        avg = stat['average']
        std = stat['std']
        plt.plot(dt_value, avg, marker='o', label=f'DT = {dt}')
        plt.fill_between(dt_value, avg - std, avg + std, alpha=0.2)

    plt.axvline(0.1, color='r', linestyle='--' )
    plt.text(x=0.1, y=0, s='End of Input Spikes', verticalalignment='bottom', color='r')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot the results with cumulative time as x-axis
def plot_statistics_with_time(stats, time_data, ylabel, title):
    plt.figure(figsize=(12, 8))
    for dt, stat in stats.items():
        times = time_data[dt]
        avg = stat['average']
        std = stat['std']
        plt.plot(times, avg, marker='o', label=f'DT = {dt}')
        plt.fill_between(times, avg - std, avg + std, alpha=0.2)

    plt.xlabel('Cumulative Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
experiments_path = 'experiments/mnist/output_metrics/'  # Change this to your actual path
dt_values = ['0.0005', '0.001', '0.002', '0.003', '0.004', '0.005', '0.006', '0.007']
latency_values = [0.02, 0.03, 0.05, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2]
prediction_confidence, accuracy_data_train, accuracy_data_test, hidden_spike_data, output_spike_data, time_data_train = load_data(experiments_path, dt_values)

prediction_confidence_stats = compute_statistics2(prediction_confidence)
accuracy_stats_train = compute_statistics(accuracy_data_train)
accuracy_stats_test = compute_statistics(accuracy_data_test)
hidden_spike_stats = compute_statistics(hidden_spike_data)
output_spike_stats = compute_statistics(output_spike_data)
cumulative_time_data_train = compute_cumulative_time(time_data_train)

print(accuracy_stats_train)
#plot_statistics_with_time(prediction_confidence_stats, cumulative_time_data_train, 'Average Prediction Confidence', 'Average Prediction Confidence during Training for Different DT Values')
plot_statistics(accuracy_stats_train, 'Accuracy (%)', 'Average of Accuracy per Epoch during Training for Different DT Values')
plot_statistics(accuracy_stats_test, 'Accuracy (%)', 'Average of Accuracy per Epoch for Different DT Values')
plot_statistics(hidden_spike_stats, 'Hidden Layer 1 Spike Counts', 'Average and Standard Deviation of Hidden Layer 1 Spike Counts per Epoch for Different DT Values')
plot_statistics(output_spike_stats, 'Output Layer Spike Counts', 'Average and Standard Deviation of Output Layer Spike Counts per Epoch for Different DT Values')
print(prediction_confidence_stats)
plot_statistics_prediction(prediction_confidence_stats, latency_values, "Prediction Confidence", "Average Prediction Confidence on test")