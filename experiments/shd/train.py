from pathlib import Path
import cupy as cp
import numpy as np

import sys

import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, "../../")  # Add repository root to python path

from Dataset import Dataset
from bats.Monitors import *
from bats.Layers import InputLayer, LIFLayer
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *

DATASET_PATH = Path("../../datasets/")

N_INPUTS = 700
SIMULATION_TIME = 0.2
LATENCY_TIMES_LIST = [0.02, 0.03, 0.05, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2]

# Hidden layer
N_NEURONS_1 = 128
TAU_S_1 = 0.100
THRESHOLD_HAT_1 = 0.005
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 30

# Output_layer
N_OUTPUTS = 20
TAU_S_OUTPUT = 0.100
THRESHOLD_HAT_OUTPUT = 0.26
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 30

# Training parameters
N_TRAINING_EPOCHS = 100

N_TRAIN_SAMPLES = 8156
N_TEST_SAMPLES = 2264
TRAIN_BATCH_SIZE = 5
TEST_BATCH_SIZE = 100
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = 0.001
LR_DECAY_EPOCH = 10  # Perform decay very n epochs
LR_DECAY_FACTOR = 1.0
MIN_LEARNING_RATE = 1e-4
TARGET_FALSE = 3
TARGET_TRUE = 15

# Plot parameters
EXPORT_METRICS = True
EXPORT_DIR = Path("output_metrics")
SAVE_DIR = Path("best_model")


#def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
#    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)

def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    k = 1.0 / n_pre
    limit = cp.sqrt(k)
    print(limit)
    weights = cp.random.uniform(-limit, limit, size=(n_post, n_pre), dtype=cp.float32)
    return weights

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

def scatter_plot_spike_times(input_spike_times, spike_times, output_spikes_times, discrete_output_spike_times, label, DT, timestep, PLOT_DIR):
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 15))

    for neuron_index in range(input_spike_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax0.scatter(input_spike_times[neuron_index], [neuron_index] * input_spike_times.shape[1], label=f'Neuron {neuron_index}')
    ax0.set_xlabel('Spike Times (s)')
    ax0.set_ylabel('Neuron Index')
    ax0.set_title('Scatter Plot of Input Layer Spike Times for Each Neuron')

    plt.subplots_adjust(hspace=0.4)

    for neuron_index in range(spike_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax1.scatter(spike_times[neuron_index], [neuron_index] * spike_times.shape[1], label=f'Neuron {neuron_index}')
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
    ax2.hlines(label, 0, 0.3, color='r', linestyles='dashed', label=f'True Label')
    plt.subplots_adjust(hspace=0.4)

    plt.savefig( PLOT_DIR / ('spike_times_plot_' + str(timestep) + ".png"))
    plt.show()

def plot_heatmap(weights, title="Weight Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Input Neurons')
    plt.ylabel('Output Neurons')
    plt.show()

def calculate_latency_prob(discrete_out_spikes: cp.ndarray, labels):
    # This is for latency
    batch_size, num_neurons, time_steps = discrete_out_spikes.shape
    # Convert LATENCY_TIMES_LIST to a CuPy array for efficient broadcasting
    latency_times_array = cp.array(LATENCY_TIMES_LIST)
    # Create a dictionary to store the results
    probabilities_list = {latency: [] for latency in LATENCY_TIMES_LIST}
    # Get the labels as a CuPy array
    # Extract spikes for the target neurons based on labels
    target_spikes = cp.array([discrete_out_spikes[i, labels[i], :] for i in range(batch_size)])
    # Loop over each latency time
    for latency in latency_times_array:
        # Create a mask for spikes that occur before the given latency time
        latency_mask = target_spikes <= latency
        # print(latency_mask)
        # Count the spikes for each batch and latency time
        count_spikes = cp.sum(latency_mask, axis=1)
        # print(count_spikes)
        # Calculate probabilities and store in the list
        probabilities_list[latency.item()] = (count_spikes / TARGET_TRUE).tolist()
    return probabilities_list

def train_spike_count(DT, np_seed, cp_seed, EXPORT_DIR, PLOT_DIR):
    print("This network will be trained using DT = " + str(DT))
    np.random.seed(np_seed)
    cp.random.seed(cp_seed)

    if EXPORT_METRICS and not EXPORT_DIR.exists():
        EXPORT_DIR.mkdir()

    if not PLOT_DIR.exists():
        PLOT_DIR.mkdir()
    # Dataset
    print("Loading datasets...")
    dataset = Dataset(path=DATASET_PATH)
    dataset_test_hypothesis = Dataset(path=DATASET_PATH)

    print("Creating network...")
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                            theta=THRESHOLD_HAT_1,
                            delta_theta=DELTA_THRESHOLD_1,
                            time_delta=DT,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_1,
                            name="Hidden layer 1")
    network.add_layer(hidden_layer)

    output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            time_delta=DT,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

    # Metrics
    training_steps = 0
    plot_count = 0
    train_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_train")
    train_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_train")
    train_silent_label_monitor = SilentLabelsMonitor()
    train_time_monitor = TimeMonitor()

    train_monitors_manager = MonitorsManager([train_loss_monitor,
                                              train_accuracy_monitor,
                                              train_silent_label_monitor,
                                              train_time_monitor],
                                             print_prefix="Train | ")

    test_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_test")
    test_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_test")
    test_learning_rate_monitor = ValueMonitor(name="Learning rate", decimal=5)
    # Only monitor LIF layers
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name, export_path=EXPORT_DIR / ("spike_count_" + str(l.name))) for l in network.layers if isinstance(l, LIFLayer)}
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

    best_acc = 0.0
    print("Training...")

    output_spike_count_target = []
    overall_average_probabilities = []

    dataset_test_hypothesis.shuffle()

    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        dataset.shuffle()

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)
        epoch_spike_target = []
        for batch_idx in range(N_TRAIN_BATCH):
            # Get next batch
            spikes, n_spikes, labels = dataset.get_train_batch(batch_idx, TRAIN_BATCH_SIZE)
            #print(spikes[spikes!=np.inf])
            if DT != 0.0:
                discrete_spikes = discrete(spikes, DT)
            else:
                discrete_spikes = spikes

            #print("Batch number : " + str(batch_idx) + "\n")
            # Inference
            network.reset()
            network.forward(spikes, n_spikes, discrete_spikes, max_simulation=SIMULATION_TIME, training=True)
            out_spikes, n_out_spikes, discrete_out_spikes = network.output_spike_trains

            # Predictions, loss and errors
            pred = loss_fct.predict(discrete_out_spikes, n_out_spikes)
            loss, errors = loss_fct.compute_loss_and_errors(discrete_out_spikes, n_out_spikes, labels)

            pred_cpu = pred.get()
            loss_cpu = loss.get()
            n_out_spikes_cpu = n_out_spikes.get()

            # Update monitors
            train_loss_monitor.add(loss_cpu)
            train_accuracy_monitor.add(pred_cpu, labels)
            train_silent_label_monitor.add(n_out_spikes_cpu, labels)

            # Compute gradient
            gradient = network.backward(errors)
            avg_gradient = [None if g is None else cp.mean(g, axis=0) for g in gradient]
            del gradient

            #print(np.max(avg_gradient[1].get().flatten()) - np.min(avg_gradient[1].get().flatten()))

            # Apply step
            deltas = optimizer.step(avg_gradient)
            del avg_gradient

            network.apply_deltas(deltas)
            del deltas

            #hidden_layer.weights = cp.clip(hidden_layer.weights, -2.0, 2.0)

            training_steps += 1
            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES

            #if batch_idx == 1:
                #print(labels)
                #plt.hist(errors.get()[0].flatten())
                #plt.show()
                #spike_times = hidden_layer.spike_trains[2].get()[batch_idx]  # first dataset in the batch
                #scatter_plot_spike_times(input_layer.spike_trains[2].get()[batch_idx], spike_times, discrete_out_spikes.get()[batch_idx], labels[batch_idx])

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                # We average the probabilities per epoch to get a rough estimate of the confidence
                #train_prediction_confidence_monitor.add(sum(epoch_spike_target) / len(epoch_spike_target))
                #print(output_spike_count_target)

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics)
                #train_monitors_manager.export()

            # Test evaluation
            if training_steps % TEST_PERIOD_STEP == 0:
                test_time_monitor.start()
                cumulative_probabilities = {latency: [] for latency in LATENCY_TIMES_LIST}
                for batch_idx in range(N_TEST_BATCH):
                    spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
                    if DT != 0.0:
                        discrete_spikes = discrete(spikes, DT)
                    else:
                        discrete_spikes = spikes
                    network.reset()
                    network.forward(spikes, n_spikes, discrete_spikes, max_simulation=SIMULATION_TIME)
                    out_spikes, n_out_spikes, discrete_out_spikes = network.output_spike_trains

                    #if batch_idx == 0:
                    #    spike_times = hidden_layer.spike_trains[0].get()[0]# first dataset in the batch
                    #    scatter_plot_spike_times(spike_times, discrete_out_spikes.get()[0], labels[0])
                        # Step 1: Create a boolean array indicating where the elements are finite
                        #finite_mask = cp.isfinite(discrete_out_spikes)
                        # Step 2: Sum along the last axis to count non-infinite numbers
                        #non_infinite_count = cp.sum(finite_mask, axis=2)
                        #selected_values = non_infinite_count[cp.arange(non_infinite_count.shape[0]), labels].reshape(
                        #     (100, 1))
                        #print(selected_values.shape)  # Should print (100, 1)
                        #print(selected_values)

                    #This is for latency
                    probabilities_list = calculate_latency_prob(discrete_out_spikes, labels)
                    # Calculate the average probabilities for each latency time using CuPy
                    average_probabilities = {latency: np.mean(probabilities) for latency, probabilities in
                                             probabilities_list.items()}

                    # Display the results
                    for latency, avg_prob in average_probabilities.items():
                        cumulative_probabilities[latency].append(avg_prob)

                    pred = loss_fct.predict(discrete_out_spikes, n_out_spikes)
                    loss = loss_fct.compute_loss(discrete_out_spikes, n_out_spikes, labels)

                    #print("-------------")
                    #print(labels)
                    #print("+")
                    #print(pred)

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
                spikes, n_spikes, labels = dataset_test_hypothesis.get_train_batch(0,
                                                                                  TEST_BATCH_SIZE)  # Using input 0
                if DT != 0.0:
                    discrete_spikes = discrete(spikes, DT)
                else:
                    discrete_spikes = spikes
                network.reset()
                network.forward(spikes, n_spikes, discrete_spikes, max_simulation=SIMULATION_TIME)
                out_spikes, n_out_spikes, discrete_out_spikes = network.output_spike_trains
                scatter_plot_spike_times(input_layer.spike_trains[2].get()[0],
                                         hidden_layer.spike_trains[2].get()[0], out_spikes.get()[0],
                                         discrete_out_spikes.get()[0], labels[0], DT, plot_count, PLOT_DIR)
                plot_count = plot_count + 1

                #plot_heatmap(output_layer.weights.get(), 'Weight Heatmap Output Layer')

                for l, mon in test_norm_monitors.items():
                    mon.add(l.weights)

                overall_average_probabilities.append({latency: np.mean(probabilities) for latency, probabilities in
                                                      cumulative_probabilities.items()})
                #print(overall_average_probabilities)


                test_learning_rate_monitor.add(optimizer.learning_rate)

                records = test_monitors_manager.record(epoch_metrics)
                test_monitors_manager.print(epoch_metrics)
                #test_monitors_manager.export()

                acc = records[test_accuracy_monitor]
                if acc > best_acc:
                    best_acc = acc
                    network.store(SAVE_DIR / ("DT = " + str(DT)))
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")
    test_monitors_manager.export()
    train_monitors_manager.export()
    return test_monitors_manager, train_monitors_manager, output_spike_count_target, overall_average_probabilities


def calculate_average_across_experiments(results):
    average_results = {}

    for dt_index, dt_value in enumerate(DT_list):
        dt_frames = [experiment[dt_index] for experiment in results]

        # Concatenate DataFrames along the rows
        concatenated_df = pd.concat(dt_frames, axis=0)

        # Calculate the mean for each group of epochs
        average_df = concatenated_df.groupby('Epochs').mean().reset_index()

        average_results[dt_value] = average_df

    return average_results


def plot_metrics_across_dt(average_results):
    # Get the metrics from the DataFrame columns, excluding 'Epochs'
    example_df = next(iter(average_results.values()))
    metrics = [col for col in example_df.columns if col != 'Epochs']

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for dt_value, avg_df in average_results.items():
            plt.plot(avg_df['Epochs'], avg_df[metric], label=f'DT = {dt_value}')

        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.title(f'{metric} Across Epochs for Different DT values')
        plt.legend()
        plt.show()

NR_EXPERIMENTS = 1

if __name__ == "__main__":
    monitors_dict = {}
    all_results = []
    all_results_test = []
    DT_list = [0.0]
    for i in range(0, NR_EXPERIMENTS):
        print("STARTING RUN NUMBER " + str(i))
        print("\n")
        max_int = np.iinfo(np.int32).max
        np_seed = np.random.randint(low=0, high=max_int)
        cp_seed = np.random.randint(low=0, high=max_int)
        print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")
        EXPORT_DIR = Path("./output_metrics/" + "experiment_" + str(i) + "_ " + str(str(np_seed) + "_" + str(cp_seed)))

        #train_model_results = []
        #test_model_results = []
        for val in DT_list:
            #for thidden in list([5, 7, 10, 12, 13, 15, 20, 25, 30]):
                #SPIKE_BUFFER_SIZE_1 = thidden
                #for toutput in list([5, 7, 10, 15, 17, 20, 23, 25, 28, 30]):
                    #SPIKE_BUFFER_SIZE_OUTPUT = toutput
            DT = val
            PLOT_DIR = Path('./scatter_plots/' + 'DT = ' + str(val))
            test_monitor, train_monitor, output_spike_target, all_discrete_spikes = train_spike_count(val, np_seed, cp_seed, EXPORT_DIR, PLOT_DIR)
            #print(all_discrete_spikes)
            df = train_monitor.return_vals()
            df2 = test_monitor.return_vals()
            df3 = pd.DataFrame(all_discrete_spikes)

            #df.to_csv(EXPORT_DIR / ("Train hidden buffer spike = " + str(thidden) + "output " + str(toutput)), index=False)
            #df2.to_csv(EXPORT_DIR / ("Test hidder buffer spike = " + str(thidden) + " output " + str(toutput)), index=False)
            #df3.to_csv(EXPORT_DIR / ("Prediction confidence hidder buffer spike = " + str(thidden) + " output buffer spike " + str(toutput)), index=False)
            df.to_csv(EXPORT_DIR / ("Train DT = " + str(DT)), index=False)
            df2.to_csv(EXPORT_DIR / ("Test DT = " + str(DT)), index=False)
            df3.to_csv(EXPORT_DIR / ("Prediction confidence DT = " + str(DT)), index=False)
        #all_results.append(train_model_results)
        #all_results_test.append(test_model_results)
    #average_results = calculate_average_across_experiments(all_results)
    #average_results_test = calculate_average_across_experiments(all_results_test)
    #plot_metrics_across_dt(average_results)
    #plot_metrics_across_dt(average_results_test)