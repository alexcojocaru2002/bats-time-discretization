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

# Dataset
DATASET_PATH = Path("../../datasets/")

N_INPUTS = 28 * 28
SIMULATION_TIME = 0.2
# LATENCY_TIMES_LIST = [0.02, 0.03, 0.05, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2]

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

# Plot parameters
EXPORT_METRICS = True
EXPORT_DIR = Path("./output_metrics")
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

def calculate_latency_prob(discrete_out_spikes: cp.ndarray, labels):
    # This is for latency
    batch_size, num_neurons, time_steps = discrete_out_spikes.shape
    # Convert LATENCY_TIMES_LIST to a CuPy array for efficient broadcasting
    # latency_times_array = cp.array(LATENCY_TIMES_LIST)
    # # Create a dictionary to store the results
    # probabilities_list = {latency: [] for latency in LATENCY_TIMES_LIST}
    # Get the labels as a CuPy array
    # Extract spikes for the target neurons based on labels
    target_spikes = cp.array([discrete_out_spikes[i, labels[i], :] for i in range(batch_size)])
    # Loop over each latency time
    # for latency in latency_times_array:
    #     # Create a mask for spikes that occur before the given latency time
    #     latency_mask = target_spikes <= latency
    #     # print(latency_mask)
    #     # Count the spikes for each batch and latency time
    #     count_spikes = cp.sum(latency_mask, axis=1)
    #     # print(count_spikes)
    #     # Calculate probabilities and store in the list
    #     probabilities_list[latency.item()] = (count_spikes / TARGET_TRUE).tolist()
    # return probabilities_list



def scatter_plot_spike_times(input_spike_times, hidden_spike_times, hidden2_spike_times, output_spikes_times, discrete_output_spike_times, label, DT, timestep, PLOT_DIR):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 15))

    for neuron_index in range(input_spike_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax0.scatter(input_spike_times[neuron_index], [neuron_index] * input_spike_times.shape[1], label=f'Neuron {neuron_index}')
    ax0.set_xlabel('Spike Times (s)')
    ax0.set_ylabel('Neuron Index')
    ax0.set_title('Scatter Plot of Input Layer Spike Times for Each Neuron')

    for neuron_index in range(hidden_spike_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax1.scatter(hidden_spike_times[neuron_index], [neuron_index] * hidden_spike_times.shape[1], label=f'Neuron {neuron_index}')
    ax1.set_xlabel('Spike Times (s)')
    ax1.set_ylabel('Neuron Index')
    ax1.set_title('Scatter Plot of Hidden Layer Spike Times for Each Neuron')

    for neuron_index in range(hidden2_spike_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax2.scatter(hidden2_spike_times[neuron_index], [neuron_index] * hidden2_spike_times.shape[1], label=f'Neuron {neuron_index}')
    ax2.set_xlabel('Spike Times (s)')
    ax2.set_ylabel('Neuron Index')
    ax2.set_title('Scatter Plot of Hidden Layer 2 Spike Times for Each Neuron')

    for neuron_index in range(discrete_output_spike_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax3.scatter(discrete_output_spike_times[neuron_index], [neuron_index] * discrete_output_spike_times.shape[1],
                    label=f'Neuron {neuron_index}', color='b')

    for neuron_index in range(output_spikes_times.shape[0]):
        # Plot spike times for the current neuron with a specific color
        ax3.scatter(output_spikes_times[neuron_index], [neuron_index] * output_spikes_times.shape[1],
                    label=f'Neuron {neuron_index}', color='r')

    ax3.set_xlabel('Spike Times (s)')
    ax3.set_ylabel('Neuron Index')
    ax3.set_title('Scatter Plot of Output Layer Spike Times for Each Neuron')
    ax3.hlines(label, 0, 0.2, color='r', linestyles='dashed', label=f'True Label')

    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.savefig(Path(PLOT_DIR) / ('spike_times_plot_' + str(timestep) + ".png"))
    plt.show()


def plot_heatmap(weights, title="Weight Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Input Neurons')
    plt.ylabel('Output Neurons')
    plt.show()

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
                            time_delta=DT,
                            delta_theta=DELTA_THRESHOLD_1,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_1,
                            name="Hidden layer 1")
    network.add_layer(hidden_layer)

    hidden_layer_2 = LIFLayer(previous_layer=hidden_layer, n_neurons=N_NEURONS_2, tau_s=TAU_S_2,
                              theta=THRESHOLD_HAT_2,
                              time_delta=DT,
                              delta_theta=DELTA_THRESHOLD_2,
                              weight_initializer=weight_initializer,
                              max_n_spike=SPIKE_BUFFER_SIZE_2,
                              name="Hidden layer 2")
    network.add_layer(hidden_layer_2)

    output_layer = LIFLayer(previous_layer=hidden_layer_2, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            time_delta=DT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

    # Metrics
    training_steps = 0
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
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
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

    best_acc = 0

    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        # dataset.shuffle()

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)

        for batch_idx in range(N_TRAIN_BATCH):
            # Get next batch
            spikes, n_spikes, labels = dataset.get_train_batch(batch_idx, TRAIN_BATCH_SIZE)
            # print(spikes[spikes!=np.inf])
            if DT != 0.0:
                discrete_spikes = discrete(spikes, DT)
            else:
                discrete_spikes = spikes

            # print("Batch number : " + str(batch_idx) + "\n")
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
            avg_gradient = [None if g is None else cp.mean(g, axis=0) for g, layer in zip(gradient, network.layers)]
            del gradient

            # plt.hist(avg_gradient[1].get().flatten())
            # plt.show()

            # Apply step
            deltas = optimizer.step(avg_gradient)
            del avg_gradient

            network.apply_deltas(deltas)
            del deltas

            training_steps += 1
            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES

            # plt.hist(errors.get()[0].flatten())
            # plt.show()

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                # We average the probabilities per epoch to get a rough estimate of the confidence
                # train_prediction_confidence_monitor.add(sum(epoch_spike_target) / len(epoch_spike_target))
                # print(output_spike_count_target)

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics)

                # train_monitors_manager.export()

            # Test evaluation
            if training_steps % TEST_PERIOD_STEP == 0:
                test_time_monitor.start()
                # cumulative_probabilities = {latency: [] for latency in LATENCY_TIMES_LIST}
                for batch_idx in range(N_TEST_BATCH):
                    spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
                    if DT != 0.0:
                        discrete_spikes = discrete(spikes, DT)
                    else:
                        discrete_spikes = spikes
                    network.reset()
                    network.forward(spikes, n_spikes, discrete_spikes, max_simulation=SIMULATION_TIME)
                    out_spikes, n_out_spikes, discrete_out_spikes = network.output_spike_trains

                    # This is for latency
                    # probabilities_list = calculate_latency_prob(discrete_out_spikes, labels)
                    # # Calculate the average probabilities for each latency time using CuPy
                    # average_probabilities = {latency: np.mean(probabilities) for latency, probabilities in
                    #                          probabilities_list.items()}

                    # Display the results
                    # for latency, avg_prob in average_probabilities.items():
                    #     cumulative_probabilities[latency].append(avg_prob)

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
                # spikes, n_spikes, labels = dataset_test_hypothesis.get_test_batch(0,
                #                                                                   TEST_BATCH_SIZE)  # Using input 0
                # if DT != 0.0:
                #     discrete_spikes = discrete(spikes, DT)
                # else:
                #     discrete_spikes = spikes
                # network.reset()
                # network.forward(spikes, n_spikes, discrete_spikes, max_simulation=SIMULATION_TIME)
                # out_spikes, n_out_spikes, discrete_out_spikes = network.output_spike_trains
                # scatter_plot_spike_times(input_layer.spike_trains[2].get()[0],
                #                          hidden_layer.spike_trains[2].get()[0], hidden_layer_2.spike_trains[2].get()[0],
                #                          out_spikes.get()[0], discrete_out_spikes.get()[0], labels[0], DT, plot_count, PLOT_DIR)
                # plot_count = plot_count + 1
                # plot_heatmap(output_layer.weights.get())

                for l, mon in test_norm_monitors.items():
                    mon.add(l.weights)

                # overall_average_probabilities.append({latency: np.mean(probabilities) for latency, probabilities in
                #                                       cumulative_probabilities.items()})
                # print(overall_average_probabilities)

                test_learning_rate_monitor.add(optimizer.learning_rate)

                records = test_monitors_manager.record(epoch_metrics)
                test_monitors_manager.print(epoch_metrics)
                # test_monitors_manager.export()

                acc = records[test_accuracy_monitor]
                if acc > best_acc:
                    best_acc = acc
                    network.store(SAVE_DIR / ("DT = " + str(DT)))
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")
    test_monitors_manager.export()
    train_monitors_manager.export()
    return test_monitors_manager, train_monitors_manager




NR_EXPERIMENTS = 1

if __name__ == "__main__":
    monitors_dict = {}
    all_results = []
    all_results_test = []
    #np_seeds = [1399766615]
    #cp_seeds = [40394794]
    DT_list = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]
    for i in range(0, NR_EXPERIMENTS):
        print("STARTING RUN NUMBER " + str(i))
        print("\n")
        max_int = np.iinfo(np.int32).max
        np_seed = np.random.randint(low=0, high=max_int)
        cp_seed = np.random.randint(low=0, high=max_int)
        #np_seed = np_seeds[i]
        #cp_seed = cp_seeds[i]
        print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")
        EXPORT_DIR = Path("./output_metrics/" + "experiment_" + str(i) + "_ " + str(str(np_seed) + "_" + str(cp_seed)))

        #train_model_results = []
        #test_model_results = []
        for val in DT_list:
            DT = val
            PLOT_DIR = Path('./scatter_plots/' + 'DT = ' + str(val))

            test_monitor, train_monitor = train_spike_count(val, np_seed, cp_seed, EXPORT_DIR, PLOT_DIR)

            df = train_monitor.return_vals()
            df2 = test_monitor.return_vals()
            # df3 = pd.DataFrame(all_discrete_spikes)

            df.to_csv(EXPORT_DIR / ("Train DT = " + str(val)), index=False)
            df2.to_csv(EXPORT_DIR / ("Test DT = " + str(val)), index=False)
            # df3.to_csv(EXPORT_DIR / ("Prediction Confidence DT = " + str(val)), index=False)
        #all_results.append(train_model_results)
        #all_results_test.append(test_model_results)
