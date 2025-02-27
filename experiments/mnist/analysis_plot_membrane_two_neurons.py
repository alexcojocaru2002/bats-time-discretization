import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the PSP kernel ε(t)
def epsilon(t, tau, tau_s):
    return (t > 0).float() * (tau * tau_s / (tau - tau_s)) * (torch.exp(-t / tau) - torch.exp(-t / tau_s))

# Define the refractory kernel η(t)
def eta(t, theta, tau_eta):
    return (t > 0).float() * theta * torch.exp(-t / tau_eta)

# Define the membrane potential with dynamic spiking
def membrane_potential_with_dynamic_spiking(
    t, pre_synaptic_spike_times, synaptic_weights, tau, tau_s, theta, tau_eta, threshold
):
    u = torch.zeros_like(t)
    self_spike_times = []

    for step, current_time in enumerate(t):
        # Contribution from pre-synaptic spikes
        for i, spike_times in enumerate(pre_synaptic_spike_times):
            weight = synaptic_weights[i]
            for spike_time in spike_times:
                if current_time >= spike_time:
                    u[step] += weight * epsilon(current_time - spike_time, tau, tau_s)

        # Contribution from self-spikes (refractory effect)
        for spike_time in self_spike_times:
            if current_time >= spike_time:
                u[step] -= eta(current_time - spike_time, theta, tau_eta)

        # Check for spiking
        if u[step] > threshold:
            self_spike_times.append(current_time.item())
            # u[step] = 0  # Hard reset (optional)

    return u, torch.tensor(self_spike_times)

# Generate Poisson spike train
def generate_poisson_spike_train(rate, duration, num_neurons):
    """
    Generate a Poisson spike train for a given rate, duration, and number of neurons.
    """
    spike_trains = []
    for _ in range(num_neurons):
        spike_times = []
        current_time = 0.0
        while current_time < duration:
            isi = np.random.exponential(1.0 / rate)  # Inter-spike interval
            current_time += isi
            if current_time < duration:
                spike_times.append(current_time)
        spike_trains.append(torch.tensor(spike_times))
    return spike_trains

# Simulation parameters
t = torch.linspace(0, 1.0, 1000)  # Time (0 to 1 second)
num_neurons = 2  # Number of pre-synaptic neurons
firing_rate = 7  # Poisson rate (Hz)

# Generate Poisson spike train for Layer 1
layer1_spike_times = generate_poisson_spike_train(rate=firing_rate, duration=1.0, num_neurons=3)
layer1_weights = torch.rand(3)  # Synaptic weights

# Layer 2 parameters
layer2_tau_s = 0.130  # PSP rise time constant
layer2_tau = 2 * layer2_tau_s  # PSP decay time constant
theta = 0.2  # Refractory strength
tau_eta = layer2_tau  # Refractory decay time constant
threshold = 0.2  # Spiking threshold

# Compute membrane potential for Layer 2 with dynamic spiking
u2, layer2_spike_times = membrane_potential_with_dynamic_spiking(
    t, layer1_spike_times, layer1_weights,
    tau=layer2_tau, tau_s=layer2_tau_s, theta=theta, tau_eta=tau_eta, threshold=threshold
)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t.numpy(), u2.numpy(), label="Layer 2 Membrane Potential", color="green")
for i, spike_times in enumerate(layer1_spike_times):
    plt.scatter(spike_times.numpy(), [0] * len(spike_times), label=f"Layer 1 Neuron {i+1} Spikes", zorder=5)
plt.scatter(layer2_spike_times.numpy(), [0] * len(layer2_spike_times), color='purple', label="Layer 2 Spikes", zorder=5)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title("Layer 2: Membrane Potential with Poisson Spike Train")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential")
plt.legend()
plt.show()
