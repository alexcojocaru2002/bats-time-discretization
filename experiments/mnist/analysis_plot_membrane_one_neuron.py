import torch
import matplotlib.pyplot as plt

# Define the PSP kernel epsilon(t)
def epsilon(t, tau, tau_s):
    return (t > 0).float() * (tau * tau_s / (tau - tau_s)) * (torch.exp(-t / tau) - torch.exp(-t / tau_s))

# Define the refractory kernel eta(t)
def eta(t, theta, tau):
    return (t > 0).float() * theta * torch.exp(-t / tau)

# Define the SRM membrane potential
def membrane_potential(t, spike_times, weights, tau, tau_s, theta, tau_eta):
    u = torch.zeros_like(t)
    for spike_time, weight in zip(spike_times, weights):
        u += weight * epsilon(t - spike_time, tau, tau_s)
    u -= torch.sum(eta(t - spike_times[:, None], theta, tau_eta), dim=0)
    return u

# Simulation parameters
t = torch.linspace(0, 100, 1000)  # Time in ms
spike_times = torch.tensor([10, 30, 50, 70])  # Spike times in ms
weights = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Synaptic weights
tau = 20.0  # PSP decay time constant in ms
tau_s = 5.0  # PSP rise time constant in ms
theta = 1.0  # Reset potential
tau_eta = 30.0  # Refractory kernel time constant in ms

# Compute membrane potential
u = membrane_potential(t, spike_times, weights, tau, tau_s, theta, tau_eta)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t.numpy(), u.numpy(), label="Membrane Potential")
plt.scatter(spike_times.numpy(), [0] * len(spike_times), color='red', label="Input Spikes", zorder=5)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title("Membrane Potential Dynamics (SRM)")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential")
plt.legend()
plt.show()
