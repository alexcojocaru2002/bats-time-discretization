from matplotlib import pyplot as plt
import numpy as np

DT_list = [0.015, 0.01, 0.005, 0.001, 0.0005]

hidden_layer_spike_count0 = [297.46, 404.45, 240.91, 283.68, 367.18]
output_layer_spike_count0 = [27.76, 29.97, 41.32, 41.83, 39.02]
accuracy0 = [6.46, 15.18, 98.15, 98.09, 98.13]
time_test0 = [1.66, 2.2, 1.5, 1.48, 1.65]

hidden_layer_spike_count1 = [262.73, 378.74, 246.69, 307.24, 429.34]
output_layer_spike_count1 = [27.93, 29.01, 41.79, 41.62, 36.46]
accuracy1 = [7.02, 17.54, 98.17, 98.24, 98.02]
time_test1 = [1.63, 2.13, 1.35, 1.37, 2.0]

hidden_layer_spike_count2 = [312.6, 369.57, 238.17, 316.06, 346.97]
output_layer_spike_count2 = [27.32, 29.14, 41.12, 41.01, 40.03]
accuracy2 = [6.33, 19.0, 98.16, 98.14, 98.11]
time_test2 = [1.68, 1.95, 1.35, 1.45, 1.77]

hidden_layer_spike_count3 = [271.7, 379.36, 245.47, 295.08, 317.73]
output_layer_spike_count3 = [28.36, 29.15, 41.82, 41.58, 41.78]
accuracy3 = [7.28, 16.55, 98.1, 98.21, 98.36]
time_test3 = [1.51, 2.15, 1.46, 1.39, 1.39]

hidden_layer_spike_count4 = [269.03, 356.16, 244.97, 285.85, 381.66]
output_layer_spike_count4 = [28.08, 29.86, 41.43, 40.79, 37.51]
accuracy4 = [7.39, 19.95, 98.07, 98.21, 98.2]
time_test4 = [1.63, 1.92, 1.4, 2.22, 1.73]

avg_hidden_layer_spike_count = np.mean([hidden_layer_spike_count0, hidden_layer_spike_count1, hidden_layer_spike_count2, hidden_layer_spike_count3, hidden_layer_spike_count4], axis=0)
avg_output_layer_spike_count = np.mean([output_layer_spike_count0, output_layer_spike_count1, output_layer_spike_count2, output_layer_spike_count3, output_layer_spike_count4], axis=0)
avg_accuracy = np.mean([accuracy0, accuracy1, accuracy2, accuracy3, accuracy4], axis=0)
avg_time_test = np.mean([time_test0, time_test1, time_test2, time_test3, time_test4], axis=0)
stdev_accuracy = np.std([accuracy0, accuracy1, accuracy2, accuracy3, accuracy4], axis=0)
'''
plt.plot(DT_list, avg_hidden_layer_spike_count, marker='o', color='red')
plt.xlabel('Step size [ms]')
plt.ylabel('Hidden Layer spike count')
plt.title('Hidden Layer spike count')
plt.show()

plt.plot(DT_list, avg_accuracy, marker='o', color='red')
plt.xlabel('Step size [ms]')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()
'''

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Average Metrics', fontsize=16)

# Subplot 1: Hidden Layer Spike Count
axs[0, 0].plot(DT_list, avg_hidden_layer_spike_count, marker='o', linestyle='-', color='b')
axs[0, 0].set_title('Average Hidden Layer Spike Count')
axs[0, 0].set_xlabel('Time step size (s)')
axs[0, 0].set_ylabel('Spike Count')
axs[0, 0].grid(True)

# Subplot 2: Output Layer Spike Count
axs[0, 1].plot(DT_list, avg_output_layer_spike_count, marker='s', linestyle='-', color='g')
axs[0, 1].set_title('Average Output Layer Spike Count')
axs[0, 1].set_xlabel('Time step size (s)')
axs[0, 1].set_ylabel('Spike Count')
axs[0, 1].grid(True)

# Subplot 3: Accuracy
axs[1, 0].plot(DT_list, avg_accuracy, marker='^', linestyle='-', color='r')
axs[1, 0].fill_between(DT_list, avg_accuracy - 2 * stdev_accuracy, avg_accuracy + 2 * stdev_accuracy, alpha=0.2, color='r')
axs[1, 0].set_title('Average Accuracy')
axs[1, 0].set_xlabel('Time step size (s)')
axs[1, 0].set_ylabel('Accuracy (%)')
axs[1, 0].grid(True)

# Subplot 4: Time Test
axs[1, 1].plot(DT_list, avg_time_test, marker='d', linestyle='-', color='m')
axs[1, 1].set_title('Average Time Test')
axs[1, 1].set_xlabel('Time step size (s)')
axs[1, 1].set_ylabel('Time (s)')
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('average_metrics.png')
plt.show()
