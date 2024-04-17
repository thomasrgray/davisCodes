import numpy as np
import os

# Define parameters
duration = 120  # Duration of the experiment in seconds
num_neurons = 15  # Number of neurons per area
num_sets = 15  # Number of sets of spike train data

# Create directories if they do not exist
os.makedirs('Desktop/Isaac_code/taste_spike_trains', exist_ok=True)
os.makedirs('Desktop/Isaac_code/olfactory_spike_trains', exist_ok=True)

# Generate and save spike train data for taste cortex
for i in range(num_sets):
    taste_firing_rate = np.random.uniform(20, 60)  # Firing rate for taste cortex (between 20 and 60 Hz)
    for neuron in range(num_neurons):
        taste_spike_times = generate_spike_train(duration, taste_firing_rate)
        np.savetxt(f'Desktop/Isaac_code/taste_spike_trains/taste_spike_train_set_{i + 1}_neuron_{neuron + 1}.txt', taste_spike_times)

# Generate and save spike train data for olfactory cortex
for i in range(num_sets):
    olfactory_firing_rate = np.random.uniform(20, 60)  # Firing rate for olfactory cortex (between 20 and 60 Hz)
    for neuron in range(num_neurons):
        olfactory_spike_times = generate_spike_train(duration, olfactory_firing_rate)
        np.savetxt(f'Desktop/Isaac_code/olfactory_spike_trains/olfactory_spike_train_set_{i + 1}_neuron_{neuron + 1}.txt', olfactory_spike_times)
