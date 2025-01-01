# Import necessary libraries
from nipype.interfaces.nipy.model import SpecifySPMModel
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.spm import DCMGenerateModel
from nipype.interfaces.spm import Level1Design, EstimateModel, EstimateContrast

# Specify experimental paradigm (e.g., on/off conditions for taste and olfactory stimuli)
paradigm_file = 'path_to_paradigm_file.txt'

# Specify anatomical and functional images
anat_file = 'path_to_anatomical_image.nii'
func_file = 'path_to_functional_image.nii'

# Step 1: Specify the SPM model
specify_model = SpecifySPMModel()
specify_model.inputs.input_units = 'secs'
specify_model.inputs.functional_runs = func_file
specify_model.inputs.event_onsets = paradigm_file
specify_model.inputs.time_repetition = TR  # Time repetition of functional images
specify_model.run()

# Step 2: Generate DCM model
dcm_generate = DCMGenerateModel()
dcm_generate.inputs.eeg_units = 'secs'
dcm_generate.inputs.estimation_units = 'secs'
dcm_generate.inputs.time_units = 'secs'
dcm_generate.inputs.subject_info = subjects_info  # Specify subject-specific information
dcm_generate.run()

# Step 3: Specify first-level design
level1 = Level1Design()
level1.inputs.timing_units = 'secs'
level1.inputs.interscan_interval = TR
level1.inputs.bases = {'hrf': {'derivs': [1, 0]}}
level1.run()

# Step 4: Estimate DCM parameters
estimate = EstimateModel()
estimate.inputs.estimation_method = {'Classical': 1}  # Bayesian estimation can also be used
estimate.run()

# Step 5: Estimate contrasts
contrast = EstimateContrast()
contrast.inputs.contrasts = contrasts  # Specify contrasts of interest
contrast.run()

##########3

import numpy as np
import matplotlib.pyplot as plt

# Load paradigm
paradigm = np.loadtxt('/Users/thomasgray/Desktop/Isaac_code/paradigm.txt', skiprows=1, dtype=int)

# Simulation parameters
total_time = paradigm[-1, 0] + paradigm[-1, 2]  # Total experiment time
tr = 2  # Time repetition (in seconds)
num_volumes = total_time // tr  # Total number of volumes

# Generate random neural activity for taste and olfactory cortex
np.random.seed(42)  # For reproducibility
taste_cortex_activity = np.random.randn(num_volumes)
olfactory_cortex_activity = np.random.randn(num_volumes)

# Plot simulated neural activity
plt.figure(figsize=(10, 4))
plt.plot(np.arange(0, total_time, tr), taste_cortex_activity, label='Taste Cortex')
plt.plot(np.arange(0, total_time, tr), olfactory_cortex_activity, label='Olfactory Cortex')
plt.xlabel('Time (seconds)')
plt.ylabel('Neural Activity')
plt.title('Simulated Neural Activity')
plt.legend()
plt.show()

################3

import numpy as np

# Function to generate spike train data based on firing rate
def generate_spike_train(duration, firing_rate):
    time_bins = np.arange(0, duration, 1.0 / firing_rate)
    spike_train = np.random.uniform(0, 1, len(time_bins)) < 1.0 / firing_rate
    spike_times = time_bins[spike_train]
    return spike_times

# Define parameters
duration = 120  # Duration of the experiment in seconds
taste_firing_rate = np.random.uniform(20, 60)  # Firing rate for taste cortex (between 20 and 60 Hz)
olfactory_firing_rate = np.random.uniform(20, 60)  # Firing rate for olfactory cortex (between 20 and 60 Hz)

# Generate spike train data
taste_spike_times = generate_spike_train(duration, taste_firing_rate)
olfactory_spike_times = generate_spike_train(duration, olfactory_firing_rate)

# Save spike train data to text files
np.savetxt('taste_spike_train.txt', taste_spike_times, fmt='%.4f')
np.savetxt('olfactory_spike_train.txt', olfactory_spike_times, fmt='%.4f')
