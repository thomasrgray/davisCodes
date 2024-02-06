import numpy as np
import matplotlib.pyplot as plt

# Load paradigm
paradigm = np.loadtxt('paradigm.txt', skiprows=1, dtype=int)

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

###########3
from nipype.interfaces.spm import DCM8, EstimateDCM
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Load spike train data for taste cortex (example for one set)
taste_spike_times = np.loadtxt('Desktop/Isaac_code/taste_spike_train_set_1.txt')

# Load spike train data for olfactory cortex (example for one set)
olfactory_spike_times = np.loadtxt('Desktop/Isaac_code/olfactory_spike_train_set_1.txt')

# Define the experimental paradigm (onset times for each stimulus)
# Modify this according to your experiment
paradigm = {
    'taste': [10, 30, 50, 70, 90],  # Example onset times for taste stimuli (in seconds)
    'olfactory': [15, 35, 55, 75, 95]  # Example onset times for olfactory stimuli (in seconds)
}

# Specify DCM model
dcm = DCM8()
dcm.inputs.spatial_resolution = 8
dcm.inputs.estimation = 'MEE'
dcm.inputs.TR = 1.0  # TR (Repetition Time) in seconds
dcm.inputs.time_units = 'secs'

# Set up DCM nodes for taste cortex
dcm_taste = dcm.clone(name='dcm_taste')
dcm_taste.inputs.specificity = [1, 0]  # Connectivity from input (stimuli) to taste cortex
dcm_taste.inputs.delays = [0] * len(paradigm['taste'])  # No delay assumed for simplicity
dcm_taste.inputs.n_order = 1

# Set up DCM nodes for olfactory cortex
dcm_olfactory = dcm.clone(name='dcm_olfactory')
dcm_olfactory.inputs.specificity = [0, 1]  # Connectivity from input (stimuli) to olfactory cortex
dcm_olfactory.inputs.delays = [0] * len(paradigm['olfactory'])  # No delay assumed for simplicity
dcm_olfactory.inputs.n_order = 1

# Run DCM for taste cortex
dcm_taste.inputs.datafile = taste_spike_times
dcm_taste.inputs.input_units = 'secs'
dcm_taste.inputs.U = [paradigm['taste']]
dcm_taste.run()

# Run DCM for olfactory cortex
dcm_olfactory.inputs.datafile = olfactory_spike_times
dcm_olfactory.inputs.input_units = 'secs'
dcm_olfactory.inputs.U = [paradigm['olfactory']]
dcm_olfactory.run()

# Estimate DCM parameters for taste cortex
estimate_taste = EstimateDCM()
estimate_taste.inputs.dcmfile = 'dcm_taste/spm_dcm_ui_CSD_MEE_cascade_1.mat'
estimate_taste.run()

# Estimate DCM parameters for olfactory cortex
estimate_olfactory = EstimateDCM()
estimate_olfactory.inputs.dcmfile = 'dcm_olfactory/spm_dcm_ui_CSD_MEE_cascade_1.mat'
estimate_olfactory.run()

# Load the DCM matrices for taste and olfactory cortex
dcm_taste_matrix = estimate_taste.result.outputs.DCM
dcm_olfactory_matrix = estimate_olfactory.result.outputs.DCM

# Plot connectivity matrices as graphs
plt.figure(figsize=(12, 6))

# Taste Cortex Connectivity Matrix
plt.subplot(1, 2, 1)
G_taste = nx.from_numpy_array(dcm_taste_matrix, create_using=nx.DiGraph)
pos_taste = nx.spring_layout(G_taste)
nx.draw(G_taste, pos_taste, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Taste Cortex Connectivity Matrix')

# Olfactory Cortex Connectivity Matrix
plt.subplot(1, 2, 2)
G_olfactory = nx.from_numpy_array(dcm_olfactory_matrix, create_using=nx.DiGraph)
pos_olfactory = nx.spring_layout(G_olfactory)
nx.draw(G_olfactory, pos_olfactory, with_labels=True, node_size=500, node_color='lightgreen', font_size=10, font_weight='bold')
plt.title('Olfactory Cortex Connectivity Matrix')

plt.tight_layout()
plt.show()
