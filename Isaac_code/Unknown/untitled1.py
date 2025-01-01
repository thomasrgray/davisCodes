import os
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Function to extract trial number, solution, and lick data
def extract_data_from_file(file_path):
    lick_data = []
    solution_names = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        data_section_started = False
        
        for line in lines:
            if data_section_started:
                line_data = line.strip().split(',')
                
                if len(line_data) >= 8:
                    lick_count = int(line_data[6])
                    lick_data.append(lick_count)
                    
                    solution_name = line_data[3].strip()
                    if solution_name.isalpha():
                        solution_names.append(solution_name)
            elif line.startswith("PRESENTATION,TUBE,CONCENTRATION"):
                data_section_started = True
    
    return lick_data, solution_names



# Folder path containing the files
folder_path = '/Users/thomasgray/Desktop/BATDATA/TG2527/'

# List of filenames to process
filenames = [
    '0702TG25_pre_preference.ms8.txt',
    '0702TG26_pre_preference.ms8.txt',
    '0702TG27_pre_preference.ms8.txt',
    '0706TG25_post_preference.ms8.txt',
    '0706TG26_post_preference.ms8.txt',
    '0706TG27_post_preference.ms8.txt'
]


# Iterate through the files and extract data
for filename in filenames:
    file_path = os.path.join(folder_path, filename)
    
    lick_data, solution_names = extract_data_from_file(file_path)
    
    # Now you can use lick_data and solution_names as needed for each file
    print(f"File: {filename}")
    print("Lick Data:", lick_data)
    print("Solution Names (TUBE):", solution_names)
    print("---------------------------")

# Initialize lists to store data
all_filenames = []
all_trial_numbers = []
all_solution_names = []
all_lick_clusters = []

# Iterate through the filenames and extract data
for filename in filenames:
    file_path = folder_path + filename
    trial_numbers, solution_names, lick_data = extract_data_from_file(file_path)
    
    all_filenames.append(filename)
    all_trial_numbers.append(trial_numbers)
    all_solution_names.append(solution_names)
    all_lick_clusters.append(lick_data)

# Now you have the extracted trial numbers, solution names, and lick data in the respective lists.


# Create a Pandas DataFrame
data = {
    'Filename': all_filenames,
    'Trial Numbers': all_trial_numbers,
    'Solution Names': all_solution_names,
    'Lick Data': all_lick_clusters
}
df = pd.DataFrame(data)

# Define the number of states for the HMM
n_states = 3  # You can adjust this based on your data

def plot_hmm_results(model, lick_clusters, trial_numbers):
    plt.figure(figsize=(12, 6))
    plt.title("HMM State Transitions")
    plt.xlabel("Trial Number")
    plt.ylabel("Lick Clusters")

    # Predict hidden states
    hidden_states = model.predict(np.array(lick_clusters).reshape(-1, 1))

    # Plot the lick clusters and their corresponding hidden states
    for trial, (cluster, state) in enumerate(zip(lick_clusters, hidden_states), start=1):
        plt.scatter(trial, cluster, c=f'C{state}', label=f"Trial {trial}", s=50)

    # Plot legend with state colors
    states = list(set(hidden_states))
    states.sort()
    legend_handles = [plt.Line2D([0], [0], marker='o', color=f'C{state}', label=f"State {state}", markersize=8) for state in states]
    plt.legend(handles=legend_handles, title="States")

    plt.show()
# Fit an HMM for each file and plot the results
for filename, lick_clusters in zip(all_filenames, all_lick_clusters):
    # Remove NaN values before fitting the HMM
    lick_clusters = [value for value in lick_clusters if not np.isnan(value)]
    X = np.array(lick_clusters).reshape(-1, 1)
    
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
    model.fit(X)
    
    # Plot HMM results
    plot_hmm_results(filename, model, lick_clusters, all_trial_numbers[all_filenames.index(filename)])

    print(f"File: {filename}")
    print("HMM Model:")
    print(model)
    print("---------------------------")
