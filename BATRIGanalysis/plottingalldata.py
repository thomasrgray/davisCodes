import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Define the data for each condition
data_condition1 = [54.06504065, -17.53246753, 13.99787911, -12.1869783, 17.55102041, 22.78761062]
data_condition2 = [0, -13.49593496, 151.7241379, -76.61169415, -47.55512943, -85.88007737]
data_condition3 = [92.57028112, 52.32273839, 141.4285714, 130.8357349]
data_condition4 = [24.3902439, -17.06924316, -25.27881041, -7.42637644]

# Combine all data
all_data = [data_condition1, data_condition2, data_condition3, data_condition4]

# Create a list of condition labels
condition_labels = ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4']

# Create an empty matrix to store p-values
num_conditions = len(all_data)
p_value_matrix = np.zeros((num_conditions, num_conditions), dtype=np.float32)

# Perform pairwise comparisons and store p-values
for i in range(num_conditions):
    for j in range(i + 1, num_conditions):
        _, p_value = stats.ttest_ind(all_data[i], all_data[j])
        p_value_matrix[i, j] = p_value

# Fill in the lower triangle of the matrix with NaNs
p_value_matrix[np.tril_indices(num_conditions, -1)] = np.nan

# Create a matrix for effect sizes (Cohen's d)
effect_size_matrix = np.zeros((num_conditions, num_conditions))

# Compute and store effect sizes (Cohen's d) for pairwise comparisons
for i in range(num_conditions):
    for j in range(i + 1, num_conditions):
        effect_size = (np.mean(all_data[i]) - np.mean(all_data[j])) / np.sqrt((np.var(all_data[i]) + np.var(all_data[j])) / 2)
        effect_size_matrix[i, j] = effect_size

# Fill in the lower triangle of the matrix with NaNs
effect_size_matrix[np.tril_indices(num_conditions, -1)] = np.nan

# Create subplots side by side
plt.figure(figsize=(14, 6))

# Plot the p-values heatmap
plt.subplot(1, 2, 1)
sns.heatmap(p_value_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, xticklabels=condition_labels,
            yticklabels=condition_labels)
plt.title('Pairwise T-Test P-Values between Conditions')

# Plot the effect sizes heatmap
plt.subplot(1, 2, 2)
sns.heatmap(effect_size_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, xticklabels=condition_labels,
            yticklabels=condition_labels)
plt.title('Pairwise Effect Sizes (Cohen\'s d) between Conditions')

# Adjust layout
plt.tight_layout()

# Print the p-values
for i in range(num_conditions):
    for j in range(i + 1, num_conditions):
        p_value = p_value_matrix[i, j]
        print(f'{condition_labels[i]} vs. {condition_labels[j]}: p = {p_value:.3f}')

# Print the effect sizes
for i in range(num_conditions):
    for j in range(i + 1, num_conditions):
        effect_size = effect_size_matrix[i, j]
        print(f'{condition_labels[i]} vs. {condition_labels[j]}: d = {effect_size:.3f}')

plt.show()

###################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define the data for each condition
data_condition1 = [54.06504065, -17.53246753, 13.99787911, -12.1869783, 17.55102041, 22.78761062]
data_condition2 = [0, -13.49593496, 151.7241379, -76.61169415, -47.55512943, -85.88007737]
data_condition3 = [92.57028112, 52.32273839, 141.4285714, 130.8357349]
data_condition4 = [24.3902439, -17.06924316, -25.27881041, -7.42637644]

# Combine all data
all_data = [data_condition1, data_condition2, data_condition3, data_condition4]

# Create a list of condition labels
condition_labels = ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4']

# Create subplots for each condition
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Linear Regression Results')

# Perform linear regression and plot for each condition
for i, data in enumerate(all_data):
    x = np.arange(len(data))  # Independent variable (e.g., time points)
    y = data  # Dependent variable (e.g., measurement values)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Plot data points
    axs[i // 2, i % 2].scatter(x, y, label='Data', color='blue')

    # Plot regression line
    regression_line = intercept + slope * x
    axs[i // 2, i % 2].plot(x, regression_line, label='Regression Line', color='red')

    axs[i // 2, i % 2].set_title(f'{condition_labels[i]}')
    axs[i // 2, i % 2].set_xlabel('Data Point')
    axs[i // 2, i % 2].set_ylabel('Value')
    axs[i // 2, i % 2].legend()

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()


#################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Define the data for each condition
data_condition1 = [54.06504065, -17.53246753, 13.99787911, -12.1869783, 17.55102041, 22.78761062]
data_condition2 = [0, -13.49593496, 151.7241379, -76.61169415, -47.55512943, -85.88007737]
data_condition3 = [92.57028112, 52.32273839, 141.4285714, 130.8357349]
data_condition4 = [24.3902439, -17.06924316, -25.27881041, -7.42637644]

# Combine all data
all_data = [data_condition1, data_condition2, data_condition3, data_condition4]

# Create a list of condition labels
condition_labels = ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4']

# Create subplots for each condition
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Linear Regression with Confidence Intervals')

# Perform linear regression and plot for each condition
for i, data in enumerate(all_data):
    x = np.arange(len(data))  # Independent variable (e.g., time points)
    y = data  # Dependent variable (e.g., measurement values)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Plot data points
    axs[i // 2, i % 2].scatter(x, y, label='Data', color='blue')

    # Plot regression line with confidence intervals
    sns.regplot(x=x, y=y, ax=axs[i // 2, i % 2], color='red', label='Regression Line')

    axs[i // 2, i % 2].set_title(f'{condition_labels[i]}')
    axs[i // 2, i % 2].set_xlabel('Data Point')
    axs[i // 2, i % 2].set_ylabel('Value')
    axs[i // 2, i % 2].legend()

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()

################################################

import numpy as np
import matplotlib.pyplot as plt

# Define the data for each condition
data_condition1 = [54.06504065, -17.53246753, 13.99787911, -12.1869783, 17.55102041, 22.78761062]
data_condition2 = [0, -13.49593496, 151.7241379, -76.61169415, -47.55512943, -85.88007737]
data_condition3 = [92.57028112, 52.32273839, 141.4285714, 130.8357349]
data_condition4 = [24.3902439, -17.06924316, -25.27881041, -7.42637644]

# Combine all data into a single array
all_data = [data_condition1, data_condition2, data_condition3, data_condition4]

# Define prior probabilities for conditions (assuming equal prior for simplicity)
prior_probabilities = np.ones(len(all_data)) / len(all_data)

# Define likelihood functions (assuming a normal distribution for simplicity)
def likelihood(data, mean, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(data - mean) ** 2 / (2 * std ** 2))

# Define observed data (you can replace this with your actual observed data)
observed_data = [10, -5, 140, -10, 15, 20]

# Calculate posterior probabilities for each condition
posterior_probabilities = []
for i, condition_data in enumerate(all_data):
    likelihoods = [likelihood(observed_data[j], np.mean(condition_data), np.std(condition_data)) for j in range(len(observed_data))]
    posterior = np.prod(likelihoods) * prior_probabilities[i]
    posterior_probabilities.append(posterior)

# Normalize the posterior probabilities
posterior_probabilities /= np.sum(posterior_probabilities)

# Condition labels
condition_labels = ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4']

# Create a bar plot of posterior probabilities
plt.figure(figsize=(8, 6))
plt.bar(condition_labels, posterior_probabilities, color='skyblue')
plt.xlabel('Conditions')
plt.ylabel('Posterior Probability')
plt.title('Posterior Probabilities of Conditions')
plt.xticks(rotation=15)

# Show the plot
plt.tight_layout()
plt.show()

####################3
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data for each condition
data_condition1 = [54.06504065, -17.53246753, 13.99787911, -12.1869783, 17.55102041, 22.78761062]
data_condition2 = [0.0, -13.49593496, 151.7241379, -76.61169415, -47.55512943, -85.88007737]
data_condition3 = [92.57028112, 52.32273839, 141.4285714, 130.8357349]
data_condition4 = [24.3902439, -17.06924316, -25.27881041, -7.42637644]

# Combine all data into a single array
all_data = np.concatenate((data_condition1, data_condition2, data_condition3, data_condition4))

# Define the observed data
observed_data = all_data  # Use all data as observed data

# Define the number of conditions
num_conditions = len(all_data)

# Define a prior distribution for condition probabilities (Dirichlet distribution)
with pm.Model() as model:
    condition_probs = pm.Dirichlet('condition_probs', a=np.ones(num_conditions))

# Define a single likelihood for all observed data
with model:
    likelihood = pm.Normal('likelihood', mu=all_data, sd=np.std(all_data), observed=observed_data)

# Perform Bayesian inference to obtain posterior probabilities
with model:
    trace = pm.sample(10000, tune=2000, target_accept=0.9)  # MCMC sampling

# Extract posterior probabilities from the trace
posterior_probs = trace['condition_probs']

# Create a violin plot to visualize posterior distributions
plt.figure(figsize=(10, 6))
sns.violinplot(data=posterior_probs, inner="stick", orient="v", palette="Set2")
plt.xlabel('Conditions')
plt.ylabel('Posterior Probability Distribution')
plt.title('Posterior Probability Distributions for Conditions')
plt.xticks(np.arange(num_conditions), [f'Condition {i+1}' for i in range(num_conditions)])
plt.tight_layout()

# Display the means of the posterior distributions
for i, mean_prob in enumerate(posterior_probs.mean(axis=0)):
    plt.text(i, -0.1, f'Mean: {mean_prob:.2f}', ha='center', va='center')

# Show the plot
plt.show()

##################################
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Define the data for each condition
data_condition1 = np.array([
    [54.06504065],
    [-17.53246753],
    [13.99787911],
    [-12.1869783],
    [17.55102041],
    [22.78761062]
])

data_condition2 = np.array([
    [0.0],
    [-13.49593496],
    [151.7241379],
    [-76.61169415],
    [-47.55512943],
    [-85.88007737]
])

data_condition3 = np.array([
    [92.57028112],
    [52.32273839],
    [141.4285714],
    [130.8357349],
    [100.8357349],
    [95.43143434]
])

data_condition4 = np.array([
    [24.3902439],
    [-17.06924316],
    [-25.27881041],
    [-7.42637644],
    [-10.2312334],
    [-5.14346677]
])

# Combine all data into a list of arrays
all_data = [data_condition1, data_condition2, data_condition3, data_condition4]

# Create labels for each condition
condition_labels = [1] * len(data_condition1) + [2] * len(data_condition2) + [3] * len(data_condition3) + [4] * len(data_condition4)

# Ensure that all data arrays have the same length
max_length = max(len(data) for data in all_data)
for i in range(len(all_data)):
    if len(all_data[i]) < max_length:
        pad_length = max_length - len(all_data[i])
        all_data[i] = np.vstack([all_data[i], np.zeros((pad_length, 1))])

# Stack data and labels
stacked_data = np.column_stack((np.vstack(all_data), np.array(condition_labels)))

# Create an HMM model for transitions from day 1 to 2
n_states = 4  # Number of hidden states (conditions)
model_day_1_to_2 = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)

# Create an HMM model for transitions from day 2 to 3
model_day_2_to_3 = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)

# Create an HMM model for transitions from day 3 to 4
model_day_3_to_4 = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)

# Create an HMM model for transitions from day 4 to 5
model_day_4_to_5 = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)

# Fit each model to the respective data
model_day_1_to_2.fit(data_condition1)
model_day_2_to_3.fit(data_condition2)
model_day_3_to_4.fit(data_condition3)
model_day_4_to_5.fit(data_condition4)

# Normalize the transition matrices
model_day_1_to_2.transmat_ /= model_day_1_to_2.transmat_.sum(axis=1)[:, np.newaxis]
model_day_2_to_3.transmat_ /= model_day_2_to_3.transmat_.sum(axis=1)[:, np.newaxis]
model_day_3_to_4.transmat_ /= model_day_3_to_4.transmat_.sum(axis=1)[:, np.newaxis]
model_day_4_to_5.transmat_ /= model_day_4_to_5.transmat_.sum(axis=1)[:, np.newaxis]

# Predict the most likely sequence of hidden states for each transition
predicted_states_day_1_to_2 = model_day_1_to_2.predict(data_condition1)
predicted_states_day_2_to_3 = model_day_2_to_3.predict(data_condition2)
predicted_states_day_3_to_4 = model_day_3_to_4.predict(data_condition3)
predicted_states_day_4_to_5 = model_day_4_to_5.predict(data_condition4)

# Plot the predicted conditions for each transition
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
for state in range(n_states):
    plt.plot(data_condition1[predicted_states_day_1_to_2 == state], label=f'Condition {state + 1}')
plt.title('Day 1 to 2')

plt.subplot(2, 2, 2)
for state in range(n_states):
    plt.plot(data_condition2[predicted_states_day_2_to_3 == state], label=f'Condition {state + 1}')
plt.title('Day 2 to 3')

plt.subplot(2, 2, 3)
for state in range(n_states):
    plt.plot(data_condition3[predicted_states_day_3_to_4 == state], label=f'Condition {state + 1}')
plt.title('Day 3 to 4')

plt.subplot(2, 2, 4)
for state in range(n_states):
    plt.plot(data_condition4[predicted_states_day_4_to_5 == state], label=f'Condition {state + 1}')
plt.title('Day 4 to 5')

plt.xlabel('Time')
plt.ylabel('Data Value')
plt.suptitle('Predicted Conditions for Different Transitions using Hidden Markov Model')
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
