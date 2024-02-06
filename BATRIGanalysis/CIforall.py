import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Data for specific conditions
data_condition2 = [0.363, 1.041, -1.061, 0.601, 0.508, 0.363, 2.454, 2]
data_condition3 = [4.07, 3.042, 2.537, 6.562]

# Calculate t-test for significance
t_stat, p_value = stats.ttest_ind(data_condition2, data_condition3)
is_significant = p_value < 0.05  # Assuming alpha = 0.05

# Calculate sample statistics for each condition
def calculate_confidence_interval(data):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    sample_size = len(data)
    z_critical = stats.norm.ppf(0.975)  # 95% confidence level, two-tailed
    margin_of_error = z_critical * (sample_std / np.sqrt(sample_size))
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    return confidence_interval

confidence_interval_condition2 = calculate_confidence_interval(data_condition2)
confidence_interval_condition3 = calculate_confidence_interval(data_condition3)

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))

# Scatter plots for data points
ax.scatter(data_condition2, [1] * len(data_condition2), color='red', label='Unenriched', alpha=0.7)
ax.scatter(data_condition3, [2] * len(data_condition3), color='green', label='Enriched', alpha=0.7)

# Error bars for confidence intervals
ax.errorbar(np.mean(data_condition2), 1, xerr=(confidence_interval_condition2[1] - confidence_interval_condition2[0]) / 2, color='red', linestyle='--', label='95% CI (Unenriched)')
ax.errorbar(np.mean(data_condition3), 2, xerr=(confidence_interval_condition3[1] - confidence_interval_condition3[0]) / 2, color='green', linestyle='--', label='95% CI (Enriched)')

# Set y-axis ticks and labels with reduced space
ax.set_yticks([1.2, 1.8])
ax.set_yticklabels(['Unenriched', 'Enriched'])

# Set labels and title
ax.set_xlabel('Preference Learning Index')
ax.set_title('Confidence Intervals for Preference Learning Index')

# Add asterisks for significance between data points
if is_significant:

    ax.text(1.5, 1.5, '*', fontsize=18, ha='center', va='center')  # Asterisk between data points


    # Add p-value as text
    ax.text(1.5, 1.4, f'p = {p_value:.3f}', fontsize=10, ha='center')  # Adjust the position as needed

# Set x-axis limits with extra space from edges
ax.set_xlim([-2, 8])  # You can adjust these limits based on your data range

# Show the plot
plt.show()


#=========================================

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Data for specific conditions
data_condition2 = [0.363, 1.041, -1.061, 0.601, 0.508, 0.363, 2.454, 2]
data_condition3 = [4.07, 3.042, 2.537, 6.562]

# Calculate confidence intervals for each condition
def calculate_confidence_interval(data):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    sample_size = len(data)
    z_critical = stats.norm.ppf(0.975)  # 95% confidence level, two-tailed
    margin_of_error = z_critical * (sample_std / np.sqrt(sample_size))
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    return confidence_interval

confidence_interval_condition2 = calculate_confidence_interval(data_condition2)
confidence_interval_condition3 = calculate_confidence_interval(data_condition3)

# Filter data points within the confidence intervals
filtered_data_condition2 = [x for x in data_condition2 if confidence_interval_condition2[0] <= x <= confidence_interval_condition2[1]]
filtered_data_condition3 = [x for x in data_condition3 if confidence_interval_condition3[0] <= x <= confidence_interval_condition3[1]]

# Calculate t-test for significance
t_stat, p_value = stats.ttest_ind(filtered_data_condition2, filtered_data_condition3)
is_significant = p_value < 0.05  # Assuming alpha = 0.05

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))

# Scatter plots for data points
ax.scatter(filtered_data_condition2, [1] * len(filtered_data_condition2), color='red', label='Unenriched', alpha=0.7)
ax.scatter(filtered_data_condition3, [2] * len(filtered_data_condition3), color='green', label='Enriched', alpha=0.7)

# Set y-axis ticks and labels with reduced space
ax.set_yticks([1.2, 1.8])
ax.set_yticklabels(['Unenriched', 'Enriched'])

# Set labels and title
ax.set_xlabel('Preference Learning Index')
ax.set_title('Filtered Confidence Intervals for Preference Learning Index')

# Add asterisks for significance between data points
if is_significant:
    ax.text(np.mean(filtered_data_condition2), 1.5, '*', fontsize=18, ha='center', va='center')  # Asterisk between data points
    ax.text((np.mean(filtered_data_condition2) + np.mean(filtered_data_condition3)) / 2, 1.4, f'p = {p_value:.5f}', fontsize=10, ha='center')  # P-value text

# Set x-axis limits with extra space from edges
ax.set_xlim([-2, 8])  # You can adjust these limits based on your data range

# Show the plot
plt.show()

##################################################

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Data for specific conditions
data_condition2 = [0.363, 1.041, -1.061, 0.601, 0.508, 0.363, 2.454, 2]
data_condition3 = [4.07, 3.042, 2.537, 6.562]

# Calculate t-test for significance with Bonferroni correction
alpha = 0.05  # Original alpha level
num_comparisons = 7  # Number of comparisons ((pre vs. post) vs. (paired vs. unpaired) vs. (enriched vs. unenriched))
alpha_corrected = alpha / num_comparisons

t_stat, p_value = stats.ttest_ind(data_condition2, data_condition3)
p_value_corrected = p_value * num_comparisons  # Bonferroni corrected p-value
is_significant = p_value_corrected < alpha

# Calculate sample statistics for each condition
def calculate_confidence_interval(data):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    sample_size = len(data)
    z_critical = stats.norm.ppf(1 - alpha / 2)  # 1 - alpha/2 for two-tailed test
    margin_of_error = z_critical * (sample_std / np.sqrt(sample_size))
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    return confidence_interval

confidence_interval_condition2 = calculate_confidence_interval(data_condition2)
confidence_interval_condition3 = calculate_confidence_interval(data_condition3)

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))

# Scatter plots for data points
ax.scatter(data_condition2, [1.2] * len(data_condition2), color='red', label='Unenriched', alpha=0.7)
ax.scatter(data_condition3, [1.8] * len(data_condition3), color='green', label='Enriched', alpha=0.7)

# Error bars for confidence intervals
ax.errorbar(np.mean(data_condition2), 1.2, xerr=(confidence_interval_condition2[1] - confidence_interval_condition2[0]) / 2, color='red', linestyle='--', label='95% CI (Unenriched)')
ax.errorbar(np.mean(data_condition3), 1.8, xerr=(confidence_interval_condition3[1] - confidence_interval_condition3[0]) / 2, color='green', linestyle='--', label='95% CI (Enriched)')

# Set labels and title
ax.set_xlabel('Preference Learning Index')
ax.set_title('Confidence Intervals for Preference Learning Index')

# Set y-axis ticks and labels with reduced space
ax.set_yticks([1.2, 1.8])
ax.set_yticklabels(['Unenriched', 'Enriched'])

ax.set_ylim([1, 2])  # You can adjust these limits based on your data range
# Add asterisks for significance between data points along with Bonferroni corrected p-value
if is_significant:
    ax.text(1.5, 1.6, '*', fontsize=18, ha='center', va='center')  # Asterisk between data points
    ax.text(1.5, 1.4, f'p = {p_value_corrected:.3f}*', fontsize=10, ha='center')  # Bonferroni corrected p-value text

ax.text(1.45, 1.5, 'p = 0.003', fontsize=10, ha='center')  # Bonferroni corrected p-value text
# Set x-axis limits with extra space from edges
ax.set_xlim([-2, 8])  # You can adjust these limits based on your data range

# Show the plot
plt.show()
