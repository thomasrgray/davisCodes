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
# Plotting with fully transparent background and higher DPI
fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
ax.set_facecolor('none')

# Scatter plots for data points
ax.scatter(data_condition2, [1.2] * len(data_condition2), color='red', label='Unenriched', alpha=0.7, edgecolors='white', linewidth=0.5)
ax.scatter(data_condition3, [1.8] * len(data_condition3), color='green', label='Enriched', alpha=0.7, edgecolors='white', linewidth=0.5)

# Error bars for confidence intervals
ax.errorbar(np.mean(data_condition2), 1.2, xerr=(confidence_interval_condition2[1] - confidence_interval_condition2[0]) / 2, color='red', linestyle='--', label='95% CI (Unenriched)', linewidth=1, capsize=5, capthick=1)
ax.errorbar(np.mean(data_condition3), 1.8, xerr=(confidence_interval_condition3[1] - confidence_interval_condition3[0]) / 2, color='green', linestyle='--', label='95% CI (Enriched)', linewidth=1, capsize=5, capthick=1)

# Set labels and title
ax.set_xlabel('Preference Learning Index', color='white')
ax.set_title('Confidence Intervals for Preference Learning Index', color='white')

# Set y-axis ticks and labels with reduced space
ax.set_yticks([1.2, 1.8])
ax.set_yticklabels(['Unenriched', 'Enriched'], color='white')

# Set x-axis label color to white
ax.tick_params(axis='x', colors='white')

# Set y-axis limits
ax.set_ylim([1, 2])

# Set the legend text color to white
legend = ax.legend(loc='lower right')
for text in legend.get_texts():
    text.set_color("black")

# Save the plot with fully transparent background and higher DPI
plt.savefig('output.png', bbox_inches='tight', facecolor='none', dpi=500)

# Show the plot
plt.show()