import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#Percent increase from pre to post for Unenriched Paired Odor
data_condition1 = [54.06504065,
-17.53246753,
13.99787911,
-12.1869783,
17.55102041,
22.78761062]
#Percent increase from pre to post for Unenriched Unaired Odor
data_condition2 = [0,
-13.49593496,
151.7241379,
-76.61169415,
-47.55512943,
-85.88007737]
#Percent increase from pre to post for Enriched Paired Odor
data_condition3 = [92.57028112,
52.32273839,
141.4285714,
130.8357349]
#Percent increase from pre to post for Enriched Unpaired Odor
data_condition4 = [24.3902439,
-17.06924316,
-25.27881041,
-7.42637644]

# Calculate sample statistics for each condition
sample_mean_condition1 = np.mean(data_condition1)
sample_std_condition1 = np.std(data_condition1, ddof=1)  # Use ddof=1 for sample standard deviation
sample_size_condition1 = len(data_condition1)
z_critical_condition1 = stats.norm.ppf(0.975)  # 95% confidence level, two-tailed
margin_of_error_condition1 = z_critical_condition1 * (sample_std_condition1 / np.sqrt(sample_size_condition1))
confidence_interval_condition1 = (sample_mean_condition1 - margin_of_error_condition1, sample_mean_condition1 + margin_of_error_condition1)

sample_mean_condition2 = np.mean(data_condition2)
sample_std_condition2 = np.std(data_condition2, ddof=1)
sample_size_condition2 = len(data_condition2)
z_critical_condition2 = stats.norm.ppf(0.975)
margin_of_error_condition2 = z_critical_condition2 * (sample_std_condition2 / np.sqrt(sample_size_condition2))
confidence_interval_condition2 = (sample_mean_condition2 - margin_of_error_condition2, sample_mean_condition2 + margin_of_error_condition2)

sample_mean_condition3 = np.mean(data_condition3)
sample_std_condition3 = np.std(data_condition3, ddof=1)
sample_size_condition3 = len(data_condition3)
z_critical_condition3 = stats.norm.ppf(0.975)
margin_of_error_condition3 = z_critical_condition3 * (sample_std_condition3 / np.sqrt(sample_size_condition3))
confidence_interval_condition3 = (sample_mean_condition3 - margin_of_error_condition3, sample_mean_condition3 + margin_of_error_condition3)

sample_mean_condition4 = np.mean(data_condition4)
sample_std_condition4 = np.std(data_condition4, ddof=1)
sample_size_condition4 = len(data_condition4)
z_critical_condition4 = stats.norm.ppf(0.975)
margin_of_error_condition4 = z_critical_condition4 * (sample_std_condition4 / np.sqrt(sample_size_condition4))
confidence_interval_condition4 = (sample_mean_condition4 - margin_of_error_condition4, sample_mean_condition4 + margin_of_error_condition4)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot for data points with labels
ax.scatter(data_condition1, [1] * len(data_condition1), color='blue', label=None, alpha=0.7)
ax.scatter(data_condition2, [2] * len(data_condition2), color='red', label=None, alpha=0.7)
ax.scatter(data_condition3, [3] * len(data_condition3), color='green', label=None, alpha=0.7)
ax.scatter(data_condition4, [4] * len(data_condition4), color='purple', label=None, alpha=0.7)

# Create vertical lines for confidence intervals
ax.errorbar(sample_mean_condition1, 1, xerr=margin_of_error_condition1, color='blue', linestyle='--', label=f'95% CI (Condition 1): [{confidence_interval_condition1[0]:.2f}, {confidence_interval_condition1[1]:.2f}]')
ax.errorbar(sample_mean_condition2, 2, xerr=margin_of_error_condition2, color='red', linestyle='--', label=f'95% CI (Condition 2): [{confidence_interval_condition2[0]:.2f}, {confidence_interval_condition2[1]:.2f}]')
ax.errorbar(sample_mean_condition3, 3, xerr=margin_of_error_condition3, color='green', linestyle='--', label=f'95% CI (Condition 3): [{confidence_interval_condition3[0]:.2f}, {confidence_interval_condition3[1]:.2f}]')
ax.errorbar(sample_mean_condition4, 4, xerr=margin_of_error_condition4, color='purple', linestyle='--', label=f'95% CI (Condition 4): [{confidence_interval_condition4[0]:.2f}, {confidence_interval_condition4[1]:.2f}]')

# Set y-axis ticks and labels
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(['Unenriched Paired Odor', 'Unenriched Unpaired Odor', 'Enriched Paired Odor', 'Enriched Unpaired Odor'])

# Set labels and title
ax.set_xlabel('Percent Change in Average Licking')
ax.set_title('Confidence Intervals for Each Condition')

# Add a legend with confidence intervals only
ax.legend()

# Show the plot
plt.show()
