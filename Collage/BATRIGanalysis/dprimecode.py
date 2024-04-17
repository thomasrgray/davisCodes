import numpy as np
import scipy.stats as stats

# Define hit rates and false alarm rates for 6 animals (replace with your data)
hit_rates = [0.8,	0.766666667,	0.833333333,	1	,1,	0.695652174] #pre eb
false_alarm_rates = [0.8,	0.733333333,	0.633333333,	1,	1,	1] #pre ct

# Initialize an empty list to store d' values
d_prime_values = []

# Calculate d' for each animal
for i in range(len(hit_rates)):
    # Calculate Z-scores for hit rate and false alarm rate
    z_hit = stats.norm.ppf(hit_rates[i])
    z_false_alarm = stats.norm.ppf(false_alarm_rates[i])
    
    # Calculate d' for the animal
    d_prime = z_hit - z_false_alarm
    
    # Append d' value to the list
    d_prime_values.append(d_prime)

# Print d' values for each animal
for i, d_prime in enumerate(d_prime_values):
    print(f"Animal {i+1}: d' = {d_prime:.4f}")

import numpy as np
import scipy.stats as stats

# Define hit rates and false alarm rates for 6 animals (replace with your data)
hit_rates = [0.633333333,	0.633333333,	0.9,	0.652173913,	0.913043478,	0.913043478] #pre eb
false_alarm_rates = [0.533333333,	0.566666667,	0.533333333,	0.565217391,	1,	0.956521739] #pre ct

# Initialize an empty list to store d' values
d_prime_values = []

# Calculate d' for each animal
for i in range(len(hit_rates)):
    # Calculate Z-scores for hit rate and false alarm rate
    z_hit = stats.norm.ppf(hit_rates[i])
    z_false_alarm = stats.norm.ppf(false_alarm_rates[i])
    
    # Calculate d' for the animal
    d_prime = z_hit - z_false_alarm
    
    # Append d' value to the list
    d_prime_values.append(d_prime)

# Print d' values for each animal
for i, d_prime in enumerate(d_prime_values):
    print(f"Animal {i+1}: d' = {d_prime:.4f}")
    
################

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define hit rates and false alarm rates for the first set of animals (replace with your data)
hit_rates_pre_eb = [0.8, 0.766666667, 0.833333333, 1, 1, 0.695652174]
false_alarm_rates_pre_ct = [0.8, 0.733333333, 0.633333333, 1, 1, 1]

# Calculate d' values for the first set of animals
d_prime_values_pre = []
for i in range(len(hit_rates_pre_eb)):
    z_hit = stats.norm.ppf(hit_rates_pre_eb[i])
    z_false_alarm = stats.norm.ppf(false_alarm_rates_pre_ct[i])
    d_prime = z_hit - z_false_alarm
    d_prime_values_pre.append(d_prime)

# Define hit rates and false alarm rates for the second set of animals (replace with your data)
hit_rates_post_eb = [0.633333333, 0.633333333, 0.9, 0.652173913, 0.913043478, 0.913043478]
false_alarm_rates_post_ct = [0.533333333, 0.566666667, 0.533333333, 0.565217391, 1, 0.956521739]

# Calculate d' values for the second set of animals
d_prime_values_post = []
for i in range(len(hit_rates_post_eb)):
    z_hit = stats.norm.ppf(hit_rates_post_eb[i])
    z_false_alarm = stats.norm.ppf(false_alarm_rates_post_ct[i])
    d_prime = z_hit - z_false_alarm
    d_prime_values_post.append(d_prime)

# Create a bar plot to compare d' values for both sets of animals
plt.figure(figsize=(10, 6))
animals = [f'Animal {i+1}' for i in range(len(hit_rates_pre_eb))]

plt.bar(animals, d_prime_values_pre, color='blue', label='Pre Training')
plt.bar(animals, d_prime_values_post, color='red', alpha=0.7, label='Post Training')
plt.xlabel('Animals')
plt.ylabel("d' Values")
plt.title('Comparison of d\' Values for Two Sets of Animals')
plt.legend()
plt.grid(True)

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

######## enriched d'
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define hit rates and false alarm rates for the first set of animals (replace with your data)
hit_rates_pre_eb = [0.8,	0.766666667,	0.533333333,	0.766666667]
false_alarm_rates_pre_ct = [0.633333333,	0.7,	0.566666667,	0.833333333]

# Calculate d' values for the first set of animals
d_prime_values_pre = []
for i in range(len(hit_rates_pre_eb)):
    z_hit = stats.norm.ppf(hit_rates_pre_eb[i])
    z_false_alarm = stats.norm.ppf(false_alarm_rates_pre_ct[i])
    d_prime = z_hit - z_false_alarm
    d_prime_values_pre.append(d_prime)

# Define hit rates and false alarm rates for the second set of animals (replace with your data)
hit_rates_post_eb = [0.866666667,	0.666666667,	0.666666667,	0.866666667]
false_alarm_rates_post_ct = [0.833333333,	0.733333333,	0.566666667,	0.8]

# Calculate d' values for the second set of animals
d_prime_values_post = []
for i in range(len(hit_rates_post_eb)):
    z_hit = stats.norm.ppf(hit_rates_post_eb[i])
    z_false_alarm = stats.norm.ppf(false_alarm_rates_post_ct[i])
    d_prime = z_hit - z_false_alarm
    d_prime_values_post.append(d_prime)

# Create a bar plot to compare d' values for both sets of animals
plt.figure(figsize=(10, 6))
animals = [f'Animal {i+1}' for i in range(len(hit_rates_pre_eb))]

plt.bar(animals, d_prime_values_pre, color='blue', label='Pre Training')
plt.bar(animals, d_prime_values_post, color='red', alpha=0.7, label='Post Training')
plt.xlabel('Animals')
plt.ylabel("d' Values")
plt.title('Comparison of d\' Values for Two Sets of Animals')
plt.legend()
plt.grid(True)

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()