import scipy.stats as stats

def calculate_d_prime(HR, FAR):
    # Avoid division by zero or one, as it can lead to undefined results
    if HR == 1:
        HR = 1 - (1 / (2 * n))
    if FAR == 0:
        FAR = 1 / (2 * n)

    # Calculate Z-scores for HR and FAR
    Z_HR = stats.norm.ppf(HR)
    Z_FAR = stats.norm.ppf(FAR)

    # Calculate d' (discrimination preference)
    d_prime = Z_HR - Z_FAR

    return d_prime

# Example usage:
HR = 0.8  # Hit Rate
FAR = 0.2  # False Alarm Rate

n = 100  # Total number of trials (you should replace this with the actual number of trials)

d_prime = calculate_d_prime(HR, FAR)
print(f"d' (discrimination preference) = {d_prime:.2f}")

# Define the initial value and the final value
initial_value = 12.2
final_value = 24.08

# Calculate the percent increase
percent_increase = ((final_value - initial_value) / initial_value) * 100

print(f"Paired Enriched Percent Increase: {percent_increase:.2f}%")

# Define the initial value and the final value
initial_value = 20.95
final_value = 19.61666667

# Calculate the percent increase
percent_increase = ((final_value - initial_value) / initial_value) * 100

print(f"Unpaired Enriched Percent Increase: {percent_increase:.2f}%")

# Define the initial value and the final value
initial_value = 16.6861413
final_value = 31.04807692

# Calculate the percent increase
percent_increase = ((final_value - initial_value) / initial_value) * 100

print(f"Paired Enriched No 0 Percent Increase: {percent_increase:.2f}%")

# Define the initial value and the final value
initial_value = 30.66725343
final_value = 26.43528743

# Calculate the percent increase
percent_increase = ((final_value - initial_value) / initial_value) * 100

print(f"Unpaired Enriched No 0 Percent Increase: {percent_increase:.2f}%")


####################33

# Define the initial value and the final value
initial_value = 21.58
final_value = 23.11

# Calculate the percent increase
percent_increase = ((final_value - initial_value) / initial_value) * 100

print(f"Paired Unenriched Percent Increase: {percent_increase:.2f}%")

# Define the initial value and the final value
initial_value = 25.81
final_value = 13.66

# Calculate the percent increase
percent_increase = ((final_value - initial_value) / initial_value) * 100

print(f"Unpaired Unenriched Percent Increase: {percent_increase:.2f}%")

# Define the initial value and the final value
initial_value = 25.56
final_value = 29.57

# Calculate the percent increase
percent_increase = ((final_value - initial_value) / initial_value) * 100

print(f"Paired Enriched No 0 Percent Increase: {percent_increase:.2f}%")

# Define the initial value and the final value
initial_value = 28.21
final_value = 20.95

# Calculate the percent increase
percent_increase = ((final_value - initial_value) / initial_value) * 100

print(f"Unpaired Enriched No 0 Percent Increase: {percent_increase:.2f}%")

