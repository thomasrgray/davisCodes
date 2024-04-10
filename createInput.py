#Automatically creates input file for BAT Reader Code
import os

# Function to extract information from the file name
def extract_info(file_name):
    # Split the file name by underscore and period
    parts = file_name.split('_')
    date = "2024/" + parts[0][:2] + "/" + parts[0][2:4]  # Extract date
    animal = parts[0][4:]  # Extract animal identifier
    condition, notes = "B" if "pre" in parts[1] else "A", "pre" if "pre" in parts[1] else "post"  # Extract condition and notes
    return animal, date, condition, notes

# Directory containing the files
directory = "/home/thomas/Desktop/Isaac/Ortho/Ortho" #rewrite to be your directory

# List to store the file information
file_info_list = []

# Iterate over files in the directory
for file_name in os.listdir(directory):
    if file_name.endswith(".txt"):
        animal, date, condition, notes = extract_info(file_name)
        file_info_list.append((animal, date, condition, notes))

# Write the file information to a text file
output_file = "/home/thomas/Desktop/Isaac/Ortho/Ortho/input_file/stink_inputfile.txt" #rewrite to be your directory
with open(output_file, "w") as file:
    # Write headers
    file.write("Animal\tDate\tCondition\tNotes\n")
    # Write data rows
    for info in file_info_list:
        file.write("\t".join(info) + "\n")

print(f"Output saved to {output_file}")
