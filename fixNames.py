#Code to fix names in BAT ms8 files
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 01:25:51 2024

@author: thomas
"""

import os
import re
import fileinput

# Define the folder path containing the text files
folder_path = '/home/thomas/Desktop/Isaac/All_data'

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file
    if filename.endswith('.txt'):
        # Create the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the file and replace '0.01m' with '0'
        with fileinput.FileInput(file_path, inplace=True) as file:
            for line in file:
                # Use regular expression to remove 'm' next to a number
                line = re.sub(r'(\d)m', r'\1', line)
                print(line, end='')

import os
import fileinput

# Define the folder path containing the text files
folder_path = '/home/thomas/Desktop/Isaac/All_data'

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file
    if filename.endswith('.txt'):
        # Create the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the file and replace 'EB' with 'carvone' and 'CT' with 'cis3hex'
        with fileinput.FileInput(file_path, inplace=True) as file:
            for line in file:
                line = line.replace('EB', 'carvone')
                line = line.replace('carh20', 'carvone')
                line = line.replace('cish20', 'cis3hex')
                line = line.replace('cis3hexh20', 'cis3hex')               
                print(line, end='')



# Display the first few rows of the DataFrame
print("First 5 rows:")
print(df.head())

# Display the last few rows of the DataFrame
print("\nLast 5 rows:")
print(df.tail())

# Display basic information about the DataFrame
print("\nDataFrame info:")
print(df.info())

# Display descriptive statistics for numerical columns
print("\nDescriptive statistics:")
print(df.describe())

# Display the column names of the DataFrame
print("\nColumn names:")
print(df.columns)

# Display the shape of the DataFrame (number of rows and columns)
print("\nDataFrame shape:")
print(df.shape)



