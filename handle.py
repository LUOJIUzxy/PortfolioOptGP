import pandas as pd
import os
import csv

csv_file_path = './Commodities/XAU_USD.csv'
df = pd.read_csv(csv_file_path, quoting=csv.QUOTE_NONE, escapechar='\\')

# Remove double quotes from all string columns
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].str.replace('"', '')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort dataframe by date in ascending order
df = df.sort_values('date')



save_path = f'./Commodities/XAU_USD/XAU_USD.csv'
        
# Ensure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the modified dataframe
df.to_csv(save_path, index=False)
print(f"Modified data saved to: {save_path}")