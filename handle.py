# %%
import pandas as pd
import os
import csv

csv_file_path = './Stocks/S&P500/S&P500_us_d.csv'
df = pd.read_csv(csv_file_path, quoting=csv.QUOTE_NONE, escapechar='\\')

# Remove double quotes from all string columns
# for column in df.select_dtypes(include=['object']):
#     df[column] = df[column].str.replace('"', '')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# If you want to change the string format of the date column
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# Sort dataframe by date in ascending order
df = df.sort_values('date')



save_path = f'./Stocks/S&P500/S&P500_us_d.csv'
        
# Ensure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the modified dataframe
df.to_csv(save_path, index=False)
print(f"Modified data saved to: {save_path}")

# %%
import csv

from datetime import datetime

def convert_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        fieldnames = ['date', 'open', 'high', 'low', 'close', 'change', 'volume']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Read the header
        header = next(reader)
        
        writer.writeheader()
        for row in reader:
            date = datetime.strptime(row[0].strip('"'), '%m/%d/%Y').strftime('%Y-%m-%d')
            new_row = {
                'date': date,
                'open': row[2].strip('"'),
                'high': row[3].strip('"'),
                'low': row[4].strip('"'),
                'close': row[1].strip('"'),
                'change': row[6].strip('"'),
                'volume': row[5].strip('"') if row[5].strip('"') else '0'
            }
            writer.writerow(new_row)

def sort_csv(file_path):
    # Read the CSV file
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    
    # Sort the data
    data.sort(key=lambda row: datetime.strptime(row['date'], '%Y-%m-%d'))
    
    # Write the sorted data back to the file
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Usage

output_file = './Stocks/Index/RUT2000/RUT2000_us_d.csv'
input_file = './Stocks/Index/RUT2000/RUT2000.csv'
convert_csv(input_file, output_file)
sort_csv(output_file)
# %%
