# %%
# Load X data:COINBASE EOD
from datetime import datetime
import pandas as pd

file_path = '../test_data/Forex/EUR_EOD/eur_usd_eod.csv'
df = pd.read_csv(file_path)
# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Get the most recent date
most_recent_date = df['date'].max()

# Calculate the date six months before the most recent date
six_months_before = most_recent_date - pd.DateOffset(months=6)

# Filter the data for the last six months
recent_six_months_data = df[df['date'] >= six_months_before]

# Split the data into separate DataFrames for each month
monthly_data = [recent_six_months_data[recent_six_months_data['date'].dt.month == month] for month in range(1, 7)]

# Adjust the months based on the most recent date's month
start_month = most_recent_date.month
monthly_data = [recent_six_months_data[recent_six_months_data['date'].dt.month == ((start_month - i - 1) % 12 + 1)] for i in range(6)]

# Save each month's data to a separate CSV file
for i, month_df in enumerate(monthly_data):
    month = (start_month - i - 1) % 12 + 1
    year = (most_recent_date.year - 1) if month > start_month else most_recent_date.year
    month_df.to_csv(f'../test_data/Forex/EUR_EOD/EUR_month_{year}_{month}.csv', index=False)

monthly_data[0].head(), monthly_data[1].head(), monthly_data[2].head(), monthly_data[3].head(), monthly_data[4].head(), monthly_data[5].head()


# %%
from datetime import datetime
import pandas as pd

file_path = '../test_data/Crypto/BTC_EOD/btc_usd_eod.csv'
df = pd.read_csv(file_path)
# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Get the most recent date
most_recent_date = df['date'].max()

# Calculate the date six months before the most recent date
six_months_before = most_recent_date - pd.DateOffset(months=6)

# Filter the data for the last six months
recent_six_months_data = df[df['date'] >= six_months_before]

# Split the data into separate DataFrames for each month
monthly_data = [recent_six_months_data[recent_six_months_data['date'].dt.month == month] for month in range(1, 7)]

# Adjust the months based on the most recent date's month
start_month = most_recent_date.month
monthly_data = [recent_six_months_data[recent_six_months_data['date'].dt.month == ((start_month - i - 1) % 12 + 1)] for i in range(6)]

# Save each month's data to a separate CSV file
for i, month_df in enumerate(monthly_data):
    month = (start_month - i - 1) % 12 + 1
    year = (most_recent_date.year - 1) if month > start_month else most_recent_date.year
    month_df.to_csv(f'../test_data/Crypto/BTC_EOD/BTC_month_{year}_{month}.csv', index=False)

monthly_data[0].head(), monthly_data[1].head(), monthly_data[2].head(), monthly_data[3].head(), monthly_data[4].head(), monthly_data[5].head()

# %%
import pandas as pd
# Load the uploaded COINBASE data
coinbase_file_path = '../test_data/Stocks/COINBASE_EOD/COIN_month_2024_3.csv'
coinbase_df = pd.read_csv(coinbase_file_path)

file_path = '../test_data/Crypto/BTC_EOD/BTC_month_2024_3.csv'
df = pd.read_csv(file_path) 

# Convert 'date' column to datetime in both DataFrames
coinbase_df['date'] = pd.to_datetime(coinbase_df['date'])
df['date'] = pd.to_datetime(df['date'])

# Merge both dataframes on 'date' to align BTC open prices with COINBASE open prices
merged_df = pd.merge(df, coinbase_df, on='date', suffixes=('_btc', '_coin'))

# Extract the aligned 'open' prices
X = merged_df['open_coin'].values
Y = merged_df['open_btc'].values

# Reshape to 2D arrays if needed
X_reshaped = X.reshape(-1, 1)
Y_reshaped = Y.reshape(-1, 1)

# Display the new shapes
X_reshaped.shape, Y_reshaped.shape

# %%
import pandas as pd


# Process COINBASE data for the first four months of 2024
for i in range(1, 5):
    # Assuming 'df' contains the BTC data as initially loaded and reversed
    btc_file_path = f'../test_data/Crypto/BTC_EOD/BTC_month_2024_{i}.csv'
    print(btc_file_path)
    btc_df = pd.read_csv(btc_file_path)
    btc_df['date'] = pd.to_datetime(btc_df['date'])
    btc_df = btc_df.iloc[::-1]  # Reverse BTC data to have dates in ascending order

    # Load the COINBASE data
    coinbase_file_path = f'../test_data/Stocks/COINBASE_EOD/COIN_month_2024_{i}.csv'
    coinbase_df = pd.read_csv(coinbase_file_path)
    coinbase_df['date'] = pd.to_datetime(coinbase_df['date'])

    # Merge the reversed BTC data with COINBASE data
    merged_df = pd.merge(btc_df, coinbase_df, on='date', suffixes=('_btc', '_coin'))

    # Extract the aligned 'open' prices
    Y = merged_df['open_coin'].values
    X = merged_df['open_btc'].values

    # Reshape to 2D arrays if needed
    Y_reshaped = Y.reshape(-1, 1)
    X_reshaped = X.reshape(-1, 1)

    # Display the new shapes and first few rows
    print(f"Month {i} - X_reshaped shape: {X_reshaped.shape}, Y_reshaped shape: {Y_reshaped.shape}")
    print(merged_df.head())

    # Optional: save the merged dataframe for each month if needed
    merged_df.to_csv(f'../test_data/Merged/Merged_BTC_COIN_month_2024_{i}.csv', index=False)
   


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gpflow

for i in range(1, 5):
    df_merged = pd.read_csv(f'../test_data/Merged/Merged_BTC_COIN_month_2024_{i}.csv')
    # Extract the aligned 'open' prices
    Y = df_merged['open_coin'].values
    X = df_merged['open_btc'].values

    # Reshape to 2D arrays if needed
    Y_reshaped = Y.reshape(-1, 1)
    X_reshaped = X.reshape(-1, 1)

    # Display the extracted arrays
    X, Y
    plt.plot(X, Y, "kx", mew=2)
    plt.show()

# %%
# Initialize a list to store the combined dataframes
combined_df_list = []

# Load and combine the four merged CSV files
for i in range(1, 5):
    # Load the merged CSV file
    merged_file_path = f'../test_data/Merged/Merged_BTC_COIN_month_2024_{i}.csv'
    merged_df = pd.read_csv(merged_file_path)
    
    # Append the dataframe to the list
    combined_df_list.append(merged_df)

# Concatenate all the dataframes
combined_df = pd.concat(combined_df_list, ignore_index=True)

# Extract the aligned 'open' prices
Y_combined = combined_df['open_coin'].values
X_combined = combined_df['open_btc'].values

# Reshape to 2D arrays if needed
Y_combined_reshaped = Y_combined.reshape(-1, 1)
X_combined_reshaped = X_combined.reshape(-1, 1)

# Display the combined shapes
X_combined_reshaped.shape, Y_combined_reshaped.shape

plt.plot(X_combined_reshaped, Y_combined_reshaped, "kx", mew=2)
plt.show()
# %%
model = gpflow.models.GPR(
    data=(X_combined_reshaped, Y_combined_reshaped),
    kernel=gpflow.kernels.SquaredExponential(),
    mean_function=gpflow.mean_functions.Linear(),
)

# %%
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(
    model.training_loss, model.trainable_variables, options=dict(maxiter=100)
)

# %%
# Load Test Data 
btc_file_path = '../test_data/Crypto/BTC_EOD/BTC_month_2024_5.csv'
btc_df_test = pd.read_csv(btc_file_path)
btc_df_test['date'] = pd.to_datetime(btc_df_test['date'])
btc_df_test = btc_df_test.iloc[::-1]

model.predict_f(btc_df_test['open'].values.reshape(-1, 1))

model.predict_y(btc_df_test['open'].values.reshape(-1, 1))
# %%
f_mean, f_var = model.predict_f(btc_df_test['open'].values.reshape(-1, 1), full_cov=False)
y_mean, y_var = model.predict_y(btc_df_test['open'].values.reshape(-1, 1))

f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)

# %%
#((28,), (28,))
btc_df_test['open'].shape, f_mean.shape, f_lower.shape, f_upper.shape, y_mean.shape, y_lower.shape, y_upper.shape
Y_combined_reshaped.shape, X_combined_reshaped.shape
# %%
# X_combined_reshaped is the training data ((83, 1), (83, 1))
# Y_combined_reshaped is the training data ((83, 1), (83, 1))
# btc_df_test['open'] is the test data ((28, 1), (28, 1))

plt.figure(figsize=(12, 6))
plt.plot(X_combined_reshaped, Y_combined_reshaped, "kx", mew=2, label="Training data")
plt.plot(btc_df_test['open'], f_mean, "-", color="C0", label="Mean")
plt.plot(btc_df_test['open'], f_lower, "--", color="C0", label="f 95percent confidence")
plt.plot(btc_df_test['open'], f_upper, "--", color="C0")
plt.fill_between(btc_df_test['open'], f_lower[:, 0], f_upper[:, 0], alpha=0.1, color="C0")

plt.plot(btc_df_test['open'], y_lower, ".", color="C0", label="Y 95percent confidence")
plt.plot(btc_df_test['open'], y_upper, ".", color="C0")
plt.fill_between(btc_df_test['open'], y_lower[:, 0], y_upper[:, 0], alpha=0.1, color="C0")
plt.legend()
# %%
