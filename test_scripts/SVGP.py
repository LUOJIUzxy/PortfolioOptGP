# %%
# Define Ticker and Timeframe


ticker = 'META'
timeframe = '1D'

train_start_date = '2021-04-14'
train_end_date = '2023-12-29'
test_start_date = '2024-01-02'
test_end_date = '2024-06-07'

# %%
from matplotlib import pyplot as plt
from matplotlib import rc
# import seaborn as sns
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20


rc('font', size=MEDIUM_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure titles

# %%
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Get the API token from the environment variable
api_token = os.getenv('API_TOKEN')
print(api_token)
# URL for the API request 640b60e3d05070.82010518
url = f'https://eodhd.com/api/eod/{ticker}.US?period=d&api_token={api_token}&fmt=json&from={train_start_date}&to={test_end_date}'
#url = 'https://eodhd.com/api/eod/EUR.FOREX?order=d&api_token=640b60e3d05070.82010518&fmt=json'

#url = 'https://eodhd.com/api/eod/SOL-USD.CC?api_token=662b7114196544.78541146&order=d&fmt=json'
# Fetching the data from the API
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)

csv_file_path = f'./Stocks/{ticker}_EOD/{ticker}_us_test.csv'
df.to_csv(csv_file_path, index=False)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from pandas.tseries.offsets import DateOffset

# Function to convert date to numerical value (e.g., day of the year)
def convert_to_day_of_year(date):
    start_date = pd.Timestamp(train_start_date)
    return (date - start_date).days

file_path = f'./Stocks/{ticker}_EOD/{ticker}_us_test.csv'
file_path_week = f'./Stocks/{ticker}_EOD/{ticker}_us_week.csv'
file_path_month = f'./Stocks/{ticker}_EOD/{ticker}_us_month.csv'
combined_df = pd.read_csv(file_path)

combined_df['date'] = pd.to_datetime(combined_df['date'])

# Extract 'open' prices and convert dates to day of the year
combined_df['day_of_year'] = combined_df['date'].apply(convert_to_day_of_year)

# Normalize the data
combined_df['open'] = (combined_df['open'] - combined_df['open'].mean()) / combined_df['open'].std()
Y_combined = combined_df['open'].values
X_combined = combined_df['day_of_year'].values

Y_combined_reshaped = Y_combined.reshape(-1, 1)
X_combined_reshaped = X_combined.reshape(-1, 1)

X_combined_tf = tf.convert_to_tensor(X_combined_reshaped, dtype=tf.float64)
Y_combined_tf = tf.convert_to_tensor(Y_combined_reshaped, dtype=tf.float64)

print(X_combined_tf.shape, Y_combined_reshaped.shape)


# Plot 
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.plot(combined_df['date'], Y_combined, label=f'{ticker} Open Price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.xticks(rotation=45)
plt.title(f'{ticker} Open Price Over Time')
plt.legend()
plt.grid(True)
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from pandas.tseries.offsets import DateOffset

# Function to convert date to numerical value (e.g., day of the year)
def convert_to_day_of_year(date):
    start_date = pd.Timestamp(train_start_date)
    return (date - start_date).days

file_path = f'./Stocks/{ticker}_EOD/{ticker}_us_test.csv'
file_path_week = f'./Stocks/{ticker}_EOD/{ticker}_us_week.csv'
file_path_month = f'./Stocks/{ticker}_EOD/{ticker}_us_month.csv'
combined_df = pd.read_csv(file_path)
weekly_df = pd.read_csv(file_path_week)
monthly_df = pd.read_csv(file_path_month)

combined_df['date'] = pd.to_datetime(combined_df['date'])
weekly_df['date'] = pd.to_datetime(weekly_df['date'])
monthly_df['date'] = pd.to_datetime(monthly_df['date'])

combined_df['week_start'] = combined_df['date'] - pd.to_timedelta(combined_df['date'].dt.dayofweek, unit='d')
weekly_avg = combined_df.groupby('week_start')['open'].transform('mean')
combined_df['open_adj'] = combined_df['open'] - weekly_avg

# Compute monthly average for each week
weekly_df['month_start'] = weekly_df['date'] - pd.to_timedelta(weekly_df['date'].dt.day - 1, unit='d')
monthly_avg = weekly_df.groupby('month_start')['open'].transform('mean')
weekly_df['open_adj'] = weekly_df['open'] - monthly_avg

# Merge adjusted weekly data back to daily data
combined_df = combined_df.merge(weekly_df[['date', 'open_adj']], on='date', suffixes=('', '_weekly'))

# Final adjusted daily data
combined_df['final_open'] = combined_df['open_adj'] - combined_df['open_adj_weekly']

# Convert to day of the year
combined_df['day_of_year'] = combined_df['date'].apply(convert_to_day_of_year)

# Normalize the final data
final_open_mean = combined_df['final_open'].mean()
final_open_std = combined_df['final_open'].std()
combined_df['final_open_normalized'] = (combined_df['final_open'] - final_open_mean) / final_open_std

Y_combined = combined_df['final_open_normalized'].values
X_combined = combined_df['day_of_year'].values

Y_combined_reshaped = Y_combined.reshape(-1, 1)
X_combined_reshaped = X_combined.reshape(-1, 1)

X_combined_tf = tf.convert_to_tensor(X_combined_reshaped, dtype=tf.float64)
Y_combined_tf = tf.convert_to_tensor(Y_combined_reshaped, dtype=tf.float64)

print(X_combined_tf.shape, Y_combined_reshaped.shape)


# Plot 
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.plot(combined_df['date'], Y_combined, label=f'{ticker} Open Price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.xticks(rotation=45)
plt.title(f'{ticker} Open Price Over Time')
plt.legend()
plt.grid(True)
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow

# Define the ticker and the date range

# Function to convert date to numerical value (e.g., day of the year)
def convert_to_day_of_year(date):
    start_date = pd.Timestamp(train_start_date)
    return (date - start_date).days

# Load the data
file_path = f'./Stocks/{ticker}_EOD/{ticker}_us_test.csv'
file_path_week = f'./Stocks/{ticker}_EOD/{ticker}_us_week.csv'
file_path_month = f'./Stocks/{ticker}_EOD/{ticker}_us_month.csv'

daily_df = pd.read_csv(file_path)
weekly_df = pd.read_csv(file_path_week)
monthly_df = pd.read_csv(file_path_month)

# Convert dates to datetime
daily_df['date'] = pd.to_datetime(daily_df['date'])
weekly_df['date'] = pd.to_datetime(weekly_df['date'])
monthly_df['date'] = pd.to_datetime(monthly_df['date'])

# Align daily data with weekly data
daily_df = daily_df.set_index('date').join(weekly_df.set_index('date'), rsuffix='_weekly', how='left').reset_index()
daily_df['open_adj'] = daily_df['open'] - daily_df['open_weekly']

# Align weekly data with monthly data
weekly_df = weekly_df.set_index('date').join(monthly_df.set_index('date'), rsuffix='_monthly', how='left').reset_index()
weekly_df['open_adj'] = weekly_df['open'] - weekly_df['open_monthly']

# Now, merge the adjusted weekly data back to the daily data
daily_df = daily_df.set_index('date').join(weekly_df[['date', 'open_adj']].set_index('date'), rsuffix='_weekly', how='left').reset_index()
daily_df['final_open'] = daily_df['open_adj'] - daily_df['open_adj_weekly']

# Convert to day of the year
daily_df['day_of_year'] = daily_df['date'].apply(convert_to_day_of_year)

# Normalize the final data
final_open_mean = daily_df['final_open'].mean()
final_open_std = daily_df['final_open'].std()
daily_df['final_open_normalized'] = (daily_df['final_open'] - final_open_mean) / final_open_std

# Prepare data for the model
Y_combined = daily_df['final_open_normalized'].values
X_combined = daily_df['day_of_year'].values

Y_combined_reshaped = Y_combined.reshape(-1, 1)
X_combined_reshaped = X_combined.reshape(-1, 1)

X_combined_tf = tf.convert_to_tensor(X_combined_reshaped, dtype=tf.float64)
Y_combined_tf = tf.convert_to_tensor(Y_combined_reshaped, dtype=tf.float64)

print(X_combined_tf.shape, Y_combined_reshaped.shape)

# Plot the normalized final open price
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.plot(daily_df['date'], Y_combined, label=f'{ticker} Open Price')
plt.xlabel('Date')
plt.ylabel('Normalized Open Price')
plt.title(f'{ticker} Open Price Over Time (Normalized)')
plt.legend()
plt.grid(True)
plt.show()

# %%
# huafen training and testing data
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import pandas as pd
from gpflow.utilities import print_summary, set_trainable
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Assuming combined_df is your DataFrame with columns 'date' and 'open_price'
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Split the data
train_start_date = '2021-04-14'
train_end_date = '2023-12-29'
test_start_date = '2024-01-02'
test_end_date = '2024-06-07'

# Training data
train_mask = (combined_df['date'] >= train_start_date) & (combined_df['date'] <= train_end_date)
train_df = combined_df[train_mask]

# Test data
test_mask = (combined_df['date'] >= test_start_date) & (combined_df['date'] <= test_end_date)
test_df = combined_df[test_mask]

# Prepare data for the model
X_train = (train_df['date'] - pd.to_datetime(train_start_date)).dt.days.values[:, None]
Y_train = train_df['open'].values[:, None]

X_test = (test_df['date'] - pd.to_datetime(train_start_date)).dt.days.values[:, None]
Y_test = test_df['open'].values[:, None]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to TensorFlow tensors
import tensorflow as tf
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float64)

# %%

# GPflow model
model = gpflow.models.GPR(
    data=(X_train_tf, Y_train_tf),
    likelihood=gpflow.likelihoods.Gaussian(variance=1e-1),
    kernel=gpflow.kernels.Exponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.Linear() ,
    mean_function=gpflow.mean_functions.Polynomial(2),
)
# Set likelihood variance training to False
set_trainable(model.likelihood.variance, False)

gpflow.utilities.print_summary(model, "notebook")

# Optimizer
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(
    model.training_loss, model.trainable_variables, options=dict(maxiter=100)
)


gpflow.utilities.print_summary(model, "notebook")

mean_test, variance_test = model.predict_f(X_test_tf)
# print(mean_test)

# Calculate the MSE on the test data
mse_test = mean_squared_error(Y_test, mean_test.numpy())
print(f'Mean Squared Error on test data: {mse_test}')


# Calculate the last day in the training data
last_day = X_combined.max()

print(last_day)
# %%
# Generate test points for the next 240 days from the last training date
Xplot = np.linspace(last_day, last_day + 240, 240)[:, None]
Xplot_tf = tf.convert_to_tensor(Xplot, dtype=tf.float64) 
X_combined_future = np.vstack([X_train_tf, X_test_tf, Xplot_tf])

# Predictions
f_mean, f_var = model.predict_f(X_combined_future, full_cov=False)
y_mean, y_var = model.predict_y(X_combined_future)

# Confidence intervals
f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(X_train, Y_train_tf, "kx", mew=2, label="Training data")
plt.plot(X_test, Y_test, 'bo', mew=2, label='Test Data')
plt.plot(X_combined_future, f_mean, "-", color="C0", label="Mean")
plt.plot(X_combined_future, f_lower, "--", color="C0", label="f 95% confidence")
plt.plot(X_combined_future, f_upper, "--", color="C0")
plt.fill_between(X_combined_future[:, 0], f_lower[:, 0], f_upper[:, 0], alpha=0.1, color="C0")

plt.plot(X_combined_future, y_lower, ".", color="C0", label="Y 95% confidence")
plt.plot(X_combined_future, y_upper, ".", color="C0")
plt.fill_between(X_combined_future[:, 0], y_lower[:, 0], y_upper[:, 0], alpha=0.1, color="C0")

# Set x-ticks to show labels correctly
start_date = pd.Timestamp(train_start_date)
num_labels = 48
x_ticks = np.linspace(0, 1400, num_labels)
labels = pd.date_range(start_date, periods=num_labels, freq="M").strftime("%b %Y")

plt.xticks(x_ticks, labels, rotation=45)
plt.xlabel('Date')
plt.ylabel('Normalized Open Price')
plt.title(f'GP Regression on {ticker} Open Price')
plt.legend()
plt.grid(True)
plt.show()



# %%
# SGPR model
def plot_model(model: gpflow.models.GPModel) -> None:
    X, Y = model.data
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    gpflow.utilities.print_summary(model, "notebook")

    Xplot = np.linspace(0.0, 1.0, 200)[:, None]

    y_mean, y_var = model.predict_y(Xplot, full_cov=False)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)

    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X, Y, "kx", mew=2)
    (mean_line,) = ax.plot(Xplot, y_mean, "-")
    color = mean_line.get_color()
    ax.plot(Xplot, y_lower, lw=0.1, color=color)
    ax.plot(Xplot, y_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=color, alpha=0.1
    )

    # Also plot the inducing variables if possible:
    iv = getattr(model, "inducing_variable", None)
    if iv is not None:
        ax.scatter(iv.Z, np.zeros_like(iv.Z), marker="^")

inducing_points = np.linspace(0, 360, 10)[:, None]

model = gpflow.models.SGPR(
    (X_combined_tf, Y_combined_reshaped),
    kernel=gpflow.kernels.SquaredExponential(),
    inducing_variable=inducing_points,
)
plot_model(model)
# %%
# VGP model
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

import gpflow

def plot_model(model: gpflow.models.GPModel) -> None:
    X, Y = model.data
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    gpflow.utilities.print_summary(model, "notebook")

    Xplot = np.linspace(0.0, 1.0, 200)[:, None]

    y_mean, y_var = model.predict_y(Xplot, full_cov=False)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)

    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X, Y, "kx", mew=2)
    (mean_line,) = ax.plot(Xplot, y_mean, "-")
    color = mean_line.get_color()
    ax.plot(Xplot, y_lower, lw=0.1, color=color)
    ax.plot(Xplot, y_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=color, alpha=0.1
    )

model = gpflow.models.VGP(
    (X_combined_tf, Y_combined_reshaped),
    kernel=gpflow.kernels.SquaredExponential(),
    likelihood=gpflow.likelihoods.StudentT(),
)
plot_model(model)

# %%
## SVGP model
# Define the SVGP model
from gpflow.utilities import print_summary, set_trainable

model = gpflow.models.SVGP(
    kernel=gpflow.kernels.SquaredExponential(),
    likelihood=gpflow.likelihoods.Gaussian(variance=1e-4),
    inducing_variable=np.linspace(0, 360, 120)[:, None],
    num_data=len(X_combined_tf),
)

# Set likelihood variance training to False
set_trainable(model.likelihood.variance, False)

opt = gpflow.optimizers.Scipy()
# Training the model
training_loss = model.training_loss_closure((X_combined_tf, Y_combined_tf))
opt.minimize(training_loss, model.trainable_variables, options=dict(maxiter=100))

print_summary(model, "notebook")

# %%
# Prediction
# Predict for the rest of 2024
# Calculate the last day in the training data
last_day = X_combined.max()

# Generate test points for the next 240 days from the last training date
X_pred = np.linspace(last_day, last_day + 240, 240)[:, None]

# X_pred = np.linspace(X_combined.max(), X_combined.max(), 1000)[:, None]
mean, variance = model.predict_f(X_pred)
lower = mean - 1.96 * np.sqrt(variance)
upper = mean + 1.96 * np.sqrt(variance)

plt.figure(figsize=(12, 6))
plt.plot(combined_df['date'], Y_combined, 'kx', mew=2, label='Observed Data')


plt.plot(pd.to_datetime('2023-05-18') + pd.to_timedelta(X_pred.squeeze(), unit='D'), mean, 'C0', lw=2, label='Mean Prediction')
plt.fill_between(pd.to_datetime('2023-05-18') + pd.to_timedelta(X_pred.squeeze(), unit='D'), lower.numpy().squeeze(), upper.numpy().squeeze(), color='C0', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('COINBASE Open Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# %%
# huafen training and testing data
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import pandas as pd
from gpflow.utilities import print_summary, set_trainable
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Assuming combined_df is your DataFrame with columns 'date' and 'open_price'
combined_df['date'] = pd.to_datetime(combined_df['date'])

# # Split the data
# train_start_date = '2021-04-14'
# train_end_date = '2023-12-29'
# test_start_date = '2024-01-02'
# test_end_date = '2024-06-07'

# Training data
train_mask = (combined_df['date'] >= train_start_date) & (combined_df['date'] <= train_end_date)
train_df = combined_df[train_mask]

# Test data
test_mask = (combined_df['date'] >= test_start_date) & (combined_df['date'] <= test_end_date)
test_df = combined_df[test_mask]

# Prepare data for the model
X_train = (train_df['date'] - pd.to_datetime(train_start_date)).dt.days.values[:, None]
Y_train = train_df['open'].values[:, None]

X_test = (test_df['date'] - pd.to_datetime(train_start_date)).dt.days.values[:, None]
Y_test = test_df['open'].values[:, None]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to TensorFlow tensors
import tensorflow as tf
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float64)

# %%
# Define the SVGP model

combined_kernel = gpflow.kernels.Exponential() * gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())

model = gpflow.models.SVGP(
    kernel=combined_kernel,
    likelihood=gpflow.likelihoods.Gaussian(variance=1e-4),
    # use
    inducing_variable=np.linspace(0, X_train.max(), 20)[:, None],
    num_data=len(X_train_tf),
)
gpflow.utilities.print_summary(model, "notebook")
# Set likelihood variance training to False
set_trainable(model.likelihood.variance, False)

# Initialize the optimizer
opt = gpflow.optimizers.Scipy()

# Define the training loss closure
training_loss = model.training_loss_closure((X_train_tf, Y_train_tf))

# Optimize the model parameters
opt.minimize(training_loss, model.trainable_variables, options=dict(maxiter=100))

# Print the summary of the trained model
print_summary(model, "notebook")

# Prediction phase
# Predict on the test data
mean_test, variance_test = model.predict_f(X_test_tf)
print(mean_test)

# Calculate the MSE on the test data
mse_test = mean_squared_error(Y_test, mean_test.numpy())
print(f'Mean Squared Error on test data: {mse_test}')

# Calculate the last day in the training data
last_day = X_train.max()

# Generate test points for the next 240 days from the last training date
X_pred = np.linspace(last_day, last_day + 240, 240)[:, None]
X_pred_scaled = scaler.transform(X_pred)
X_combined_future = np.vstack([X_train_tf, X_test_tf, X_pred_scaled])

# Make predictions
mean, variance = model.predict_f(tf.convert_to_tensor(X_combined_future, dtype=tf.float64))

# Calculate the confidence interval
lower = mean - 1.96 * np.sqrt(variance)
upper = mean + 1.96 * np.sqrt(variance)

num_days = len(X_test) + len(X_pred)
dates_combined = pd.date_range(start=test_start_date, periods=num_days, freq='D')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], Y_train, 'kx', mew=2, label='Training Data')
plt.plot(test_df['date'], Y_test, 'bo', mew=2, label='Test Data')
plt.plot(dates_combined, mean.numpy().flatten(), 'C0', lw=2, label='Mean Prediction')
plt.fill_between(dates_combined, lower.numpy().flatten(), upper.numpy().flatten(), color='C0', alpha=0.2, label='95% Confidence Interval')

iv = getattr(model, "inducing_variable", None)
if iv is not None:
    plt.scatter(pd.to_datetime(train_start_date) + pd.to_timedelta(iv.Z.numpy().squeeze(), unit='D'), np.zeros_like(iv.Z.numpy().squeeze()), marker="^", color='r', label='Inducing Variables')

plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title(f'{ticker} Open Price Prediction')
plt.legend()
plt.grid(True)
plt.show()


# %%
