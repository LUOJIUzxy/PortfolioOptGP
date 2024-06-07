# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from pandas.tseries.offsets import DateOffset

# Function to convert date to numerical value (e.g., day of the year)
def convert_to_day_of_year(date):
    start_date = pd.Timestamp('2021-04-14')
    return (date - start_date).days

# Load the combined data
file_path = './Stocks/COINBASE_EOD/coin_us_eod.csv'
combined_df = pd.read_csv(file_path)

# Convert 'date' column to datetime
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Extract 'open' prices and convert dates to day of the year
combined_df['day_of_year'] = combined_df['date'].apply(convert_to_day_of_year)

# Normalize the data
combined_df['open'] = (combined_df['open'] - combined_df['open'].mean()) / combined_df['open'].std()
Y_combined = combined_df['open'].values
X_combined = combined_df['day_of_year'].values

# Reshape to 2D arrays as required by GPflow
Y_combined_reshaped = Y_combined.reshape(-1, 1)
X_combined_reshaped = X_combined.reshape(-1, 1)

# Convert to TensorFlow tensors
X_combined_tf = tf.convert_to_tensor(X_combined_reshaped, dtype=tf.float64)
Y_combined_tf = tf.convert_to_tensor(Y_combined_reshaped, dtype=tf.float64)

# Display the combined shapes
print(X_combined_tf.shape, Y_combined_reshaped.shape)

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(combined_df['date'], Y_combined, label='COINBASE Open Price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('COINBASE Open Price Over Time')
plt.legend()
plt.grid(True)
plt.show()

# %%

# GPflow model
model = gpflow.models.GPR(
    data=(X_combined_tf, Y_combined_reshaped),
    kernel=gpflow.kernels.SquaredExponential(),
    mean_function=gpflow.mean_functions.Polynomial(2),
)

# Optimizer
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(
    model.training_loss, model.trainable_variables, options=dict(maxiter=100)
)
# Calculate the last day in the training data
last_day = X_combined.max()

print(last_day)
# %%
# Generate test points for the next 240 days from the last training date
Xplot = np.linspace(last_day, last_day + 245, 245)[:, None]
Xplot_tf = tf.convert_to_tensor(Xplot, dtype=tf.float64)

# Predictions
f_mean, f_var = model.predict_f(Xplot_tf, full_cov=False)
y_mean, y_var = model.predict_y(Xplot_tf)

# Confidence intervals
f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(X_combined_tf, Y_combined_reshaped, "kx", mew=2, label="Training data")
plt.plot(Xplot_tf, f_mean, "-", color="C0", label="Mean")
plt.plot(Xplot_tf, f_lower, "--", color="C0", label="f 95% confidence")
plt.plot(Xplot_tf, f_upper, "--", color="C0")
plt.fill_between(Xplot_tf[:, 0], f_lower[:, 0], f_upper[:, 0], alpha=0.1, color="C0")

plt.plot(Xplot_tf, y_lower, ".", color="C0", label="Y 95% confidence")
plt.plot(Xplot_tf, y_upper, ".", color="C0")
plt.fill_between(Xplot_tf[:, 0], y_lower[:, 0], y_upper[:, 0], alpha=0.1, color="C0")

# Set x-ticks to show labels correctly
start_date = pd.Timestamp('2021-04-14')
num_labels = 60
x_ticks = np.linspace(0, 1400, num_labels)
labels = pd.date_range(start_date, periods=num_labels, freq="M").strftime("%b %Y")

plt.xticks(x_ticks, labels, rotation=45)
plt.xlabel('Date')
plt.ylabel('Normalized Open Price')
plt.title('GP Regression on COINBASE Open Price')
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
X_train = (train_df['date'] - pd.to_datetime('2021-04-14')).dt.days.values[:, None]
Y_train = train_df['open'].values[:, None]

X_test = (test_df['date'] - pd.to_datetime('2021-04-14')).dt.days.values[:, None]
Y_test = test_df['open'].values[:, None]

# Convert to TensorFlow tensors
import tensorflow as tf
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float64)

# %%
# Define the SVGP model
model = gpflow.models.SVGP(
    kernel=gpflow.kernels.RationalQuadratic(),
    likelihood=gpflow.likelihoods.Gaussian(variance=1e-4),
    inducing_variable=np.linspace(0, X_train.max(), 1000)[:, None],
    num_data=len(X_train_tf),
)

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

# Calculate the MSE on the test data
mse_test = mean_squared_error(Y_test, mean_test.numpy())
print(f'Mean Squared Error on test data: {mse_test}')

# Calculate the last day in the training data
last_day = X_train.max()

# Generate test points for the next 240 days from the last training date
X_pred = np.linspace(last_day, last_day + 240, 240)[:, None]

# Make predictions
mean, variance = model.predict_f(X_pred)

# Calculate the confidence interval
lower = mean - 1.96 * np.sqrt(variance)
upper = mean + 1.96 * np.sqrt(variance)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], Y_train, 'kx', mew=2, label='Training Data')
plt.plot(test_df['date'], Y_test, 'bo', mew=2, label='Test Data')
plt.plot(pd.to_datetime('2021-04-14') + pd.to_timedelta(X_pred.squeeze(), unit='D'), mean, 'C0', lw=2, label='Mean Prediction')
plt.fill_between(pd.to_datetime('2021-04-14') + pd.to_timedelta(X_pred.squeeze(), unit='D'), lower.numpy().squeeze(), upper.numpy().squeeze(), color='C0', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('COINBASE Open Price Prediction')
plt.legend()
plt.grid(True)
plt.show()


# %%
