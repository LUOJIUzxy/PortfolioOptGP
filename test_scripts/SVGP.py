# %%
# Time Series Forecasting with GPflow
# %%
# Time Series Forecasting with GPflow

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow

# Function to convert date to numerical value (e.g., day of the year)
def convert_to_day_of_year(date):
    return date.day_of_year

# Load the combined data
file_path = './Stocks/COINBASE_EOD/coin_us_data.csv'
combined_df = pd.read_csv(file_path)

# Convert 'date' column to datetime
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Extract 'open' prices and convert dates to day of the year
combined_df['day_of_year'] = combined_df['date'].apply(convert_to_day_of_year)
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

# GPflow model (assuming the model setup follows here)

# %%
model = gpflow.models.GPR(
    data=(X_combined_tf, Y_combined_reshaped),
    kernel=gpflow.kernels.SquaredExponential(),
    mean_function=gpflow.mean_functions.Polynomial(2),
)

# %%
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(
    model.training_loss, model.trainable_variables, options=dict(maxiter=100)
)

# %%
# Generate test points for the next 28 days
Xplot = np.linspace(1, 365, 240)[: , None]
Xplot_tf = tf.convert_to_tensor(Xplot, dtype=tf.float64)
Xplot_tf.shape

# %%
f_mean, f_var = model.predict_f(Xplot_tf, full_cov=False)
y_mean, y_var = model.predict_y(Xplot_tf)

f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)

# %%
# X_combined_reshaped is the training data ((83, 1), (83, 1))
# Y_combined_reshaped is the training data ((83, 1), (83, 1))
# btc_df_test['open'] is the test data ((28, 1), (28, 1))

plt.figure(figsize=(12, 6))
plt.plot(X_combined_tf, Y_combined_reshaped, "kx", mew=2, label="Training data")
plt.plot(Xplot_tf, f_mean, "-", color="C0", label="Mean")
plt.plot(Xplot_tf, f_lower, "--", color="C0", label="f 95percent confidence")
plt.plot(Xplot_tf, f_upper, "--", color="C0")
plt.fill_between(Xplot_tf[:, 0], f_lower[:, 0], f_upper[:, 0], alpha=0.1, color="C0")

plt.plot(Xplot_tf, y_lower, ".", color="C0", label="Y 95percent confidence")
plt.plot(Xplot_tf, y_upper, ".", color="C0")
plt.fill_between(Xplot_tf[:, 0], y_lower[:, 0], y_upper[:, 0], alpha=0.1, color="C0")

# Set x-ticks to show labels correctly

plt.legend()
# %%
# SGPR model
def plot_model(model: gpflow.models.GPModel) -> None:
    X, Y = model.data
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    gpflow.utilities.print_summary(model, "notebook")

    Xplot = np.linspace(0, 240, 200)[:, None]

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

    Xplot = np.linspace(0, 240, 200)[:, None]

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
model = gpflow.models.SVGP(
    kernel=gpflow.kernels.SquaredExponential(),
    likelihood=gpflow.likelihoods.Gaussian(),
    inducing_variable=np.linspace(0, 360, 20)[:, None],
    num_data=len(X_combined_tf),
)
opt = gpflow.optimizers.Scipy()
# Training the model
training_loss = model.training_loss_closure((X_combined_tf, Y_combined_tf))
opt.minimize(training_loss, model.trainable_variables, options=dict(maxiter=100))
gpflow.utilities.print_summary(model, "notebook")
# %%
# Prediction
# Predict for the rest of 2024
X_pred = np.linspace(X_combined.min(), X_combined.max(), 1000)[:, None]
mean, variance = model.predict_f(X_pred)
lower = mean - 1.96 * np.sqrt(variance)
upper = mean + 1.96 * np.sqrt(variance)

plt.figure(figsize=(12, 6))
plt.plot(combined_df['date'], Y_combined, 'kx', mew=2, label='Observed Data')
plt.plot(pd.to_datetime('2024-01-01') + pd.to_timedelta(X_pred.squeeze(), unit='D'), mean, 'C0', lw=2, label='Mean Prediction')
plt.fill_between(pd.to_datetime('2024-01-01') + pd.to_timedelta(X_pred.squeeze(), unit='D'), lower.numpy().squeeze(), upper.numpy().squeeze(), color='C0', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('COINBASE Open Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# %%
