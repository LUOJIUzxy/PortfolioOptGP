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
