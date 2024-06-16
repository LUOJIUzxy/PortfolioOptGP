# %%
# Setting Parameters 

import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary, set_trainable
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from matplotlib import rc
import requests
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


ticker = 'META'
ticker1 = 'AAPL'
ticker2 = 'MSFT'
ticker3 = 'GOOGL'
ticker4 = 'AMZN'
ticker5 = 'TSLA'
ticker6 = 'FB'
ticker7 = 'NVDA'
ticker8 = 'PYPL'
ticker9 = 'NFLX'
ticker10 = 'S&P500'

tickers = [ticker1]
timeframes = ['d', 'w', 'm']
timeframe_map = {'d': 'Daily', 'w': 'Weekly', 'm': 'Monthly'}

train_start_date = '2021-04-14'
train_end_date = '2023-12-29'
test_start_date = '2024-01-02'
test_end_date = '2024-06-07'

inducing_points_svgp = 20

kernel_combinations = [
    gpflow.kernels.SquaredExponential(),
    gpflow.kernels.Matern12(),
    gpflow.kernels.RationalQuadratic(),
    gpflow.kernels.Exponential(),
    gpflow.kernels.SquaredExponential() + gpflow.kernels.Matern12(),
    gpflow.kernels.Exponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.Linear(),
    gpflow.kernels.Exponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
    gpflow.kernels.SquaredExponential() * gpflow.kernels.Matern12(),
]

X_daily_tf = None
Y_daily_tf = None
X_weekly_tf = None
Y_weekly_tf = None
X_monthly_tf = None
Y_monthly_tf = None

# %%
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
rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure titles

# %%
# Funtion for Getting the data from the API
def fetch_and_save_data(ticker, period, train_start_date, test_end_date):
    # URL for the API request 640b60e3d05070.82010518
    load_dotenv()
    # Get the API token from the environment variable
    api_token = os.getenv('API_TOKEN')
    url = f'https://eodhd.com/api/eod/{ticker}.US?period={period}&api_token={api_token}&fmt=json&from={train_start_date}&to={test_end_date}'

    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)

    csv_file_path = f'./Stocks/{ticker}_EOD/{ticker}_us_{period}.csv'
    df.to_csv(csv_file_path, index=False)

for ticker in tickers:
    for timeframe in timeframes:
        fetch_and_save_data(ticker, timeframe, train_start_date, test_end_date)

# %%
#url = 'https://eodhd.com/api/eod/EUR.FOREX?order=d&api_token=640b60e3d05070.82010518&fmt=json'

#url = 'https://eodhd.com/api/eod/SOL-USD.CC?api_token=662b7114196544.78541146&order=d&fmt=json'

def convert_to_day_of_year(date):
    start_date = pd.Timestamp(train_start_date)
    return (date - start_date).days

def split_and_scale_data(df, train_start_date, train_end_date, test_start_date, test_end_date):
    df['date'] = pd.to_datetime(df['date'])

    # Training data
    train_mask = (df['date'] >= train_start_date) & (df['date'] <= train_end_date)
    train_df = df[train_mask]

    # Test data
    test_mask = (df['date'] >= test_start_date) & (df['date'] <= test_end_date)
    test_df = df[test_mask]

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
    X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float64)
    X_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float64)
    Y_test_tf = tf.convert_to_tensor(Y_test, dtype=tf.float64)
    
    return X_train_tf, Y_train_tf, X_test_tf, Y_test_tf, train_df, test_df

def normalize_and_reshape(df, _df, X_tf, Y_tf, column='open'):
    # Normalize the data
    mean = df[column].mean()
    std = df[column].std()

    Y_tf = (Y_tf - mean) / std
    
    return X_tf, Y_tf, _df['date'], mean, std

def denormalize(Y_tf, mean, std):
    return Y_tf * std + mean

def process_data_for_ticker(ticker, timeframes, train_start_date, test_end_date):
    data = {}

    for timeframe in timeframes:
        fetch_and_save_data(ticker, timeframe, train_start_date, test_end_date)

        file_path = f'./Stocks/{ticker}_EOD/{ticker}_us_{timeframe}.csv'
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].apply(convert_to_day_of_year)

        X_train_tf, Y_train_tf, X_test_tf, Y_test_tf, train_df, test_df = split_and_scale_data(df, train_start_date, train_end_date, test_start_date, test_end_date)

        X_tf_train, Y_tf_train, dates_train, mean_train, std_train = normalize_and_reshape(df, train_df, X_train_tf, Y_train_tf, column='open')

        X_tf_test, Y_tf_test, dates_test, mean_test, std_test = normalize_and_reshape(df, test_df, X_test_tf, Y_test_tf, column='open')

        data[timeframe] = (X_tf_train, Y_tf_train, X_tf_test, Y_tf_test, dates_train, dates_test, mean_train, std_train)

        print(f"{ticker} - {timeframe}: X shape {X_tf_train.shape + X_tf_test.shape}, Y shape {Y_tf_train.shape + Y_tf_test.shape}")
    
    return data

# Plotting the data
def plot_data(ticker, Y_tf_train, Y_tf_test, dates_train, dates_test, title='Open Price Over Time', denormalize_data=True, mean=0, std=1):
    if denormalize_data:
        Y_tf_train = denormalize(Y_tf_train, mean, std)
        Y_tf_test = denormalize(Y_tf_test, mean, std)
    
    dates_train = dates_train.dt.strftime('%y-%m')
    dates_test = dates_test.dt.strftime('%y-%m')

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, Y_tf_train.numpy(), label='Training Data', color='red')
    plt.plot(dates_test, Y_tf_test.numpy(), label='Test Data', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.xticks(rotation=45)
    plt.title(f'{title} Open Price Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(X_tf_train, Y_tf_train, X_tf_test, Y_tf_test, kernel_combinations):
    best_kernel = None
    best_mse = float('inf')
    best_model = None
    for kernel in kernel_combinations:
        # GPflow model
        model = gpflow.models.GPR(
            data=(X_tf_train, Y_tf_train),
            likelihood=gpflow.likelihoods.Gaussian(variance=1e-1),
            kernel=gpflow.kernels.Exponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.Linear() ,
            mean_function=gpflow.mean_functions.Polynomial(2),
        )
        # Set likelihood variance training to False
        set_trainable(model.likelihood.variance, False)

        # Optimizer
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            model.training_loss, model.trainable_variables, options=dict(maxiter=100)
        )
        #print(opt_logs)

        # gpflow.utilities.print_summary(model, "notebook")

        mean_test, variance_test = model.predict_f(X_tf_test)
        # print(mean_test)

        # Calculate the MSE on the test data
        mse_test = mean_squared_error(Y_tf_test, mean_test.numpy())
        #print(f'Kernel: {kernel}, Mean Squared Error on test data: {mse_test}')

        # Update the best kernel if the current kernel has a lower MSE
        if mse_test < best_mse:
            best_mse = mse_test
            best_kernel = kernel
            best_model = model

    return best_kernel, best_mse, best_model

def predict_combined(alpha, beta, daily_model, weekly_model, monthly_model, X):
    mean_daily, var_daily = daily_model.predict_f(X)
    mean_weekly, var_weekly = weekly_model.predict_f(X)
    mean_monthly, var_monthly = monthly_model.predict_f(X)
    combined_mean = alpha * mean_daily + beta * mean_weekly + (1 - alpha - beta) * mean_monthly
    combined_variance = alpha * var_daily + beta * var_weekly + (1 - alpha - beta) * var_monthly
    return combined_mean, combined_variance

# Function to compute the loss for given alpha and beta
def loss_fn(weights, daily_model, weekly_model, monthly_model, X, Y):
    alpha, beta = weights
    combined_mean, _ = predict_combined(alpha, beta, daily_model, weekly_model, monthly_model, X)
    mse = mean_squared_error(Y, combined_mean)
    return mse

# %%
from scipy.optimize import minimize

daily_df = pd.read_csv(f'./Stocks/{ticker}_EOD/{ticker}_us_d.csv')
weekly_df = pd.read_csv(f'./Stocks/{ticker}_EOD/{ticker}_us_w.csv')
monthly_df = pd.read_csv(f'./Stocks/{ticker}_EOD/{ticker}_us_m.csv')
for ticker in tickers:
    data = process_data_for_ticker(ticker, timeframes, train_start_date, test_end_date)

    X_daily_tf_train, Y_daily_tf_train, X_daily_tf_test, Y_daily_tf_test, dates_daily_train, dates_daily_test, mean_daily, std_daily = data['d']
    X_weekly_tf_train, Y_weekly_tf_train, X_weekly_tf_test, Y_weekly_tf_test, dates_weekly_train, dates_weekly_test,mean_weekly, std_weekly = data['w']
    X_monthly_tf_train, Y_monthly_tf_train, X_monthly_tf_test, Y_monthly_tf_test, dates_monthly_train, dates_monthly_test, mean_monthly, std_monthly = data['m']


    for timeframe in timeframes:
        X_tf_train, Y_tf_train, X_tf_test, Y_tf_test, dates_train, dates_test, mean_train,std_train = data[timeframe]
        plot_data(ticker, Y_tf_train, Y_tf_test, dates_train, dates_test, title=f'{ticker} - {timeframe_map[timeframe]}',denormalize_data=True, mean=mean_train, std=std_train)

    daily_kernel, daily_mse, daily_model = train_model(X_daily_tf_train, Y_daily_tf_train, X_daily_tf_test, Y_daily_tf_test, kernel_combinations)
    print(f'Best Kernel {ticker} Daily: {daily_kernel}, Best MSE {ticker} Daily: {daily_mse}')
    print(f'Best Model {ticker} Daily: {daily_model}')

    weekly_kernel, weekly_mse, weekly_model = train_model(X_weekly_tf_train, Y_weekly_tf_train, X_weekly_tf_test, Y_weekly_tf_test, kernel_combinations)
    print(f'Best Kernel {ticker} Weekly: {weekly_kernel}, Best MSE {ticker} Weekly: {weekly_mse}')
    print(f'Best Model {ticker} Weekly: {weekly_model}')

    monthly_kernel, monthly_mse, monthly_model = train_model(X_monthly_tf_train, Y_monthly_tf_train, X_monthly_tf_test, Y_monthly_tf_test, kernel_combinations)
    print(f'Best Kernel {ticker} Monthly: {monthly_kernel}, Best MSE {ticker} Monthly: {monthly_mse}')
    print(f'Best Model {ticker} Monthly: {monthly_model}')

    # train linear model = daily + weekly + monthly
    # Function to predict using the linear combination of the models
    X_combined_daily = np.vstack([X_daily_tf_train, X_daily_tf_test])
    Y_combined_daily = np.vstack([Y_daily_tf_train, Y_daily_tf_test])
    X_combined_weekly = np.vstack([X_weekly_tf_train, X_weekly_tf_test])
    X_combined_monthly = np.vstack([X_monthly_tf_train, X_monthly_tf_test])
    # Initial guess for alpha and beta
    initial_weights = [0.33, 0.33]

    # Bounds for alpha and beta
    bounds = [(0, 1), (0, 1)]

    # Constraint for alpha + beta <= 1
    constraints = {'type': 'ineq', 'fun': lambda x: 1 - sum(x)}

    # Optimize the weights
    result = minimize(
        lambda weights: loss_fn(weights, daily_model, weekly_model, monthly_model, X_combined_daily, Y_combined_daily),
        initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    alpha_opt, beta_opt = result.x
    print(f'Optimal alpha: {alpha_opt}, Optimal beta: {beta_opt}')

    # Predict on the combined data using the optimal weights
    combined_mean, combined_mvariance = predict_combined(alpha_opt, beta_opt, daily_model, weekly_model, monthly_model, X_combined_daily)
    print(combined_mean)
# %%

# Calculate the last day in the training data

last_day = dates_test.max()
print(type(last_day))
future_dates = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=240, freq='D')
print(future_dates)

# %%
# Generate test points for the next 240 days from the last training date

Xplot_tf = tf.convert_to_tensor(future_dates.values[:, None], dtype=tf.float64) 

X_combined_future = np.vstack([X_combined_daily, Xplot_tf])
print(X_combined_future)

# Predictions using the linear combination of the models
# Predict on the combined data using the optimal weights
f_mean, f_var = predict_combined(alpha_opt, beta_opt, daily_model, weekly_model, monthly_model, X_combined_future)


# Confidence intervals
f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(X_daily_tf_train, Y_daily_tf_train, "kx", mew=2, label="Training data")
plt.plot(Y_daily_tf_test, Y_daily_tf_test, 'bo', mew=2, label='Test Data')
plt.plot(X_combined_future, f_mean, "-", color="C0", label="Mean")
plt.plot(X_combined_future, f_lower, "--", color="C0", label="f 95% confidence")
plt.plot(X_combined_future, f_upper, "--", color="C0")
plt.fill_between(X_combined_future[:, 0], f_lower[:, 0], f_upper[:, 0], alpha=0.1, color="C0")


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

kernel_combinations = [
    gpflow.kernels.SquaredExponential(),
    gpflow.kernels.Matern12(),
    gpflow.kernels.RationalQuadratic(),
    gpflow.kernels.Exponential(),
    gpflow.kernels.SquaredExponential() + gpflow.kernels.Matern12(),
    gpflow.kernels.Exponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.Linear(),
    gpflow.kernels.Exponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
    gpflow.kernels.SquaredExponential() * gpflow.kernels.Matern12(),
]

best_kernel = None
best_mse = float('inf')
best_model = None

for kernel in kernel_combinations:

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

    mean_test, variance_test = model.predict_f(X_test_tf)
    # print(mean_test)

    # Calculate the MSE on the test data
    mse_test = mean_squared_error(Y_test, mean_test.numpy())
    print(f'Kernel: {kernel}, Mean Squared Error on test data: {mse_test}')

    # Update the best kernel if the current kernel has a lower MSE
    if mse_test < best_mse:
        best_mse = mse_test
        best_kernel = kernel
        best_model = model



# %%
# Define the SVGP model


kernel_combinations = [
    gpflow.kernels.SquaredExponential(),
    gpflow.kernels.Matern12(),
    gpflow.kernels.RationalQuadratic(),
    gpflow.kernels.Exponential(),
    gpflow.kernels.SquaredExponential() + gpflow.kernels.Matern12(),
    gpflow.kernels.Exponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.Linear(),
    gpflow.kernels.Exponential() + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
    gpflow.kernels.SquaredExponential() * gpflow.kernels.Matern12(),
    gpflow.kernels.Exponential() * gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
]

best_kernel = None
best_mse = float('inf')
best_model = None

for kernel in kernel_combinations:

    model = gpflow.models.SVGP(
        kernel=kernel,
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
    print(f'Kernel: {kernel}, Mean Squared Error on test data: {mse_test}')

    # Update the best kernel if the current kernel has a lower MSE
    if mse_test < best_mse:
        best_mse = mse_test
        best_kernel = kernel
        best_model = model

print(f'Best Kernel: {best_kernel}, Best MSE: {best_mse}')
print(f'Best Model: {best_model}')

# Calculate the last day in the training data
last_day = X_train.max()

# Generate test points for the next 240 days from the last training date
X_pred = np.linspace(last_day, last_day + 240, 240)[:, None]
X_pred_scaled = scaler.transform(X_pred)
X_combined_future = np.vstack([X_train_tf, X_test_tf, X_pred_scaled])

# Make predictions
mean, variance = best_model.predict_f(tf.convert_to_tensor(X_combined_future, dtype=tf.float64))

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

iv = getattr(best_model, "inducing_variable", None)
if iv is not None:
    plt.scatter(pd.to_datetime(train_start_date) + pd.to_timedelta(iv.Z.numpy().squeeze(), unit='D'), np.zeros_like(iv.Z.numpy().squeeze()), marker="^", color='r', label='Inducing Variables')

plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title(f'{ticker} Open Price Prediction')
plt.legend()
plt.grid(True)
plt.show()


# %%
