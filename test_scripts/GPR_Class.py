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
from scipy.optimize import minimize

# Setting plot styles
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20
rc('font', size=MEDIUM_SIZE)
rc('axes', titlesize=BIGGER_SIZE)
rc('axes', labelsize=BIGGER_SIZE)
rc('xtick', labelsize=MEDIUM_SIZE)
rc('ytick', labelsize=BIGGER_SIZE)
rc('legend', fontsize=MEDIUM_SIZE)
rc('figure', titlesize=MEDIUM_SIZE)

class StockPredictor:
    def __init__(self, tickers, train_start_date, train_end_date, test_start_date, test_end_date, kernel_combinations, inducing_points_svgp=20):
        self.tickers = tickers
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.kernel_combinations = kernel_combinations
        self.inducing_points_svgp = inducing_points_svgp
        self.initial_weights = [0.33, 0.33]
        self.bounds = [(0, 1), (0, 1)]
        self.constraints = {'type': 'ineq', 'fun': lambda x: 1 - sum(x)}

    def fetch_and_save_data(self, ticker, period):
        load_dotenv()
        api_token = os.getenv('API_TOKEN')
        url = f'https://eodhd.com/api/eod/{ticker}.US?period={period}&api_token={api_token}&fmt=json&from={self.train_start_date}&to={self.test_end_date}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        csv_file_path = f'./Stocks/{ticker}_EOD/{ticker}_us_{period}.csv'
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df.to_csv(csv_file_path, index=False)

    def convert_to_day_of_year(self, date):
        start_date = pd.Timestamp(self.train_start_date)
        return (date - start_date).days

    def normalize_and_reshape(self, df, column='open'):
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / std
        Y = df[column].values
        X = df['day_of_year'].values
        Y_reshaped = Y.reshape(-1, 1)
        X_reshaped = X.reshape(-1, 1)
        X_tf = tf.convert_to_tensor(X_reshaped, dtype=tf.float64)
        Y_tf = tf.convert_to_tensor(Y_reshaped, dtype=tf.float64)
        return X_tf, Y_tf, df['date'], mean, std

    def denormalize(self, Y_tf, mean, std):
        return Y_tf * std + mean

    def process_data(self, ticker, period):
        self.fetch_and_save_data(ticker, period)
        file_path = f'./Stocks/{ticker}_EOD/{ticker}_us_{period}.csv'
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].apply(self.convert_to_day_of_year)
        X_tf, Y_tf, dates, mean, std = self.normalize_and_reshape(df)
        return X_tf, Y_tf, dates, mean, std

    def plot_data(self, X_tf, Y_tf, dates, title, mean, std, filename):
        dates_formatted = dates.dt.strftime('%y-%m')
        Y_tf = self.denormalize(Y_tf, mean, std)
        plt.figure(figsize=(12, 6))
        plt.plot(dates_formatted, Y_tf.numpy(), label=title)
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.xticks(rotation=45)
        plt.title(f'{title} Open Price Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def train_model(self, X_tf, Y_tf):
        best_kernel = None
        best_mse = float('inf')
        best_model = None
        for kernel in self.kernel_combinations:
            model = gpflow.models.GPR(data=(X_tf, Y_tf), kernel=kernel, mean_function=gpflow.mean_functions.Constant())
            set_trainable(model.likelihood.variance, False)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
            mean_test, _ = model.predict_f(X_tf)
            mse_test = mean_squared_error(Y_tf, mean_test.numpy())
            if mse_test < best_mse:
                best_mse = mse_test
                best_kernel = kernel
                best_model = model
        return best_kernel, best_mse, best_model

    def predict_combined(self, alpha, beta, daily_model, weekly_model, monthly_model, X):
        mean_daily, var_daily = daily_model.predict_f(X)
        mean_weekly, var_weekly = weekly_model.predict_f(X)
        mean_monthly, var_monthly = monthly_model.predict_f(X)
        combined_mean = alpha * mean_daily + beta * mean_weekly + (1 - alpha - beta) * mean_monthly
        combined_variance = alpha * var_daily + beta * var_weekly + (1 - alpha - beta) * var_monthly
        return combined_mean, combined_variance

    def loss_fn(self, weights, daily_model, weekly_model, monthly_model, X, Y):
        alpha, beta = weights
        combined_mean, _ = self.predict_combined(alpha, beta, daily_model, weekly_model, monthly_model, X)
        mse = mean_squared_error(Y, combined_mean)
        return mse

    def generate_future_dates(self, df, days=240):
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        future_df = pd.DataFrame({'date': future_dates})
        future_df['day_of_year'] = future_df['date'].apply(self.convert_to_day_of_year)
        X_pred = future_df['day_of_year'].values
        X_pred_reshaped = X_pred.reshape(-1, 1)
        X_pred_tf = tf.convert_to_tensor(X_pred_reshaped, dtype=tf.float64)
        return X_pred_tf

    def plot_pred_data(self, X_daily_tf, Y_daily_tf, X_combined_future, f_mean, f_lower, f_upper, title, mean, std, filename):
        Y_daily_tf = self.denormalize(Y_daily_tf, mean, std)
        f_mean = self.denormalize(f_mean, mean, std)
        f_lower = self.denormalize(f_lower, mean, std)
        f_upper = self.denormalize(f_upper, mean, std)
        plt.figure(figsize=(12, 6))
        plt.plot(X_daily_tf, Y_daily_tf, "kx", mew=2, label="Training data")
        plt.plot(X_combined_future, f_mean, "-", color="C0", label="Mean")
        plt.plot(X_combined_future, f_lower, "--", color="C0", label="f 95% confidence")
        plt.plot(X_combined_future, f_upper, "--", color="C0")
        plt.fill_between(X_combined_future[:, 0], f_lower[:, 0], f_upper[:, 0], alpha=0.1, color="C0")
        start_date = pd.Timestamp(self.train_start_date)
        num_labels = 48
        x_ticks = np.linspace(0, 1400, num_labels)
        labels = pd.date_range(start_date, periods=num_labels, freq="M").strftime("%b %Y")
        plt.xticks(x_ticks, labels, rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Normalized Open Price')
        plt.title(f'GP Regression on {title} Open Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def optimize_weights(self, daily_model, weekly_model, monthly_model, X_tf, Y_tf):
        result = minimize(lambda weights: self.loss_fn(weights, daily_model, weekly_model, monthly_model, X_tf, Y_tf), self.initial_weights, bounds=self.bounds, constraints=self.constraints, method='SLSQP')
        return result.x

    def run(self, timeframes):
        for ticker in self.tickers:
            data = {}
            for timeframe in timeframes:
                X_tf, Y_tf, dates, mean, std = self.process_data(ticker, timeframe)
                data[timeframe] = (X_tf, Y_tf, dates, mean, std)
                self.plot_data(X_tf, Y_tf, dates, title=f'{ticker} - {timeframe}', mean=mean, std=std, filename=f'{ticker}_{timeframe}.png')
            
            X_daily_tf, Y_daily_tf, _, mean_daily, std_daily = data['d']
            X_weekly_tf, Y_weekly_tf, _, mean_weekly, std_weekly = data['w']
            X_monthly_tf, Y_monthly_tf, _, mean_monthly, std_monthly = data['m']

            daily_kernel, daily_mse, daily_model = self.train_model(X_daily_tf, Y_daily_tf)
            weekly_kernel, weekly_mse, weekly_model = self.train_model(X_weekly_tf, Y_weekly_tf)
            monthly_kernel, monthly_mse, monthly_model = self.train_model(X_monthly_tf, Y_monthly_tf)

            alpha_opt, beta_opt = self.optimize_weights(daily_model, weekly_model, monthly_model, X_daily_tf, Y_daily_tf)

            X_combined_future = np.vstack([X_daily_tf, self.generate_future_dates(pd.read_csv(f'./Stocks/{ticker}_EOD/{ticker}_us_d.csv'), days=240)])

            f_mean, f_var = self.predict_combined(alpha_opt, beta_opt, daily_model, weekly_model, monthly_model, X_combined_future)

            f_lower = f_mean - 1.96 * np.sqrt(f_var)
            f_upper = f_mean + 1.96 * np.sqrt(f_var)

            self.plot_pred_data(X_daily_tf, Y_daily_tf, X_combined_future, f_mean, f_lower, f_upper, title=ticker, mean=mean_daily, std=std_daily, filename=f'{ticker}_predict.png')

# Parameters
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

tickers = [ticker1, ticker2, ticker3]

train_start_date = '2021-04-14'
train_end_date = '2023-12-29'
test_start_date = '2024-01-02'
test_end_date = '2024-06-07'

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

timeframes = ['d', 'w', 'm']

predictor = StockPredictor(tickers, train_start_date, train_end_date, test_start_date, test_end_date, kernel_combinations)
predictor.run(timeframes)
