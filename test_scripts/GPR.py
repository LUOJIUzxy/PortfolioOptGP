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
        # self.fetch_and_save_data(ticker, period)
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
            model = gpflow.models.GPR(data=(X_tf, Y_tf), likelihood=gpflow.likelihoods.Gaussian(variance=1e-1), kernel=kernel, mean_function=gpflow.mean_functions.Polynomial(2))
            set_trainable(model.likelihood.variance, False)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
            mean_test, _ = model.predict_f(X_tf)
            mse_test = mean_squared_error(Y_tf, mean_test.numpy())
            if mse_test < best_mse:
                best_mse = mse_test
                best_kernel = kernel
                best_model = model
            
        # print(best_kernel)
        # print_summary(best_model)
        return best_kernel, best_mse, best_model
    
    def train_svgp_model(self, X_tf, Y_tf, last_date):
        best_kernel = None
        best_mse = float('inf')
        best_model = None
        for kernel in self.kernel_combinations:
            print(last_date)
            inducing_points = np.linspace(0, last_date, self.inducing_points_svgp)[:, None][0]
            print(inducing_points)
            svgp_model = gpflow.models.SVGP(kernel=kernel, likelihood=gpflow.likelihoods.Gaussian(variance=1e-4), inducing_variable=X_tf[:self.inducing_points_svgp])
             # Set likelihood variance training to False
            set_trainable(svgp_model.likelihood.variance, False)
            opt = gpflow.optimizers.Scipy()
            training_loss = svgp_model.training_loss_closure((X_tf, Y_tf))
            opt.minimize(training_loss, svgp_model.trainable_variables, options=dict(maxiter=100))
            mean_test_svgp, _ = svgp_model.predict_f(X_tf)
            mse_test_svgp = mean_squared_error(Y_tf, mean_test_svgp.numpy())
            if mse_test_svgp < best_mse:
                best_mse = mse_test_svgp
                best_kernel = kernel
                best_model = svgp_model
        return best_kernel, best_mse, best_model

    def predict_combined(self, alpha, beta, daily_model, weekly_model, monthly_model, X):
        f_mean_daily, f_var_daily = daily_model.predict_f(X, full_cov=False)
        f_mean_weekly, f_var_weekly = weekly_model.predict_f(X, full_cov=False)
        f_mean_monthly, f_var_monthly = monthly_model.predict_f(X, full_cov=False)

        y_mean_daily, y_var_daily = daily_model.predict_y(X)
        y_mean_weekly, y_var_weekly = weekly_model.predict_y(X)
        y_mean_monthly, y_var_monthly = monthly_model.predict_y(X)
        f_combined_mean = alpha * f_mean_daily + beta * f_mean_weekly + (1 - alpha - beta) * f_mean_monthly
        f_combined_variance = alpha * f_var_daily + beta * f_var_weekly + (1 - alpha - beta) * f_var_monthly

        y_combined_mean = alpha * y_mean_daily + beta * y_mean_weekly + (1 - alpha - beta) * y_mean_monthly
        y_combined_variance = alpha * y_var_daily + beta * y_var_weekly + (1 - alpha - beta) * y_var_monthly


        return f_combined_mean, f_combined_variance, y_combined_mean, y_combined_variance

    def loss_fn(self, weights, daily_model, weekly_model, monthly_model, X, Y):
        alpha, beta = weights
        f_combined_mean, f_combined_variance, y_combined_mean, y_combined_variance = self.predict_combined(alpha, beta, daily_model, weekly_model, monthly_model, X)
        mse = mean_squared_error(Y, f_combined_mean)
        return mse

    def generate_future_dates(self, df, period='d', total_days=90):
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max()

        # Calculate the number of future dates based on the period
        if period == 'd':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=total_days, freq='D')
        elif period == 'w':
            num_weeks = total_days // 7
            future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=num_weeks, freq='W')
        elif period == 'm':
            num_months = total_days // 30
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_months, freq='M')
        else:
            raise ValueError("Period must be 'd', 'w', or 'm'")

        future_df = pd.DataFrame({'date': future_dates})
        future_df['day_of_year'] = future_df['date'].apply(self.convert_to_day_of_year)
        X_pred = future_df['day_of_year'].values
        X_pred_reshaped = X_pred.reshape(-1, 1)
        X_pred_tf = tf.convert_to_tensor(X_pred_reshaped, dtype=tf.float64)

        print(X_pred_tf.shape)
        return X_pred_tf
    
    def upsample_predictions(dates, predictions, period='d'):
        upsampled_predictions = []
        if period == 'w':
            for date, pred in zip(dates, predictions):
                upsampled_predictions.extend([pred] * 7)  # Repeat for 7 days
        elif period == 'm':
            for date, pred in zip(dates, predictions):
                days_in_month = pd.Period(date.strftime('%Y-%m')).days_in_month
                upsampled_predictions.extend([pred] * days_in_month)  # Repeat for days in the month
        else:
            upsampled_predictions = predictions  # No upsampling needed for daily
        return np.array(upsampled_predictions)

    def plot_pred_data(self, X_daily_tf, Y_daily_tf, X_combined_future, f_mean, f_lower, f_upper, y_mean, y_lower, y_upper, title, mean, std, filename, best_model):
        Y_daily_tf = self.denormalize(Y_daily_tf, mean, std)
        f_mean = self.denormalize(f_mean, mean, std)
        f_lower = self.denormalize(f_lower, mean, std)
        f_upper = self.denormalize(f_upper, mean, std)
        y_mean = self.denormalize(y_mean, mean, std)
        y_lower = self.denormalize(y_lower, mean, std)
        y_upper = self.denormalize(y_upper, mean, std)
        plt.figure(figsize=(12, 6))
        plt.plot(X_daily_tf, Y_daily_tf, "kx", mew=2, label="Training data")
        plt.plot(X_combined_future, f_mean, "-", color="C0", label="Predicted f Mean")
        plt.plot(X_combined_future, f_lower, "--", color="C0", label="f 95% confidence")
        plt.plot(X_combined_future, f_upper, "--", color="C0")
        plt.plot(X_combined_future, y_mean, "-", color="C0", label="Predicted Y Mean")
        plt.plot(X_combined_future, y_lower, ":", color="C1", label="y 95% confidence")
        plt.plot(X_combined_future, y_upper, ":", color="C1")
        plt.fill_between(X_combined_future[:, 0], f_lower[:, 0], f_upper[:, 0], alpha=0.1, color="C0")
        plt.fill_between(X_combined_future[:, 0], y_lower[:, 0], y_upper[:, 0], alpha=0.1, color="C1")
        start_date = pd.Timestamp(self.train_start_date)
        num_labels = 48
        x_ticks = np.linspace(0, 1400, num_labels)
        labels = pd.date_range(start_date, periods=num_labels, freq="M").strftime("%b %Y")

        # if 'SVGP' in filename:
        #     iv = getattr(best_model, "inducing_variable", None)
        #     if iv is not None:
        #         plt.scatter(pd.to_datetime(train_start_date) + pd.to_timedelta(iv.Z.numpy().squeeze(), unit='D'), np.zeros_like(iv.Z.numpy().squeeze()), marker="^", color='r', label='Inducing Variables')

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
                self.plot_data(X_tf, Y_tf, dates, title=f'{ticker} - {timeframe}', mean=mean, std=std, filename=f'./plots/{ticker}_{timeframe}.png')
            
            X_daily_tf, Y_daily_tf, _, mean_daily, std_daily = data['d']
            X_weekly_tf, Y_weekly_tf, _, mean_weekly, std_weekly = data['w']
            X_monthly_tf, Y_monthly_tf, _, mean_monthly, std_monthly = data['m']

            daily_kernel, daily_mse, daily_model = self.train_model(X_daily_tf, Y_daily_tf)
            weekly_kernel, weekly_mse, weekly_model = self.train_model(X_weekly_tf, Y_weekly_tf)
            monthly_kernel, monthly_mse, monthly_model = self.train_model(X_monthly_tf, Y_monthly_tf)
            
            print(f'{ticker} - Best Kernel Daily: {daily_kernel}, Best MSE Daily: {daily_mse}')
            print(f'{ticker} - Best Kernel Weekly: {weekly_kernel}, Best MSE Weekly: {weekly_mse}')
            print(f'{ticker} - Best Kernel Monthly: {monthly_kernel}, Best MSE Monthly: {monthly_mse}')

            alpha_opt, beta_opt = self.optimize_weights(daily_model, weekly_model, monthly_model, X_daily_tf, Y_daily_tf)
            
            X_combined_daily = np.vstack([X_daily_tf, self.generate_future_dates(pd.read_csv(f'./Stocks/{ticker}_EOD/{ticker}_us_d.csv'), total_days=90)])
            X_combined_weekly = np.vstack([X_weekly_tf, self.generate_future_dates(pd.read_csv(f'./Stocks/{ticker}_EOD/{ticker}_us_w.csv'), total_days=90, period='w')])
            X_combined_monthly = np.vstack([X_monthly_tf, self.generate_future_dates(pd.read_csv(f'./Stocks/{ticker}_EOD/{ticker}_us_m.csv'), total_days=90, period='m')])
            
            f_mean_daily, f_var_daily, y_mean_daily, y_var_daily = self.predict_combined(1, 0, daily_model, weekly_model, monthly_model, X_combined_daily)
            f_mean_weekly, f_var_weekly, y_mean_weekly, y_var_weekly = self.predict_combined(0, 1, daily_model, weekly_model, monthly_model, X_combined_weekly)
            f_mean_monthly, f_var_monthly, y_mean_monthly, y_var_monthly = self.predict_combined(0, 0, daily_model, weekly_model, monthly_model, X_combined_monthly)

            print(f_mean_daily.shape, f_mean_weekly.shape, f_mean_monthly.shape)

            # Upsample weekly and monthly predictions to daily frequency
      

            f_lower_daily = f_mean_daily - 1.96 * np.sqrt(f_var_daily)
            f_upper_daily = f_mean_daily + 1.96 * np.sqrt(f_var_daily)

            f_lower_weekly = f_mean_weekly - 1.96 * np.sqrt(f_var_weekly)
            f_upper_weekly = f_mean_weekly + 1.96 * np.sqrt(f_var_weekly)

            f_lower_monthly = f_mean_monthly - 1.96 * np.sqrt(f_var_monthly)
            f_upper_monthly = f_mean_monthly + 1.96 * np.sqrt(f_var_monthly)

            y_lower_daily = y_mean_daily - 1.96 * np.sqrt(y_var_daily)
            y_upper_daily = y_mean_daily + 1.96 * np.sqrt(y_var_daily)

            y_lower_weekly = y_mean_weekly - 1.96 * np.sqrt(y_var_weekly)
            y_upper_weekly = y_mean_weekly + 1.96 * np.sqrt(y_var_weekly)

            y_lower_monthly = y_mean_monthly - 1.96 * np.sqrt(y_var_monthly)
            y_upper_monthly = y_mean_monthly + 1.96 * np.sqrt(y_var_monthly)


            self.plot_pred_data(X_daily_tf, Y_daily_tf, X_combined_daily, f_mean_daily, f_lower_daily, f_upper_daily, y_mean_daily, y_lower_daily, y_upper_daily, title=ticker, mean=mean_daily, std=std_daily, filename=f'./plots/{ticker}_GPR_predict_daily.png', best_model=daily_model)
            self.plot_pred_data(X_weekly_tf, Y_weekly_tf, X_combined_weekly, f_mean_weekly, f_lower_weekly, f_upper_weekly, y_mean_weekly, y_lower_weekly, y_upper_weekly, title=ticker, mean=mean_weekly, std=std_weekly, filename=f'./plots/{ticker}_GPR_predict_weekly.png', best_model=weekly_model)
            self.plot_pred_data(X_monthly_tf, Y_monthly_tf, X_combined_monthly, f_mean_monthly, f_lower_monthly, f_upper_monthly, y_mean_monthly, y_lower_monthly, y_upper_monthly, title=ticker, mean=mean_monthly, std=std_monthly, filename=f'./plots/{ticker}_GPR_predict_monthly.png', best_model=daily_model)
           
            # print(f'Optimal weights for {ticker} GPR: alpha = {alpha_opt}, beta = {beta_opt}')

            # last_date = dates.apply(self.convert_to_day_of_year).max()
            # daily_kernel_svgp, daily_mse_svgp, daily_model_svgp = self.train_svgp_model(X_daily_tf, Y_daily_tf, last_date)
            # weekly_kernel_svgp, weekly_mse_svgp, weekly_model_svgp = self.train_svgp_model(X_weekly_tf, Y_weekly_tf, last_date)
            # monthly_kernel_svgp, monthly_mse_svgp, monthly_model_svgp = self.train_svgp_model(X_monthly_tf, Y_monthly_tf, last_date)

            # alpha_opt_svgp, beta_opt_svgp = self.optimize_weights(daily_model_svgp, weekly_model_svgp, monthly_model_svgp, X_daily_tf, Y_daily_tf)
            # model = alpha_opt_svgp * daily_model_svgp + beta_opt_svgp * weekly_kernel_svgp + (1 - alpha_opt_svgp - beta_opt_svgp) * monthly_model_svgp
            # f_mean_svgp, f_var_svgp = self.predict_combined(alpha_opt_svgp, beta_opt_svgp, daily_model_svgp, weekly_model_svgp, monthly_model_svgp, X_combined_future)

            # f_lower_svgp = f_mean_svgp - 1.96 * np.sqrt(f_var_svgp)
            # f_upper_svgp = f_mean_svgp + 1.96 * np.sqrt(f_var_svgp)

            # self.plot_pred_data(X_daily_tf, Y_daily_tf, X_combined_future, f_mean_svgp, f_lower_svgp, f_upper_svgp, title=ticker, mean=mean_daily, std=std_daily, filename=f'./plots/{ticker}_SVGP_predict.png', best_model=daily_model_svgp)
            # print(f'Optimal weights for {ticker} (SVGP): alpha = {alpha_opt_svgp}, beta = {beta_opt_svgp}')

# Parameters
ticker1 = 'AAPL'
ticker2 = 'MSFT'
ticker3 = 'NFLX'
ticker4 = 'META'
ticker5 = 'TSLA'
ticker6 = 'FB'
ticker7 = 'NVDA'
ticker8 = 'PYPL'
ticker9 = 'BAC'
ticker10 = 'ARM'
ticker11 = 'S&P500'

tickers = [ticker3, ticker4]
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
inducing_points_svgp = 20
predictor = StockPredictor(tickers, train_start_date, train_end_date, test_start_date, test_end_date, kernel_combinations, inducing_points_svgp)
predictor.run(timeframes)
