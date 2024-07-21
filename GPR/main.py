# main.py
import gpflow
from gpflow.utilities import print_summary, set_trainable
import numpy as np
import tensorflow as tf
from data_handler import DataHandler
from model_trainer import ModelTrainer
from predictor import Predictor
from optimizer import Optimizer
from visualizer import Visualizer

class StockPredictor:
    def __init__(self, tickers, train_start_date, train_end_date, test_start_date, test_end_date, kernel_combinations, lambda_=0.01, predict_Y='return'):
        self.tickers = tickers
        self.data_handler = DataHandler(train_start_date, train_end_date, test_start_date, test_end_date)
        self.model_trainer = ModelTrainer(kernel_combinations)
        self.predictor = Predictor()
        self.optimizer = Optimizer(lambda_)
        self.visualizer = Visualizer()
        self.predict_Y = predict_Y

    def run(self, timeframes):
        for ticker in self.tickers:
            data = {}
            for timeframe in timeframes:
                X_tf, Y_tf, dates, mean, std = self.data_handler.process_data(ticker, timeframe, self.predict_Y)
                data[timeframe] = (X_tf, Y_tf, dates, mean, std)
                self.visualizer.plot_data(X_tf, Y_tf, dates, title=f'{ticker} - {timeframe}', mean=mean, std=std, filename=f'../plots/{ticker}_{timeframe}.png')
            
            X_daily_tf, Y_daily_tf, _, mean_daily, std_daily = data['d']
            X_weekly_tf, Y_weekly_tf, _, mean_weekly, std_weekly = data['w']
            X_monthly_tf, Y_monthly_tf, _, mean_monthly, std_monthly = data['m']

            # Train models
            daily_kernel, daily_mse, daily_model = self.model_trainer.train_model(X_daily_tf, Y_daily_tf)
            weekly_kernel, weekly_mse, weekly_model = self.model_trainer.train_model(X_weekly_tf, Y_weekly_tf)
            monthly_kernel, monthly_mse, monthly_model = self.model_trainer.train_model(X_monthly_tf, Y_monthly_tf)

            print(f'{ticker} - Best Kernel Daily: {daily_kernel}, Best MSE Daily: {daily_mse}')
            print_summary(daily_model)
            print(f'{ticker} - Best Kernel Weekly: {weekly_kernel}, Best MSE Weekly: {weekly_mse}')
            print_summary(weekly_model)
            print(f'{ticker} - Best Kernel Monthly: {monthly_kernel}, Best MSE Monthly: {monthly_mse}')
            print_summary(monthly_model)

            # Make predictions
            f_mean_daily, f_var_daily, y_mean_daily, y_var_daily = self.predictor.predict_single(daily_model, X_daily_tf)
            f_mean_weekly, f_var_weekly, y_mean_weekly, y_var_weekly = self.predictor.predict_single(weekly_model, X_weekly_tf)
            f_mean_monthly, f_var_monthly, y_mean_monthly, y_var_monthly = self.predictor.predict_single(monthly_model, X_monthly_tf)

            # Upsample predictions
            f_mean_weekly_upsampled = self.predictor.upsample_predictions(X_daily_tf, X_weekly_tf, f_mean_weekly, period='w')
            f_mean_monthly_upsampled = self.predictor.upsample_predictions(X_daily_tf, X_monthly_tf, f_mean_monthly, period='m')

            # Optimize weights
            alpha_opt, beta_opt = self.optimizer.optimize_weights(Y_daily_tf, f_mean_daily, f_mean_weekly_upsampled, f_mean_monthly_upsampled)
            print(f'Optimal weights for {ticker} GPR: alpha = {alpha_opt}, beta = {beta_opt}')

            # Generate future dates and make predictions
            X_combined_daily = np.vstack([X_daily_tf, self.data_handler.generate_future_dates(ticker, 'd', 30)])
            X_combined_weekly = np.vstack([X_weekly_tf, self.data_handler.generate_future_dates(ticker, 'w', 30)])
            X_combined_monthly = np.vstack([X_monthly_tf, self.data_handler.generate_future_dates(ticker, 'm', 30)])

            X_combined_daily = tf.convert_to_tensor(X_combined_daily, dtype=tf.float64)
            X_combined_weekly = tf.convert_to_tensor(X_combined_weekly, dtype=tf.float64)
            X_combined_monthly = tf.convert_to_tensor(X_combined_monthly, dtype=tf.float64)

            f_mean_combined, f_var_combined, y_mean_combined, y_var_combined = self.predictor.predict_combined(
                alpha_opt, beta_opt, daily_model, weekly_model, monthly_model, 
                X_combined_daily, X_combined_weekly, X_combined_monthly
            )

            # Calculate confidence intervals
            f_lower_combined = f_mean_combined - 1.96 * np.sqrt(f_var_combined)
            f_upper_combined = f_mean_combined + 1.96 * np.sqrt(f_var_combined)
            y_lower_combined = y_mean_combined - 1.96 * np.sqrt(y_var_combined)
            y_upper_combined = y_mean_combined + 1.96 * np.sqrt(y_var_combined)

            # Plot combined predictions
            self.visualizer.plot_pred_data(
                X_daily_tf, Y_daily_tf, X_combined_daily, 
                f_mean_combined, f_lower_combined, f_upper_combined, 
                y_mean_combined, y_lower_combined, y_upper_combined, 
                title=ticker, mean=mean_daily, std=std_daily, 
                filename=f'../plots/{ticker}_GPR_predict_combined.png'
            )

if __name__ == "__main__":
    # Parameters
    ticker1 = 'AAPL'
    ticker2 = 'MSFT'
    ticker3 = 'NFLX'
    ticker4 = 'META'
    ticker5 = 'TSLA'
    ticker7 = 'NVDA'
    ticker8 = 'PYPL'
    ticker9 = 'BAC'
    ticker10 = 'ARM'
    tickers = [ticker1, ticker2]
    train_start_date = '2024-02-01'
    train_end_date = '2024-04-26'
    test_start_date = '2024-04-29'
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
    lambda_ = 0.1
    predict_Y = 'return'

    predictor = StockPredictor(tickers, train_start_date, train_end_date, test_start_date, test_end_date, kernel_combinations, lambda_, predict_Y)
    predictor.run(timeframes)