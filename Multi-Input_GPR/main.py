import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import deepcopy
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm
import os
import random

from Optimization.optimizer import Optimizer
from utils.visualizer import Visualizer
from utils.data_handler import DataHandler
from models.model_trainer import ModelTrainer

from Strategies.sharpe_strategy import SharpeRatioStrategy
from Strategies.max_return_strategy import MaxReturnStrategy
from Strategies.min_volatility_strategy import MinVolatilityStrategy

from Portfolio.portfolio import Portfolio

import pandas as pd
import tensorflow as tf

from statsmodels.tsa.arima.model import ARIMA

class MultiInputGPR:
    
    def __init__(self, ticker, features, train_start_date, train_end_date, test_start_date, test_end_date, kernel_combinations, kernels, threshold, removal_percentage, window_size, predict_Y='return', isFixedLikelihood=False):
        self.ticker = ticker
        self.features = features
        self.kernel_combinations = kernel_combinations
        self.kernels = kernels
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.data_handler = DataHandler(train_start_date, train_end_date, test_start_date, test_end_date, window_size)
        self.model_trainer = ModelTrainer(kernel_combinations)
        self.visualizer = Visualizer()
        self.predict_Y = predict_Y
        self.threshold = threshold
        self.removal_percentage = removal_percentage
        self.isFixed = isFixedLikelihood

    # X as input with the shape of (n, 1), Y as output with the shape of (n, 1)
    def calculate_correlation(self, X, Y):
        # Convert TensorFlow tensors to numpy arrays
        if isinstance(X, tf.Tensor):
            X = X.numpy()
        if isinstance(Y, tf.Tensor):
            Y = Y.numpy()

        X = X.reshape(-1)  # Reshape X to 1D array
        Y = Y.reshape(-1)  # Reshape Y to 1D array

        # Calculate correlation
        corr_matrix = np.corrcoef([X, Y])

        # print("\nCorrelation Matrix:")
        # print(corr_matrix)

        return corr_matrix[0, 1]

    # X as input with the shape of (n, m), Y as output with the shape of (n, 1)
    def full_correlations(self, X, Y):
        # Convert TensorFlow tensors to numpy arrays
        if isinstance(X, tf.Tensor):
            X = X.numpy()
        if isinstance(Y, tf.Tensor):
            Y = Y.numpy()

        # Ensure Y is 1D
        Y = Y.reshape(-1)

        # Combine X and Y for correlation calculation
        combined = np.column_stack((X, Y))

        # Calculate full correlation matrix
        full_corr = np.corrcoef(combined.T)

        print("\nFull Correlation Matrix:")
        print(full_corr)

        # Print correlations between each X column and Y
        for i in range(X.shape[1]):
            corr_XiY = full_corr[i, -1]
            #print(f"Correlation between X{i+1} and Y: {corr_XiY:.4f}")

        return full_corr  

    def remove_random_points(self, X, Y, removal_percentage):
        total_points = X.shape[0]
        points_to_remove = int(total_points * removal_percentage)
        
        # Create a mask of True values
        mask = np.ones(total_points, dtype=bool)
        
        # Randomly set some values to False
        remove_indices = random.sample(range(total_points), points_to_remove)
        mask[remove_indices] = False
        
        # Apply the mask to X and Y
        X_reduced = X[mask]
        Y_reduced = Y[mask]
        
        # Keep track of removed points for later comparison
        X_removed = X[~mask]
        Y_removed = Y[~mask]
        
        return X_reduced, Y_reduced, X_removed, Y_removed
    
    # Create a composite kernel that uses different kernels for different input dimensions
    def create_composite_kernel(self, input_dimension, kernel1_class, kernel2_class):
        # Kernel for the first input dimension (all other assets)
        # Create Kernel1 here

        # Set the variance to fixed 1.0
    
        # k1 = kernel1_class(active_dims=slice(0, input_dimension - 2), variance=1.0)
        # gpflow.set_trainable(k1.variance, False)
        k1 = kernel1_class(active_dims=slice(0, input_dimension - 1))
        #k1 = kernel1_class(active_dims=slice(0, 1))
    
        # Kernel for the second dimension (time)
        # Create the Kernel2 here
        k2 = kernel2_class(active_dims=slice(input_dimension - 1, input_dimension))
        #k2 = gpflow.kernels.slice(kernel2, slice(input_dimension - 1, input_dimension))
        
        # Combine the kernels
        return k1 * k2 
        
    def run_step_1(self) -> None:
    
        #1. Fetch the actual values of to-be-predicted stock data(e.g. APPL stock price)
        X_AAPL_tf, Y_AAPL_tf, AAPL_dates, (AAPL_mean, AAPL_std), (x_mean, x_std) = self.data_handler.process_data("Stocks", self.ticker, "d", self.train_start_date, self.train_end_date, self.predict_Y, isFetch=False)

        #2. Fetch input data(e.g. Brent Oil / MSFT stock price)
        _X = []
        for feature in self.features:
            if feature == "Brent_Oil" or feature == "DXY" or feature == "XAU_USD":
                X_tf, Y_tf, dates, (y_mean, y_std), (x_mean, x_std) = self.data_handler.process_data("Commodities", feature, "d", self.train_start_date, self.train_end_date, "close", isFetch=False)
            else:
                X_tf, Y_tf, dates, (y_mean, y_std), (x_mean, x_std) = self.data_handler.process_data("Stocks", feature, "d", self.train_start_date, self.train_end_date, "close", isFetch=False)
            visualizer = Visualizer()
            visualizer.plot_data(X_tf, Y_tf, dates, title=f'{feature} - Day', mean=y_mean, std=y_std, filename=f'../plots/multi-input/{feature}_Day.png')

            # Calculate correlation between X and Y
            corr = self.calculate_correlation(Y_tf * y_std + y_mean, Y_AAPL_tf * AAPL_std + AAPL_mean)
            print(f"Correlation between {feature} and {self.ticker}: {corr:.4f}")

            if np.abs(corr) > self.threshold:
                _X.append(Y_tf)
                print(f"Selected {feature} for training")
            else:
                print(f"Discarded {feature} for training")
        

        #3. Concatenate multi-dimenstional input data X = [X1, X2, ...], as days * features vector
        # X_AAPL_tf should be equal to X_tf
        
        _X.append(X_AAPL_tf)
        X = self.data_handler.concatenate_X(_X)

        Y = Y_AAPL_tf
        self.full_correlations(X, Y)


        #4. Train the model with the input data and the actual values of to-be-predicted stock data
        for kernel in self.kernels:
            model = gpflow.models.GPR(
                (X, Y), kernel=deepcopy(kernel), noise_variance=1e-5
            )
            model = ModelTrainer.train_model(model)

        ##Step1 finished
        # predict the mean and variance of the to-be-predicted stock data using the trained model, with input X vector of days * features
        f_mean, f_cov = model.predict_f(X, full_cov=False)

        # Calculate mse
        mse_test = mean_squared_error(Y, f_mean.numpy())
        print(f"Mean Squared Error: {mse_test:.4f}")

        f_mean = f_mean * AAPL_std + AAPL_mean
        Y_actual = Y * AAPL_std + AAPL_mean
        f_cov = f_cov * AAPL_std ** 2

        visualizer.plot_GP(X_AAPL_tf, Y_actual, f_mean, f_cov, title=f"{self.ticker} / Day, predicted by features", filename=f'../plots/multi-input/predicted_{self.ticker}.png')
   
    def run_step_2(self) -> None:
    
        #1. Fetch the actual values of to-be-predicted stock data(e.g. APPL stock price)
        X_AAPL_tf, Y_AAPL_tf, AAPL_dates, (AAPL_mean, AAPL_std), (x_mean, x_std) = self.data_handler.process_data("Stocks", self.ticker, "d", self.train_start_date, self.train_end_date, self.predict_Y, isFetch=False)

        #2. Fetch input data(e.g. Brent Oil / MSFT stock price)
        _X = []
        for feature in self.features:
            if feature == "Brent_Oil" or feature == "DXY" or feature == "XAU_USD":
                X_tf, Y_tf, dates, (y_mean, y_std), (x_mean, x_std)= self.data_handler.process_data("Commodities", feature, "d", self.train_start_date, self.train_end_date, "close", isFetch=False)
            else:
                X_tf, Y_tf, dates, (y_mean, y_std), (x_mean, x_std) = self.data_handler.process_data("Stocks", feature, "d", self.train_start_date, self.train_end_date, "close", isFetch=False)
            visualizer = Visualizer()
            visualizer.plot_data(X_tf, Y_tf, dates, title=f'{feature} - Day', mean=y_mean, std=y_std, filename=f'../plots/multi-input/{feature}_Day.png')

            # Calculate correlation between X and Y
            corr = self.calculate_correlation(Y_tf * y_std + y_mean, Y_AAPL_tf * AAPL_std + AAPL_mean)
            print(f"Correlation between {feature} and {self.ticker}: {corr:.4f}")

            if np.abs(corr) > self.threshold:
                _X.append(Y_tf)
                print(f"Selected {feature} for training")
            else:
                print(f"Discarded {feature} for training")
        

        #3. Concatenate multi-dimenstional input data X = [X1, X2, ...], as days * features vector
        # X_AAPL_tf should be equal to X_tf
        
        _X.append(X_AAPL_tf)
        X = self.data_handler.concatenate_X(_X)

        Y = Y_AAPL_tf

        # Remove random points
        X_reduced, Y_reduced, X_removed, Y_removed = self.remove_random_points(X, Y, self.removal_percentage)

        # 4. Train the model with the reduced data
        for kernel in self.kernels:
            model = gpflow.models.GPR(
                (X_reduced, Y_reduced), kernel=deepcopy(kernel), noise_variance=1e-5
            )
            model = ModelTrainer.train_model(model)

        # Predict on all points, including removed ones
        f_mean, f_cov = model.predict_f(X, full_cov=False)

        # Denormalize predictions and actual values
        f_mean = f_mean * AAPL_std + AAPL_mean
        Y_actual = Y * AAPL_std + AAPL_mean
        Y_removed_actual = Y_removed * AAPL_std + AAPL_mean
        f_cov = f_cov * AAPL_std ** 2

        # Calculate MSE for all points and removed points
        mse_all = mean_squared_error(Y_actual, f_mean.numpy())
        mse_removed = mean_squared_error(Y_removed_actual, f_mean.numpy()[~np.isin(X, X_reduced).all(axis=1)])

        print(f"Mean Squared Error (all points): {mse_all:.4f}")
        print(f"Mean Squared Error (removed points): {mse_removed:.4f}")
       

        # Plot results
        self.visualizer.plot_GP_with_removed(
            X_AAPL_tf, Y_actual, f_mean, f_cov, 
            X_removed[:, -1], Y_removed_actual,  # Assuming the last column of X is the date
            title=f"{self.ticker} / Day, with {self.removal_percentage * 100} percentage points removed",
            filename=f'../plots/multi-input/predicted_{self.ticker}_with_removed.png'
        )

    # Predict the future values of the to-be-predicted stock data
    # Fetch Train + Test data 
    def run_step_3(self) -> None:
        #1. Fetch the actual values of to-be-predicted stock data(e.g. APPL stock price)
        X_AAPL_tf, Y_AAPL_tf, AAPL_dates, (AAPL_mean, AAPL_std), (x_mean, x_std) = self.data_handler.process_data("Stocks", self.ticker, "d", self.train_start_date, self.train_end_date, self.predict_Y, isFetch=True, isDenoised=False, isFiltered=True)
        X_AAPL_full_tf, Y_AAPL_full_tf, AAPL_full_dates, (AAPL_full_mean, AAPL_full_std), (x_mean_full, x_std_full) = self.data_handler.process_data("Stocks", self.ticker, "d", self.train_start_date, self.test_end_date, "return", isFetch=True, isDenoised=False, isFiltered=False)

        #2. Fetch input data(e.g. Brent Oil / MSFT stock price)
        # X columns vector for training
        _X = []
        # X columns vector for testing
        X_full = []
        for feature in self.features:
            if feature == "Brent_Oil" or feature == "DXY" or feature == "XAU_USD":
                X_tf, Y_tf, dates, (y_mean, y_std), (x_mean, x_std) = self.data_handler.process_data("Commodities", feature, "d", self.train_start_date, self.train_end_date, self.predict_Y, isFetch=False, isDenoised=False, isFiltered=True)
                X_full_tf, Y_full_tf, full_dates, (y_full_mean, y_full_std), (x_full_mean, x_full_std)= self.data_handler.process_data("Commodities", feature, "d", self.train_start_date, self.test_end_date, "return", isFetch=False, isDenoised=False, isFiltered=False)
            elif feature == "SP500" or feature == "NasDaq100":
                X_tf, Y_tf, dates, (y_mean, y_std), (x_mean, x_std) = self.data_handler.process_data("Stocks/index", feature, "d", self.train_start_date, self.train_end_date, self.predict_Y, isFetch=False, isDenoised=False, isFiltered=True)
                X_full_tf, Y_full_tf, full_dates, (y_full_mean, y_full_std), (x_full_mean, x_full_std) = self.data_handler.process_data("Stocks/Index", feature, "d", self.train_start_date, self.test_end_date, "return", isFetch=False, isDenoised=False, isFiltered=False)
            else:
                X_tf, Y_tf, dates, (y_mean, y_std), (x_mean, x_std) = self.data_handler.process_data("Stocks", feature, "d", self.train_start_date, self.train_end_date, self.predict_Y, isFetch=True, isDenoised=False, isFiltered=True)
                X_full_tf, Y_full_tf, full_dates, (y_full_mean, y_full_std), (x_full_mean, x_full_std) = self.data_handler.process_data("Stocks", feature, "d", self.train_start_date, self.test_end_date, "return", isFetch=True, isDenoised=False, isFiltered=False)
            visualizer = Visualizer()
            visualizer.plot_data(X_tf, Y_tf, dates, title=f'{feature} - Day', mean=y_mean, std=y_std, filename=f'../plots/multi-input/{feature}_Day.png')

            # Calculate correlation between X and Y
            corr = self.calculate_correlation(Y_tf * y_std + y_mean, Y_AAPL_tf * AAPL_std + AAPL_mean)
            print(f"Correlation between {feature} and {self.ticker}: {corr:.4f}")

            if np.abs(corr) > self.threshold:
                _X.append(Y_tf)
                X_full.append(Y_full_tf)
                print(f"Selected {feature} for training")
            else:
                print(f"Discarded {feature} for training")

        #3. Concatenate multi-dimenstional input data X = [X1, X2, ...], as days * features vector
        # X_AAPL_tf should be equal to X_tf
        
        ## Add Time as a feature
        _X.append(X_AAPL_tf)
        X = self.data_handler.concatenate_X(_X)

        Y = Y_AAPL_tf
        self.full_correlations(X, Y)

        #4. Train the model with the input data and the actual values of to-be-predicted stock data
        self.kernel_combinations = [
            self.create_composite_kernel(X.shape[1], k1, k2)
            for k1, k2 in self.kernel_combinations
        ]
        
        for composite_kernel in self.kernel_combinations:
            if self.isFixed:
                model = gpflow.models.GPR(
                    (X, Y), kernel=deepcopy(composite_kernel), noise_variance=1e-3
                )
                
                model = ModelTrainer.train_model(model)
            else:
                model = ModelTrainer.train_likelihood(X, Y, composite_kernel)

        # add test data to predict as well
        X_full.append(X_AAPL_full_tf)
        X_full = self.data_handler.concatenate_X(X_full)
        # predict the mean and variance of the to-be-predicted stock data using the trained model, with input X vector of days * features
        f_mean, f_cov = model.predict_f(X_full, full_cov=False)
        print(f"f_mean: {f_mean[-1]}")
        print(f"f_cov: {f_cov[-1]}")

        # Calculate mse for all train and test data
        # Actually should only calculate the test period data
        mse_test = mean_squared_error(Y_AAPL_full_tf, f_mean.numpy())
        print(f"Mean Squared Error Normalized: {mse_test:.4f}")

        print(f"Mean: {AAPL_full_mean}, Std: {AAPL_full_std}")
        f_mean = f_mean * AAPL_full_std + AAPL_full_mean
        Y_actual = Y_AAPL_full_tf * AAPL_full_std + AAPL_full_mean
        f_cov = f_cov * AAPL_full_std ** 2

        # Calculate denormalised mse for all train and test data
        # Actually should only calculate the test period data
        mse_test = mean_squared_error(Y_actual, f_mean.numpy())
        print(f"Mean Squared Error DeNormalized: {mse_test:.4f}")

        visualizer.plot_GP(X_AAPL_full_tf, Y_actual, f_mean, f_cov, title=f"{self.ticker} / Day, predicted by features", filename=f'../plots/multi-input/future_predictions_{self.ticker}.png')

        # Return two-day predictions
        return [f_mean[-5:], f_cov[-5:], Y_actual[-5:]]
    
    def run_arima(self) -> None:
        
        df = self.data_handler.process_df("Stocks", self.ticker, "d", self.train_start_date, self.train_end_date, 'close')
        df_test = self.data_handler.process_df("Stocks", self.ticker, "d", self.test_start_date, self.test_end_date, 'close')
        # Fit the ARIMA model
        model = ARIMA(df, order=(3, 1, 0))
        model_fit = model.fit()

        # Make predictions
        forecast = model_fit.forecast(steps=5)

        # Print the forecasted values
        print(forecast)

        mse = mean_squared_error(df_test, forecast)
        print(f"Mean Squared Error: {mse:.4f}")


if __name__ == "__main__":
    train_start_date = '2024-02-10'
    train_end_date = '2024-05-10'
    test_start_date = '2024-05-13'
    test_end_date = '2024-05-17'

    ticker1 = 'JPM'
    ticker2 = 'AAPL'
    ticker3 = 'AMGN'
    ticker4 = 'HLT'
    ticker5 = 'COST'
    assets = ['MSFT', 'TSLA', 'GOOGL', 'JNJ', 'XOM', 'PLD', 'NEE', 'Brent_Oil', 'DXY', 'BAC', 'SP500', 'NasDaq100']
    portolio_assets = [ticker1, ticker2, ticker3, ticker4, ticker5]

    timeframes = ['d', 'w', 'm']
    predict_Y = 'return'

    # Risk-free rate (daily)
    risk_free_rate = 0.01 / 252  

    max_volatility_threshold = 0.02  
    min_return_threshold = 0.005 

    l1 = 0.01
    l2 = 0.00
    broker_fee = 0.00001
    if_add_broker_fee_as_regularisation = True


    kernels = [
        gpflow.kernels.Exponential(),
    ]

    kernel_combinations = [  
        (gpflow.kernels.Exponential, gpflow.kernels.Exponential),
       # (gpflow.kernels.Exponential(), gpflow.kernels.SquaredExponential()),
        
    ]

    predicted_values = []
    predicted_variances = []
    predicted_Y_values = []

    for to_be_predicted in portolio_assets:
        print(f"Predicting {to_be_predicted}")
        multiInputGPR = MultiInputGPR(
            ticker=to_be_predicted, 
            features=assets,
            train_start_date=train_start_date, 
            train_end_date=train_end_date, 
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            kernel_combinations=kernel_combinations, 
            window_size=3,
            kernels=kernels,
            threshold=0.30,
            predict_Y=predict_Y,
            removal_percentage=0.1,
            isFixedLikelihood=False
        )
        predicted = multiInputGPR.run_step_3()
        
        # [ [asset1 over 5 days], [asset2 over 5 days], ...]
        predicted_values.append(predicted[0])
        predicted_variances.append(predicted[1])
        predicted_Y_values.append(predicted[2])
    
    for to_be_predicted in portolio_assets:
        multiInputGPR.run_arima()   
     

    optimal_weights_min_volatility = []
    optimal_weights_max_return = []
    optimal_weights_max_sharpe = []
    

    optimizer = Optimizer(lambda_l1=l1, lambda_l2=l2, trx_fee=broker_fee, if_tx_penalty=if_add_broker_fee_as_regularisation) 
    portfolio = Portfolio(portolio_assets, predicted_values, predicted_variances, optimizer, risk_free_rate=risk_free_rate, lambda_=0.01, broker_fee=broker_fee, if_cml=True)
    
    optimal_weights_max_sharpe, volatilities_max_sharpe = portfolio.evaluate_portfolio(strategy_name='sharpe', max_volatility=max_volatility_threshold, min_return=min_return_threshold)
    portfolio.backtest_portfolio(historical_returns=predicted_Y_values, strategy_name='sharpe', optimal_weights=optimal_weights_max_sharpe, predicted_volatilities=volatilities_max_sharpe)
    
    optimal_weights_max_return, volatilities_max_return = portfolio.evaluate_portfolio(strategy_name='max_return', max_volatility=max_volatility_threshold, min_return=min_return_threshold)
    portfolio.backtest_portfolio(historical_returns=predicted_Y_values, strategy_name='max_return', optimal_weights=optimal_weights_max_return, predicted_volatilities=volatilities_max_return)
    

    optimal_weights_min_volatility, volatilities_min_volatility = portfolio.evaluate_portfolio(strategy_name='min_volatility', max_volatility=max_volatility_threshold, min_return=min_return_threshold)
    portfolio.backtest_portfolio(historical_returns=predicted_Y_values, strategy_name='min_volatility', optimal_weights=optimal_weights_min_volatility, predicted_volatilities=volatilities_min_volatility)    
    
    plt.show()

