import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import deepcopy, print_summary
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm
import os

from visualizer import Visualizer
from data_handler import DataHandler
from model_trainer import ModelTrainer

import numpy as np
import tensorflow as tf

class MultiInputGPR:
    
    def __init__(self, ticker, features, train_start_date, train_end_date, kernel_combinations, predict_Y='close'):
        self.ticker = ticker
        self.features = features
        self.kernel_combinations = kernel_combinations
        self.data_handler = DataHandler(train_start_date, train_end_date)
        self.model_trainer = ModelTrainer(kernel_combinations)
        self.visualizer = Visualizer()
        self.predict_Y = predict_Y

# X as input with the shape of (n, 2), Y as output with the shape of (n, 1)
    def calculate_correlations(self, X, Y):
        # Convert TensorFlow tensors to numpy arrays
        if isinstance(X, tf.Tensor):
            X = X.numpy()
        if isinstance(Y, tf.Tensor):
            Y = Y.numpy()

        X1 = X[:, 0]
        X2 = X[:, 1]
        Y = Y.reshape(-1)  # Reshape Y to 1D array

        corr_X1Y = np.corrcoef(X1, Y)[0, 1]
        corr_X2Y = np.corrcoef(X2, Y)[0, 1]

        print(f"Correlation between X1 and Y: {corr_X1Y:.4f}")
        print(f"Correlation between X2 and Y: {corr_X2Y:.4f}")

        # Full correlation matrix
        full_corr = np.corrcoef([X1, X2, Y])
        print("\nFull Correlation Matrix:")
        print(full_corr)

        return corr_X1Y, corr_X2Y, full_corr

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

        # print(f"Correlation between X and Y: {corr_matrix[0, 1]:.4f}")

        print("\nCorrelation Matrix:")
        print(corr_matrix)

        return corr_matrix[0, 1]

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
            print(f"Correlation between X{i+1} and Y: {corr_XiY:.4f}")

        return full_corr  


    def plot_2d_kernel_prediction(self, ax: Axes) -> None:
    
        data = {}
        #1. Fetch the actual values of to-be-predicted stock data(e.g. APPL stock price)
        X_AAPL_tf, Y_AAPL_tf, AAPL_dates, AAPL_mean, AAPL_std = self.data_handler.process_data("Stocks", self.ticker, "d", self.predict_Y, isFetch=True)
        data["d"] = (X_AAPL_tf, Y_AAPL_tf, AAPL_dates, AAPL_mean, AAPL_std)

        #2. Fetch input data(e.g. Brent Oil / MSFT stock price)
        _X = []
        for feature in self.features:
            if feature == "Brent_Oil" or feature == "DXY" or feature == "XAU_USD":
                X_tf, Y_tf, dates, mean, std = self.data_handler.process_data("Commodities", feature, "d", "close", isFetch=False)
            else:
                X_tf, Y_tf, dates, mean, std = self.data_handler.process_data("Stocks", feature, "d", "close", isFetch=False)
            visualizer = Visualizer()
            visualizer.plot_data(X_tf, Y_tf, dates, title=f'{feature} - Day', mean=mean, std=std, filename=f'../plots/multi-input/{feature}_Day.png')

            # Calculate correlation between X and Y
            corr = self.calculate_correlation(Y_tf * std + mean, Y_AAPL_tf * AAPL_std + AAPL_mean)
            print(f"Correlation between {feature} and {self.ticker}: {corr:.4f}")

            if np.abs(corr) > 0.15:
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
        for kernel in self.kernel_combinations:
            model = gpflow.models.GPR(
                (X, Y), kernel=deepcopy(kernel), noise_variance=1e-5
            )
            model = ModelTrainer.train_model(model)

        ##Step1 finished
        # predict the mean and variance of the to-be-predicted stock data using the trained model, with input X vector of days * features
        f_mean, f_cov = model.predict_f(X, full_cov=False)

        # Calculate mse
        mse_test = mean_squared_error(Y_tf, f_mean.numpy())
        print(f"Mean Squared Error: {mse_test:.4f}")

        f_mean = f_mean * AAPL_std + AAPL_mean
        Y_actual = Y * AAPL_std + AAPL_mean
        f_cov = f_cov * AAPL_std ** 2
        visualizer.plot_GP(X_AAPL_tf, Y_actual, f_mean, f_cov, title=f"{self.ticker} / Day, predicted by features", filename=f'../plots/multi-input/predicted_{self.ticker}.png')

   

    # # Original data
    # X_1 = X_tf.numpy().reshape(-1)
    # X_2 = Y_tf.numpy().reshape(-1)

    # print(X_1.shape)
    # print(X_2.shape)

    # n_grid = 20

    # #Generate future dates, 20
    # Xplot_1 = np.linspace(X_1.max(), X_1.max() + 20, n_grid)  # Assuming day of year continues
    # X1_combined = np.concatenate([X_1, Xplot_1])
    # #Xplot_2 = np.linspace(-1.5, 1.5, n_grid)
    # Xplot_2 = Y_tf_2.numpy().reshape(-1)
    # X2_combined = np.concatenate([X_2, Xplot_2])

    # X1_mesh, X2_mesh = np.meshgrid(X1_combined, X2_combined)

   
    # Xplot = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])


    # f_mean_all, _ = model.predict_f(Xplot, full_cov=False)
    # f_mean_reshaped = f_mean_all.numpy().reshape(X1_mesh.shape)

    # print(f_mean_all.shape, Xplot.shape)

    # plt.plot(Xplot[: 0], f_mean_all, "kx", mew=2, label="Training data")


    # ax.set_title(f"Day, MSFT Price vs. Apple Stock Price\n"
    #              f"Corr(X1,Y)={corr_X1Y:.4f}, Corr(X2,Y)={corr_X2Y:.4f}\n"
    #              )
    
    # ax.set_xlabel('Day of Year')
    # ax.set_ylabel('AAPL Price')
    # # ax.set_zlabel('Predicted APPL Price')

# 1. Fetch the actual values of to-be-predicted stock data(e.g. APPL stock price) 
# 2. Fetch input data(e.g. Brent Oil / MSFT stock price)
# 3. Concatenate multi-dimenstional input data X = [X1, X2, ...], as days * features vector
        #e.g. [X0 = 70 days index, X1 = 70 days * 1 feature, X2 = 70 days * 1 feature....]
        # Here only 2D, one feature at a time
# 4. Train the model with the input data and the actual values of to-be-predicted stock data
# 5. Predict the mean and variance of the to-be-predicted stock data using the trained model
# 6. Plot the predicted mean and variance of the to-be-predicted stock data as Y-axis, and the days_of_year as X-axis
# Calculate MSE between the actual values and the predicted denormalized mean values
# 7. Calculate the correlation between the input data and the actual values of to-be-predicted stock data: 2 x 2 matrix
# 8. Plot the coloful correlation matrix with all the correlation values
# 9. Select all the highly-correlated features and re-train the model with multiple selected features
# 10. Repeat the steps 5 to 8 with the re-trained model
# 11. Missing Values
# 12. Predict future values of the to-be-predicted stock data using the re-trained
        
# Functions:
# 1. Fetch the actual values of to-be-predicted stock data(e.g. APPL stock price)
# 2. Fetch input data(e.g. Brent Oil / MSFT stock price)
# 3. Concatenate multi-dimenstional input data X = [X1, X2, ...], as days * features vector
        #Args: 1. features list: list[], 2. start_date: str, 3. end_date: str, 4. feature_names list: list[]

# 4. Train the model with the input data and the actual values of to-be-predicted stock data
# 5. Normalize
# 6. Denormalize
        
if __name__ == "__main__":
    train_start_date = '2024-02-10'
    train_end_date = '2024-05-10'

    to_be_predicted = 'AAPL'
    assets = ['MSFT', 'Brent_Oil', 'DXY', 'BAC', 'SP500', 'NasDaq100', 'XAU_USD']

    timeframes = ['d', 'w', 'm']
    predict_Y = 'close'


    kernel_combinations = [  
        gpflow.kernels.Exponential(), 
    ]

    multiInputGPR = MultiInputGPR(
        ticker=to_be_predicted, 
        features=assets,
        train_start_date=train_start_date, 
        train_end_date=train_end_date, 
        kernel_combinations=kernel_combinations, 
        predict_Y=predict_Y
    )

    multiInputGPR.plot_2d_kernel_prediction(plt.gca() )
    plt.show()

