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

import numpy as np
import tensorflow as tf

# X as input with the shape of (n, 2), Y as output with the shape of (n, 1)
def calculate_correlations(X, Y):
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
def calculate_correlation(X, Y):
    # Convert TensorFlow tensors to numpy arrays
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    if isinstance(Y, tf.Tensor):
        Y = Y.numpy()

    X = X.reshape(-1)  # Reshape X to 1D array
    Y = Y.reshape(-1)  # Reshape Y to 1D array

    # Calculate correlation
    corr_matrix = np.corrcoef([X, Y])

    print(f"Correlation between X and Y: {corr_matrix[0, 1]:.4f}")

    print("\nCorrelation Matrix:")
    print(corr_matrix)

    return corr_matrix

# What is a Kernel?
# A kernel is a function that defines the covariance between two points in the input space.
# The kernel defines what kind of shapes can take, and it is one of the primary ways you fit your model to your data.
# Technically, a kernel is a function that takes values and returns a covariance matrix telling us how those 
#  coordinates relate to each other. However, for many users it may be more useful to develop an intuitive understanding of how the different kernels behave than to study the maths.
# A kernel is sometimes also known as a covariance function.

# What is a Model?
def train_model(model: gpflow.models.GPR) -> gpflow.models.GPR:
    #model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel), noise_variance=1e-3)
    gpflow.set_trainable(model.likelihood, False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    print_summary(model)

    return model

def plot_2d_kernel_prediction(ax: Axes) -> None:
  
    data = {}
    data_handler = DataHandler('2024-02-01', '2024-05-10', '2024-05-13', '2024-05-10')
    #1. Fetch the actual values of to-be-predicted stock data(e.g. APPL stock price)
    X_AAPL_tf, Y_AAPL_tf, AAPL_dates, AAPL_mean, AAPL_std = data_handler.process_data("Stocks", "AAPL", "d", "close", isFetch=True)
    data["d"] = (X_AAPL_tf, Y_AAPL_tf, AAPL_dates, AAPL_mean, AAPL_std)

    #2. Fetch input data(e.g. Brent Oil / MSFT stock price)
    X_tf, Y_tf, dates, mean, std = data_handler.process_data("Stocks", "NasDaq100", "d", "close", isFetch=False)
    visualizer = Visualizer()
    visualizer.plot_data(X_tf, Y_tf, dates, title='NasDaq100 - Day', mean=mean, std=std, filename='../plots/NasDaq100_Day.png')

    # Calculate correlation between X and Y
    corr = calculate_correlation(Y_tf, Y_AAPL_tf)
    
    #3. Concatenate multi-dimenstional input data X = [X1, X2, ...], as days * features vector
    X = data_handler.concatenate_X([X_tf, Y_tf])

    Y = Y_AAPL_tf

    # Add correlation information to the plot title
    ax.set_title(f"Example data fit\nCorr(X1,Y)={corr_X1Y:.4f}, Corr(X2,Y)={corr_X2Y:.4f}")

    kernel = gpflow.kernels.Exponential()
    model = gpflow.models.GPR(
        (X, Y), kernel=deepcopy(kernel), noise_variance=1e-5
    )
    model = train_model(model)

    ##Step1 finished
    # Plot the data
    f_mean, f_cov = model.predict_f(X, full_cov=False)

    # Calculate mse
    mse_test = mean_squared_error(Y_tf, f_mean.numpy())
    print(f"Mean Squared Error: {mse_test:.4f}")

    f_mean = f_mean * AAPL_std + AAPL_mean
    Y_actual = Y * AAPL_std + AAPL_mean
    f_cov = f_cov * AAPL_std ** 2
    visualizer.plot_GP(X_tf, Y_actual, f_mean, f_cov, title="APPL / Day, predicted by NasDaq100", filename=f'../plots/NasDaq100_APPL_GP.png')

   

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
        

parameters = [0.1, 0.5]
kernel = gpflow.kernels.Exponential()
# plot_2d_kernel(kernel, save_path='../plots/2d_kernel_plot3.png')
plot_2d_kernel_prediction(plt.gca() )
plt.show()