import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import deepcopy, print_summary
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm
import os

from visualizer import Visualizer
from data_handler import DataHandler

import numpy as np
import tensorflow as tf

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

def plot_2d_kernel_prediction(ax: Axes) -> None:
  
    data = {}
    data_handler = DataHandler('2024-02-01', '2024-04-26', '2024-04-29', '2024-05-10')
    X_AAPL_tf, Y_AAPL_tf, AAPL_dates, AAPL_mean, AAPL_std = data_handler.process_data("AAPL", "d", "close")
    data["d"] = (X_AAPL_tf, Y_AAPL_tf, AAPL_dates, AAPL_mean, AAPL_std)

    X, X_tf, Y_tf, dates, mean, std = data_handler.process_2D_X("MSFT", "2024-02-01", "2024-05-10", "close")

    visualizer = Visualizer()
    visualizer.plot_data(X_tf, Y_tf, dates, title=f'MSFT - Day', mean=mean, std=std, filename=f'../plots/MSFT_Day.png')

    Y = Y_AAPL_tf
   
    # Calculate correlations
    corr_X1Y, corr_X2Y, full_corr = calculate_correlations(X, Y)

    # Add correlation information to the plot title
    ax.set_title(f"Example data fit\nCorr(X1,Y)={corr_X1Y:.4f}, Corr(X2,Y)={corr_X2Y:.4f}")

    model = gpflow.models.GPR(
        (X, Y), kernel=deepcopy(kernel), noise_variance=1e-3
    )
    model = train_model(X, Y, model)

    X_2, X_tf_2, Y_tf_2, dates_2, mean_2, std_2 = data_handler.process_2D_X("MSFT",  "2024-05-10", "2024-06-07", "close")

    # Original data
    X_1 = X_tf.numpy().reshape(-1)
    X_2 = Y_tf.numpy().reshape(-1)

    print(X_1.shape)
    print(X_2.shape)

    n_grid = 20

    #Generate future dates, 20
    Xplot_1 = np.linspace(X_1.max(), X_1.max() + 20, n_grid)  # Assuming day of year continues
    X1_combined = np.concatenate([X_1, Xplot_1])
    #Xplot_2 = np.linspace(-1.5, 1.5, n_grid)
    Xplot_2 = Y_tf_2.numpy().reshape(-1)
    X2_combined = np.concatenate([X_2, Xplot_2])

    X1_mesh, X2_mesh = np.meshgrid(X1_combined, X2_combined)

   
    Xplot = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])


    f_mean_all, _ = model.predict_f(Xplot, full_cov=False)
    f_mean_reshaped = f_mean_all.numpy().reshape(X1_mesh.shape)

    print(f_mean_all.shape, Xplot.shape)

    plt.plot(Xplot[: 0], f_mean_all, "kx", mew=2, label="Training data")
    # surface = ax.plot_surface(X1_mesh, X2_mesh, f_mean_reshaped, cmap="coolwarm", alpha=0.2)
    # scatter = ax.scatter(X[:, 0], X[:, 1], Y[:, 0], c=Y[:, 0], 
    #                      cmap='gist_gray', s=50, edgecolors='black')
    
    # plt.colorbar(surface, ax=ax, label='Predicted Value', pad=0.1)
    # plt.colorbar(scatter, ax=ax, label='Actual Value', pad=0.15)

    ax.set_title(f"Day, MSFT Price vs. Apple Stock Price\n"
                 f"Corr(X1,Y)={corr_X1Y:.4f}, Corr(X2,Y)={corr_X2Y:.4f}\n"
                 )
    
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('AAPL Price')
    # ax.set_zlabel('Predicted APPL Price')
    

def train_model(X: np.ndarray, Y: np.ndarray, model: gpflow.models.GPR) -> gpflow.models.GPR:
    model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel), noise_variance=1e-3)
    gpflow.set_trainable(model.likelihood, False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    print_summary(model)

    return model



def plot_2d_kernel(kernel: gpflow.kernels.Kernel, save_path: str = None) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))
    plot_2d_kernel_prediction(ax, kernel)
    
    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    else:
        plt.show()

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