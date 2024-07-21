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

def plot_2d_kernel_prediction(ax: Axes, kernel: gpflow.kernels.Kernel) -> None:
  
    data = {}
    data_handler = DataHandler('2024-02-01', '2024-04-26', '2024-04-29', '2024-05-10')
    X_AAPL_tf, Y_AAPL_tf, AAPL_dates, AAPL_mean, AAPL_std = data_handler.process_data("AAPL", "d", "close")
    data["d"] = (X_AAPL_tf, Y_AAPL_tf, AAPL_dates, AAPL_mean, AAPL_std)

    X, X_Oil_tf, Y_Oil_tf, Oil_dates, oil_mean, oil_std = data_handler.process_2D_X("XAU_USD", "2024-02-01", "2024-05-10", "close")

    visualizer = Visualizer()
    visualizer.plot_data(X_Oil_tf, Y_Oil_tf, Oil_dates, title=f'XAU_USD - Day', mean=oil_mean, std=oil_std, filename=f'../plots/XAU_USD_Day.png')

    Y = Y_AAPL_tf
    model = gpflow.models.GPR(
        (X, Y), kernel=deepcopy(kernel), noise_variance=1e-3
    )
    # Calculate correlations
    corr_X1Y, corr_X2Y, full_corr = calculate_correlations(X, Y)

    # Add correlation information to the plot title
    ax.set_title(f"Example data fit\nCorr(X1,Y)={corr_X1Y:.4f}, Corr(X2,Y)={corr_X2Y:.4f}")

    gpflow.set_trainable(model.likelihood, False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)

    print_summary(model)


    X_2, X_tf_2, Y_tf_2, dates_2, mean_2, std_2 = data_handler.process_2D_X("XAU_USD",  "2024-05-13", "2024-06-10", "close")


    n_grid = 20

    #Generate future dates, 20
    Xplot_1 = np.linspace(100, 125, n_grid)
    #Xplot_2 = np.linspace(-1.5, 1.5, n_grid)
    Xplot_2 = Y_tf_2.numpy().reshape(-1)
    Xplot1, Xplot2 = np.meshgrid(Xplot_1, Xplot_2)

    print(Xplot1.shape)
    print(Xplot2.shape)

    Xplot = np.stack([Xplot1, Xplot2], axis=-1)
    Xplot = Xplot.reshape([n_grid * n_grid, 2])


    X_all = np.vstack((X, Xplot))

    f_mean_all, _ = model.predict_f(X_all, full_cov=False)

    f_mean_original = f_mean_all[:len(X)]
    f_mean_new = f_mean_all[len(X):].numpy().reshape((n_grid, n_grid))

    print(f_mean_all.shape)

    mse = np.mean((Y.numpy() - f_mean_original.numpy()) ** 2)

    ax.plot_surface(Xplot1, Xplot2, f_mean_new, cmap="coolwarm", alpha=0.7)
    scatter = ax.scatter(X[:, 0], X[:, 1], Y[:, 0], c=f_mean_original.numpy(), 
                         cmap='coolwarm', s=50, edgecolors='black')
    
    plt.colorbar(scatter, ax=ax, label='Predicted Value')

    ax.set_title(f"Day, Gold Spot Price vs. Apple Stock Price\n"
                 f"Corr(X1,Y)={corr_X1Y:.4f}, Corr(X2,Y)={corr_X2Y:.4f}\n"
                 f"MSE={mse:.4f}")
    
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Gold Spot Price')
    ax.set_zlabel('Predicted APPL Price')

    print(f"Mean Squared Error: {mse:.4f}")


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

parameters = [0.1, 0.5]
kernel = gpflow.kernels.SquaredExponential()
plot_2d_kernel(kernel, save_path='../plots/2d_kernel_plot1.png')