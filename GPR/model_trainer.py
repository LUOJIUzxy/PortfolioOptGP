import gpflow
from gpflow.utilities import print_summary, set_trainable
from sklearn.metrics import mean_squared_error
import tensorflow as tf

class ModelTrainer:
    def __init__(self, kernel_combinations):
        self.kernel_combinations = kernel_combinations

    def train_model(self, X_tf, Y_tf):
        best_kernel = None
        best_mse = float('inf')
        best_model = None
        for kernel in self.kernel_combinations:
            model = gpflow.models.GPR(data=(X_tf, Y_tf), kernel=kernel)
            model.likelihood.variance.assign(1e-5)
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
