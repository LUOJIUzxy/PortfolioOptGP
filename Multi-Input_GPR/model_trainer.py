from copy import deepcopy
import gpflow
from gpflow.utilities import print_summary, set_trainable
from sklearn.metrics import mean_squared_error
import tensorflow as tf

class ModelTrainer:
    def __init__(self, kernel_combinations):
        self.kernel_combinations = kernel_combinations
    
# What is a Kernel?
# A kernel is a function that defines the covariance between two points in the input space.
# The kernel defines what kind of shapes can take, and it is one of the primary ways you fit your model to your data.
# Technically, a kernel is a function that takes values and returns a covariance matrix telling us how those 
#  coordinates relate to each other. However, for many users it may be more useful to develop an intuitive understanding of how the different kernels behave than to study the maths.
# A kernel is sometimes also known as a covariance function.
    def train_model(model: gpflow.models.GPR) -> gpflow.models.GPR:
        #model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel), noise_variance=1e-3)
        gpflow.set_trainable(model.likelihood, False)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        print_summary(model)

        return model
    
    def train_likelihood(X, Y, composite_kernel, starting_variances=[1e-5, 1e-3, 1e-1, 1.0]):
        best_model = None
        best_loss = float('inf')

        for start_var in starting_variances:
            model = gpflow.models.GPR((X, Y), kernel=deepcopy(composite_kernel), noise_variance=start_var)
            
            # Allow the likelihood variance to be trainable
            gpflow.set_trainable(model.likelihood, True)
            
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(model.training_loss, model.trainable_variables)
            
            # Get the final loss
            final_loss = opt_logs.fun
            print(f"\nModel trained with starting variance {start_var}:")
            print_summary(model)
            print(f"Final loss: {final_loss}")

            # Keep the model with the lowest loss
            if final_loss < best_loss:
                best_model = model
                best_loss = final_loss

        print("\nBest model:")
        print_summary(best_model)
        print(f"Best loss: {best_loss}")

        return best_model

    def train_best_model(self, X_tf, Y_tf):
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
