from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import numpy as np

class Optimizer:
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
        self.initial_weights = [0.33, 0.33]
        # Bounds for alpha and beta
        self.bounds = [(0, 1), (0, 1)]
        self.constraints = {'type': 'ineq', 'fun': lambda x: 1 - sum(x)}

    def loss_fn(self, weights, Y, f_mean_daily, f_mean_weekly, f_mean_monthly):
        alpha, beta = weights
        f_combined_mean = alpha * f_mean_daily + beta * f_mean_weekly + (1 - alpha - beta) * f_mean_monthly
        mse = mean_squared_error(Y, f_combined_mean)
        l1_regularization = self.lambda_ * (np.abs(alpha) + np.abs(beta))
        return mse + l1_regularization

    def optimize_weights(self, Y_tf, f_mean_daily, f_mean_weekly, f_mean_monthly):
        result = minimize(
            lambda weights: self.loss_fn(weights, Y_tf, f_mean_daily, f_mean_weekly, f_mean_monthly),
            self.initial_weights,
            bounds=self.bounds,
            constraints=self.constraints,
            method='SLSQP'
        )
        return result.x