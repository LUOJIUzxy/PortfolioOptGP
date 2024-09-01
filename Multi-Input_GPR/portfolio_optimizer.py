from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import numpy as np

class Optimizer:
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
        self.initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        # Bounds for alpha and beta
        self.bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        self.constraints = constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        self.mu = None  # Predicted returns
        self.Sigma = None  # Covariance matrix
        self.r_f = None  # Risk-free rate

    def set_predictions(self, predicted_means, predicted_variances, r_f):
        # Convert TensorFlow tensors to NumPy arrays
        self.mu = np.array([mean.numpy() for mean in predicted_means])
        variances = np.array([var.numpy() for var in predicted_variances])
        
        # Constructing a diagonal covariance matrix from variances (assuming no covariance between assets)
        self.Sigma = np.diag(variances)
        
        # Set the risk-free rate
        self.r_f = r_f
    
    def set_predictions_cml(self, predicted_returns, predicted_variances, r_f):
        # Convert TensorFlow tensors to NumPy arrays and calculate cumulative returns
        self.mu = np.array([np.prod([1 + ret.numpy() for ret in asset_returns]) - 1 
                            for asset_returns in predicted_returns])
        variances = np.array([np.sum([var.numpy() for var in asset_variances]) 
                              for asset_variances in predicted_variances])
        
        # Constructing a diagonal covariance matrix from variances (assuming no covariance between assets)
        self.Sigma = np.diag(variances)
        
        # Set the risk-free rate
        self.r_f = r_f
    
    # Objective function (negative Sharpe ratio)
    def objective(self, w):
        if self.mu is None or self.Sigma is None or self.r_f is None:
            raise ValueError("Predictions and covariance matrix must be set before optimization.")
        portfolio_return = np.dot(self.mu, w)
        portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))
        sharpe_ratio = (portfolio_return - self.r_f) / portfolio_volatility
        return -sharpe_ratio

    def returns_objective(self, w):
            return -np.dot(self.mu, w)  # Negative because we are minimizing by default
    
    def uncertainty_objective(self, w):
            return np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))
    
    def maximize_returns(self, max_volatility):
        volatility_constraint = {'type': 'ineq', 'fun': lambda w: max_volatility - np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))}
        constraints = [self.constraints, volatility_constraint]

        result = minimize(
            self.returns_objective,
            self.initial_weights,
            bounds=self.bounds,
            constraints=constraints,
            method='SLSQP'
        )
        return result.x

    def minimize_uncertainty(self, min_return):
        return_constraint = {'type': 'ineq', 'fun': lambda w: np.dot(self.mu, w) - min_return}
        constraints = [self.constraints, return_constraint]
        
        result = minimize(
            self.uncertainty_objective,
            self.initial_weights,
            bounds=self.bounds,
            constraints=constraints,
            method='SLSQP'
        )
        return result.x

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
    
    def optimize_portfolio(self):
        if self.mu is None or self.Sigma is None or self.r_f is None:
            raise ValueError("Predictions and covariance matrix must be set before optimization.")
        result = minimize(
            self.objective,
            self.initial_weights,
            bounds=self.bounds,
            constraints=self.constraints,
            method='SLSQP'
        )
        return result.x
    
    def calculate_portfolio_performance(self, weights):
        """
        Calculate the portfolio return and volatility based on the given weights.
        """
        portfolio_return = np.dot(self.mu, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.Sigma, weights)))
        return portfolio_return, portfolio_volatility