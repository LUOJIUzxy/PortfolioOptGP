from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

class Optimizer:
    def __init__(self, lambda_l1=0.00, lambda_l2=0.00, trx_fee=0.0005, if_tx_penalty=True):
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_tx = trx_fee
        self.initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        self.constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        self.mu = None  # Predicted asset returns
        self.Sigma = None  # Covariance matrix (variances or full covariance)
        self.r_f = None  # Risk-free rate
        self.previous_weights = self.initial_weights  # Initialize previous weights
        self.if_tx_penalty = if_tx_penalty

    def set_predictions(self, predicted_means, predicted_variances, r_f):
        self.mu = np.array(predicted_means)
        self.Sigma = np.diag(np.array(predicted_variances))  # Assume there's no covariance between assets, covariance matrix is diagonal, for a single day
        self.r_f = r_f

    def set_predictions_cml(self, predicted_means, predicted_variances, r_f):
        """
        Set cumulative predictions based on asset-level returns over multiple periods.
        
        :param asset_returns: List of lists, where each sublist is the asset returns over multiple periods.
        :param asset_variances: List of lists, where each sublist is the asset variances over multiple periods.
        :param r_f: Risk-free rate for Sharpe ratio calculation.
        """
        # Calculate cumulative returns for each asset: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        self.mu = np.array([np.prod([1 + r for r in asset_return_list]) - 1 for asset_return_list in predicted_means])

        # Calculate cumulative variances by summing variances over multiple periods
        cumulative_variances = [np.sum(asset_variance_list) for asset_variance_list in predicted_variances]
        self.Sigma = np.diag(np.array(cumulative_variances))  # Diagonal covariance matrix

        self.r_f = r_f
    
    def set_cml_log_return(self, predicted_log_returns, predicted_variances, r_f):
        """
        Set cumulative predictions based on asset-level log returns over multiple periods.
        
        :param predicted_log_returns: List of lists, where each sublist is the asset log returns over multiple periods.
        :param r_f: Risk-free rate for Sharpe ratio calculation.
        """
        # Calculate cumulative returns for each asset: sum of log returns
        self.mu = np.array([np.sum(asset_log_return_list) for asset_log_return_list in predicted_log_returns])

        # Calculate cumulative variances by summing variances over multiple periods
        cumulative_variances = [np.sum(asset_variance_list) for asset_variance_list in predicted_variances]
        self.Sigma = np.diag(np.array(cumulative_variances))

        self.r_f = r_f

    """Set the previous predicted returns for calculate broker fees."""
    def set_previous_weights(self, previous_weights):
        """
        Set the previous portfolio weights.

        :param previous_weights: Numpy array of previous weights.
        """
        self.previous_weights = np.array(previous_weights)
    
    def regularization(self, w):
        """
        Compute the regularization penalty based on the selected regularization types.

        :param w: Portfolio weights.
        :return: Regularization penalty.
        """
        residual = 0.0
        if self.lambda_l1 > 0:
            residual += self.lambda_l1 * np.sum(np.abs(w))
        if self.lambda_l2 > 0:
            residual += self.lambda_l2 * np.sum(w ** 2)
        return residual
    
    def transaction_cost_penalty(self, w):
        """
        Compute the transaction cost penalty based on the change in weights.

        :param w: Current portfolio weights.
        :return: Transaction cost penalty.
        """
        # Calculate the sum of absolute changes in weights
        # What is the type of w?
        weight_changes = np.abs(w - self.previous_weights)
        
        penalty = self.lambda_tx * np.sum(weight_changes)
        return penalty

    def total_penalty(self, w):
        """
        Compute the total penalty (regularization + transaction costs).

        :param w: Portfolio weights.
        :return: Total penalty.
        """
        reg_penalty = self.regularization(w)

        if self.if_tx_penalty:
            tx_penalty = self.transaction_cost_penalty(w)
        else:    
            tx_penalty = 0.0

        return reg_penalty + tx_penalty
    
    def objective(self, w):
        """
        Objective function to maximize the Sharpe ratio with L1 and/or L2 regularization.
        
        :param w: Weights of the assets in the portfolio.
        :return: Negative Sharpe ratio (since we are minimizing) + L1 and/or regularization term.
        """
        if self.mu is None or self.Sigma is None or self.r_f is None:
            raise ValueError("Predictions and covariance matrix must be set before optimization.")
        
        # Portfolio return and volatility
        portfolio_return = np.dot(self.mu, w)
        portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.r_f) / portfolio_volatility

        # regularization term
        # Only L1/L2 regularization
        #regularization = self.regularization(w)
        # L1/L2 regularization + transaction costs
        regularization = self.total_penalty(w)
        
        return -sharpe_ratio + regularization # Negative because we want to maximize it
    
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
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        optimized_weights = result.x
        # Update previous_weights for next optimization
        self.set_previous_weights(optimized_weights)
        return optimized_weights

    # for maximizing returns
    def returns_objective(self, w):
        portfolio_return = np.dot(self.mu, w)
        
        # Only L1/L2 regularization
        #regulization = self.regularization(w)

        # L1/L2 regularization + transaction costs
        regulization = self.total_penalty(w)
        
        return -portfolio_return + regulization  # Negative because we are minimizing by default, but we want to maximize returns
    
    # for minimizing uncertainty
    def uncertainty_objective(self, w):
        portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))
        # Only L1/L2 regularization
        #regulization = self.regularization(w)

        # L1/L2 regularization + transaction costs
        regulization = self.total_penalty(w)
        return portfolio_volatility + regulization
    
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
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        optimized_weights = result.x
        # Update previous_weights for next optimization
        self.set_previous_weights(optimized_weights)
        return optimized_weights

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
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        optimized_weights = result.x
        # Update previous_weights for next optimization
        self.set_previous_weights(optimized_weights)
        return optimized_weights

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
    
    def calculate_portfolio_performance(self, weights):
        """
        Calculate the portfolio return and volatility based on the given weights.
        """
        portfolio_return = np.dot(self.mu, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.Sigma, weights)))
        return portfolio_return, portfolio_volatility
     