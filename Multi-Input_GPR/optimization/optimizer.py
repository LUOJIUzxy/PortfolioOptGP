from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

class Optimizer:
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
        self.initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        self.constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        self.mu = None  # Predicted asset returns
        self.Sigma = None  # Covariance matrix (variances or full covariance)
        self.r_f = None  # Risk-free rate

    def set_predictions(self, predicted_means, predicted_variances, r_f):
        self.mu = np.array(predicted_means)
        self.Sigma = np.diag(np.array(predicted_variances))  
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
    
    def objective(self, w):
        """
        Objective function to maximize the Sharpe ratio.
        
        :param w: Weights of the assets in the portfolio.
        :return: Negative Sharpe ratio (since we are minimizing).
        """
        if self.mu is None or self.Sigma is None or self.r_f is None:
            raise ValueError("Predictions and covariance matrix must be set before optimization.")
        
        # Portfolio return and volatility
        portfolio_return = np.dot(self.mu, w)
        portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(self.Sigma, w)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.r_f) / portfolio_volatility
        
        return -sharpe_ratio  # Negative because we want to maximize it
    
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
    
    def calculate_portfolio_performance(self, weights):
        """
        Calculate the portfolio return and volatility based on the given weights.
        """
        portfolio_return = np.dot(self.mu, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.Sigma, weights)))
        return portfolio_return, portfolio_volatility
    
    '''
    Below are methods for Backtesting: calculating cumulative return and portfolio returns with time-varying weights.
    '''
    
    def calculate_cumulative_return(self, portfolio_returns):
        """
        Calculate cumulative return of the portfolio over time.
        
        :param portfolio_returns: Array or list of portfolio returns for each period (e.g., daily returns).
        :return: Cumulative return as a percentage.
        """
        # Convert periodic returns to cumulative return
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        return cumulative_return
    

    '''
    asset_data = {
        'Asset_A': [0.01, 0.02, -0.005],
        'Asset_B': [0.005, 0.01, -0.002],
        'Asset_C': [0.02, 0.015, 0.0]
    }, asset_returns = pd.DataFrame(asset_data)

    weights_data = {
        'Asset_A': [0.4, 0.35, 0.45],
        'Asset_B': [0.3, 0.35, 0.25],
        'Asset_C': [0.3, 0.3, 0.3]
    }, weights_df = pd.DataFrame(weights_data)

    predicted_volatilities_data = {
        'Asset_A': [0.02, 0.025, 0.018],
        'Asset_B': [0.015, 0.017, 0.014],
        'Asset_C': [0.01, 0.012, 0.011]
    }

    predicted_volatilities = pd.DataFrame(predicted_volatilities_data)
    '''
    def calculate_portfolio_returns_with_time_varying_weights(asset_returns, weights_df, predicted_volatilities):
        """
        Calculate portfolio returns and portfolio volatility using time-varying weights and predicted volatilities.
        
        Return: (portfolio_returns, portfolio_volatility)
        """
        # Ensure that asset_returns, weights_df, and predicted_volatilities have the same number of rows (same periods)
        assert asset_returns.shape[0] == weights_df.shape[0] == predicted_volatilities.shape[0], "Mismatch in time periods."
        
        # Calculate portfolio returns for each day
        portfolio_returns = (asset_returns * weights_df).sum(axis=1)
        
        # Calculate portfolio variance and volatility for each day
        portfolio_volatility = np.sqrt(
            (weights_df ** 2 * predicted_volatilities ** 2).sum(axis=1)
        )
        
        return pd.Series(portfolio_returns, index=asset_returns.index), pd.Series(portfolio_volatility, index=asset_returns.index)
        
    