# This file contains the SharpeRatioStrategy class, which is a subclass of the Strategy class.
# The SharpeRatioStrategy class is used to optimize a portfolio to maximize the Sharpe ratio.
# The optimize method of the SharpeRatioStrategy class calls the optimize_portfolio method of the optimizer object to optimize the portfolio.
from Strategies.strategy import Strategy
from Optimization.optimizer import Optimizer
import numpy as np
from scipy.stats import norm

class DynamicStrategy(Strategy):
    """
    Strategy to dynamically optimize the portfolio.
    """
    def probability_A_greater_than_B_cdf(mu_A, sigma_A, mu_B, sigma_B):
        # Mean and standard deviation of the difference of the two distributions
        mu_diff = mu_A - mu_B
        sigma_diff = np.sqrt(sigma_A**2 + sigma_B**2)
        
        # Calculate the probability that A > B using the CDF of the standard normal distribution
        prob = 1 - norm.cdf(0, loc=mu_diff, scale=sigma_diff)
        return prob

    def optimize(self, optimizer: Optimizer, strategy_name, max_volatility, min_return):
        """
        Optimize portfolio to keep portfolio allocation contant for all time points.
        
        :param optimizer: Optimizer instance for portfolio optimization.
        :param strategy_name: The name of the strategy being applied.
        :param max_volatility: Maximum volatility constraint.
        :param min_return: Minimum return constraint.
        :return: Optimal portfolio weights.
        """
        # Perform optimization (here we assume `optimizer` has a method for optimizing Sharpe ratio)
        optimal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])



        return optimal_weights

