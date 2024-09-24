# This file contains the SharpeRatioStrategy class, which is a subclass of the Strategy class.
# The SharpeRatioStrategy class is used to optimize a portfolio to maximize the Sharpe ratio.
# The optimize method of the SharpeRatioStrategy class calls the optimize_portfolio method of the optimizer object to optimize the portfolio.
from Strategies.strategy import Strategy
from Optimization.optimizer import Optimizer
import numpy as np

class ConstantStrategy(Strategy):
    """
    Strategy to maximize the Sharpe ratio.
    """

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

