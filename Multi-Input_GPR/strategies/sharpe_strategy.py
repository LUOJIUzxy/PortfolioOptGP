# This file contains the SharpeRatioStrategy class, which is a subclass of the Strategy class.
# The SharpeRatioStrategy class is used to optimize a portfolio to maximize the Sharpe ratio.
# The optimize method of the SharpeRatioStrategy class calls the optimize_portfolio method of the optimizer object to optimize the portfolio.
from Strategies.strategy import Strategy

class SharpeRatioStrategy(Strategy):
    """
    Strategy to maximize the Sharpe ratio.
    """

    def optimize(self, max_volatility=None, min_return=None):
        """
        Optimize portfolio to maximize Sharpe ratio.
        """
        if self.optimizer:
            return self.optimizer.optimize_portfolio()
        else:
            raise ValueError("Optimizer not set.")
