from strategies.strategy import Strategy

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
