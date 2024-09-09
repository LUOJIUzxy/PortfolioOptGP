from Strategies.strategy import Strategy

class MinVolatilityStrategy(Strategy):
    """
    Strategy to minimize portfolio volatility given a minimum return constraint.
    """

    def optimize(self, min_return=None):
        """
        Optimize portfolio to minimize volatility under a minimum return constraint.
        
        :param min_return: The minimum required return for the portfolio.
        """
        if self.optimizer:
            if min_return is None:
                raise ValueError("min_return must be provided for MinVolatilityStrategy.")
            return self.optimizer.minimize_uncertainty(min_return)
        else:
            raise ValueError("Optimizer not set.")
