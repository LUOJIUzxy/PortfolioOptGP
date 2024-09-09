from Strategies.strategy import Strategy

class MaxReturnStrategy(Strategy):
    """
    Strategy to maximize portfolio returns within a volatility constraint.
    """

    def optimize(self, max_volatility=None):
        """
        Optimize portfolio to maximize returns under a maximum volatility constraint.
        
        :param max_volatility: The maximum allowable volatility for the portfolio.
        """
        if self.optimizer:
            if max_volatility is None:
                raise ValueError("max_volatility must be provided for MaxReturnStrategy.")
            return self.optimizer.maximize_returns(max_volatility)
        else:
            raise ValueError("Optimizer not set.")
