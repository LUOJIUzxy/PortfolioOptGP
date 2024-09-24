from Strategies.strategy import Strategy
from Optimization.optimizer import Optimizer

class MaxReturnStrategy(Strategy):
    """
    Strategy to maximize portfolio returns within a volatility constraint.
    """

    def optimize(self, optimizer: Optimizer, strategy_name, max_volatility, min_return):
        """
        Optimize portfolio to maximize returns given a volatility constraint.
        
        :param optimizer: Optimizer instance for portfolio optimization.
        :param strategy_name: The name of the strategy being applied.
        :param max_volatility: Maximum volatility constraint.
        :param min_return: Minimum return constraint.
        :return: Optimal portfolio weights.
        """
       
        optimal_weights = optimizer.maximize_returns(max_volatility=max_volatility)

        # Apply broker fee adjustments if the flag is enabled
        #optimal_weights = self.apply_broker_fee(optimal_weights)

        return optimal_weights
