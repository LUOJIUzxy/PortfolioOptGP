from Strategies.strategy import Strategy
from Optimization.optimizer import Optimizer

class MinVolatilityStrategy(Strategy):
    """
    Strategy to minimize portfolio volatility given a minimum return constraint.
    """

    def optimize(self, optimizer: Optimizer, max_volatility, min_return):
        """
        Optimize portfolio to maximize returns given a volatility constraint.
        
        :param optimizer: Optimizer instance for portfolio optimization.
        :param strategy_name: The name of the strategy being applied.
        :param max_volatility: Maximum volatility constraint.
        :param min_return: Minimum return constraint.
        :return: Optimal portfolio weights.
        """
        # Perform optimization to minimize portfolio volatility (constrained by volatility)
        optimal_weights = optimizer.minimize_uncertainty(min_return=min_return)

        # Apply broker fee adjustments if the flag is enabled
        #optimal_weights = self.apply_broker_fee(optimal_weights)

        return optimal_weights
