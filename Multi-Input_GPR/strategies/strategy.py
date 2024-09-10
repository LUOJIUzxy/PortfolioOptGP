from abc import ABC, abstractmethod
from Optimization.optimizer import Optimizer

class Strategy(ABC):
    """
    Abstract base class for different portfolio optimization strategies.
    """

    def __init__(self, broker_fee=0, apply_broker_fee=False):
        """
        Initialize the Strategy with optional broker fees.
        
        :param broker_fee: Percentage fee applied to each trade (default 0).
        :param apply_broker_fee: Flag to indicate whether broker fees should be applied (default False).
        """
        self.broker_fee = broker_fee
        self.apply_broker_fee_flag = apply_broker_fee


    # def set_optimizer(self, optimizer: Optimizer):
    #     """Set the optimizer for the strategy."""
    #     self.optimizer = optimizer
    
    def apply_broker_fee(self, weights):
        """
        Apply broker fee to the portfolio weights.
        
        :param weights: Portfolio weights.
        :return: Adjusted weights after applying broker fees if enabled.
        """
        if self.apply_broker_fee_flag and self.broker_fee > 0:
            return weights * (1 - self.broker_fee)
        return weights

    @abstractmethod
    def optimize(self, optimizer, strategy_name, max_volatility, min_return):
        """
        Abstract method to be implemented by subclasses for optimizing portfolio weights.
        """
        pass

    def calculate_portfolio_performance(self, weights):
        """
        Common utility to calculate the portfolio performance for given weights.
        """
        if self.optimizer:
            return self.optimizer.calculate_portfolio_performance(weights)
        else:
            raise ValueError("Optimizer not set.")
