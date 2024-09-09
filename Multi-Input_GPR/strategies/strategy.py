from abc import ABC, abstractmethod
from Optimization.optimizer import Optimizer

class Strategy(ABC):
    """
    Abstract base class for different portfolio optimization strategies.
    """

    def __init__(self):
        """Initialize common strategy parameters."""
        self.optimizer = None

    def set_optimizer(self, optimizer: Optimizer):
        """Set the optimizer for the strategy."""
        self.optimizer = optimizer

    @abstractmethod
    def optimize(self, max_volatility=None, min_return=None):
        """
        Abstract method to optimize the portfolio.
        Each subclass should implement this method based on its specific strategy.
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
