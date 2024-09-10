from Strategies.strategy import Strategy
from Strategies.sharpe_strategy import SharpeRatioStrategy
from Strategies.min_volatility_strategy import MinVolatilityStrategy
from Strategies.max_return_strategy import MaxReturnStrategy

import numpy as np
import pandas as pd

from Portfolio.returns import Return
from Portfolio.volatilities import Volatility

class Portfolio:
    def __init__(self, assets, asset_returns, predicted_volatilities, optimizer, risk_free_rate=0.01/252, lambda_=0.01, broker_fee=0, regularization=False, if_cml=False):
        """
        Initialize the Portfolio with a list of assets, optimizer, strategy, and optional parameters for fees and regularization.
        
        :param assets: List of asset tickers.
        :param optimizer: Optimizer instance for portfolio optimization.
        :param strategy: Strategy instance to handle different optimization strategies.
        :param risk_free_rate: The risk-free rate used in optimization (default is 1% annualized).
        :param lambda_: Regularization parameter.
        :param broker_fee: Broker fee percentage applied to each trade.
        :param regularization: Boolean to indicate whether regularization is applied.
        """
        self.assets = assets
        self.optimizer = optimizer
        # Default to a basic strategy if none is provided
        self.risk_free_rate = risk_free_rate
        self.lambda_ = lambda_
        self.broker_fee = broker_fee
        self.regularization = regularization

        self.returns = asset_returns
        self.variances = predicted_volatilities

        self.if_cml = if_cml

        # Strategy mapping - maps strategy names to their corresponding classes
        self.strategy_mapping = {
            'sharpe': SharpeRatioStrategy,
            'max_return': MaxReturnStrategy,
            'min_volatility': MinVolatilityStrategy
        }

    """Set the predicted returns and variances for optimization."""
    def set_returns(self):
        # self.returns = returns
        # self.variances = variances
        if self.if_cml:
            self.optimizer.set_predictions_cml(self.returns, self.variances, self.risk_free_rate)
        else: self.optimizer.set_predictions(self.returns, self.variances, self.risk_free_rate)

    def select_strategy(self, strategy_name):
        """
        Select the strategy class based on the strategy name.
        
        :param strategy_name: Name of the strategy to use (e.g., 'sharpe', 'max_return').
        :return: An instance of the selected strategy.
        """
        strategy_class = self.strategy_mapping.get(strategy_name)
        if strategy_class is None:
            raise ValueError(f"Strategy '{strategy_name}' is not recognized.")
        
        # Instantiate the selected strategy, passing in the broker fee
        return strategy_class(broker_fee=self.broker_fee)

    # Do the portfolio optimization
    def get_optimal_weights(self, strategy_name='sharpe', max_volatility=0.02, min_return=0.005):
        """
        Get optimal portfolio weights based on the selected strategy.
        
        :param strategy_name: Name of the optimization strategy to use (e.g., 'sharpe').
        :param max_volatility: Maximum volatility constraint.
        :param min_return: Minimum return constraint.
        :return: Optimal portfolio weights.
        """
        if self.regularization:
            self.optimizer.lambda_ = self.lambda_  # Apply regularization if required

        # Select the appropriate strategy based on the strategy name
        strategy = self.select_strategy(strategy_name)

        # Optimize portfolio weights using the strategy
        optimal_weights = strategy.optimize(self.optimizer, strategy_name, max_volatility, min_return)

        return optimal_weights

    def calculate_performance(self, weights):
        """Calculate portfolio return and volatility for given weights."""
        return self.optimizer.calculate_portfolio_performance(weights)

    def evaluate_portfolio(self, strategy_name='sharpe', max_volatility=0.02, min_return=0.005):
        """Evaluate the portfolio using the selected strategy and calculate performance."""
        self.set_returns()
        optimal_weights = self.get_optimal_weights(strategy_name, max_volatility, min_return)
        portfolio_return, portfolio_volatility = self.calculate_performance(optimal_weights)

        print(f"Optimal weights ({strategy_name}): {optimal_weights}")
        print(f"Portfolio return: {portfolio_return:.4f}, Portfolio volatility: {portfolio_volatility:.4f}")
        
        return optimal_weights, portfolio_return, portfolio_volatility

    def backtest_portfolio(self, historical_returns, historical_volatilities, strategy_name='sharpe', max_volatility=0.02, min_return=0.005):
        """
        Perform backtesting of the portfolio using historical data.

        :param historical_returns: DataFrame of historical returns for the assets.
        :param historical_volatilities: DataFrame of historical volatilities for the assets.
        :param strategy_name: Name of the strategy to use for optimization.
        :param max_volatility: Maximum volatility constraint.
        :param min_return: Minimum return constraint.
        """
        # Step 1: Get the optimal weights based on the historical data
        optimal_weights = self.get_optimal_weights(strategy_name, max_volatility, min_return)

        # Step 2: Create new Return and Volatility calculators with historical data and optimal weights
        historical_return_calculator = Return(historical_returns, pd.DataFrame([optimal_weights]*historical_returns.shape[0], columns=historical_returns.columns))
        historical_volatility_calculator = Volatility(historical_volatilities, pd.DataFrame([optimal_weights]*historical_volatilities.shape[0], columns=historical_volatilities.columns))

        # Step 3: Calculate portfolio returns for each day using the Return class
        portfolio_returns = historical_return_calculator.calculate_portfolio_returns()

        # Step 4: Calculate portfolio volatility for each day using the Volatility class
        portfolio_volatility = historical_volatility_calculator.calculate_portfolio_volatility()

        # Step 5: Calculate cumulative returns over the backtesting period using the Return class
        cumulative_return = historical_return_calculator.calculate_cumulative_return(portfolio_returns)
        #cumulative_return = np.prod(1 + portfolio_returns) - 1

        # Print backtesting results
        print(f"Backtest Results - {strategy_name}")
        print(f"Daily Portfolio Returns:\n{portfolio_returns}")
        print(f"Cumulative Return: {cumulative_return:.4%}")
        print(f"Portfolio Volatility:\n{portfolio_volatility}")

        return portfolio_returns, cumulative_return, portfolio_volatility
