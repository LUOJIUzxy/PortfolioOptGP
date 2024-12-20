from Strategies.strategy import Strategy
from Strategies.sharpe_strategy import SharpeRatioStrategy
from Strategies.min_volatility_strategy import MinVolatilityStrategy
from Strategies.max_return_strategy import MaxReturnStrategy
from Strategies.constant_baseline_strategy import ConstantStrategy
from Strategies.dynamic_strategy import DynamicStrategy

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


from Portfolio.returns import Return
from Portfolio.volatilities import Volatility

class Portfolio:
    def __init__(self, assets, asset_returns, predicted_volatilities, optimizer, risk_free_rate=0.01/252, lambda_=0.01, broker_fee=0):
        """
        Initialize the Portfolio with a list of assets, optimizer, strategy, and optional parameters for fees and regularization.
        
        :param assets: List of asset tickers.
        :param asset_returns: Asset returns for each day. [ [asset1 over 5 days], [asset2 over 5 days], ...]
        :param predicted_volatilities: Predicted volatilities for each asset. [ [asset1 over 5 days], [asset2 over 5 days], ...]
        :param optimizer: Optimizer instance for portfolio optimization.
        :param risk_free_rate: The risk-free rate used in optimization (default is 1% annualized).
        :param lambda_: Regularization parameter.
        :param broker_fee: Broker fee percentage applied to each trade.
        """
        self.assets = assets
        self.optimizer = optimizer
        # Default to a basic strategy if none is provided
        self.risk_free_rate = risk_free_rate
        self.lambda_ = lambda_
        self.broker_fee = broker_fee


        self.returns = asset_returns
        self.variances = predicted_volatilities  # Initialize previous weights

        # Strategy mapping - maps strategy names to their corresponding classes
        self.strategy_mapping = {
            'sharpe': SharpeRatioStrategy,
            'max_return': MaxReturnStrategy,
            'min_volatility': MinVolatilityStrategy,
            'constant': ConstantStrategy,
            'dynamic': DynamicStrategy,
        }

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
    def get_optimal_weights(self, strategy_name='sharpe', max_volatility=0.02, min_return=0.005, prob_threshold=0.05, mu_A=None, cov_A=None, mu_B=None, cov_B=None, previous_weights=None):
        """
        Get optimal portfolio weights based on the selected strategy.
        
        :param strategy_name: Name of the optimization strategy to use (e.g., 'sharpe').
        :param max_volatility: Maximum volatility constraint.
        :param min_return: Minimum return constraint.
        :return: Optimal portfolio weights.
        """

        # Select the appropriate strategy based on the strategy name
        strategy = self.select_strategy(strategy_name)

        # Optimize portfolio weights using the strategy
        if strategy_name == 'dynamic':
            optimal_weights = strategy.optimize(self.optimizer, max_volatility, prob_threshold, mu_A, cov_A, mu_B, cov_B, previous_weights, broker_fee=self.broker_fee)
        else:
            optimal_weights = strategy.optimize(self.optimizer,  max_volatility, min_return)

        

        return optimal_weights

        
    def calculate_performance(self, weights):
        """Calculate portfolio return and volatility for given weights."""
        return self.optimizer.calculate_portfolio_performance(weights)

    def evaluate_portfolio(self, strategy_name='sharpe', max_volatility=0.02, min_return=0.005, prob_threshold=0.05, isLogReturn=True, cov=None):
        """Evaluate the portfolio using the selected strategy and calculate performance."""
        print(f" ============================================== Predicted Results: {strategy_name} =========================================================== ")
        optimal_weights = []
        predicted_volatilities = []

        cov_matrixs = []
        daily_returns = []

        #Loop over each day in the dataset
        for day in range(0, len(self.returns[0])):
            # returns for set predictions purpose
            returns = []
            volatilities = []

            # std_devs，return, cov_matrix for calculating multivariate normal distribution
            std_devs = []
            daily_return = []
            cov_matrix = None
            
            # Update the optimizer with the current day's returns, update cumulative returns for each day
            if day == 0:
                for i in range(len(self.returns)):
                    returns.append(self.returns[i][0][0])
                    volatilities.append(self.variances[i][0][0])
                    std_devs.append(np.sqrt(self.variances[i][0][0]))
                    daily_return.append(self.returns[i][0][0])
                self.optimizer.set_predictions(returns, volatilities, self.risk_free_rate)
                
            else:
                # Loop over every asset, and get the returns and volatilities for the current day
                for i in range(len(self.returns)):
                    returns.append(self.returns[i][:(day+1)])
                    volatilities.append(self.variances[i][:(day+1)])
                    std_devs.append(np.sqrt(self.variances[i][day][0]))
                    daily_return.append(self.returns[i][day][0])
                
               
                if isLogReturn:
                    self.optimizer.set_cml_log_return(returns, volatilities, self.risk_free_rate)
                else:
                    self.optimizer.set_predictions_cml(returns, volatilities, self.risk_free_rate)
            
            # Calculate the multi-variate normal distribution for the current day
            daily_returns.append(daily_return)
            cov_matrix = np.outer(np.array(std_devs), np.array(std_devs)) * cov
            cov_matrixs.append(cov_matrix)
            joint_distribution = multivariate_normal(mean=np.array(daily_return), cov=cov_matrix)

            if day == 0:
                optimal_weights_daily = self.get_optimal_weights(strategy_name, max_volatility, min_return, prob_threshold, mu_A=None, cov_A=None, mu_B=np.array(daily_return), cov_B=cov_matrix, previous_weights=None)
            else:
                # Get the optimal weights for the current day
                # mu_A: previous day's returns, cov_A: previous day's covariance matrix, mu_B: predicted current day's returns, cov_B: predicted current day's covariance matrix, previous_weights: previous day's optimal weights
                optimal_weights_daily = self.get_optimal_weights(strategy_name, max_volatility, min_return, prob_threshold, mu_A=np.array(daily_returns[-2]), cov_A=cov_matrixs[-2], mu_B=np.array(daily_return), cov_B=cov_matrix, previous_weights=optimal_weights[-1])



            # Calculate the portfolio return and volatility for the current day
            portfolio_return, portfolio_volatility = self.calculate_performance(optimal_weights_daily)

            print(f"Day {(day+1)}: Optimal weights ({strategy_name}): {optimal_weights_daily}")

            if isLogReturn:
                portfolio_return = np.exp(portfolio_return) - 1
                print(f"Day {(day+1)}: Predicted Portfolio Log Return (Cumulative): {portfolio_return:.4%}, Predicted Portfolio volatility: {portfolio_volatility:.4%}")
            else:
                print(f"Day {(day+1)}: Predicted Portfolio Return (Cumulative): {portfolio_return:.4%}, Predicted Portfolio volatility: {portfolio_volatility:.4%}")

            optimal_weights.append(optimal_weights_daily)
            predicted_volatilities.append(portfolio_volatility)
        
        # shape of optimal_weights: (5, 5) -> 5 days, 5 assets, [ [day 1 weights], [day 2 weights], ...]
        return optimal_weights, predicted_volatilities

    def backtest_portfolio(self, historical_returns, strategy_name='sharpe', optimal_weights=None, predicted_volatilities=None):
        """
        Perform backtesting of the portfolio using historical data.

        :param historical_returns: DataFrame of historical returns for the assets.
        :param historical_volatilities: DataFrame of historical volatilities for the assets.
        :param strategy_name: Name of the strategy to use for optimization.
        :param max_volatility: Maximum volatility constraint.
        :param min_return: Minimum return constraint.
        """
         # Print backtesting results
        print(f" ============================================== Backtest Results: {strategy_name} =========================================================== ")
       
        # Step 2: Create new Return and Volatility calculators with historical data and optimal weights
        
        historical_return_calculator = Return(historical_returns, optimal_weights, transaction_cost_rate=self.broker_fee)
       

        # Step 3: Calculate portfolio returns for each day using the Return class
        portfolio_returns, transaction_costs = historical_return_calculator.calculate_portfolio_returns()

        
        sharpe_ratios = []
        return_cmls = []
        transaction_costs_cmls = []

        # Display daily returns and transaction costs
        # Loop over each day in the dataset
        for i, (ret, trx, vars) in enumerate(zip(portfolio_returns, transaction_costs, predicted_volatilities)):
            daily_sharpe = (ret - self.risk_free_rate) / vars
            sharpe_ratios.append(daily_sharpe)
            print(f"Day {i+1}: Daily Portfolio Net Return = {ret:.4%}, Transaction Cost = {trx:.6%}, Portfolio Variance = {vars:.6%}, Daily Sharpe Ratio = {daily_sharpe:.4f}")

            return_cml = historical_return_calculator.calculate_cumulative_return(portfolio_returns[:i+1])
            return_cmls.append(return_cml)
            transaction_costs_cml = historical_return_calculator.calculate_cumulative_transaction_costs(transaction_costs[:i+1])
            transaction_costs_cmls.append(transaction_costs_cml)

        # Step 5: Calculate cumulative returns over the backtesting period using the Return class
        cumulative_return = historical_return_calculator.calculate_cumulative_return(portfolio_returns)
        print(f"Cumulative Return: {cumulative_return:.4%}")
        
        cumulative_trx_costs = historical_return_calculator.calculate_cumulative_transaction_costs()
        print(f"Cumulative Transaction Costs (based on changes in weights): {cumulative_trx_costs:.6%}")

        cumulative_variance = np.sum(predicted_volatilities)
        print(f"Cumulative Portfolio Variance: {cumulative_variance:.6%}")

        # Step 6: Calculate the Sharpe ratio for the portfolio
        sharpe_ratio = (cumulative_return - self.risk_free_rate) / cumulative_variance
        print(f"Sharpe Ratio: {sharpe_ratio:.6f}")

           

        
        return return_cmls, transaction_costs_cmls
