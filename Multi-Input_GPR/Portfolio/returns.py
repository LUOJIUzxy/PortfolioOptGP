import pandas as pd
import numpy as np

#每次要计算任何return是，都生成一个Return类的实例
# Initialize的时候，取决于asset_returns和weights_df
# e.g. 1. 计算 预测的portfolio return时，使用optimized weights 和 predicted returns. 
# 2. Backtesting时，使用optimized weights + true assets return

class Return:
    def __init__(self, asset_returns, weights, transaction_cost_rate=0):
        """
        Initialize the Return class with asset returns and portfolio weights.
        
        :param asset_returns: List or numpy array of asset returns.
        :param weights_df: List or numpy array of portfolio weights, where each row is for a specific time period.
        """
        asset_returns = np.array(asset_returns)
        asset_returns = np.squeeze(asset_returns).T # (5, 2)
        weights = np.array(weights)
        
        if asset_returns.shape != weights.shape:
            print(asset_returns.shape, weights.shape)
            raise ValueError("The shapes of asset_returns and weights must match (same number of days and assets).")
        
        self.asset_returns = asset_returns
        self.weights = weights
        self.transaction_cost_rate = transaction_cost_rate


    def calculate_portfolio_returns(self):
        """
        Calculate real portfolio returns based on time-varying weights for backtesting.
        
        :return: List of portfolio returns with length of days.
        """
        portfolio_returns = []
        transaction_costs = []
        previous_weights = np.zeros_like(self.weights[0])  # Assuming starting with no position

        for i in range(self.asset_returns.shape[0]):
            current_weights = self.weights[i]
            daily_return = np.dot(current_weights, self.asset_returns[i])
            
            if i == 0:
                # Assuming initial allocation from zero; calculate transaction costs accordingly
                trx_cost = self.transaction_cost_rate * np.sum(np.abs(current_weights))
            else:
                # Transaction costs based on changes from previous weights
                weight_change = np.abs(current_weights - previous_weights)
                trx_cost = self.transaction_cost_rate * np.sum(weight_change)
            
            # Net return after subtracting transaction costs
            net_return = daily_return - trx_cost
            
            # Append to lists
            portfolio_returns.append(net_return)
            transaction_costs.append(trx_cost)
            
            # Update previous_weights for next iteration
            previous_weights = current_weights

            # Debugging statements (optional, can be removed in production)
            # print(f"Day {i+1}: Gross Return = {daily_return:.4%}, Transaction Cost = {trx_cost:.6f}, Net Return = {net_return:.4%}")
        
        return portfolio_returns, transaction_costs

    '''
    Portfolio_returns来自于calculate_portfolio_returns()是一个Series 
    所以每次想要计算cml, 都需要先计算 daily portfolio_returns
    '''
    def calculate_cumulative_return(self, portfolio_returns=None):
        """
        Calculate the cumulative return of the portfolio over time.
        
        :param portfolio_returns: Series of portfolio returns.
        :return: Cumulative return as a percentage.
        """
        # Calculate daily portfolio returns
        if portfolio_returns is None:
            portfolio_returns, _ = self.calculate_portfolio_returns()

        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate cumulative return using the formula (1 + r1)(1 + r2)...(1 + rn) - 1
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        return cumulative_return

    def calculate_cumulative_transaction_costs(self, transaction_costs=None):
        """
        Calculate cumulative transaction costs over time.
        
        :param transaction_costs: List or array of daily transaction costs. If None, calculates it.
        :return: Cumulative transaction costs as a float.
        """
        if transaction_costs is None:
            _, transaction_costs = self.calculate_portfolio_returns()
        
        cumulative_trx_costs = np.sum(transaction_costs)
        return cumulative_trx_costs
    
    def get_daily_transaction_costs(self, transaction_costs=None):
        """
        Get the daily transaction costs.
        
        :param transaction_costs: List or array of daily transaction costs. If None, calculates it.
        :return: Numpy array of daily transaction costs.
        """
        if transaction_costs is None:
            _, transaction_costs = self.calculate_portfolio_returns()
        return np.array(transaction_costs)

    def get_daily_portfolio_returns(self, portfolio_returns=None):
        """
        Get the daily net portfolio returns.
        
        :param portfolio_returns: List or array of daily portfolio returns. If None, calculates it.
        :return: Numpy array of daily portfolio returns.
        """
        if portfolio_returns is None:
            portfolio_returns, _ = self.calculate_portfolio_returns()
        return np.array(portfolio_returns)