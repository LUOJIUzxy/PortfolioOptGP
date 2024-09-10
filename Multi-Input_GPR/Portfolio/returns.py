import pandas as pd
import numpy as np

#每次要计算任何return是，都生成一个Return类的实例
# Initialize的时候，取决于asset_returns和weights_df
# e.g. 1. 计算 预测的portfolio return时，使用optimized weights 和 predicted returns. 
# 2. Backtesting时，使用optimized weights + true assets return

class Return:
    def __init__(self, asset_returns, weights):
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


    def calculate_portfolio_returns(self):
        """
        Calculate portfolio returns based on time-varying weights.
        
        :return: Pandas Series of portfolio returns.
        """
        portfolio_returns = []
        # Calculate portfolio returns for each day: weighted sum of asset returns
        for i in range(self.asset_returns.shape[0]):
            portfolio_returns.append(np.sum(self.asset_returns[i] * self.weights[i]))
        
        return portfolio_returns

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
            portfolio_returns = self.calculate_portfolio_returns()

        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate cumulative return using the formula (1 + r1)(1 + r2)...(1 + rn) - 1
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        return cumulative_return
