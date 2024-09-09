import pandas as pd
import numpy as np

#每次要计算任何return是，都生成一个Return类的实例
# Initialize的时候，取决于asset_returns和weights_df
# e.g. 1. 计算 预测的portfolio return时，使用optimized weights 和 predicted returns. 
# 2. Backtesting时，使用optimized weights + true assets return

class Return:
    def __init__(self, asset_returns, weights_df):
        """
        Initialize the Return class with asset returns and portfolio weights.
        
        :param asset_returns: DataFrame of asset returns.
        :param weights_df: DataFrame of portfolio weights, where each row is for a specific time period.
        """
        self.asset_returns = asset_returns
        self.weights_df = weights_df

    def calculate_portfolio_returns(self):
        """
        Calculate portfolio returns based on time-varying weights.
        
        :return: Pandas Series of portfolio returns.
        """
        # Calculate portfolio returns for each day: weighted sum of asset returns
        portfolio_returns = (self.asset_returns * self.weights_df).sum(axis=1)
        return pd.Series(portfolio_returns, index=self.asset_returns.index)

    '''
    Portfolio_returns来自于calculate_portfolio_returns()是一个Series 
    所以每次想要计算cml, 都需要先计算 daily portfolio_returns
    '''
    def calculate_cumulative_return(self, portfolio_returns):
        """
        Calculate the cumulative return of the portfolio over time.
        
        :param portfolio_returns: Series of portfolio returns.
        :return: Cumulative return as a percentage.
        """
        # Convert periodic returns to cumulative return
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        return cumulative_return
