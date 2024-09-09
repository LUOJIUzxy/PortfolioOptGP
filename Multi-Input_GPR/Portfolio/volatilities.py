import numpy as np
import pandas as pd

class Volatility:
    def __init__(self, predicted_volatilities, weights_df):
        """
        Initialize the Volatility class with predicted volatilities and portfolio weights.
        
        :param predicted_volatilities: DataFrame of predicted volatilities for each asset.
        :param weights_df: DataFrame of portfolio weights, where each row is for a specific time period.
        """
        self.predicted_volatilities = predicted_volatilities
        self.weights_df = weights_df

    def calculate_portfolio_volatility(self):
        """
        Calculate portfolio volatility using time-varying weights and predicted volatilities.
        
        :return: Pandas Series of portfolio volatility for each day.
        """
        # Calculate portfolio variance and volatility for each day
        portfolio_variance = (self.weights_df ** 2 * self.predicted_volatilities ** 2).sum(axis=1)
        portfolio_volatility = np.sqrt(portfolio_variance)
        return pd.Series(portfolio_volatility, index=self.predicted_volatilities.index)
