import pandas as pd
import numpy as np

class Features:
    def __init__(self) -> None:
        pass
    
    def simple_imbalance(masked_bid_amount, masked_ask_amount):
        simple_imbalance = (masked_bid_amount - masked_ask_amount) / (masked_bid_amount + masked_ask_amount)
        return simple_imbalance


    def weighted_imbalance(bid, ask, levels=5, window=300):
        """
        Calculate the weighted imbalance as a rolling factor.
        
        Parameters:
        - bid, ask: DataFrames with shape (86400, 5)
        - levels: number of price levels to consider
        - window: rolling window size in seconds
        
        Returns:
        - DataFrame with shape (86400, 5) containing the rolling imbalance factor
        """
        ans = pd.DataFrame(index=bid.index, columns=bid.columns)
        weights = np.array([levels - i + 1 for i in range(1, levels + 1)])
        
        for col in bid.columns:
            weighted_bids = np.zeros(bid.shape[0])
            weighted_asks = np.zeros(ask.shape[0])
            
            for i in range(bid.shape[0]):
                bids = bid[col].iloc[i:i+levels].values * weights[:len(bid[col].iloc[i:i+levels])]
                asks = ask[col].iloc[i:i+levels].values * weights[:len(ask[col].iloc[i:i+levels])]
                
                weighted_bids[i] = np.sum(bids)
                weighted_asks[i] = np.sum(asks)
            
            rolling_bids = pd.Series(weighted_bids).rolling(window=window, min_periods=1).sum()
            rolling_asks = pd.Series(weighted_asks).rolling(window=window, min_periods=1).sum()
            
            imbalance = (rolling_bids - rolling_asks) / (rolling_bids + rolling_asks)
            ans[col] = imbalance.replace([np.inf, -np.inf], 0).fillna(0)
        
        return ans

    def multi_level_weighted_imbalance(bid_price, ask_price, bid_amount, ask_amount, levels=10, window=300):
        """
        Calculate the multi-level weighted imbalance as a rolling factor.
        
        Parameters:
        - bid_price, ask_price, bid_amount, ask_amount: DataFrames with shape (86400, 5)
        - levels: number of price levels to consider
        - window: rolling window size in seconds
        
        Returns:
        - DataFrame with shape (86400, 5) containing the rolling imbalance factor
        """
        mid_price = (bid_price + ask_price) / 2
        result = pd.DataFrame(index=bid_price.index, columns=bid_price.columns)
        
        # Calculate weights for each level
        level_weights = np.array([(levels - i) for i in range(levels)])
        
        for col in bid_price.columns:
            weighted_bids = np.zeros(bid_price.shape[0])
            weighted_asks = np.zeros(ask_price.shape[0])
            
            for i in range(bid_price.shape[0]):
                # Calculate the weighted bid and ask for each time step
                weighted_bid = (bid_price[col].iloc[i:i+levels].values / mid_price[col].iloc[i]) * bid_amount[col].iloc[i:i+levels].values * level_weights[:len(bid_price[col].iloc[i:i+levels])]
                weighted_ask = (ask_price[col].iloc[i:i+levels].values / mid_price[col].iloc[i]) * ask_amount[col].iloc[i:i+levels].values * level_weights[:len(ask_price[col].iloc[i:i+levels])]
                
                # Replace inf and nan with 0
                weighted_bid = np.nan_to_num(weighted_bid, nan=0.0, posinf=0.0, neginf=0.0)
                weighted_ask = np.nan_to_num(weighted_ask, nan=0.0, posinf=0.0, neginf=0.0)
                
                weighted_bids[i] = np.sum(weighted_bid)
                weighted_asks[i] = np.sum(weighted_ask)
            
            # Calculate rolling sum
            rolling_bid = pd.Series(weighted_bids).rolling(window=window, min_periods=1).sum()
            rolling_ask = pd.Series(weighted_asks).rolling(window=window, min_periods=1).sum()
            
            # Calculate imbalance
            imbalance = (rolling_bid - rolling_ask) / (rolling_bid + rolling_ask)
            
            # Replace inf with 0
            imbalance = imbalance.replace([np.inf, -np.inf], 0).fillna(0)
            
            result[col] = imbalance
        
        return result

