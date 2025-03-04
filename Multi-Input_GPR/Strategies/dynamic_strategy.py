# This file contains the SharpeRatioStrategy class, which is a subclass of the Strategy class.
# The SharpeRatioStrategy class is used to optimize a portfolio to maximize the Sharpe ratio.
# The optimize method of the SharpeRatioStrategy class calls the optimize_portfolio method of the optimizer object to optimize the portfolio.
from Strategies.strategy import Strategy
from Optimization.optimizer import Optimizer
import numpy as np
from scipy.stats import norm, multivariate_normal

class DynamicStrategy(Strategy):
    """
    Strategy to dynamically optimize the portfolio.
    """
    def probability_A_greater_than_B_cdf(self, mu_A, sigma_A, mu_B, sigma_B):
        # Mean and standard deviation of the difference of the two distributions
        mu_diff = mu_A - mu_B
        sigma_diff = np.sqrt(sigma_A**2 + sigma_B**2)
        
        # Calculate the probability that A > B using the CDF of the standard normal distribution
        prob = 1 - norm.cdf(0, loc=mu_diff, scale=sigma_diff)
        return prob
    
    # Multivariate normal version
    # A should be previous time point, B should be current time point
    def probability_A_greater_than_B_mvnorm(self, mu_A, cov_A, mu_B, cov_B, num_samples=10000):
        """
        Estimate the probability that A > B for two multivariate normal distributions A ~ N(mu_A, cov_A)
        and B ~ N(mu_B, cov_B) using Monte Carlo sampling.

        Args:
            mu_A (np.ndarray): Mean vector of distribution A.
            cov_A (np.ndarray): Covariance matrix of distribution A.
            mu_B (np.ndarray): Mean vector of distribution B.
            cov_B (np.ndarray): Covariance matrix of distribution B.
            num_samples (int): Number of Monte Carlo samples to use (default: 100000).

        Returns:
            float: Estimated probability that A > B (element-wise comparison).
        """
        # Ensure the inputs are numpy arrays
        mu_A = np.array(mu_A)
        cov_A = np.array(cov_A)
        mu_B = np.array(mu_B)
        cov_B = np.array(cov_B)

        # Generate samples from the multivariate normal distributions
        # Incorporate covariance matrices，not just diagonal matrices(correlation as well)
        # five assets joint distribution, contribute to my portfolio
        samples_A = multivariate_normal.rvs(mean=mu_A, cov=cov_A, size=num_samples)
        samples_B = multivariate_normal.rvs(mean=mu_B, cov=cov_B, size=num_samples)

        # Count how often samples from A are greater than samples from B in all dimensions
        count_A_greater_B = np.sum(np.all(samples_A > samples_B, axis=1))

        # Estimate the probability
        prob = count_A_greater_B / num_samples
        print("Probability A > B: ", prob)

        return prob

    # (mu_A is None && prev_optimal_weights is None) if it is the first time point
    # (mu_A != None && prev_optimal_weights == None) cannot happen at the same time
    # def optimize(self, optimizer: Optimizer, max_volatility, prob_threshold, mu_A, cov_A, mu_B, cov_B, prev_optimal_weights=None):
    #     """
    #     Optimize portfolio to keep portfolio allocation contant for all time points.
        
    #     :param optimizer: Optimizer instance for portfolio optimization.
    #     :param strategy_name: The name of the strategy being applied.
    #     :param max_volatility: Maximum volatility constraint.
    #     :param min_return: Minimum return constraint.
    #     :return: Optimal portfolio weights.
    #     """
    #     # print("mu_A: ", mu_A)
    #     # print("cov_A: ", cov_A)
    #     # print("mu_B: ", mu_B)
    #     # print("cov_B: ", cov_B)
    #     # print("prev_optimal_weights: ", prev_optimal_weights)
    #     # print("probability threshold: ", prob_threshold)
    #     if mu_A is None:
    #         # If it is the first time point, optimize the portfolio to maximize returns
    #         optimal_weights = optimizer.maximize_returns(max_volatility)
    #     else:
    #         # mu_A: previous day's returns, cov_A: previous day's covariance matrix, mu_B: predicted current day's returns, cov_B: predicted current day's covariance matrix, previous_weights: previous day's optimal weights
    #         # prob: 
    #         prob = self.probability_A_greater_than_B_mvnorm(mu_B, cov_B, mu_A, cov_A, num_samples=10000)
    #         if prob >= prob_threshold:
    #             print("Probability is greater than threshold, optimizing for returns")
    #             optimal_weights = optimizer.maximize_returns(max_volatility)
    #         else:
    #             # optimal weights stay the same as the previous time point
    #             print("Probability is less than threshold, keeping the same weights as the previous time point")
    #             optimal_weights = prev_optimal_weights

    #     return optimal_weights
        
    def optimize(self, optimizer: Optimizer, max_volatility, min_return, mu_A, cov_A, mu_B, cov_B, prev_optimal_weights=None, broker_fee=0.001):
        
        if mu_A is None:
            print("Allocate the first day")
            # optimal_weights = optimizer.minimize_uncertainty(0.0)
            optimal_weights = optimizer.maximize_returns(max_volatility)
            return optimal_weights
            # return np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        else:
            mu_A = np.array(mu_A)
            print("mu_A: ", mu_A)
            mu_B = np.array(mu_B)
            print("mu_B: ", mu_B)
            prev_optimal_weights = np.array(prev_optimal_weights)
            print("prev_optimal_weights: ", prev_optimal_weights)
            
            expected_return_A = np.dot(mu_A, prev_optimal_weights)
            expected_return_B = np.dot(mu_B, prev_optimal_weights)

            if expected_return_A < expected_return_B:
                # Since with the same allocation weights, the next day's return is decreasing, so we need to maximize the returns
                print("Expected return A is greater than expected return B, maximizing returns")
                optimal_weights = optimizer.maximize_returns(max_volatility)
            else:

                # Since with the same allocation weights, the next day's return is increasing, so we need to be more conservative, minimize the volatility
                print("Expected return A is less than expected return B, minimizing uncertainty")
                optimal_weights = optimizer.minimize_uncertainty(expected_return_B - expected_return_A)
                # Incorporate transaction costs to the proposed weights from minimize_uncertainty startegy here
                # The transaction costs are based on the weights of the previous time point
                transaction_costs =  np.sum(broker_fee * np.abs(optimal_weights - prev_optimal_weights))
                realized_return = expected_return_B - expected_return_A - transaction_costs
                if realized_return > 0:
                    print("Realized return is greater than 0, keeping the weights")
                    pass
                else:
                    print("Realized return is less than 0, reverting to previous weights")
                    optimal_weights = prev_optimal_weights

        return optimal_weights