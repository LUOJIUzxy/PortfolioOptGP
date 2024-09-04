from strategies.strategy import Strategy

class Portfolio:
    def __init__(self, assets, optimizer, strategy: Strategy, risk_free_rate=0.01/252, lambda_=0.01, broker_fee=0, regularization=False):
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
        self.strategy = strategy if strategy else Strategy()  # Default to a basic strategy if none is provided
        self.risk_free_rate = risk_free_rate
        self.lambda_ = lambda_
        self.broker_fee = broker_fee
        self.regularization = regularization
        self.returns = None
        self.variances = None

    def set_returns(self, returns, variances):
        """Set the predicted returns and variances for optimization."""
        self.returns = returns
        self.variances = variances
        self.optimizer.set_predictions(self.returns, self.variances, self.risk_free_rate)

    def apply_broker_fee(self, weights):
        """Apply broker fee to the portfolio."""
        if self.broker_fee > 0:
            return weights * (1 - self.broker_fee)
        return weights

    def get_optimal_weights(self, strategy_name='sharpe', max_volatility=0.02, min_return=0.005):
        """Get optimal weights based on the selected strategy."""
        if self.regularization:
            self.optimizer.lambda_ = self.lambda_  # Apply regularization if required

        # Call the strategy to determine the optimal weights
        optimal_weights = self.strategy.optimize(self.optimizer, strategy_name, max_volatility, min_return)
        
        # Apply broker fee adjustments if necessary
        optimal_weights = self.apply_broker_fee(optimal_weights)
        return optimal_weights

    def calculate_performance(self, weights):
        """Calculate portfolio return and volatility for given weights."""
        return self.optimizer.calculate_portfolio_performance(weights)

    def evaluate_portfolio(self, strategy_name='sharpe', max_volatility=0.02, min_return=0.005):
        """Evaluate the portfolio using the selected strategy and calculate performance."""
        optimal_weights = self.get_optimal_weights(strategy_name, max_volatility, min_return)
        portfolio_return, portfolio_volatility = self.calculate_performance(optimal_weights)

        print(f"Optimal weights ({strategy_name}): {optimal_weights}")
        print(f"Portfolio return: {portfolio_return:.4f}, Portfolio volatility: {portfolio_volatility:.4f}")
        
        return optimal_weights, portfolio_return, portfolio_volatility

    def backtest_portfolio(self):
        """Perform a backtest on the optimized portfolio using historical data."""
        # Implement backtesting logic here
        pass
