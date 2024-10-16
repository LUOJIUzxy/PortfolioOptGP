import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

class Visualizer:
    def __init__(self):
        self.setup_plot_style()

    def setup_plot_style(self):
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        rc('text', usetex=True)
        #rc('text.latex', preamble=r'\usepackage[sc]{mathpazo}')
        SMALL_SIZE = 8
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 20
        rc('font', size=MEDIUM_SIZE)
        rc('axes', titlesize=BIGGER_SIZE)
        rc('axes', labelsize=BIGGER_SIZE)
        rc('xtick', labelsize=MEDIUM_SIZE)
        rc('ytick', labelsize=BIGGER_SIZE)
        rc('legend', fontsize=MEDIUM_SIZE)
        rc('figure', titlesize=MEDIUM_SIZE)

    def plot_data(self, X_tf, Y_tf, dates, title, mean, std, filename):
        dates_formatted = dates.dt.strftime(f'%y-%m-%d')
        Y_tf = Y_tf * std + mean
        plt.figure(figsize=(12, 6))
        plt.plot(dates_formatted, Y_tf, label=title)
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.xticks(rotation=45)
        plt.title(f'{title}, Daily Return Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename, format='eps')
        plt.close()
    
    def plot_GP(self, X_tf, Y_tf, f_mean, f_cov, title, filename):
        f_lower = f_mean - 1.96 * f_cov
        f_upper = f_mean + 1.96 * f_cov


        plt.figure(figsize=(12, 6))
        plt.plot(X_tf, Y_tf, "kx", mew=2, label="Training data")
        plt.plot(X_tf, f_mean, color="C0", label="Predicted Mean")
        plt.plot(X_tf, f_lower, "--", color="C1", label="95% confidence")
        plt.plot(X_tf, f_upper, "--", color="C1")
        plt.fill_between(X_tf[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.2)
        plt.xlabel('Date')
        plt.ylabel('APPL Close Price')
        plt.title(f'GP Regression on {title} Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
    
    def plot_GP_with_removed(self, X, Y_actual, f_mean, f_cov, X_removed, Y_removed, title, filename):
        plt.figure(figsize=(12, 6))
        X = X.numpy() if hasattr(X, 'numpy') else X
        Y_actual = Y_actual.numpy() if hasattr(Y_actual, 'numpy') else Y_actual
        f_mean = f_mean.numpy() if hasattr(f_mean, 'numpy') else f_mean
        f_cov = f_cov.numpy() if hasattr(f_cov, 'numpy') else f_cov
        X_removed = X_removed.numpy() if hasattr(X_removed, 'numpy') else X_removed
        Y_removed = Y_removed.numpy() if hasattr(Y_removed, 'numpy') else Y_removed

        plt.plot(X, Y_actual, 'kx', label='Actual data')
        plt.plot(X, f_mean, '.', color='blue', label='Predicted mean')
        plt.fill_between(
            X.ravel(),
            f_mean.ravel() - 1.96 * np.sqrt(f_cov.ravel()),
            f_mean.ravel() + 1.96 * np.sqrt(f_cov.ravel()),
            color='C1', alpha=0.2, label=f'95% confidence interval'
        )
        plt.scatter(X_removed, Y_removed, color='red', s=50, label='Removed points')
        plt.title(title)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plot_pred_data(self, X_daily_tf, Y_daily_tf, X_combined_future, f_mean, f_lower, f_upper, y_mean, y_lower, y_upper, title, mean, std, filename):
        Y_daily_tf = Y_daily_tf * std + mean
        f_mean = f_mean * std + mean
        f_lower = f_lower * std + mean
        f_upper = f_upper * std + mean
        y_mean = y_mean * std + mean
        y_lower = y_lower * std + mean
        y_upper = y_upper * std + mean
        
        plt.figure(figsize=(12, 6))
        plt.plot(X_daily_tf, Y_daily_tf, "kx", mew=2, label="Training data")
        plt.plot(X_combined_future, f_mean, "-", color="C0", label="Predicted f Mean")
        plt.plot(X_combined_future, f_lower, "--", color="C0", label="f 95% confidence")
        plt.plot(X_combined_future, f_upper, "--", color="C0")
        plt.plot(X_combined_future, y_mean, "-", color="C0", label="Predicted Y Mean")
        plt.plot(X_combined_future, y_lower, ":", color="C1", label="y 95% confidence")
        plt.plot(X_combined_future, y_upper, ":", color="C1")
        plt.fill_between(X_combined_future[:, 0], f_lower[:, 0], f_upper[:, 0], alpha=0.1, color="C0")
        plt.fill_between(X_combined_future[:, 0], y_lower[:, 0], y_upper[:, 0], alpha=0.1, color="C1")
        
        plt.xlabel('Date')
        plt.ylabel('Normalized Return')
        plt.title(f'GP Regression on {title} Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
    
    def plot_backtest_cml(self, baseline_cmls, sharpe_cmls, max_return_cmls, min_vol_cmls, dynamic_cmls, y_label, title, filename):
        days = range(0, len(baseline_cmls))  # Create a list of days
        
        plt.figure(figsize=(12, 6))
        plt.plot(days, baseline_cmls, label="Baseline", color="black")
        plt.plot(days, sharpe_cmls, label="Sharpe", color="blue")
        plt.plot(days, max_return_cmls, label="Max Return", color="green")
        plt.plot(days, min_vol_cmls, label="Min Volatility", color="red")
        plt.plot(days, dynamic_cmls, label="Dynamic", color="orange")
        plt.xlabel('Day')
        plt.xlim(0, len(baseline_cmls))
        plt.xticks(range(0, len(baseline_cmls) + 1, max(1, len(baseline_cmls) // 10)))  # Set X-axis ticks
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()