import matplotlib.pyplot as plt
from matplotlib import rc

class Visualizer:
    def __init__(self):
        self.setup_plot_style()

    def setup_plot_style(self):
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        rc('text', usetex=True)
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
        plt.savefig(filename)
        plt.close()
    
    def plot_GP(self, X_tf, Y_tf, f_mean, f_cov, title, filename):
        f_lower = f_mean - 1.96 * f_cov
        f_upper = f_mean + 1.96 * f_cov


        plt.figure(figsize=(12, 6))
        plt.plot(X_tf, Y_tf, "kx", mew=2, label="Training data")
        plt.plot(X_tf, f_mean, color="C0", label="Mean")
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