import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from cycler import cycler
import itertools


class Visualizer:
    def __init__(self):
        self.setup_plot_style()

    def setup_plot_style(self):
        # Font settings
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        rc('text', usetex=True)
        rc('text.latex', preamble=r'\usepackage{mathpazo}')
        
        # Font sizes
        SMALL_SIZE = 8
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 20
        
        # Apply font sizes
        rc('font', size=BIGGER_SIZE)
        rc('axes', titlesize=BIGGER_SIZE)
        rc('axes', labelsize=BIGGER_SIZE)
        rc('xtick', labelsize=BIGGER_SIZE)
        rc('ytick', labelsize=BIGGER_SIZE)
        rc('legend', fontsize=BIGGER_SIZE)
        rc('figure', titlesize=BIGGER_SIZE)
        
        # TUM colors from TUM LaTeX template
        tum_colors = {
            'TUMBlue': '#0065BD',
            'line1': '#66c2a5',
            'line2': '#fc8d62',
            'line3': '#8da0cb',
            'line4': '#e78ac3',
            'line5': '#a6d854',
            'TUMAccentBlue': '#64A0C8'
        }
        
        # Set color cycle to match TUM corporate design
        plt.rcParams['axes.prop_cycle'] = cycler(color=[
            tum_colors['line1'],
            tum_colors['line2'],
            tum_colors['line3'],
            tum_colors['line4'],
            tum_colors['line5']
        ])
        
        # Figure settings
        plt.rcParams['figure.figsize'] = (10, 7)  # Default figure size
        plt.rcParams['figure.dpi'] = 100  # Screen display DPI
        plt.rcParams['savefig.dpi'] = 300  # Saved figure DPI
        plt.rcParams['figure.constrained_layout.use'] = True  # Better layout 
        
        # Axes settings
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['lines.markersize'] = 8     # Size of the circular markers
        plt.rcParams['lines.markeredgewidth'] = 1 # Edge width of markers
        
        # Legend settings
        plt.rcParams['legend.frameon'] = True     # Legend box is visible
        plt.rcParams['legend.framealpha'] = 1.0   # Fully opaque legend box
        plt.rcParams['legend.edgecolor'] = 'gray' # Light gray edge for legend box
        
        plt.rcParams['axes.grid'] = False
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linestyle'] = '--'
        
        # Tick settings
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['xtick.minor.width'] = 0.6
        plt.rcParams['ytick.minor.width'] = 0.6
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        
        # Legend settings
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.9
        plt.rcParams['legend.edgecolor'] = 'none'
        plt.rcParams['legend.fancybox'] = True
        
        # Save settings
        plt.rcParams['savefig.format'] = 'pdf'  # Default save format
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.transparent'] = True
        
        # Additional LaTeX settings
        # plt.rcParams['text.latex.preview'] = True
        # plt.rcParams['axes.unicode_minus'] = False  # Proper minus signs
        
        return tum_colors 

    def plot_data(self, X_tf, Y_tf, dates, title, mean, std, filename):
        tum_colors = self.setup_plot_style()

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
        tum_colors = self.setup_plot_style()
        

        f_lower = f_mean - 1.96 * f_cov
        f_upper = f_mean + 1.96 * f_cov


        plt.figure(figsize=(12, 6))
        plt.plot(X_tf, Y_tf, "kx", mew=2, label="Training data")
        plt.plot(X_tf, f_mean, color=tum_colors['TUMAccentBlue'], label="Predicted Mean")
        plt.plot(X_tf, f_lower, "--", color=tum_colors['TUMBlue'], label="95% confidence")
        plt.plot(X_tf, f_upper, "--", color=tum_colors['TUMBlue'])
        plt.fill_between(X_tf[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.2)
        plt.xlabel('Date')
        plt.ylabel('APPL Close Price')
        plt.title(f'GP Regression on {title} Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
    
    def plot_GP_with_removed(self, X, Y_actual, f_mean, f_cov, X_removed, Y_removed, title, filename):
        tum_colors = self.setup_plot_style()

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
        tum_colors = self.setup_plot_style()

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
        tum_colors = self.setup_plot_style()

        days = range(0, len(baseline_cmls))  # Create a list of days
        
        plt.figure(figsize=(10, 7))
        plt.plot(days, baseline_cmls, "*--", label="Baseline", color=tum_colors['line1'])
        plt.plot(days, sharpe_cmls, "o--", label="Sharpe", color=tum_colors['line2'])
        plt.plot(days, max_return_cmls, "o--", label="Max Return", color=tum_colors['line3'])
        plt.plot(days, min_vol_cmls, "o--", label="Min Volatility", color=tum_colors['line4'])
        plt.plot(days, dynamic_cmls, "o--", label="Dynamic", color=tum_colors['line5'])
        plt.xlabel('Day')
        plt.xlim(-0.1, len(baseline_cmls) - 0.8)  # Set X-axis limits
        plt.xticks(range(0, len(baseline_cmls), max(1, len(baseline_cmls) // 10)))  # Set X-axis ticks
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()


    def plot_strategy_returns(self, returns, title, filename):
        tum_colors = self.setup_plot_style()

        plt.figure(figsize=(10, 7))
        plt.plot(returns, label=title)
        plt.xlabel('Day')
        plt.ylabel('Cumulative Return')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        

    def plot_asset_allocations(self, allocations, asset_names, time_points, strategy_name, filename_base):
        tum_colors = self.setup_plot_style()

        # Ensure allocations is a NumPy array
        allocations = np.array(allocations)

        # Assign colors to assets
        asset_colors = [
            tum_colors['line1'],
            tum_colors['line2'],
            tum_colors['line3'],
            tum_colors['line4'],
            tum_colors['line5']
        ]
        # In case there are more than 5 assets, repeat colors
        if len(asset_names) > len(asset_colors):
            asset_colors = list(itertools.islice(itertools.cycle(asset_colors), len(asset_names)))

        n_time_points = len(time_points)
        fig, axs = plt.subplots(1, n_time_points, figsize=(n_time_points * 4, 4))

        # Ensure axs is iterable
        if n_time_points == 1:
            axs = [axs]
        
        for idx, (ax, time_point) in enumerate(zip(axs, time_points)):
            # Get allocations for this time point
            allocation = allocations[idx, :]

            # Create custom legend labels with percentages
            
            legend_labels = [f'{name} ({allocation[i]*100:.1f}%)' for i, name in enumerate(asset_names)]
            
            
            # Function to format percentages, hide small slices
            def autopct_func(pct):
                return ('%1.1f%%' % pct) if pct > 5 else ''

            wedges, texts, autotexts = ax.pie(
                allocation,
                colors=asset_colors,
                autopct=autopct_func,
                startangle=90
            )

            # Add legend to each subplot
            # ax.legend(wedges, legend_labels, loc='upper right', bbox_to_anchor=(1, 1))

            ax.set_title(f'Day {time_point}')
            # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.axis('equal')  

        # Add a legend outside the subplots
        fig.legend(wedges, asset_names, loc='center right', bbox_to_anchor=(1.05, 0.5))
        plt.subplots_adjust(right=0.85)  # Adjust the right boundary to make room for the legend

        fig.suptitle(f'Asset Allocation - {strategy_name}')
        plt.savefig(filename_base, bbox_inches='tight')
        plt.close()


    def plot_arim_comparison():
        # Sample data for demonstration
        days = np.arange(1, 6)
        actual_values = [180, 182, 183, 185, 184]
        gpr_predictions = [181, 182.5, 183.5, 184.5, 185]
        arima_predictions = [180.5, 181.5, 182.5, 183, 183.5]

        plt.figure(figsize=(10, 6))
        plt.plot(days, actual_values, label='Actual Values', marker='o')
        plt.plot(days, gpr_predictions, label='GPR Predictions', marker='x')
        plt.plot(days, arima_predictions, label='ARIMA Predictions', marker='^')

        plt.xlabel('Day')
        plt.ylabel('Value')
        plt.title('Comparison of Predicted Values from GPR and ARIMA Models')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_comparison.png', dpi=300)
        plt.show()

