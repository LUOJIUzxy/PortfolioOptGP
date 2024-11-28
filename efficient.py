import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import rc

# Generate random data for individual assets
np.random.seed(42)
n_assets = 6
returns = np.random.normal(0.10, 0.02, n_assets)
risk = np.random.normal(0.15, 0.05, n_assets)

# Generate efficient frontier curve points
x = np.linspace(0, 0.3, 100)
y = 0.05 + np.sqrt(x) * 0.3  # Simplified curve formula

# Create plot
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{mathpazo}')

# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20

# Apply font sizes
rc('font', size=MEDIUM_SIZE)
rc('axes', titlesize=BIGGER_SIZE)
rc('axes', labelsize=BIGGER_SIZE)
rc('xtick', labelsize=MEDIUM_SIZE)
rc('ytick', labelsize=MEDIUM_SIZE)
rc('legend', fontsize=MEDIUM_SIZE)
rc('figure', titlesize=MEDIUM_SIZE)

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
plt.figure(figsize=(10, 6))

# Plot efficient frontier
plt.plot(x, y, 'k-', label='Efficient Frontier')

# Plot individual assets
plt.scatter(risk, returns, c=tum_colors['line1'], label='Individual Assets')

# Plot CAL (Capital Allocation Line)
rf_rate = 0.02  # Risk-free rate
tangency_point = (0.15, 0.15)  # Example tangency portfolio
plt.plot([0, tangency_point[0]], [rf_rate, tangency_point[1]], 'k--', label='Best possible CAL')

# Mark tangency portfolio
plt.scatter(tangency_point[0], tangency_point[1], c=tum_colors['line2'], label='Tangency Portfolio')

# Customize plot
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Show plot
plt.savefig('./plots/multi-input/efficient_frontier.png', format='png')