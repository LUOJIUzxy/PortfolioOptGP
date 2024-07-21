import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = Visualizer()

    @patch('visualizer.plt.figure')
    @patch('visualizer.plt.savefig')
    def test_plot_data(self, mock_savefig, mock_figure):
        X = np.array([[1], [2], [3]])
        Y = np.array([[1], [2], [3]])
        dates = pd.date_range(start='2024-01-01', periods=3)
        
        self.visualizer.plot_data(X, Y, dates, 'Test', 0, 1, 'test.png')
        
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once_with('test.png')
