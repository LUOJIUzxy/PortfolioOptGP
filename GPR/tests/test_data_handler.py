import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from data_handler import DataHandler

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.data_handler = DataHandler('2024-01-01', '2024-03-31', '2024-04-01', '2024-04-30')

    @patch('data_handler.requests.get')
    def test_fetch_and_save_data(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [{'date': '2024-01-01', 'open': 100, 'close': 101}]
        mock_get.return_value = mock_response

        with patch('data_handler.pd.DataFrame.to_csv') as mock_to_csv:
            self.data_handler.fetch_and_save_data('AAPL', 'd')
            mock_to_csv.assert_called_once()

    def test_process_data(self):
        with patch('data_handler.DataHandler.fetch_and_save_data'), \
             patch('data_handler.pd.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=5),
                'open': [100, 101, 102, 103, 104],
                'close': [101, 102, 103, 104, 105]
            })
            mock_read_csv.return_value = mock_df

            X, Y, dates, mean, std = self.data_handler.process_data('AAPL', 'd', 'close')
            self.assertEqual(X.shape, (5, 1))
            self.assertEqual(Y.shape, (5, 1))
            self.assertEqual(len(dates), 5)
            self.assertIsInstance(mean, float)
            self.assertIsInstance(std, float)

    def test_generate_future_dates(self):
        with patch('data_handler.pd.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=5)
            })
            mock_read_csv.return_value = mock_df

            future_dates = self.data_handler.generate_future_dates('AAPL', 'd', 30)
            self.assertEqual(future_dates.shape, (30, 1))
