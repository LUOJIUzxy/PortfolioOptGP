import unittest
import numpy as np
import tensorflow as tf
from predictor import Predictor

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor()

    def test_predict_single(self):
        mock_model = MagicMock()
        mock_model.predict_f.return_value = (np.array([[1.0]]), np.array([[0.1]]))
        mock_model.predict_y.return_value = (np.array([[1.1]]), np.array([[0.2]]))

        X = tf.constant([[1.0]])
        f_mean, f_var, y_mean, y_var = self.predictor.predict_single(mock_model, X)

        np.testing.assert_array_equal(f_mean, np.array([[1.0]]))
        np.testing.assert_array_equal(f_var, np.array([[0.1]]))
        np.testing.assert_array_equal(y_mean, np.array([[1.1]]))
        np.testing.assert_array_equal(y_var, np.array([[0.2]]))