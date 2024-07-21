import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tensorflow as tf
from model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.model_trainer = ModelTrainer([MagicMock()])

    @patch('model_trainer.gpflow.models.GPR')
    def test_train_model(self, mock_gpr):
        mock_model = MagicMock()
        mock_model.predict_f.return_value = (np.array([[1.0]]), np.array([[0.1]]))
        mock_gpr.return_value = mock_model

        X = tf.constant([[1.0], [2.0], [3.0]])
        Y = tf.constant([[1.0], [2.0], [3.0]])

        kernel, mse, model = self.model_trainer.train_model(X, Y)
        
        self.assertIsNotNone(kernel)
        self.assertIsInstance(mse, float)
        self.assertEqual(model, mock_model)