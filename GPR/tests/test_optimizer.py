import unittest
import numpy as np
from optimizer import Optimizer

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = Optimizer()

    def test_loss_fn(self):
        weights = [0.3, 0.3]
        Y = np.array([[1.0], [2.0], [3.0]])
        f_mean_daily = np.array([[1.1], [2.1], [3.1]])
        f_mean_weekly = np.array([[1.2], [2.2], [3.2]])
        f_mean_monthly = np.array([[1.3], [2.3], [3.3]])

        loss = self.optimizer.loss_fn(weights, Y, f_mean_daily, f_mean_weekly, f_mean_monthly)
        self.assertIsInstance(loss, float)

    def test_optimize_weights(self):
        Y = np.array([[1.0], [2.0], [3.0]])
        f_mean_daily = np.array([[1.1], [2.1], [3.1]])
        f_mean_weekly = np.array([[1.2], [2.2], [3.2]])
        f_mean_monthly = np.array([[1.3], [2.3], [3.3]])

        weights = self.optimizer.optimize_weights(Y, f_mean_daily, f_mean_weekly, f_mean_monthly)
        self.assertEqual(len(weights), 2)
        self.assertTrue(all(0 <= w <= 1 for w in weights))