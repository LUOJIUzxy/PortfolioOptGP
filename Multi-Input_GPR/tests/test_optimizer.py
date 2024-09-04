import unittest
from optimization.optimizer import Optimizer

class TestOptimizer(unittest.TestCase):
    def test_optimize_portfolio(self):
        optimizer = Optimizer(lambda_=0.01)
        optimizer.set_predictions([0.02, 0.03], [0.01, 0.02], 0.01/252)
        optimal_weights = optimizer.optimize_portfolio()
        self.assertAlmostEqual(sum(optimal_weights), 1, places=4)

if __name__ == "__main__":
    unittest.main()
