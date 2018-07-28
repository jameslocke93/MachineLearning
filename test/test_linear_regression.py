import unittest
from machine_learning.linear_regression import LinearRegression
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.linear = LinearRegression(epochs=100, alpha=0.001)

    def test_model_train(self):
        features = np.ndarray(shape=(3, 1),
                              buffer=np.array([0.5, 1.0, 1.5]),
                              dtype='float64')
        labels = np.ndarray(shape=(3, 1),
                            buffer=np.array([1.0, 2.0, 3.0]),
                            dtype='float64')
        self.linear.train(features=features, label=labels)

    def test_model_test(self):
        features = np.ndarray(shape=(3, 1),
                              buffer=np.array([2.0, 2.5, 3.0]),
                              dtype='float64')
        labels = np.ndarray(shape=(3, 1),
                            buffer=np.array([4.0, 5.0, 6.0]),
                            dtype='float64')
        self.linear.test(features, labels)

    def test_model_multivariate_train(self):
        features = np.ndarray(shape=(3, 2),
                              buffer=np.array([0.5, 0.5, 1.0, 1.0, 1.5, 1.5]),
                              dtype='float64')
        label = np.ndarray(shape=(3, 1),
                           buffer=np.array([1.0, 2.0, 3.0]),
                           dtype='float64')
        self.linear.train(features, label)

    def test_model_multivariate_test(self):
        features = np.ndarray(shape=(3, 2),
                              buffer=np.array([2.0, 2.0, 2.5, 2.5, 3.0, 3.0]),
                              dtype='float64')
        label = np.ndarray(shape=(3, 1),
                           buffer=np.array([4.0, 5.0, 6.0]),
                           dtype='float64')
        self.linear.test(features, label)


if __name__ == '__main__':
    unittest.main()
