import unittest
import numpy as np
from machine_learning.polynomial_regression import PolynomialRegression


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.polynomial = PolynomialRegression(epochs=100, alpha=0.01)

    def test_model_train(self):
        features = np.ndarray(shape=(3, 1),
                              buffer=np.array([0.5, 1.0, 1.5]),
                              dtype='float64')
        labels = np.ndarray(shape=(3, 1),
                            buffer=np.array([1.0, 2.0, 3.0]),
                            dtype='float64')
        self.polynomial.train(features=features, labels=labels, num_coeffs=2)

    def test_model_test(self):
        features = np.ndarray(shape=(3, 1),
                              buffer=np.array([2.0, 2.5, 3.0]),
                              dtype='float64')
        labels = np.ndarray(shape=(3, 1),
                            buffer=np.array([4.0, 5.0, 6.0]),
                            dtype='float64')
        self.polynomial.test(features=features, labels=labels, num_coeffs=2)

    def test_model_multivariate_train(self):
        features = np.ndarray(shape=(3, 2),
                              buffer=np.array([0.5, 0.5, 1.0, 1.0, 1.5, 1.5]),
                              dtype='float64')
        label = np.ndarray(shape=(3, 1),
                           buffer=np.array([1.0, 2.0, 3.0]),
                           dtype='float64')
        self.polynomial.train(features=features, labels=label, num_coeffs=2)

    def test_model_multivariate_test(self):
        features = np.ndarray(shape=(3, 2),
                              buffer=np.array([2.0, 2.0, 2.5, 2.5, 3.0, 3.0]),
                              dtype='float64')
        label = np.ndarray(shape=(3, 1),
                           buffer=np.array([4.0, 5.0, 6.0]),
                           dtype='float64')
        self.polynomial.test(features=features, labels=label, num_coeffs=2)


if __name__ == '__main__':
    unittest.main()
