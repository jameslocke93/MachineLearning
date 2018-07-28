import unittest
from machine_learning.neural_network import NeuralNetwork
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.test_nn = NeuralNetwork(problem_type='Regression', alpha=0.001, epochs=100)

    def test_model_train(self):
        weights, biases = self.test_nn.gen_weights_bias([2, 3], 1, 1)
        features = np.ndarray(shape=(3, 1), buffer=np.array([0.5, 1.0, 1.5]), dtype='float64')
        labels = np.ndarray(shape=(3, 1), buffer=np.array([1.0, 2.0, 3.0]), dtype='float64')
        self.test_nn.train(features, labels, weights, biases, activation='Sigmoid')

    def test_model_test(self):
        weights, biases = self.test_nn.gen_weights_bias([2, 3], 1, 1)
        self.test_nn.test(features=np.ndarray(shape=(2, 1), buffer=np.array([2.0, 2.5]), dtype='float64'),
                          labels=np.ndarray(shape=(2, 1), buffer=np.array([4.0, 5.0]), dtype='float64'),
                          weights=weights, biases=biases, activation='Sigmoid')

    def test_model_multivariate_train(self):
        # Input dim has to be the same size as number of features??
        weights, biases = self.test_nn.gen_weights_bias(dimensions=[2, 3], output_dim=1, input_dim=2)
        features = np.ndarray(shape=(3, 2),
                              buffer=np.array([0.5, 0.5, 1.0,
                                               1.0, 1.5, 1.5]),
                              dtype='float64')
        labels = np.ndarray(shape=(3, 1),
                            buffer=np.array([1.0, 2.0, 3.0]),
                            dtype='float64')
        self.test_nn.train(features, labels, weights, biases, activation='Sigmoid')

    def test_model_multivariate_test(self):
        weights, biases = self.test_nn.gen_weights_bias(dimensions=[2, 3], output_dim=1, input_dim=2)
        features = np.ndarray(shape=(3, 2),
                              buffer=np.array([0.5, 0.5, 1.0,
                                               1.0, 1.5, 1.5]),
                              dtype='float64')
        labels = np.ndarray(shape=(3, 1),
                            buffer=np.array([1.0, 2.0, 3.0]),
                            dtype='float64')
        self.test_nn.test(features, labels, weights, biases, activation='Sigmoid')


if __name__ == '__main__':
    unittest.main()
