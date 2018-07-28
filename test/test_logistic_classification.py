import unittest
from machine_learning.logistic_classification import LogisticClassification
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.logistic = LogisticClassification(epochs=100, alpha=0.001)

    def test_model_train_binary(self):
        features = np.ndarray(shape=(6, 1),
                              buffer=np.array([0.5, 1.0, 1.5,
                                               2.0, 2.5, 3.0]),
                              dtype='float64')
        labels = np.ndarray(shape=(6, 1),
                            buffer=np.array([1.0, 1.0, 1.0,
                                             2.0, 2.0, 2.0]),
                            dtype='float64')
        self.logistic.train(features=features, label=labels)

    def test_model_test_binary(self):
        features = np.ndarray(shape=(6, 1),
                              buffer=np.array([0.5, 1.0, 1.5,
                                               2.0, 2.5, 3.0]),
                              dtype='float64')
        labels = np.ndarray(shape=(6, 1),
                            buffer=np.array([1.0, 1.0, 1.0,
                                             2.0, 2.0, 2.0]),
                            dtype='float64')
        self.logistic.test(features=features, label=labels)


if __name__ == '__main__':
    unittest.main()
