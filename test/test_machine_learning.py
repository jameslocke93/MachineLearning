import unittest
from machine_learning.machine_learning import MachineLearning


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ml = MachineLearning(dtype='float64')

    def test_gen_weights_biases(self):
        ground_truth_weights = {
            'weights_0': (1, 2),
            'weights_1': (2, 3),
            'weights_2': (3, 1)
        }
        ground_truth_biases = {
            'biases_0': [2],
            'biases_1': [3],
            'biases_2': [1]
        }
        test_weights, test_biases = self.ml.gen_weights_bias([2, 3], 1, 1)
        weights = dict()
        biases = dict()
        for key, value in test_weights.items():
            weights[key] = value.shape

        for key, value in test_biases.items():
            biases[key] = value.get_shape().as_list()

        self.assertEqual(ground_truth_weights, weights, 'Weight dimensions are incorrect')
        self.assertEqual(ground_truth_biases, biases, 'Bias dimensions are incorrect')


if __name__ == '__main__':
    unittest.main()
