import unittest
from machine_learning.load_data import LoadData
import pandas as ps
import numpy as np
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_equal


class MyTestCase(unittest.TestCase):

    def test_clean_data(self):
        ground_truth_output = ps.DataFrame({
            'col_1': [1.0, 4.0],
            'col_2': [2.0, 5.0],
            'col_3': [3.0, 6.0]
        })
        input_data = {
            'col_1': [1.0, 4.0, 7.0],
            'col_2': [2.0, 5.0, 8.0],
            'col_3': [3.0, 6.0, np.nan]
        }
        input_data_frame = ps.DataFrame(data=input_data)
        output_data = LoadData.clean_data(input_data_frame)
        assert_frame_equal(output_data, ground_truth_output)

    def test_load_data(self):
        ground_truth_output = ps.DataFrame({
            'col_1': [1.0, 4.0],
            'col_2': [2.0, 5.0],
            'col_3': [3.0, 6.0]
        })
        output_data = LoadData.read_file(file_name='data/test_load_data.csv',
                                         delimiter=',',
                                         columns=['col_1', 'col_2', 'col_3'])
        assert_frame_equal(output_data, ground_truth_output)

    def test_clean_mean_data(self):
        ground_truth_output = ps.DataFrame({
            'col_1': [2.0, 2.0, 2.0],
            'col_2': [2.0, 2.0, 2.0]
        })
        input_data = ps.DataFrame({
            'col_1': [2.0, 2.0, 2.0],
            'col_2': [2.0, 2.0, np.nan]
        })
        output_data = LoadData.clean_data(input_data, clean_method='Mean')
        assert_frame_equal(output_data, ground_truth_output)

    def test_load_clean_data(self):
        ground_truth_output = ps.DataFrame({
            'col_1': [1.0, 4.0],
            'col_2': [2.0, 5.0],
            'col_3': [3.0, 6.0]
        })
        unclean_data = LoadData.read_file(file_name='data/test_missing_data.csv',
                                          delimiter=',',
                                          columns=['col_1', 'col_2', 'col_3'])
        output_data = LoadData.clean_data(unclean_data)
        assert_frame_equal(output_data, ground_truth_output)

    def test_delimiter_data(self):
        ground_truth_output = ps.DataFrame({
            'col_1': [2.0, 5.0],
            'col_2': [3.0, 6.0],
            'col_3': [4.0, 7.0]
        })
        output_data = LoadData.read_file(file_name='data/test_delimiter_data.csv',
                                         delimiter='|',
                                         columns=['col_1', 'col_2', 'col_3'])
        assert_frame_equal(ground_truth_output, output_data)

    def test_split_test_train(self):
        ground_truth_train_features = np.ndarray(
            shape=(3, 2),
            buffer=np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]),
            dtype='float64'
            )
        ground_truth_train_labels = np.ndarray(
            shape=(3, 1),
            buffer=np.array([3.0, 3.0, 3.0]),
            dtype='float64'
        )
        ground_truth_test_features = np.ndarray(
            shape=(1, 2),
            buffer=np.array([1.0, 2.0]),
            dtype='float64'
        )
        ground_truth_test_labels = np.ndarray(
            shape=(1, 1),
            buffer=np.array([3.0]),
            dtype='float64'
        )
        data = LoadData.read_file(file_name='data/test_train_data.csv',
                                  delimiter=',',
                                  columns=['col_1', 'col_2', 'col_3'])
        train_features, train_labels, test_features, test_labels = LoadData.split_test_train(data=data,
                                                                                             training_split=0.75,
                                                                                             features=['col_1', 'col_2'],
                                                                                             labels=['col_3'])
        assert_array_equal(ground_truth_train_features, train_features)
        assert_array_equal(ground_truth_train_labels, train_labels)
        assert_array_equal(ground_truth_test_features, test_features)
        assert_array_equal(ground_truth_test_labels, test_labels)


if __name__ == '__main__':
    unittest.main()
