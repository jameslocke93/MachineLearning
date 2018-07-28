import numpy as np
import pandas as ps


class LoadData:
    @staticmethod
    def read_file(file_name, delimiter, columns):
        if delimiter == ',':
            file_data = ps.DataFrame(ps.read_csv(file_name), columns=columns)
        else:
            file_data = ps.DataFrame(ps.read_csv(file_name, sep=delimiter), columns=columns)
        return file_data

    @staticmethod
    def clean_data(data, clean_method='Remove'):
        clean_data = ps.DataFrame()
        if data.isnull().values.any():
            if clean_method == 'Remove':
                clean_data = data.dropna()
            elif clean_method == 'Mean':
                clean_data = data.fillna(data.mean())
        return clean_data

    @staticmethod
    def normalise(data):
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        return np.divide(np.subtract(data, data_min), np.subtract(data_max, data_min))

    @staticmethod
    def unnormalise(data):
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        return np.add(np.multiply(data, np.subtract(data_max, data_min)), data_min)

    # Need to update as_matrix to use .values syntax
    @staticmethod
    def split_test_train(data, training_split, features, labels):
        num_entries = len(data.index)
        num_training = int(num_entries * training_split)

        train_features = ps.DataFrame.as_matrix(data[:num_training][features])
        train_labels = ps.DataFrame.as_matrix(data[:num_training][labels])
        test_features = ps.DataFrame.as_matrix(data[num_training:][features])
        test_labels = ps.DataFrame.as_matrix(data[num_training:][labels])

        return train_features, train_labels, test_features, test_labels
