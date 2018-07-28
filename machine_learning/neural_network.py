import tensorflow as tf
import matplotlib.pyplot as plt
from machine_learning.machine_learning import MachineLearning


class NeuralNetwork(MachineLearning):

    def __init__(self, problem_type, epochs, alpha, display_step=0, dtype='float64'):
        MachineLearning.__init__(self, epochs=epochs, alpha=alpha, display_step=display_step, dtype=dtype)

        problem_type_set = frozenset(['Classification', 'Regression'])
        # Want to change these if statements to try throw type errors
        if problem_type not in problem_type_set:
            self.problem_type = 'Regression'
        else:
            self.problem_type = problem_type

    def train(self, features, labels, weights, biases, activation='RELU'):
        _features = features
        _labels = labels

        loss_dict = dict()

        x = tf.placeholder(dtype=self.dtype, shape=[len(_features), len(_features[0])])
        # I don't like having double None need to work out values
        y = tf.placeholder(dtype=self.dtype, shape=[None, None])

        pred = self.neural_network_model(_features, weights, biases, activation)

        loss = tf.losses.mean_squared_error(labels=_labels, predictions=pred)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                train_data = {x: _features, y: _labels}
                sess.run(optimizer, feed_dict=train_data)
                loss_dict[epoch] = sess.run(loss, feed_dict=train_data)
                if self.display_step != 0 and (epoch + 1) % self.display_step == 0:
                    c = sess.run(loss, feed_dict={x: _features, y: _labels})
                    print("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(c),
                          "W=", sess.run(weights), "b=", sess.run(biases))
            saver.save(sess, self.model_path)

        plt.plot(list(loss_dict.keys()), list(loss_dict.values()), label='Loss Curve')
        # plt.show()

    @staticmethod
    def neural_network_model(features, weights, biases, activation, problem_type='Regression'):
        current_pred = tf.nn.sigmoid(tf.matmul(features, weights['weights_0']) + biases['biases_0'])
        if activation == 'RELU':
            activation_function = tf.nn.relu
        else:
            activation_function = tf.nn.sigmoid

        for layer in range(1, len(weights) - 1):
            weight_name = 'weights_' + str(layer)
            bias_name = 'biases_' + str(layer)
            prev_pred = current_pred
            current_pred = activation_function(tf.matmul(prev_pred, weights[weight_name]) + biases[bias_name])
        if problem_type == 'Classification':
            pass
        else:
            return tf.matmul(current_pred, weights['weights_' + str(len(weights) - 1)]) + \
                   biases['biases_' + str(len(biases) - 1)]

    def test(self, features, labels, weights, biases, activation='Relu'):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            for key, _ in weights.items():
                weights[key] = sess.run(key + ':0')
            for key, _ in biases.items():
                biases[key] = sess.run(key + ':0')
            pred = self.neural_network_model(features, weights, biases, activation)

            plt.scatter(range(len(features)), labels, marker='x', c='red', label='Ground Truth')
            plt.scatter(range(len(features)), sess.run(pred), marker='o', c='blue', label='Predictions')
            # plt.show()
