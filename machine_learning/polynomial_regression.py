import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from machine_learning.machine_learning import MachineLearning


class PolynomialRegression(MachineLearning):

    def __init__(self, epochs, alpha, display_step=0, dtype='float64'):
        MachineLearning.__init__(self, epochs=epochs, alpha=alpha, display_step=display_step, dtype=dtype)

    def train(self, features, labels, num_coeffs):
        _features = features
        _label = labels

        loss_dict = dict()

        x = tf.placeholder(dtype=self.dtype)
        y = tf.placeholder(dtype=self.dtype)

        weights = tf.Variable([1.0] * (num_coeffs * len(features[0])), name="weights_1", dtype="float64")
        bias = tf.Variable(1.0, name="biases_1", dtype="float64")

        # weights_dict, bias_dict = self.gen_weights_bias(dimensions=[len(features[0]), 1])
        # weights = weights_dict.get('weights_1')
        # bias = bias_dict.get('biases_1')

        pred = tf.add(self.poly_fit(_features, weights, num_coeffs), bias)

        loss = tf.losses.mean_squared_error(labels=y, predictions=pred)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                train_data = {x: _features, y: _label}
                sess.run(optimizer, feed_dict=train_data)
                loss_dict[epoch] = sess.run(loss, feed_dict=train_data)
                if self.display_step != 0 and (epoch + 1) % self.display_step == 0:
                    c = sess.run(loss, feed_dict={x: _features, y: _label})
                    print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(c),
                          "W=", sess.run(weights), "b=", sess.run(bias))
            saver.save(sess, self.model_path)

        plt.plot(list(loss_dict.keys()), list(loss_dict.values()), label='Loss Curve')
        plt.show()

    def test(self, features, labels, num_coeffs):
        weights = tf.Variable([1.0] * (num_coeffs * len(features[0])), name="weights_1", dtype="float64")
        bias = tf.Variable(1.0, name="biases_1", dtype="float64")

        predictions = np.zeros(labels.size, dtype=self.dtype)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            weights = sess.run('weights_1:0')
            bias = sess.run('biases_1:0')
            for i in range(len(features)):
                predictions[i] = tf.add(self.poly_fit(features[i], weights, num_coeffs), bias).eval()

            plt.scatter(range(len(features)), labels, marker='x', c='red', label='Ground Truth')
            plt.scatter(range(len(features)), predictions, marker='o', c='blue', label='Predictions')
            plt.show()

    def poly_fit(self, features, weights, num_coeffs):
        terms = []
        if features[0].dtype != self.dtype:
            num_features = len(features[0])
        else:
            num_features = 1

        for feat in range(num_features):
            for i in range(num_coeffs):
                index = i
                if feat != 0:
                    index = feat * num_coeffs + i
                term = tf.cast(tf.multiply(weights[index], tf.pow(features[feat], i + 1)), tf.float64)
                terms.append(term)
        return tf.add_n(terms)
