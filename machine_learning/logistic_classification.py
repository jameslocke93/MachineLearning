from machine_learning.machine_learning import MachineLearning
import tensorflow as tf
import matplotlib.pyplot as plt

#TODO: This is just a test

class LogisticClassification(MachineLearning):

    # TODO: Needs a parameter for the activation function
    def __init__(self, epochs, alpha, display_step=0, dtype='float64'):
        MachineLearning.__init__(self, epochs=epochs, alpha=alpha, display_step=display_step, dtype=dtype)

    def train(self, features, label):
        _features = features
        _label = label

        loss_dict = dict()

        x = tf.placeholder(dtype=self.dtype)
        y = tf.placeholder(dtype=self.dtype)

        weights = tf.Variable([1.0] * len(features[0]), name="weights_1", dtype="float64")
        bias = tf.Variable(1.0, name="biases_1", dtype="float64")

        # weights_dict, bias_dict = self.gen_weights_bias(dimensions=[len(features[0]), 1])
        # weights = weights_dict.get('weights_1')
        # bias = bias_dict.get('biases_1')

        pred = tf.sigmoid(tf.add(tf.multiply(x, weights), bias))

        loss = tf.reduce_mean(-y * tf.log(pred) - (1 - y) * tf.log(1 - pred))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha).minimize(loss)

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

            # plt.plot(list(loss_dict.keys()), list(loss_dict.values()), label='Loss Curve')
            plt.scatter(range(len(features)), label, marker='x', c='red', label='Ground Truth')
            plt.scatter(range(len(features)), sess.run(pred), marker='o', c='blue', label='Predictions')
            plt.show()

    def test(self, features, label):
        weights = tf.Variable([1.0] * len(features[0]), name="weights_1", dtype="float64")
        bias = tf.Variable(1.0, name="biases_1", dtype="float64")

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            weights = sess.run('weights_1:0')
            bias = sess.run('biases_1:0')
            pred = tf.sigmoid(tf.add(tf.multiply(features, weights), bias))
            _tmp = sess.run(pred)
            plt.scatter(range(len(features)), label, marker='x', c='red', label='Ground Truth')
            plt.scatter(range(len(features)), sess.run(pred), marker='o', c='blue', label='Predictions')
            plt.show()
