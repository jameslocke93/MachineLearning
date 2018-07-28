import tensorflow as tf


class MachineLearning:

    def __init__(self, epochs, alpha, display_step=0, dtype='float64'):
        self.model_path = '/Users/jameslocke/PycharmProjects/MachineLearning/data/model.ckpt'
        self.epochs = epochs
        self.alpha = alpha
        self.display_step = display_step
        self.dtype = dtype

    def gen_weights_bias(self, dimensions, output_dim=0, input_dim=0):
        weights = dict()
        biases = dict()
        if input_dim != 0:
            weights['weights_0'] = tf.Variable(tf.random_normal([input_dim, dimensions[0]],
                                                                dtype=self.dtype), name='weights_0')
            biases['biases_0'] = tf.Variable(tf.random_normal([dimensions[0]], dtype=self.dtype), name='biases_0')

        for depth in range(1, len(dimensions)):
            weight_name = 'weights_' + str(depth)
            bias_name = 'biases_' + str(depth)

            weights[weight_name] = tf.Variable(
                tf.random_normal([dimensions[depth - 1], dimensions[depth]], dtype=self.dtype), name=weight_name)
            biases[bias_name] = tf.Variable(tf.random_normal([dimensions[depth]], dtype=self.dtype), name=bias_name)

        out_weight_name = 'weights_' + str(len(dimensions))
        out_bias_name = 'biases_' + str(len(dimensions))
        if output_dim != 0:
            weights[out_weight_name] = tf.Variable(
                tf.random_normal([dimensions[-1], output_dim], dtype=self.dtype), name=out_weight_name)
            biases[out_bias_name] = tf.Variable(tf.random_normal([output_dim], dtype=self.dtype), name=out_bias_name)
        return weights, biases
