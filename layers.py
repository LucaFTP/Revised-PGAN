import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore

class MinibatchStdev(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        avg_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(avg_stddev, (shape[0], shape[1], shape[2], 1))
        return tf.concat([inputs, minibatch_stddev], axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] + 1)

class PixelNormalization(tf.keras.layers.Layer):
    def call(self, inputs):
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.math.rsqrt(mean_square + 1.0e-8)
        return inputs * l2

class WeightScaling(tf.keras.layers.Layer):
    def __init__(self, shape, gain=np.sqrt(2), **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.gain = gain

    def build(self, input_shape):
        fan_in = np.prod(self.shape)
        self.wscale = self.gain / np.sqrt(fan_in)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32) * self.wscale

class Bias(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        return inputs + self.bias

class WeightedSum(tf.keras.layers.Layer):
    def __init__(self, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    def call(self, inputs):
        assert len(inputs) == 2
        return (1.0 - self.alpha) * inputs[0] + self.alpha * inputs[1]