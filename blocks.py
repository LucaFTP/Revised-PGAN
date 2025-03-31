import numpy as np
import tensorflow as tf
from layers import WeightScaling, Bias, PixelNormalization

def get_activation(name):
    if name == 'leaky_relu':
        return tf.keras.layers.LeakyReLU(0.2)
    elif name == 'tanh':
        return tf.keras.layers.Activation('tanh')
    elif name == 'relu':
        return tf.keras.layers.ReLU()
    else:
        return None

def apply_weight_scaled_dense(x, units, gain=np.sqrt(2), activation=None, pixelnorm=False):
    init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    in_filters = x.shape[-1]
    x = tf.keras.layers.Dense(units, use_bias=False, kernel_initializer=init)(x)
    x = WeightScaling(shape=(in_filters,), gain=gain)(x)
    x = Bias()(x)
    if activation:
        x = get_activation(activation)(x)
    if pixelnorm:
        x = PixelNormalization()(x)
    return x

def apply_weight_scaled_conv(x, filters, kernel_size, gain=np.sqrt(2), strides=(1, 1), activation=None, pixelnorm=False):
    init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    in_filters = x.shape[-1]
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False,
                               padding="same", kernel_initializer=init)(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias()(x)
    if activation:
        x = get_activation(activation)(x)
    if pixelnorm:
        x = PixelNormalization()(x)
    return x

def apply_regressor_conv(x, filters, kernel_size, strides=(1,1), pooling=None, activation='leaky_relu'):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = get_activation(activation)(x)
    if pooling == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    elif pooling == 'avg':
        x = tf.keras.layers.AveragePooling2D(pool_size=4, strides=1)(x)
    return x