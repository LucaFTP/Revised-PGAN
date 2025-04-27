import keras
from tensorflow.keras import backend as K # type: ignore

from model_utils.layers import(
    WeightScaling,
    Bias,
    PixelNormalization
)

def WeightScalingDense(x, filters, gain, use_pixelnorm=False, activate=None):
    init = keras.initializers.RandomNormal(mean=0., stddev=1.)
    in_filters = K.int_shape(x)[-1]
    x = keras.layers.Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate=='LeakyReLU':
        x = keras.layers.LeakyReLU(0.2)(x)
    elif activate=='tanh':
        x = keras.layers.Activation('tanh')(x)

    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x

def WeightScalingConv(x, filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=(1,1)):
    init = keras.initializers.RandomNormal(mean=0., stddev=1.)
    in_filters = K.int_shape(x)[-1]
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate=='LeakyReLU':
        x = keras.layers.LeakyReLU(0.2)(x)
    elif activate=='tanh':
        x = keras.layers.Activation('tanh')(x)
    
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x 

def RegressorConv(x, filters, kernel_size, pooling=None, activate=None, strides=(1,1)):

    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", dtype='float32')(x)
    x = keras.layers.BatchNormalization()(x)
    if activate=='LeakyReLU':
        x = keras.layers.LeakyReLU(0.01)(x)
    if pooling=='max':
        x = keras.layers.MaxPooling2D(pool_size = 2, strides = 2)(x)
    elif pooling=='avg':
        x = keras.layers.AveragePooling2D(pool_size = 4, strides = 1)(x)
    return x 