import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore

from utils import compute_depth_from_res
from layers import MinibatchStdev, WeightedSum
from blocks import apply_weight_scaled_conv, apply_weight_scaled_dense

class DiscriminatorBuilder:
    def __init__(self, config:dict):
        self.min_res     = config.get('MIN_RES', 4)
        self.gain        = config.get('GAIN', np.sqrt(2))
        self.activation  = config.get('ACTIVATION', 'leaky_relu')
        self.filters     = config.get('FILTERS', [512, 256, 128, 64, 32, 16, 8])
        
        self.initial_res = config['INITIAL_RES']
        self.depth = compute_depth_from_res(self.initial_res, self.min_res)

    def build_base_discriminator(self):
        input_shape = (self.initial_res, self.initial_res, 1)
        img_input = tf.keras.layers.Input(shape=input_shape, name="disc_input")
        x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(img_input)

        # Apply blocks from current resolution to 4x4
        for i in range(self.depth):
            f_in = self.filters[i]
            f_out = self.filters[i + 1]
            x = apply_weight_scaled_conv(x, filters=f_in, kernel_size=(3, 3),
                                         gain=self.gain, activation=self.activation)
            x = apply_weight_scaled_conv(x, filters=f_out, kernel_size=(3, 3),
                                         gain=self.gain, activation=self.activation)
            x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

        # Final 4x4 block
        x = MinibatchStdev()(x)
        x = apply_weight_scaled_conv(x, filters=self.filters[self.depth],
                                     kernel_size=(3, 3), gain=self.gain, activation=self.activation)
        x = apply_weight_scaled_conv(x, filters=self.filters[self.depth],
                                     kernel_size=(4, 4), gain=self.gain,
                                     activation=self.activation, strides=(4, 4))
        x = tf.keras.layers.Flatten()(x)
        x = apply_weight_scaled_dense(x, units=1, gain=1.0)

        return Model(img_input, x, name="discriminator")
    
    def build_fade_in_discriminator(self, old_discriminator, current_depth):
        """Builds a new discriminator with a fade-in layer at the top"""
        assert current_depth > 0, "Cannot fade in at depth 0"

        # 1. New input shape (double resolution)
        res = self.min_res * (2 ** current_depth)  # es: 4 * 2^5 = 128
        input_shape = (res, res, 1)
        img_input = tf.keras.Input(shape=input_shape, name=f"disc_input_{res}x{res}")
        x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(img_input)

        # 2. Branch 1 – Old pathway (downsample + use previous discriminator block)
        x1 = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        x1 = old_discriminator(x1)

        # 3. Branch 2 – New block at higher res
        f_in = self.filters[current_depth]
        f_out = self.filters[current_depth - 1]

        x2 = apply_weight_scaled_conv(x, filters=f_in, kernel_size=(1, 1),
                                      gain=self.gain, activation=self.activation)
        x2 = apply_weight_scaled_conv(x2, filters=f_in, kernel_size=(3, 3),
                                      gain=self.gain, activation=self.activation)
        x2 = apply_weight_scaled_conv(x2, filters=f_out, kernel_size=(3, 3),
                                      gain=self.gain, activation=self.activation)
        x2 = tf.keras.layers.AveragePooling2D(pool_size=2)(x2)

        # 4. Weighted sum (alpha will be a variable controlled externally)
        x = WeightedSum()([x1, x2])

        # 5. Add remaining old discriminator layers (skip Input layer)
        for layer in old_discriminator.layers[1:]:
            x = layer(x)

        return Model(inputs=img_input, outputs=x, name=f"discriminator_fadein_{res}")

class GeneratorBuilder:
    def __init__(self, config:dict):
        self.min_res     = config.get('MIN_RES', 4)
        self.gain        = config.get('GAIN', np.sqrt(2))
        self.activation  = config.get('ACTIVATION', 'leaky_relu')
        self.filters     = config.get('FILTERS', [512, 256, 128, 64, 32, 16, 8])

        self.latent_dim  = config['LATENT_DIM']
        self.initial_res = config['INITIAL_RES']

    def build_base_generator(self):
        noise = tf.keras.Input(shape=(self.latent_dim,), name="noise_input")
        mass = tf.keras.Input(shape=(1,), name="mass_input")

        x = tf.keras.layers.Concatenate(name="latent_merge")([noise, mass])
        x = apply_weight_scaled_dense(
            x,
            filters=self.min_res * self.min_res * self.filters[0],
            gain=self.gain / 4,
            activation=self.activation,
            use_pixelnorm=False
        )
        x = tf.keras.layers.Reshape((self.min_res, self.min_res, self.filters[0]), name="reshape")(x)

        depth = int(np.log2(self.initial_res // self.min_res))
        for i in range(depth):
            f_in = self.filters[i]       # Input channels
            f_out = self.filters[i + 1]  # Output channels
            x = tf.keras.layers.UpSampling2D()(x)  # Increase in resolution

            x = apply_weight_scaled_conv(x, filters=f_in, kernel_size=(3, 3),
                                         gain=self.gain, activation=self.activation, use_pixelnorm=True)

            x = apply_weight_scaled_conv(x, filters=f_out, kernel_size=(3, 3),
                                         gain=self.gain, activation=self.activation, use_pixelnorm=True)

        x = apply_weight_scaled_conv(x, filters=1, kernel_size=(1, 1),
                                    gain=1., activation='tanh', use_pixelnorm=False)

        x = tf.keras.layers.Lambda(lambda x: (x + 1) / 2, name="rescale_0_1")(x)

        return tf.keras.Model(inputs=[noise, mass], outputs=x, name=f"generator_{self.initial_res}")

    def build_fade_in_generator(self, old_generator, current_depth):
        assert current_depth > 0, "Cannot fade in at depth 0"

        res = self.min_res * (2 ** current_depth)
        input_noise = tf.keras.Input(shape=(self.latent_dim,), name="noise_input")
        input_mass = tf.keras.Input(shape=(1,), name="mass_input")

        old_out = old_generator([input_noise, input_mass])
        x1 = tf.keras.layers.UpSampling2D()(old_out)

        x = old_generator.layers[-5].output
        x = tf.keras.layers.UpSampling2D()(x)

        f_out = self.filters[current_depth]
        f_in = self.filters[current_depth - 1]

        x = apply_weight_scaled_conv(x, filters=f_in, kernel_size=(3, 3),
                                     gain=self.gain, activation=self.activation, use_pixelnorm=True)
        x = apply_weight_scaled_conv(x, filters=f_out, kernel_size=(3, 3),
                                     gain=self.gain, activation=self.activation, use_pixelnorm=True)

        x2 = old_generator.layers[-4](x)  # Conv
        x2 = old_generator.layers[-3](x)  # WeightScaling
        x2 = old_generator.layers[-2](x)  # Bias
        x2 = old_generator.layers[-1](x)  # Tanh + Rescale

        x = WeightedSum()([x1, x2])

        return tf.keras.Model(inputs=[input_noise, input_mass], outputs=x, name=f"generator_fadein_{res}")