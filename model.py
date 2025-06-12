import numpy as np
import tensorflow as tf
from keras import Model, models

from model_utils.blocks import(
    WeightScalingConv,
    WeightScalingDense,
    RegressorConv
)
from model_utils.layers import(
    MinibatchStdev,
    WeightedSum
)

class PGAN(Model):
    def __init__(
            self,
            pgan_config: dict
            ):
        super(PGAN, self).__init__()
        
        self.latent_dim = pgan_config.get('latent_dim');        self.d_steps = pgan_config.get('d_steps')
        self.gp_weight  = pgan_config.get('gp_weight', 10);     self.drift_weight = pgan_config.get('drift_weight', 0.001)
        self.min_resolution = pgan_config.get('min_res', 4);    self.mass_loss_weight = pgan_config.get('mass_loss_weight', 1)

        # Filters
        self.filters = [512, 256, 128, 64, 32, 16, 8]
        self.regressor_filters = [50, 50, 50, 50, 20, 10, 10]
        self.regressor_filters_2 = [50, 50, 50, 20, 10, 10, 10]
        
        self.discriminator = self.init_discriminator()
        self.generator = self.init_generator()
        self.regressor = models.load_model("regressor_results/best_regressor_new_mass_range.keras") #  self.init_regressor()  #

    def call(self, inputs):
        return

    def init_discriminator(self):
        init_shape = (self.min_resolution, self.min_resolution, 1)
        img_input = tf.keras.layers.Input(shape=init_shape)
        img_input = tf.keras.ops.cast(img_input, tf.float32)

        # fromGrayScale
        x = WeightScalingConv(img_input, filters = self.filters[0], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU') # 4 x 4 x 512
        
        # Add Minibatch end of discriminator
        x = MinibatchStdev()(x) # 4 x 4 x 513

        x = WeightScalingConv(x, filters = self.filters[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 4 x 4 x 512
        
        x = WeightScalingConv(x, filters = self.filters[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', strides=(4,4)) # 1 x 1 x 512

        x = tf.keras.layers.Flatten()(x)
        
        x = WeightScalingDense(x, filters=1, gain=1.)
        d_model = Model(img_input, x, name='discriminator')

        return d_model

    # Fade in upper resolution block
    def fade_in_discriminator(self):

        input_shape = list(self.discriminator.input.shape)
        # 1. Double the input resolution. 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3]) # 8 x 8 x 2
        img_input = tf.keras.layers.Input(shape=input_shape)
        img_input = tf.keras.ops.cast(img_input, tf.float32)

        # 2. Add pooling layer 
        #    Reuse the existing “FromGrayScale” block defined as “x1" -- SKIP CONNECTION (ALREADY STABILIZED -> 1-alpha)
        x1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2))(img_input) # 4 x 4 x 1
        x1 = self.discriminator.layers[1](x1) # Conv2D FromGrayScale # 4 x 4 x 512
        x1 = self.discriminator.layers[2](x1) # WeightScalingLayer # 4 x 4 x 512
        x1 = self.discriminator.layers[3](x1) # Bias # 4 x 4 x 512
        x1 = self.discriminator.layers[4](x1) # LeakyReLU # 4 x 4 x 512

        # 3.  Define a "fade in" block (x2) with a new "fromGrayScale" and two 3x3 convolutions.
        # symmetric
        x2 = WeightScalingConv(img_input, filters=self.filters[self.n_depth], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 256

        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 256
        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 512

        x2 = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2))(x2) # 4 x 4 x 512

        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator 
        for i in range(5, len(self.discriminator.layers)):
            x2 = self.discriminator.layers[i](x2)
        self.discriminator_stabilize = Model(img_input, x2, name='discriminator')

        # 5. Add existing discriminator layers. 
        for i in range(5, len(self.discriminator.layers)):
            x = self.discriminator.layers[i](x)
        self.discriminator = Model(img_input, x, name='discriminator')

    # Change to stabilized(c. state) discriminator 
    def stabilize_discriminator(self):
        self.discriminator = self.discriminator_stabilize
        
    def init_regressor(self):
        
        init_shape = (self.min_resolution, self.min_resolution, 1)
        img_input = tf.keras.layers.Input(shape = init_shape)
        img_input = tf.keras.ops.cast(img_input, tf.float32)

        #  [(I - F +2 *P) / S] +1 = 4 x 4 x 50

        x = RegressorConv(img_input, self.regressor_filters[0], kernel_size = 1, pooling=None, activate='LeakyReLU', strides=(1,1))
        
        # print(x.shape) # 4 x 4 x 50
        x = RegressorConv(x, self.regressor_filters[0], kernel_size = 3, pooling='avg', activate='LeakyReLU', strides=(1,1)) 
        # print(x.shape) # should be 1 x 1 x 50
        x = tf.keras.layers.Flatten()(x) # 50
        x = tf.keras.layers.Dense(units = 16)(x) # 16
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        
        x = tf.keras.layers.Dense(units = 1)(x) # 1

        c_model = Model(img_input, x, name='regressor')

        return c_model

    def fade_in_regressor(self):

        input_shape = list(self.regressor.input.shape)
        
        # 1. Double the input resolution. 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3]) # 8 x 8 x 2
        img_input = tf.keras.layers.Input(shape=input_shape)
        img_input = tf.keras.ops.cast(img_input, tf.float32)

        # 2. Add pooling layer 
        x1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2))(img_input) 
        x1 = self.regressor.layers[1](x1) # Conv2D 
        x1 = self.regressor.layers[2](x1) # BatchNormalization 
        x1 = self.regressor.layers[3](x1) # LeakyReLU

        # 3.  Define a "fade in" block (x2) with a new "fromGrayScale" and two 3x3 convolutions.
        if self.n_depth!=5:
            x2 = RegressorConv(img_input, self.regressor_filters_2[self.n_depth], kernel_size=1, pooling=None, activate='LeakyReLU', strides=(1,1))
            x2 = RegressorConv(x2, self.regressor_filters[self.n_depth], kernel_size=3, pooling='max', activate='LeakyReLU', strides=(1,1))
        else:
            x2 = RegressorConv(img_input, self.regressor_filters[self.n_depth], kernel_size=3, pooling='max', activate='LeakyReLU', strides=(1,1))

        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator 
        for i in range(4, len(self.regressor.layers)):
            x2 = self.regressor.layers[i](x2)
        self.regressor_stabilize = Model(img_input, x2, name='regressor')

        # 5. Add existing discriminator layers. 
        for i in range(4, len(self.regressor.layers)):
            x = self.regressor.layers[i](x)
        self.regressor = Model(img_input, x, name='regressor')

    # Change to stabilized(c. state) discriminator 
    def stabilize_regressor(self):
        self.regressor = self.regressor_stabilize

    def init_generator(self):
        noise = tf.keras.layers.Input(shape=(self.latent_dim,), name='noise_input') # None, latent_dim
        mass  = tf.keras.layers.Input(shape=(1,), name='mass_input') # None, 1
        merge = tf.keras.layers.Concatenate()([noise, mass]) # None, latent_dim+1

        # Actual size(After doing reshape) is just FILTERS[0], so divide gain by 4
        x = WeightScalingDense(merge, filters=self.min_resolution*self.min_resolution*self.filters[0], gain=np.sqrt(2)/4, activate='LeakyReLU', use_pixelnorm=False) 
        
        x = tf.keras.layers.Reshape((self.min_resolution, self.min_resolution, self.filters[0]))(x)

        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=False) # 4 x 4 x 512
        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(2,2), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True) # 4 x 4 x 512

        # Gain should be 1 as its the last layer
        x = WeightScalingConv(x, filters=1, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False) # change to tanh and understand gain 1 if training unstable
        x = (x + 1)/2 # Limits the values between 0 and 1

        g_model = Model([noise,mass], x, name='generator')

        return g_model

    # Fade in upper resolution block
    def fade_in_generator(self):
        
        # 1. Get the node above the “toGrayScale” block 
        block_end = self.generator.layers[-5].output
        
        # 2. Upsample block_end
        block_end = tf.keras.layers.UpSampling2D((2,2))(block_end) # 8 x 8 x 512

        # 3. Reuse the existing “toGrayScale” block defined as“x1”. --- SKIP CONNECTION (ALREADY STABILIZED)
        x1 = self.generator.layers[-4](block_end) # Conv2d
        x1 = self.generator.layers[-3](x1) # WeightScalingLayer
        x1 = self.generator.layers[-2](x1) # Bias
        x1 = self.generator.layers[-1](x1) # tanh
        x1 = (x1 + 1)/2 # Limits the values between 0 and 1

        # 4. Define a "fade in" block (x2) with two 3x3 convolutions and a new "toRGB".
        x2 = WeightScalingConv(block_end, filters=self.filters[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True) # 8 x 8 x 512 
        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True) # 8 x 8 x 512   
        
        # "toGrayScale"
        x2 = WeightScalingConv(x2, filters=1, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False) # 
        x2 = (x2 + 1)/2 # Limits the values between 0 and 1

        # Define stabilized(c. state) generator
        self.generator_stabilize = Model(self.generator.input, x2, name='generator')

        # 5.Then "WeightedSum" x1 and x2 to smoothly put the "fade in" block.
        x = WeightedSum()([x1, x2])
        self.generator = Model(self.generator.input, x, name='generator')

    # Change to stabilized(c. state) generator 
    def stabilize_generator(self):
        self.generator = self.generator_stabilize

    def compile(self, d_optimizer, g_optimizer, r_optimizer):
        super(PGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.r_optimizer = r_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        diff = fake_images - real_images
        interpolated = real_images + epsilon * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        
        real_images, real_mass = data
        batch_size = tf.shape(real_images)[0]
        indices = tf.random.shuffle(tf.range(2 * batch_size))

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator([random_latent_vectors, real_mass], training=False)
            combined_images = tf.concat([generated_images, real_images], axis=0)
            shuffled_combined_images = tf.gather(combined_images, indices)
                        
            with tf.GradientTape() as d_tape:

                # Train discriminator
                pred_logits = self.discriminator(shuffled_combined_images, training=True)
                unshuffled_pred_logits = tf.gather(pred_logits, tf.argsort(indices))  

                # Wasserstein Loss
                d_fake = tf.reduce_mean(unshuffled_pred_logits[:batch_size])
                d_real = tf.reduce_mean(unshuffled_pred_logits[batch_size:])

                d_cost = d_fake - d_real

                # Gradient Penalty
                gp = self.gradient_penalty(batch_size, real_images, generated_images)

                # Drift added by PGGAN paper
                drift = tf.reduce_mean(tf.square(pred_logits))

                # WGAN-GP
                d_loss = d_cost + (self.gp_weight * gp) + (self.drift_weight * drift)

            d_gradient = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_weights))
        '''
        with tf.GradientTape() as r_tape:

            # Train regressor
            pred_mass = self.regressor(real_images, training=True)

            # Loss on mass 
            r_loss = tf.keras.losses.MeanAbsoluteError()(real_mass, pred_mass)

        r_gradient = r_tape.gradient(r_loss, self.regressor.trainable_weights) 
        self.r_optimizer.apply_gradients(zip(r_gradient, self.regressor.trainable_weights))
        '''
        with tf.GradientTape() as g_tape:
            
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
                        
            gen = self.generator([random_latent_vectors, real_mass], training=True)
            predictions  = self.discriminator(gen, training=False)
            predictions_mass = self.regressor(gen, training=False)
            
            # Total generator loss
            mass_loss = tf.keras.losses.MeanAbsoluteError()(real_mass, predictions_mass)
            
            g_cost = tf.reduce_mean(predictions)
            g_loss = -g_cost + self.mass_loss_weight * mass_loss
            
        # Get the gradients
        g_gradient = g_tape.gradient(g_loss , self.generator.trainable_weights)
        # Update the weights 
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_weights))
        
        return {'d_loss': d_loss, 'g_loss': g_loss, 'mass_loss': mass_loss}
