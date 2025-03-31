import keras
import tensorflow as tf
tf.test.gpu_device_name()

from model_utils.layers import WeightedSum
from model_utils.builders import DiscriminatorBuilder, GeneratorBuilder

class PGAN(keras.Model):
    def __init__(self, configuration:dict, regressor:keras.Model):
        super(PGAN, self).__init__()
        self.gen_builder  = GeneratorBuilder(configuration)
        self.disc_builder = DiscriminatorBuilder(configuration)
        self.generator      = self.gen_builder.build_base_generator()
        self.discriminator  = self.disc_builder.build_base_discriminator()

        self.regressor  = regressor
        self.d_steps    = configuration['D_STEPS']
        self.latent_dim = configuration['LATENT_DIM']
        self.mass_loss_weight = configuration['MASS_LOSS_WEIGHT']
        
    def compile(self, d_optimizer, g_optimizer):
        super(PGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        
    def fade_in_discriminator(self):
        self.n_depth += 1
        self.discriminator = self.disc_builder.build_fade_in_discriminator(
            old_discriminator=self.discriminator,
            current_depth=self.n_depth
        )

    def fade_in_generator(self):
        self.n_depth += 1
        self.generator = self.gen_builder.build_fade_in_generator(
            old_generator=self.generator,
            current_depth=self.n_depth
        )

    def stabilize_discriminator(self):
        old_discriminator = self.discriminator
        stabilized_discriminator = tf.keras.models.clone_model(old_discriminator)
        stabilized_discriminator.set_weights(old_discriminator.get_weights())

        new_input = stabilized_discriminator.input
        x = new_input

        for layer in stabilized_discriminator.layers:
            if isinstance(layer, WeightedSum):
                continue
            x = layer(x)

        self.discriminator = tf.keras.Model(inputs=new_input, outputs=x, name="discriminator_stabilized")
            
    def stabilize_generator(self):
        old_generator = self.generator
        stabilized_generator = tf.keras.models.clone_model(old_generator)
        stabilized_generator.set_weights(old_generator.get_weights())

        new_input = stabilized_generator.input
        x = new_input

        for layer in stabilized_generator.layers:
            if isinstance(layer, WeightedSum):
                continue
            x = layer(x)

        self.generator = tf.keras.Model(inputs=new_input, outputs=x, name="generator_stabilized")

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """
        Calculates the gradient penalty.
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
        
        with tf.GradientTape() as g_tape:
            
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
                        
            gen = self.generator([random_latent_vectors, real_mass], training=True)
            predictions = self.discriminator(gen, training = False)
            predictions_mass = self.regressor.predict(gen)
            
            # Total generator loss
            mass_loss = tf.keras.losses.MeanAbsoluteError()(real_mass, predictions_mass)
            
            g_cost = tf.reduce_mean(predictions)
            g_loss = -g_cost + self.mass_loss_weight * mass_loss
            
            
        # Get the gradients
        g_gradient = g_tape.gradient(g_loss , self.generator.trainable_weights)
        # Update the weights 
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_weights))
        
        return {'d_loss': d_loss, 'g_loss': g_loss, 'mass_loss': mass_loss}