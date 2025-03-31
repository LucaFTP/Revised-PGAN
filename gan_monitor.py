import numpy as np
import tensorflow as tf
from keras import backend as K

from model_utils.layers import WeightedSum
from utils import (prepare_fake_images,
                   calculate_fid,
                   plot_images
)

# Saves generated images and updates alpha in WeightedSum layers
class GANMonitor(tf.keras.callbacks.Callback):
    
    def __init__(
            self,
            num_img:int,
            latent_dim:int,
            image_path:str,
            checkpoint_dir:str,
            fid_model,
            fid_real_par:tuple[float, float],
            prefix:str = ''
            ):

        self.prefix  = prefix
        self.image_path = image_path; self.checkpoint_dir = checkpoint_dir
        self.random_latent_vectors = tf.random.normal(shape=[num_img, latent_dim])
        self.mass = tf.convert_to_tensor(np.round(tf.random.uniform(
                        shape=[num_img, 1], minval=14,maxval=14.95),2))
        
        self.fid_scores = []
        self.fid_model = fid_model
        self.mu1, self.cov1 = fid_real_par

    def set_prefix(self, prefix=''):
        self.prefix = prefix
        
    def set_steps(self, steps_per_epoch, epochs):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.steps = self.steps_per_epoch * self.epochs

    def on_epoch_begin(self, epoch):
        self.n_epoch = epoch
        checkpoint_path = f"{self.checkpoint_dir}/pgan_{self.n_epoch:04d}.weights.h5"
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch):
        self.n_epoch = epoch

        try:
            prefix_number, prefix_state = self.prefix.split("_")
            prefix_number = int(prefix_number)
        except ValueError:
            print(f"[GANMonitor] Unexpected prefix format: '{self.prefix}'. Skipping FID/weights saving.")
            return

        # Plot
        if epoch % 20 == 0:
            save_path = f'{self.image_path}/plot_{self.prefix}_{epoch:03d}.png'
            plot_images(self.model.generator([self.random_latent_vectors, self.mass]), save_path=save_path)

        # Save weights + FID
        if prefix_number == 5 and prefix_state == 'final' and epoch % 10 == 0:
            generated_imgs = self.model.generator([self.random_latent_vectors, self.mass])
            synthetic_images = prepare_fake_images(generated_imgs)
            fid = calculate_fid(self.fid_model, self.mu1, self.cov1, synthetic_images)
            self.fid_scores.append(fid)

            print(f"[GANMonitor] Epoch {epoch}: FID = {fid:.4f}")
            self.model.save_weights(self.checkpoint_path)
            print(f"[GANMonitor] Weights saved at {self.checkpoint_path}")
            
    def on_batch_begin(self, batch):
        
        # Update alpha in WeightedSum layers
        # alpha usually goes from 0 to 1 evenly over ALL the epochs for that depth.
        alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / float(self.steps - 1) #1/219  to 1*110+109/220 for 2 epochs
        
        # print(f'!!! From GANMonitor: Steps: {self.steps}, Epoch: {self.n_epoch}, Steps per epoch: {self.steps_per_epoch}, Batch: {batch}, Alpha: {alpha}')
        
        for layer in self.model.generator.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)
        for layer in self.model.discriminator.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)
                