import os, sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore

from model_utils.layers import WeightedSum
from generic_utils import (
    prepare_fake_images,
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
            plot_every:int,
            checkpoint_dir:str,
            fid_model,
            fid_real_par:tuple[float, float],
            prefix:str = '',
            **kwargs
            ):

        # Prefix for correct saving path
        self.prefix  = prefix;   self.image_path = image_path;   self.checkpoint_dir = checkpoint_dir

        # In case of milestone, define an initial epoch different from 0
        self.milestone = kwargs.get('milestone', None)
        self.delta = int(self.milestone) if self.milestone is not None else 0

        # Latent vectors initialization
        self.plot_every = plot_every
        self.random_latent_vectors = tf.random.normal(shape=[num_img, latent_dim])
        self.mass = tf.convert_to_tensor(np.round(tf.random.uniform(
                        shape=[num_img, 1], minval=1, maxval=16), 2))
        
        # FID score tracking
        self.fid_scores = [];   self.fid_model = fid_model;   self.mu1, self.cov1 = fid_real_par

    def set_prefix(self, prefix:str):
        self.prefix = prefix
        
    def set_steps(self, steps_per_epoch, epochs):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs + self.delta
        self.steps = self.steps_per_epoch * self.epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.n_epoch = epoch + self.delta
        checkpoint_path = f"{self.checkpoint_dir}/pgan_{self.prefix}_{self.n_epoch:04d}.weights.h5"
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs=None):
        self.n_epoch = epoch

        # Plot
        if (epoch + self.delta) % self.plot_every == 0 and epoch != 0:
            save_path = f'{self.image_path}/plot_{self.prefix}_{epoch + self.delta:03d}.png'
            plot_images(self.model.generator([self.random_latent_vectors, self.mass]), save_path=save_path)

        # Save weights + FID
        try:
            prefix_number, prefix_state = self.prefix.split("_")
            prefix_number = int(prefix_number)
        except ValueError:
            print(f"[GANMonitor] Unexpected prefix format: '{self.prefix}'. Skipping FID/weights saving.")
            return

        if (epoch + self.delta) % 15 == 0 and epoch != 0:
            # generated_imgs = self.model.generator([self.random_latent_vectors, self.mass])
            # synthetic_images = prepare_fake_images(generated_imgs)
            # fid = calculate_fid(self.fid_model, self.mu1, self.cov1, synthetic_images)
            # self.fid_scores.append(fid)

            # print(f"[GANMonitor] Epoch {epoch}: FID = {fid:.4f}")
            self.model.save_weights(self.checkpoint_path)
            print(f"[GANMonitor] Weights saved at {self.checkpoint_path}")
            
    def on_batch_begin(self, batch, logs=None):
        
        # Update alpha in WeightedSum layers
        # alpha usually goes from 0 to 1 evenly over ALL the epochs for that depth.
        alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / float(self.steps - 1) #1/219  to 1*110+109/220 for 2 epochs
        # print(f"[GANMonitor] Steps: {self.steps}, Epoch: {self.n_epoch}, Steps per epoch: {self.steps_per_epoch}, Batch: {batch}, Alpha: {alpha}")
        
        for layer in self.model.generator.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)
        for layer in self.model.discriminator.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)

class FadeInLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, min_lr, fade_in_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.fade_in_steps = fade_in_steps

    def __call__(self, step):
        # decrescita lineare durante il fade-in
        decay = (1.0 - tf.minimum(step / self.fade_in_steps, 1.0))
        return self.min_lr + decay * (self.initial_lr - self.min_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "fade_in_steps": self.fade_in_steps,
        }
    
class FirstBatchStderrFilter(tf.keras.callbacks.Callback):
    '''
    Suppress stderr output for the first training batch to remove the
    tens of thousands gpu_timer warnings.
    '''
    def on_train_batch_begin(self, batch, logs=None):
        if batch == 0:
            self._orig_fd = os.dup(sys.stderr.fileno())
            self._devnull = open('/dev/null', 'w')
            os.dup2(self._devnull.fileno(), sys.stderr.fileno())

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0:
            os.dup2(self._orig_fd, sys.stderr.fileno())
            os.close(self._orig_fd)
            self._devnull.close()