import math
import numpy as np
import tensorflow as tf

from model import PGAN
from gan_monitor import GANMonitor
from data_utils import CustomDataGen

class PGANTrainer:
    def __init__(
            self,
            meta_data,
            config: dict,
            pgan: PGAN,
            cbk: GANMonitor,
            loss_out_path: str
            ):
        
        self.cbk = cbk
        self.pgan = pgan
        self.config = config
        self.meta_data = meta_data
        self.loss_out_path = loss_out_path

        self.start_size  = config['START_SIZE'];   self.end_size = config['END_SIZE']
        self.batch_sizes = config['BATCH_SIZE'];   self.epochs   = config['EPOCHS']

        self.g_lr = config.get('G_LR', 1e-3);      self.beta_1 = config.get('BETA_1', 0.0)
        self.d_lr = config.get('D_LR', 1e-3);      self.beta_2 = config.get('BETA_2', 0.999)

        self.init_optimizers()
    
    def init_optimizers(self):
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.g_lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=1e-8)
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.d_lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=1e-8)
        self.pgan.compile(d_optimizer=self.d_optimizer, g_optimizer=self.g_optimizer)

    def _make_dataset(self, size, batch_size):
        return CustomDataGen(
            self.meta_data, X_col='id', y_col='mass', batch_size=batch_size,
            target_size=(size, size), shuffle=True
        )

    def _fit_and_log(self, dataset, prefix, steps, epochs):
        self.cbk.set_prefix(prefix)
        self.cbk.set_steps(steps_per_epoch=steps, epochs=epochs)
        history = self.pgan.fit(dataset, epochs=epochs, callbacks=[self.cbk])
        np.save(f'{self.loss_out_path}/history_{prefix}.npy', history.history)
        return history

    def train(self):
        print(f"Starting training at {self.start_size}x{self.start_size}")
        dataset = self._make_dataset(self.start_size, self.batch_sizes[0])
        self._fit_and_log(dataset, 'init', len(dataset), self.epochs)

        max_depth = int(math.log(self.end_size, 2))
        initial_depth = int(math.log(self.start_size, 2))
        for n_depth in range(initial_depth, max_depth):
            current_size = self.start_size * (2 ** n_depth)
            print(f"\n>> Fading in at size {current_size}x{current_size}")
            self.pgan.n_depth = n_depth
            dataset = self._make_dataset(current_size, self.batch_sizes[n_depth])

            self.pgan.fade_in_generator()
            self.pgan.fade_in_discriminator()
            self.init_optimizers()

            self._fit_and_log(dataset, f'{n_depth}_fade_in', len(dataset), self.epochs)

            print(f"\n>> Stabilizing at size {current_size}x{current_size}")
            self.pgan.stabilize_generator()
            self.pgan.stabilize_discriminator()
            self.init_optimizers()

            self._fit_and_log(dataset, f'{n_depth}_stabilize', len(dataset), self.epochs)

        print(f"\n>> Final training at size {self.end_size}x{self.end_size}")
        dataset = self._make_dataset(self.end_size, self.batch_sizes[max_depth])
        self._fit_and_log(dataset, f'{max_depth}_final', len(dataset), 2 * self.epochs)

        return self.pgan