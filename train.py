import keras
import numpy as np

from model import PGAN
from data_utils import CustomDataGen
from gan_monitor import GANMonitor, FadeInLRSchedule

class PGANTrainer:
    def __init__(
            self,
            meta_data,
            config: dict,
            pgan: PGAN,
            cbk: GANMonitor,
            loss_out_path: str,
            **kwargs
            ):
        
        self.pgan = pgan;                   self.cbk = cbk
        self.meta_data = meta_data;         self.config = config
        self.loss_out_path = loss_out_path; self.verbose = kwargs.get('verbose', 1)

        self.start_size  = config['start_size'];   self.end_size = config['end_size']
        self.batch_sizes = config['batch_size'];   self.epochs   = config['epochs']

        self.g_lr = config.get('G_LR', 1e-3);      self.beta_1 = config.get('BETA_1', 0.0)
        self.d_lr = config.get('D_LR', 1e-3);      self.beta_2 = config.get('BETA_2', 0.999)
        self.r_lr = config.get('R_LR', 1e-3);      self.fade_in_epochs = config.get('fade_in_epochs', 250)

        self.rescale_eps = config.get('EPS', 1e-6); self.mult_factor = config.get('MULT', 2.5)
    
    def init_optimizers(self, fade_in=False, steps=None):
        if fade_in:
            g_lr_schedule = FadeInLRSchedule(
                initial_lr=self.g_lr,
                min_lr=self.g_lr * 0.1,
                fade_in_steps=self.fade_in_epochs * steps
            )
            d_lr_schedule = FadeInLRSchedule(
                initial_lr=self.d_lr,
                min_lr=self.d_lr * 0.1,
                fade_in_steps=self.fade_in_epochs * steps
            )
            r_lr_schedule = FadeInLRSchedule(
                initial_lr=self.r_lr,
                min_lr=self.r_lr * 0.1,
                fade_in_steps=self.fade_in_epochs * steps
            )
        else:
            g_lr_schedule = self.g_lr
            d_lr_schedule = self.d_lr
            r_lr_schedule = self.r_lr

        self.g_optimizer = keras.optimizers.Adam(
            learning_rate=g_lr_schedule, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=1e-8)
        self.d_optimizer = keras.optimizers.Adam(
            learning_rate=d_lr_schedule, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=1e-8)
        self.r_optimizer = keras.optimizers.Adam(
            learning_rate=r_lr_schedule, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=1e-8)
        self.pgan.compile(d_optimizer=self.d_optimizer, g_optimizer=self.g_optimizer, r_optimizer=self.r_optimizer)

    def _make_dataset(self, size, batch_size):
        return CustomDataGen(
            self.meta_data, X_col='id', y_col='mass', batch_size=batch_size,
            target_size=(size, size), shuffle=True, epsilon=self.rescale_eps, mult_factor=self.mult_factor
        )

    def _fit_and_log(self, dataset, prefix, steps, epochs):
        self.cbk.set_prefix(prefix)
        self.cbk.set_steps(steps_per_epoch=steps, epochs=epochs)
        history = self.pgan.fit(dataset, epochs=epochs, callbacks=[self.cbk], verbose=self.verbose)
        np.save(f'{self.loss_out_path}/history_{prefix}.npy', history.history)
        return history

    def train(self):
        max_depth = int(np.log2(self.end_size/2))
        initial_depth = int(np.log2(self.start_size/2))
        current_size = self.start_size

        if self.start_size == self.pgan.min_resolution:
            self.pgan.n_depth = 0

        for n_depth in range(1, initial_depth):
            self.pgan.n_depth = n_depth
            self.pgan.fade_in_generator()
            self.pgan.fade_in_discriminator()
            self.pgan.fade_in_regressor()

            self.pgan.stabilize_generator()
            self.pgan.stabilize_discriminator()
            self.pgan.stabilize_regressor()
        
        print(f"Starting training at {self.start_size}x{self.start_size}")
        self.init_optimizers()
        dataset = self._make_dataset(self.start_size, self.batch_sizes[0])
        history_init = self._fit_and_log(dataset, f'{self.pgan.n_depth}_init', len(dataset), self.epochs)

        if initial_depth == max_depth:
            return self.pgan

        for n_depth in range(initial_depth, max_depth):
            current_size = current_size * 2
            print(f"\n>> Fading in at size {current_size}x{current_size}")
            self.pgan.n_depth = n_depth
            dataset = self._make_dataset(current_size, self.batch_sizes[n_depth - initial_depth])

            self.pgan.fade_in_generator()
            self.pgan.fade_in_discriminator()
            self.pgan.fade_in_regressor()
            self.init_optimizers(fade_in=True, steps=len(dataset))

            history_fade_in = self._fit_and_log(dataset, f'{n_depth}_fade_in', len(dataset), self.fade_in_epochs)

            print(f"\n>> Stabilizing at size {current_size}x{current_size}")
            self.pgan.stabilize_generator()
            self.pgan.stabilize_discriminator()
            self.pgan.stabilize_regressor()
            self.init_optimizers()

            history_stabilize = self._fit_and_log(dataset, f'{n_depth}_stabilize', len(dataset), self.epochs)

        print(f"\n>> Final training at size {self.end_size}x{self.end_size}")
        dataset = self._make_dataset(self.end_size, self.batch_sizes[max_depth - initial_depth])
        history_final_step = self._fit_and_log(dataset, f'{self.pgan.n_depth}_final', len(dataset), 2 * self.epochs)

        return self.pgan
