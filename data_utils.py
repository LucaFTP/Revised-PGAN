import os
import numpy as np
import pandas as pd
import tensorflow as tf

from generic_utils import dynamic_range_opt

def create_folders(version:str):
    
    CKPT_OUTPUT_PATH = '/leonardo_work/uTS25_Fontana/Gan_results_and_ckpts/GAN_ckpts_' + version
    IMG_OUTPUT_PATH  = 'results/results_' + version + '/Images'
    LOSS_OUTPUT_PATH = 'results/results_' + version + '/Loss'

    try:
        os.mkdir('results/results_' + version)
    except FileExistsError:
        pass
    
    try:
        os.mkdir(CKPT_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(IMG_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(LOSS_OUTPUT_PATH)
    except FileExistsError:
        pass

    return CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, LOSS_OUTPUT_PATH

def get_unique(data):
    for col in data.columns[1:]:
        print(f'\n"{col}" has {len(data[col].unique())} unique values: {data[col].unique()}')

def load_meta_data(csv_file, redshift, show=False):
    meta_data = pd.read_csv(csv_file)
    meta_data = meta_data[meta_data['redshift']<=redshift]
    # meta_data = meta_data[meta_data['redshift']>=0.2]

    meta_data = meta_data[['id','redshift', 'mass', 'simulation', 'snap', 
                           'ax', 'rot']].drop_duplicates()
    
    # Showing what all is in my data
    if show:
        get_unique(meta_data)
    
    return meta_data
        
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(
            self,
            meta_data: pd.DataFrame,
            batch_size: int,
            target_size: int,
            X_col: str = 'id',
            y_col: str = 'mass',
            shuffle=True,
            **kwargs
            ):
        super().__init__()
        
        self.meta_data = meta_data.copy();  self.n = len(self.meta_data);   self.shuffle = shuffle
        
        self.X_col = X_col;     self.batch_size = batch_size
        self.y_col = y_col;     self.target_size = target_size
        
        self.eps = kwargs.get('epsilon', 1e-6); self.mult_factor = kwargs.get('mult_factor', 2.5)

        self.data_dir = "/leonardo_scratch/fast/uTS25_Fontana/redshift_zero_folder/"

    def on_epoch_end(self):
        if self.shuffle:
            print('\n Shuffling the data..')
            self.meta_data = self.meta_data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, img_id, mass, target_size):

        file_name = self.data_dir + img_id + '.npy'

        img = np.load(file_name).astype('float32')
        # img = physical_units_zoom(img=img, mass=mass, side_length=3000)
        # img = img[362:662, 332:662]
        img = tf.image.resize(np.expand_dims(img, axis=-1), target_size).numpy()
        img = dynamic_range_opt(img, epsilon=self.eps, mult_factor=self.mult_factor)
        
        return img
    
    def __get_output(self, label):
        return label
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_col_batch = batches[self.X_col]
        y_col_batch = batches[self.y_col]

        X_batch = np.asarray([self.__get_input(x, y, self.target_size) for x, y in zip(X_col_batch, y_col_batch)])
        y_batch = np.asarray([self.__get_output(10**(y - 13.8)) for y in y_col_batch])

        return X_batch, y_batch

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.n)
        meta_data_batch = self.meta_data[start:end]
        X, y = self.__get_data(meta_data_batch)
        return X, y
    
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))


### --------------------- TF DATASET VERSION --------------------- ###
class CustomDataTF:
    def __init__(
            self,
            meta_data: pd.DataFrame,
            target_size: tuple[int, int],
            X_col="id",
            y_col="mass",
            epsilon=1e-6,
            mult_factor=2.5
            ):
        self.meta_data = meta_data.copy()
        self.target_size = target_size
        self.data_dir = "/leonardo_scratch/fast/uTS25_Fontana/redshift_zero_folder/"
        self.X_col = X_col;     self.y_col = y_col
        self.eps = epsilon;     self.mult_factor = mult_factor

    def __len__(self):
        return len(self.meta_data)

    def _load_example(self, img_id, mass):
        file_name = tf.strings.join([self.data_dir, img_id, ".npy"])
        img = tf.numpy_function(np.load, [file_name], tf.float32)
        img = tf.reshape(img, [1024, 1024])
        # img = img[362:662, 332:662]
        img = tf.expand_dims(img, axis=-1)
        img = tf.image.resize(img, self.target_size)

        img = dynamic_range_opt(img, epsilon=self.eps, mult_factor=self.mult_factor)
        label = tf.math.pow(tf.constant(10.0, dtype=tf.float64), (mass - 13.8))
        return img, label

    def get_dataset(
            self, batch_size: int, shuffle: bool = True
            ) -> tf.data.Dataset:
        ids = self.meta_data[self.X_col].values
        masses = self.meta_data[self.y_col].values

        ds = tf.data.Dataset.from_tensor_slices((ids, masses))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.meta_data))

        ds = ds.map(lambda img_id, m: self._load_example(img_id, m),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
    
if __name__ == "__main__":
    # Example usage
    meta_data = load_meta_data(redshift=0.25, show=True)
    print("Loaded meta data.")
    print(f"Maximum mass: {meta_data['mass'].max()} \t Minimum mass: {meta_data['mass'].min()}")
    # data_gen = CustomDataGen(meta_data, batch_size=32, target_size=(128, 128))
    data_gen = CustomDataTF(meta_data, target_size=(128, 128), epsilon=5e-6).get_dataset(batch_size=32)
    print(f"Number of batches per epoch: {len(data_gen)}")

    for X, y in data_gen:
        print(f"Batch X shape: {X.shape}, Batch y shape: {y.shape}")
        print(f"Masses: {y}")
        print(f"Maximum value in images: {np.max(X)}")
        print(f"Minimum value in images: {np.min(X)}")
        break  # Just to show the first batch
