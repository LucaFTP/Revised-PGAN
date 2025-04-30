import os
import numpy as np
import pandas as pd
import tensorflow as tf

from generic_utils import dynamic_range_opt

def create_folders(version:str):
    
    CKPT_OUTPUT_PATH = '/leonardo_scratch/fast/uTS25_Fontana/GAN_ckpts' + version
    IMG_OUTPUT_PATH  = 'results_' + version + '/Images'
    LOSS_OUTPUT_PATH = 'results_' + version + '/Loss'

    try:
        os.mkdir('results_' + version)
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

def load_meta_data(redshift, show=False):
    meta_data = pd.read_csv("mainframe.csv")
    meta_data=meta_data[meta_data['redshift']==redshift]

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

        self.data_dir = "/leonardo_scratch/fast/uTS25_Fontana/ALL_ROT_npy_version/1024x1024/"

    def on_epoch_end(self):
        if self.shuffle:
            print('\n Shuffling the data..')
            self.meta_data = self.meta_data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, img_id, target_size):

        file_name = self.data_dir + img_id + '.npy'

        img = np.load(file_name).astype('float32')
        img = tf.image.resize(np.expand_dims(img, axis=-1), target_size).numpy()

        img = dynamic_range_opt(img, epsilon=self.eps, mult_factor=self.mult_factor)
        
        return img
    
    def __get_output(self, label):
        return label
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_col_batch = batches[self.X_col]
        y_col_batch = batches[self.y_col]

        X_batch = np.asarray([self.__get_input(x, self.target_size) for x in X_col_batch])
        y_batch = np.asarray([self.__get_output(y) for y in y_col_batch])
        
        return X_batch, y_batch
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.n)
        meta_data_batch = self.meta_data[start:end]
        X, y = self.__get_data(meta_data_batch)
        return X, y
    
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))
