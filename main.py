import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3 # type: ignore

from model import PGAN; from train import PGANTrainer; from gan_monitor import GANMonitor
from utils import prepare_real_images; from data_utils import load_meta_data, create_folders

from config import CONFIG
[print(*item) for item in CONFIG.items()]

version = '_z_th_' + str(CONFIG['z_th'])
CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, LOSS_OUTPUT_PATH = create_folders(version=version)

meta_data = load_meta_data(CONFIG['z_th'], show=True)
print(f"Data Shape: {meta_data.shape}")

# Computing the Fid parameters associated with the real dataset
fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
mu1, sigma1 = prepare_real_images(fid_model=fid_model, meta_data=meta_data, target_size=CONFIG['END_SIZE'])

pgan = PGAN(configuration=CONFIG, regressor=trained_R)
cbk  = GANMonitor(
    num_img=len(meta_data), latent_dim=CONFIG['LATENT_DIM'],
    image_path=IMG_OUTPUT_PATH, checkpoint_dir=CKPT_OUTPUT_PATH,
    fid_model=InceptionV3, fid_real_par=(mu1, sigma1)
    )

trainer = PGANTrainer(meta_data, CONFIG, pgan, cbk, LOSS_OUTPUT_PATH)
trained_pgan = trainer.train()

np.save(f"results{version}/fid_scores", cbk.fid_scores)