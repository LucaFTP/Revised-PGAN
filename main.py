import argparse
import matplotlib
from tensorflow.keras.applications.inception_v3 import InceptionV3 # type: ignore

from model import PGAN
from config import CONFIG
from train import PGANTrainer
from gan_monitor import GANMonitor
from generic_utils import prepare_real_images
from data_utils import load_meta_data, create_folders

# Parser creation
parser = argparse.ArgumentParser(
    description="Options for the execution of the code."
)
parser.add_argument("-v", "--verbose", type=int, default=1, required=False,
                    help="0 = silent, 1 = progress bar, 2 = one line per epoch.")
args = parser.parse_args()

REDSHIFT    = CONFIG.get('z_th')
D_STEPS     = CONFIG.get('D_STEPS', 5)
EPOCHS      = CONFIG.get('EPOCHS')
NOISE_DIM   = CONFIG.get('LATENT_DIM', 6)
START_SIZE  = CONFIG.get('START_SIZE')
END_SIZE    = CONFIG.get('END_SIZE')
BATCH_SIZE  = CONFIG.get('BATCH_SIZE')

GP_WEIGHT   = CONFIG.get('GP_WEIGHT', 10)
DRIFT_WEIGHT= CONFIG.get('DRIFT_WEIGHT', 0.001)
MASS_LOSS_WEIGHT = CONFIG.get('MASS_LOSS_WEIGHT', 1)

version = '_z_' + str(REDSHIFT)
CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, LOSS_OUTPUT_PATH = create_folders(version=version)

meta_data = load_meta_data(REDSHIFT, show=True)
print(f"Data Shape: {meta_data.shape}")

# Computing the Fid parameters associated with the real dataset
fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
mu1, sigma1 = prepare_real_images(fid_model=fid_model, meta_data=meta_data, target_size=END_SIZE)

pgan = PGAN(
    latent_dim=NOISE_DIM,
    d_steps=D_STEPS,
    gp_weight=GP_WEIGHT,
    drift_weight=DRIFT_WEIGHT,
    mass_loss_weight=MASS_LOSS_WEIGHT
    )

cbk = GANMonitor(
    num_img=350,
    latent_dim = NOISE_DIM,
    plot_every=25,
    fid_model=fid_model,
    fid_real_par=(mu1, sigma1),
    checkpoint_dir=CKPT_OUTPUT_PATH,
    image_path=IMG_OUTPUT_PATH
    )

trainer = PGANTrainer(
    meta_data=meta_data,
    config=CONFIG,
    pgan=pgan,
    cbk=cbk,
    loss_out_path=LOSS_OUTPUT_PATH,
    verbose=args.verbose
    )
trainer.train()