import os
import json
import matplotlib
from argparse import ArgumentParser
from tensorflow.keras.applications.inception_v3 import InceptionV3 # type: ignore

from model import PGAN
from train import PGANTrainer
from gan_monitor import GANMonitor
from generic_utils import prepare_real_images
from data_utils import load_meta_data, create_folders

# Parser creation
parser = ArgumentParser(
    description="Options for the execution of the code."
)
parser.add_argument(
    "-c",
    "--config-filepath",
    help="The config filepath for the model/trainer config (Litteral filepath form this file)",
    type=str,
    required=True,
)
parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=1,
    required=False,
    help="0 = silent, 1 = progress bar, 2 = one line per epoch."
)
parser.add_argument(
    "-m",
    "--model-milestone",
    type=str,
    default=None,
    required=False,
    help="The model milestone to load. If None, the latest model will be loaded."
)
args = parser.parse_args()

if not os.path.isfile(args.config_filepath):
        raise FileNotFoundError(args.config_filepath)

with open(args.config_filepath, "r") as f:
    config_file = json.load(f)

train_config = config_file.get('train_config')
model_config = config_file.get('model_config')

version = config_file.get('version')
CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, LOSS_OUTPUT_PATH = create_folders(version=version)

meta_data = load_meta_data(train_config.get('z_th'), show=True)
print(f"Data Shape: {meta_data.shape}")

# Computing the Fid parameters associated with the real dataset
fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
mu1, sigma1 = prepare_real_images(fid_model=fid_model, meta_data=meta_data, target_size=train_config.get('end_size'))

pgan = PGAN(pgan_config=model_config, version=version)

cbk = GANMonitor(
    num_img=150,
    latent_dim=model_config.get('latent_dim'),
    plot_every=15,
    fid_model=fid_model,
    fid_real_par=(mu1, sigma1),
    checkpoint_dir=CKPT_OUTPUT_PATH,
    image_path=IMG_OUTPUT_PATH
    )

trainer = PGANTrainer(
    meta_data=meta_data,
    config=train_config,
    pgan=pgan,
    cbk=cbk,
    loss_out_path=LOSS_OUTPUT_PATH,
    verbose=args.verbose,
    milestone=args.model_milestone
    )
trainer.train()