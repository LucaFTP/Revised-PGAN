import os
import json
import matplotlib
from argparse import ArgumentParser
from tensorflow.keras.applications.inception_v3 import InceptionV3 # type: ignore

from train import PGANTrainer
from gan_monitor import GANMonitor
from generic_utils import prepare_real_images, parser
from data_utils import load_meta_data, create_folders

def parse_arguments(parser: ArgumentParser) -> ArgumentParser:
    """
    Function to parse the arguments for the script.
    """
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
        help="The model milestone to load. If None, the training will restart from the beginning."
    )
    parser.add_argument(
        "-r",
        "--trained-regressor",
        action="store_true",
        help="If True, the regressor will be a pre-trained model. If False, the regressor will be trained."
    )
    return parser

@parser(
    "Progressive GAN training script",
    "Progressive GAN training script",
    parse_arguments,
)
def main(command_line_args: ArgumentParser) -> None:
    if not os.path.isfile(command_line_args.config_filepath):
            raise FileNotFoundError(command_line_args.config_filepath)

    with open(command_line_args.config_filepath, "r") as f:
        config_file = json.load(f)

    train_config = config_file.get('train_config')
    model_config = config_file.get('model_config')

    version = config_file.get('version')
    CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, LOSS_OUTPUT_PATH = create_folders(version=version)

    meta_data = load_meta_data(train_config.get('z_th'), show=True)
    print(f"Data Shape: {meta_data.shape}")

    # Computing the Fid parameters associated with the real dataset
    fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # mu1, sigma1 = prepare_real_images(fid_model=fid_model, meta_data=meta_data, target_size=train_config.get('end_size'))

    cbk = GANMonitor(
        num_img=150,
        latent_dim=model_config.get('latent_dim'),
        plot_every=15,
        fid_model=fid_model,
        fid_real_par=(0, 0),
        checkpoint_dir=CKPT_OUTPUT_PATH,
        image_path=IMG_OUTPUT_PATH
        )

    trainer = PGANTrainer(
        meta_data=meta_data,
        config=train_config,
        pgan_config=model_config,
        trained_reg=version if command_line_args.trained_regressor else None,
        cbk=cbk,
        loss_out_path=LOSS_OUTPUT_PATH,
        verbose=command_line_args.verbose,
        milestone=command_line_args.model_milestone
        )
    trainer.train()

if __name__ == "__main__":
    main()