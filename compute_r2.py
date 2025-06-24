import json
import keras
import timeit
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from argparse import ArgumentParser
from sklearn import metrics, model_selection

from model import PGAN
from train_regressor import build_regressor
from data_utils import CustomDataGen, load_meta_data

to_mass_range = lambda x: 10**(x - 13.8)

def compute_tstr_trst(
        train_config: dict, model: PGAN, meta_data: pd.DataFrame, num_imgs: int=2500, batch_size: int=64
    ) -> np.ndarray:

    
    random_latent_vectors = tf.random.normal(shape=[num_imgs, model.latent_dim])
    random_mass = to_mass_range(np.round(tf.random.uniform([num_imgs, 1], minval=13.8, maxval=15.), 2))

    time_0 = timeit.default_timer()
    print(f"Generating {num_imgs} images with latent vectors and mass...")
    generated_imgs = model.generator.predict([random_latent_vectors, random_mass])  # num_images x end_size x end_size x 1
    print(f"Image generation completed in {timeit.default_timer() - time_0:.2f} seconds.")
    
    x_train, x_val, y_train, y_val = model_selection.train_test_split(generated_imgs, random_mass, test_size=0.3)

    regressor_model = build_regressor(image_size=generated_imgs.shape[1], filters=[50, 50, 50, 20, 10])
    regressor_model.compile(keras.optimizers.Adam(learning_rate=5e-4), loss=keras.losses.MeanSquaredError())

    early_stop =  keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    history = regressor_model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=100, batch_size=batch_size, callbacks=[early_stop])
    
    real_dataset = CustomDataGen(
        meta_data[meta_data['mass'] <= 15.], batch_size=batch_size, target_size=(train_config["end_size"],train_config["end_size"]),
        shuffle=True, epsilon=train_config["epsilon"], mult_factor=train_config["mult_factor"]
        )

    r2_score_dist, true_mass, pred_mass = [], [], []
    for X, y in tqdm(real_dataset):
        y_true = y
        X_real = X
        if X_real.shape[0] != batch_size:
            break
        y_pred = regressor_model.predict(X_real, verbose=0)
        true_mass.append(y_true)
        pred_mass.append(np.squeeze(y_pred))
        r2_score_dist.append(metrics.r2_score(y_true, y_pred))
    return np.asanyarray(true_mass).flatten(), np.asanyarray(pred_mass).flatten(), np.asanyarray(r2_score_dist).flatten()

def main():

    parser = ArgumentParser(
        description="Set parameters for script execution."
    )
    parser.add_argument(
        "c",
        "--config-file",
        type=str,
        required=True,
        help="Configuration file to use for the model. Only the filename, the folder path is added automatically."
        )
    parser.add_argument(
        "-e",
        "--best-epoch",
        type=str,
        required=True,
        help="Best epoch to load weights from. It should be a string with lenght four representing the epoch number (e.g., '1485' or '0050')."
        )
    args = parser.parse_args()

    with open(f"config_file_dir/{args.config_file}", "r") as f:
        config = json.load(f)
    
    model_config = config["model_config"]
    train_config = config["train_config"]
    version = config["version"]

    meta_data = load_meta_data(train_config["z_th"], show=True)

    model = PGAN(pgan_config=model_config, version=version)
    for n_depth in range(1, int(np.log2(train_config["end_size"]/2))):
        model.n_depth = n_depth

        model.fade_in_generator()
        model.fade_in_discriminator()

        model.stabilize_generator()
        model.stabilize_discriminator()

    best_epoch = args.best_epoch
    print(f"Loading weights from epoch {best_epoch} for version {version}...")
    ckpt_path = f"/leonardo_scratch/fast/uTS25_Fontana/GAN_ckpts_{version}/pgan_5_init_{best_epoch}.weights.h5"
    model.load_weights(ckpt_path)

    y_true, y_pred, r2_scores = compute_tstr_trst(train_config,model, meta_data, num_imgs=10000, batch_size=128)
    print(f"Mean R2 score: {np.mean(r2_scores):.3f}")
    print(f"Standard deviation of R2 scores: {np.std(r2_scores):.3f}")
    np.save(f"results/results_{version}/r2_scores_{version}.npy", r2_scores)
    np.save(f"results/results_{version}/y_true_{version}.npy", y_true)
    np.save(f"results/results_{version}/y_pred_{version}.npy", y_pred)

if __name__ == "__main__":
    main()