import json
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics, model_selection

from model import PGAN
from data_utils import CustomDataGen, load_meta_data

def compute_tstr_trst(
        model: PGAN, meta_data: pd.DataFrame, num_imgs: int=2500, batch_size: int=64
    ) -> np.ndarray:

    random_latent_vectors = tf.random.normal(shape=[num_imgs, model.latent_dim])
    random_mass = np.round(tf.random.uniform([num_imgs, 1], minval=1., maxval=16.),2)

    generated_imgs = model.generator.predict([random_latent_vectors, random_mass])  # num_images x end_size x end_size x 1

    x_train, x_val, y_train, y_val = model_selection.train_test_split(generated_imgs, random_mass, test_size=0.3)

    tstr_regressor = keras.models.load_model("regressor_results/best_regressor_new_mass_range.keras")
    tstr_regressor.compile(keras.optimizers.Adam(learning_rate=5e-4),  loss=keras.losses.MeanSquaredError())

    early_stop =  keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    history = tstr_regressor.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=100, batch_size=batch_size, callbacks=[early_stop]) 

    real_dataset = CustomDataGen(meta_data, batch_size=batch_size, target_size=(tstr_regressor.input_shape[1], tstr_regressor.input_shape[2]), shuffle=True)
    r2_score_dist = np.empty((len(real_dataset),))
    for i, (X, y) in enumerate(real_dataset):
        y_true = y
        X_real = X
        y_pred = tstr_regressor.predict(X_real, verbose=0)
        r2_score_dist[i] = metrics.r2_score(y_true, y_pred)
        if i == len(real_dataset) - 1: break
    return r2_score_dist

def main():

    with open("config_new_mass_range.json", "r") as f:
        config = json.load(f)
    
    model_config = config["model_config"]
    train_config = config["train_config"]
    version = config["version"]

    meta_data = load_meta_data(train_config["z_th"], show=True)

    model = PGAN(pgan_config=model_config)
    for n_depth in range(1, int(np.log2(train_config["end_size"]/2))):
        model.n_depth = n_depth

        model.fade_in_generator()
        model.fade_in_discriminator()

        model.stabilize_generator()
        model.stabilize_discriminator()
    
    best_epoch = "0855"
    ckpt_path = f"/leonardo_scratch/fast/uTS25_Fontana/GAN_ckpts_{version}/pgan_5_init_{best_epoch}.weights.h5"
    model.load_weights(ckpt_path)

    r2_scores = compute_tstr_trst(model, meta_data, num_imgs=5000, batch_size=128)
    print(f"Mean R2 score: {np.mean(r2_scores):.3f}")
    print(f"Standard deviation of R2 scores: {np.std(r2_scores):.3f}")
    np.save(f"r2_scores_{version}.npy", r2_scores)

if __name__ == "__main__":
    main()