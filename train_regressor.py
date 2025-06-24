import os
import json
import keras
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from data_utils import CustomDataGen, load_meta_data

def RegressorConv(
        x: keras.layers.Layer,
        filters: int,
        kernel_size: int,
        pooling: str = None,
        activate: str = None,
        strides: tuple = (1, 1)
    ) -> keras.layers.Layer:

    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", dtype='float32')(x)
    x = keras.layers.BatchNormalization()(x)
    if activate == 'LeakyReLU':
        x = keras.layers.LeakyReLU(0.01)(x)
    if pooling == 'max':
        x = keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    elif pooling == 'avg':
        x = keras.layers.AveragePooling2D(pool_size=4, strides=1)(x)
    return x

def build_regressor(image_size: int, filters: list) -> keras.Model:
    input_shape = (image_size, image_size, 1)
    img_input = keras.layers.Input(shape=input_shape, name="reg_input")
    x = img_input
    
    for depth in range(5):
        x = RegressorConv(x, filters[depth], kernel_size=1, pooling=None,  activate='LeakyReLU', strides=(1,1))
        x = RegressorConv(x, filters[depth], kernel_size=3, pooling='max', activate='LeakyReLU', strides=(1,1))

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=16)(x)
    x = keras.layers.LeakyReLU(0.01)(x)
    x = keras.layers.Dense(units=1)(x)

    return keras.Model(img_input, x, name='regressor')

def main(
        config_filepath: str,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
        ) -> None:

    if not os.path.isfile(config_filepath):
        raise FileNotFoundError(config_filepath)
    
    with open(config_filepath, "r") as f:
        config_file = json.load(f)

    version = config_file.get('version')
    train_config = config_file.get('train_config')

    meta_data = load_meta_data(train_config['z_th'], show=True)
    print(f"Data Shape: {meta_data.shape}")
    train_meta, val_meta = train_test_split(meta_data, test_size=0.2, random_state=42, shuffle=True)

    size = train_config['end_size']; batch_size = batch_size
    print(f"Training with image size: {size} and batch size: {batch_size}")
    train_dataset = CustomDataGen(
                train_meta, batch_size=batch_size, target_size=(size, size),
                epsilon=train_config.get('epsilon'), mult_factor=train_config.get('mult_factor')
                )
    valid_dataset = CustomDataGen(
                val_meta, batch_size=batch_size, target_size=(size, size),
                epsilon=train_config.get('epsilon'), mult_factor=train_config.get('mult_factor')
            )

    REGRESSOR_FILTERS = [50, 50, 50, 20, 10]
    regressor_model = build_regressor(image_size=size, filters=REGRESSOR_FILTERS)
    regressor_model.compile(keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.MeanSquaredError())
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f"results/regressor_results/best_regressor_{version}.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )

    history = regressor_model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        verbose=verbose,
        callbacks=[early_stop, checkpoint_callback]
    )
    print("Training ended.")
    print("Saving history...")
    np.save(f"regressor_results/history_final_step_{version}.npy", history.history)

if __name__ == "__main__":
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
        "-e",
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        help="0 = silent, 1 = progress bar, 2 = one line per epoch."
    )
    args = parser.parse_args()
    main(
        config_filepath=args.config_filepath,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose
    )