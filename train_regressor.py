import keras
import numpy as np
from sklearn.model_selection import train_test_split

from config import CONFIG
from data_utils import CustomDataGen, load_meta_data

def RegressorConv(x, filters, kernel_size, pooling=None, activate=None, strides=(1,1)):

    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", dtype='float32')(x)
    x = keras.layers.BatchNormalization()(x)
    if activate=='LeakyReLU':
        x = keras.layers.LeakyReLU(0.01)(x)
    if pooling=='max':
        x = keras.layers.MaxPooling2D(pool_size = 2, strides = 2)(x)
    elif pooling=='avg':
        x = keras.layers.AveragePooling2D(pool_size = 4, strides = 1)(x)
    return x 

def build_regressor(image_size: int, filters: list) -> keras.Model:
    input_shape = (image_size, image_size, 1)
    img_input = keras.layers.Input(shape=input_shape, name="reg_input")
    x = img_input
    
    for depth in range(5):
        x = RegressorConv(x, filters[depth], kernel_size=1, pooling=None, activate='LeakyReLU', strides=(1,1))
        x = RegressorConv(x, filters[depth], kernel_size=3, pooling='max', activate='LeakyReLU', strides=(1,1))

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=16)(x)
    x = keras.layers.LeakyReLU(0.01)(x)
    x = keras.layers.Dense(units=1)(x)

    return keras.Model(img_input, x, name='regressor')

def main():

    meta_data = load_meta_data(CONFIG['z_th'], show=True)
    print(f"Data Shape: {meta_data.shape}")
    train_meta, val_meta = train_test_split(meta_data, test_size=0.2, random_state=42, shuffle=True)

    size = CONFIG['START_SIZE']; batch_size = 32
    train_dataset = CustomDataGen(
                train_meta, X_col='id', y_col='mass', batch_size=batch_size,
                target_size=(size, size), shuffle=True
            )
    valid_dataset = CustomDataGen(
                val_meta, X_col='id', y_col='mass', batch_size=batch_size,
                target_size=(size, size), shuffle=True
            )

    REGRESSOR_FILTERS = [50, 50, 50, 20, 10]
    regressor_model = build_regressor(image_size=size, filters=REGRESSOR_FILTERS)
    regressor_model.compile(keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.MeanSquaredError())
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f"regressor_results/best_regressor_{CONFIG['z_th']}.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )

    history = regressor_model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=100,
        verbose=1,
        callbacks=[early_stop, checkpoint_callback]
    )
    print("Training terminato.")
    print("Salvataggio della history...")
    np.save(f"regressor_results/history_final_step_{CONFIG['z_th']}.npy", history.history)

if __name__ == "__main__":
    main()