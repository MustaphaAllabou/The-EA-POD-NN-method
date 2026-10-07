"""TensorFlow / Keras backend."""
import numpy as np
import tensorflow as tf

EXTENSION = ".keras"


def set_seed(seed):
    tf.keras.utils.set_random_seed(seed)


def build(input_dim, output_dim, neurons, hidden_layers, learning_rate):
    layers = [tf.keras.Input(shape=(input_dim,))]
    layers += [tf.keras.layers.Dense(neurons, activation="relu") for _ in range(hidden_layers)]
    layers += [tf.keras.layers.Dense(output_dim)]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def train(X_train, Y_train, X_val, Y_val, neurons, config):
    model = build(X_train.shape[1], Y_train.shape[1], neurons,
                  config.hidden_layers, config.learning_rate)
    callbacks = []
    if config.patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience, restore_best_weights=True))
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                        epochs=config.epochs, batch_size=config.batch_size,
                        callbacks=callbacks, verbose=config.verbose)
    return model, history.history


def predict(model, X):
    return np.asarray(model(tf.convert_to_tensor(X, dtype=tf.float32), training=False))


def save(model, path):
    model.save(path + EXTENSION)


def load(path):
    return tf.keras.models.load_model(path + EXTENSION)
