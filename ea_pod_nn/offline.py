"""EA-POD-NN offline phase: reduced bases + training of the two networks.

Network 1 maps (mu, t) to the POD coefficients on Brb,
network 2 maps (mu, t) to the coefficients of the projection error on Ber.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .pod import build_reduced_bases, plot_eigenvalues


@dataclass
class NNConfig:
    neurons: int = 20               # neurons per hidden layer, network 1
    neurons_er: Optional[int] = None  # network 2; default 2 * Ner
    hidden_layers: int = 11
    epochs: int = 10
    batch_size: int = 10
    learning_rate: float = 1e-3
    val_fraction: float = 0.2       # fraction of simulations used for validation
    test_fraction: float = 0.1      # fraction of simulations kept for the final test
    patience: Optional[int] = None  # early stopping on val_loss (None = off)
    seed: int = 0


class MinMax:
    """Min-max scaling, per column (axis=0) or global (axis=None)."""

    def __init__(self, vmin, vmax):
        self.min, self.max = np.asarray(vmin, float), np.asarray(vmax, float)

    @classmethod
    def fit(cls, x, axis=None):
        return cls(np.min(x, axis=axis), np.max(x, axis=axis))

    def transform(self, x):
        return (x - self.min) / np.where(self.max > self.min, self.max - self.min, 1.0)

    def inverse(self, x):
        return x * (self.max - self.min) + self.min


def split_by_simulation(sim_id, val_fraction, test_fraction, seed=0):
    """Train/val/test masks; all time steps of a simulation go to the same set."""
    sims = np.random.default_rng(seed).permutation(np.unique(sim_id))
    n_test = max(1, int(round(test_fraction * len(sims)))) if test_fraction > 0 else 0
    n_val = max(1, int(round(val_fraction * len(sims)))) if val_fraction > 0 else 0
    if n_test + n_val >= len(sims):
        raise ValueError(f"Not enough simulations ({len(sims)}) for this split")
    test, val = sims[:n_test], sims[n_test:n_test + n_val]
    return ~np.isin(sim_id, np.r_[test, val]), np.isin(sim_id, val), np.isin(sim_id, test)


def build_network(input_dim, output_dim, neurons, hidden_layers, learning_rate):
    import tensorflow as tf

    layers = [tf.keras.Input(shape=(input_dim,))]
    layers += [tf.keras.layers.Dense(neurons, activation="relu") for _ in range(hidden_layers)]
    layers += [tf.keras.layers.Dense(output_dim)]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_ea_pod_nn(snapshots, epsilon_pod, epsilon_err, config=None,
                    output_dir="results/model", show_plots=False):
    """Full offline phase. Everything needed online is saved in ``output_dir``."""
    import tensorflow as tf

    config = config or NNConfig()
    os.makedirs(output_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(config.seed)

    # 1. Reduced bases
    rb, er, Er = build_reduced_bases(snapshots.S, epsilon_pod, epsilon_err)
    Brb, Ber = rb.basis, er.basis
    plot_eigenvalues(rb.eigenvalues, "Snapshots", os.path.join(output_dir, "eigenvalues.png"))
    plot_eigenvalues(er.eigenvalues, "Projection error",
                     os.path.join(output_dir, "eigenvalues_error.png"))

    # 2. Reduced coefficients and scaling
    Urb = (Brb.T @ snapshots.S).T
    Uer = (Ber.T @ Er).T
    p_scale = MinMax.fit(snapshots.P, axis=0)
    rb_scale, er_scale = MinMax.fit(Urb), MinMax.fit(Uer)
    X = p_scale.transform(snapshots.P)
    Y, Y_er = rb_scale.transform(Urb), er_scale.transform(Uer)

    train, val, test = split_by_simulation(snapshots.sim_id, config.val_fraction,
                                           config.test_fraction, config.seed)
    print(f"[NN] samples: train {train.sum()}, val {val.sum()}, test {test.sum()}")

    # 3. Training
    callbacks = []
    if config.patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience, restore_best_weights=True))
    neurons_er = config.neurons_er or 2 * er.n_modes
    histories = {}
    networks = {}
    for name, target, n_out, neurons in (("rb", Y, rb.n_modes, config.neurons),
                                         ("er", Y_er, er.n_modes, neurons_er)):
        net = build_network(X.shape[1], n_out, neurons, config.hidden_layers,
                            config.learning_rate)
        histories[name] = net.fit(X[train], target[train], epochs=config.epochs,
                                  batch_size=config.batch_size,
                                  validation_data=(X[val], target[val]),
                                  callbacks=callbacks, verbose=1).history
        if test.any():
            print(f"[NN] network {name}: test MSE = "
                  f"{net.evaluate(X[test], target[test], verbose=0):.3e}")
        net.save(os.path.join(output_dir, f"network_{name}.keras"))
        networks[name] = net

    # 4. Everything the online phase needs
    np.savez(os.path.join(output_dir, "reduced_model.npz"),
             Brb=Brb, Ber=Ber, variables=np.array(snapshots.variables),
             p_min=p_scale.min, p_max=p_scale.max,
             rb_min=rb_scale.min, rb_max=rb_scale.max,
             er_min=er_scale.min, er_max=er_scale.max,
             eigenvalues=rb.eigenvalues, eigenvalues_error=er.eigenvalues)

    fig, ax = plt.subplots()
    for name, color in (("rb", "b"), ("er", "g")):
        ax.semilogy(histories[name]["loss"], color, label=f"train ({name})")
        ax.semilogy(histories[name]["val_loss"], color + "--", label=f"validation ({name})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE loss")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
    if show_plots:
        plt.show()
    print(f"[NN] reduced model saved in {output_dir}")
    return networks, histories
