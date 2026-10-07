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

from .networks import get_backend
from .pod import build_reduced_bases, plot_eigenvalues


@dataclass
class NNConfig:
    backend: str = "keras"          # "keras" (TensorFlow) or "torch" (PyTorch)
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
    verbose: int = 1


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


def train_ea_pod_nn(snapshots, epsilon_pod, epsilon_err, config=None,
                    output_dir="results/model", show_plots=False):
    """Full offline phase. Everything needed online is saved in ``output_dir``."""
    config = config or NNConfig()
    backend = get_backend(config.backend)
    backend.set_seed(config.seed)
    os.makedirs(output_dir, exist_ok=True)

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
    targets = {"rb": rb_scale.transform(Urb), "er": er_scale.transform(Uer)}
    neurons = {"rb": config.neurons, "er": config.neurons_er or 2 * er.n_modes}

    train, val, test = split_by_simulation(snapshots.sim_id, config.val_fraction,
                                           config.test_fraction, config.seed)
    print(f"[NN] backend {config.backend}: train {train.sum()}, val {val.sum()}, "
          f"test {test.sum()} samples")

    # 3. Training of the two networks
    networks, histories = {}, {}
    for name, Y in targets.items():
        print(f"[NN] training network {name} ({neurons[name]} neurons per layer)")
        model, histories[name] = backend.train(X[train], Y[train], X[val], Y[val],
                                               neurons[name], config)
        if test.any():
            mse = np.mean((backend.predict(model, X[test]) - Y[test]) ** 2)
            print(f"[NN] network {name}: test MSE = {mse:.3e}")
        backend.save(model, os.path.join(output_dir, f"network_{name}"))
        networks[name] = model

    # 4. Everything the online phase needs
    np.savez(os.path.join(output_dir, "reduced_model.npz"),
             backend=config.backend,
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
