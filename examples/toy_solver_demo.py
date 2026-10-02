#!/usr/bin/env python3
"""Run the whole EA-POD-NN chain with a cheap analytical "solver".

No simulation software is needed: this shows how to plug a solver into the
method and lets you test the installation in a few seconds.

    python examples/toy_solver_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_pod_nn import EAPODNNPredictor, HFSolver, NNConfig, train_ea_pod_nn  # noqa: E402


class ToySolver(HFSolver):
    """Two travelling waves on a 1D mesh; mu = (amplitude, speed)."""

    variables = ("h", "u", "v")

    def __init__(self, n_nodes=300):
        self.x = np.linspace(0.0, 1.0, n_nodes)

    def run(self, mu, run_id, worker_id=0):
        amplitude, speed = mu
        times = np.linspace(0.0, 1.0, 20)
        X, T = np.meshgrid(self.x, times, indexing="ij")
        h = 1.0 + amplitude * np.exp(-((X - speed * T - 0.2) / 0.1) ** 2)
        u = speed * (h - 1.0)
        v = 0.1 * amplitude * np.sin(np.pi * X) * T
        return times, np.concatenate([h, u, v])


if __name__ == "__main__":
    solver = ToySolver()
    rng = np.random.default_rng(0)
    mus = np.column_stack([rng.uniform(0.5, 1.5, 30), rng.uniform(0.2, 0.6, 30)])

    snapshots = solver.compute_snapshots(mus, "results_toy/snapshots")
    train_ea_pod_nn(snapshots, epsilon_pod=1e-3, epsilon_err=1e-1,
                    config=NNConfig(neurons=32, hidden_layers=4, epochs=300, batch_size=32),
                    output_dir="results_toy/model")

    mu_new = np.array([1.1, 0.45])
    times, reference = solver.run(mu_new, run_id="reference")
    prediction = EAPODNNPredictor.load("results_toy/model").predict(mu_new, times)
    rel = np.linalg.norm(reference - prediction) / np.linalg.norm(reference)
    print(f"Relative error on a new parameter: {rel:.2%}")
