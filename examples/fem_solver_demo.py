#!/usr/bin/env python3
"""Run the whole EA-POD-NN chain with a finite element solver of the
steady advection-diffusion problem (P1 + SUPG, see ``examples/fem.py``):

    -div(lambda(mu) grad u) + alpha * w0 . grad u = f   in (0,1)^2

Parameters: mu_vec = (mu, alpha), mu in [1, 10] (diffusion), alpha (advection).

    python examples/fem_solver_demo.py              # TensorFlow / Keras
    python examples/fem_solver_demo.py torch        # PyTorch
    python examples/fem_solver_demo.py solver       # only test the FE solver
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from fem import MU_MAX, MU_MIN, Problem  # noqa: E402

try:
    from ea_pod_nn import EAPODNNPredictor, HFSolver, NNConfig, train_ea_pod_nn  # noqa: E402
except ImportError:  # allows "solver" mode without the package
    HFSolver = object

ALPHA_MIN, ALPHA_MAX = 0.5, 2.0
RESULTS = "results_toy"
FIG_DIR = os.path.join(RESULTS, "figures_fem")


class FEMAdvDiffSolver(HFSolver):
    """Steady P1-SUPG advection-diffusion solver; mu = (mu, alpha).

    Steady problem -> a single "time" instant t = 0. The snapshot matrix has
    shape (n_nodes, 1) and contains the full nodal field (with lifting u = 1).
    """

    variables = ("u",)

    def __init__(self, n=40):
        self.n = n
        self._cache = {}

    def problem(self, alpha):
        key = round(float(alpha), 12)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = Problem(self.n, alpha)
        return self._cache[key]

    def run(self, mu, run_id, worker_id=0):
        mu_d, alpha = mu
        pb = self.problem(alpha)
        u = pb.full(pb.solve(mu_d))
        times = np.array([0.0])
        return times, u[:, None]


# ---------------------------------------------------------------- figures
def plot_field(ax, pb, u, title, vmin=None, vmax=None, cmap="viridis"):
    tc = ax.tricontourf(pb.x, pb.y, pb.tris, u, levels=40, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(tc, ax=ax, shrink=0.85)


def test_solver(solver):
    """Sanity checks of the FE solver + figures."""
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) solutions for several parameters
    cases = [(1.0, 1.0), (5.0, 1.0), (10.0, 1.0), (1.0, 0.5), (1.0, 2.0), (10.0, 2.0)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (m, a) in zip(axes.ravel(), cases):
        _, U = solver.run((m, a), run_id="test")
        pb = solver.problem(a)
        u = U[:, 0]
        assert np.all(np.isfinite(u))
        assert np.allclose(u[pb.x < 1e-12], 1.0)          # Dirichlet BC
        plot_field(ax, pb, u, rf"$\mu$={m:g}, $\alpha$={a:g}  (max {u.max():.3g})")
        print(f"  mu={m:5.2f} alpha={a:4.2f}  min={u.min():8.4f}  max={u.max():8.4f}")
    fig.suptitle("FE solution (P1 + SUPG) of the advection-diffusion problem")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fem_solutions.png"), dpi=150)
    plt.close(fig)

    # 2) mesh convergence (reference: fine mesh, compared at coarse nodes)
    mu_t, a_t = 3.0, 1.0
    ref_pb = Problem(160, a_t)
    u_ref = ref_pb.full(ref_pb.solve(mu_t))
    ns, errs = [10, 20, 40, 80], []
    for n in ns:
        pb = Problem(n, a_t)
        u = pb.full(pb.solve(mu_t))
        step = 160 // n
        xs = np.arange(0, 161, step)
        I, J = np.meshgrid(xs, xs)
        u_r = u_ref[(I + J * 161).ravel()]
        errs.append(np.linalg.norm(u - u_r) / np.linalg.norm(u_r))
    rates = np.log(np.array(errs[:-1]) / np.array(errs[1:])) / np.log(2)
    print("  mesh convergence (rel. nodal error):",
          ", ".join(f"n={n}: {e:.2e}" for n, e in zip(ns, errs)))
    print("  observed rates:", np.round(rates, 2))
    fig, ax = plt.subplots(figsize=(5, 4))
    hs = 1.0 / np.array(ns)
    ax.loglog(hs, errs, "o-", label="FE error vs n=160")
    ax.loglog(hs, errs[0] * (hs / hs[0]) ** 2, "k--", label=r"$O(h^2)$")
    ax.loglog(hs, errs[0] * (hs / hs[0]), "k:", label=r"$O(h)$")
    ax.set_xlabel("h")
    ax.set_ylabel("relative error")
    ax.set_title(rf"Mesh convergence ($\mu$={mu_t}, $\alpha$={a_t})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fem_convergence.png"), dpi=150)
    plt.close(fig)


def plot_prediction(solver, mu_new, reference, prediction):
    pb = solver.problem(mu_new[1])
    ref, pred = reference[:, 0], prediction[:, 0]
    vmin, vmax = ref.min(), ref.max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_field(axes[0], pb, ref, "FE reference", vmin, vmax)
    plot_field(axes[1], pb, pred, "EA-POD-NN prediction", vmin, vmax)
    plot_field(axes[2], pb, np.abs(ref - pred), "|error|", cmap="magma")
    fig.suptitle(rf"New parameter $\mu$={mu_new[0]:g}, $\alpha$={mu_new[1]:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prediction_vs_reference.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "keras"
    solver = FEMAdvDiffSolver(n=40)

    print("== Testing the finite element solver")
    test_solver(solver)
    if mode == "solver":
        print(f"Figures saved in {FIG_DIR}/")
        sys.exit(0)

    backend = mode
    output_dir = f"{RESULTS}/model_fem_{backend}"
    rng = np.random.default_rng(0)
    n_train = 40
    mus = np.column_stack([rng.uniform(MU_MIN, MU_MAX, n_train),
                           rng.uniform(ALPHA_MIN, ALPHA_MAX, n_train)])

    snapshots = solver.compute_snapshots(mus, f"{RESULTS}/snapshots_fem")
    train_ea_pod_nn(snapshots, epsilon_pod=1e-3, epsilon_err=1e-1,
                    config=NNConfig(backend=backend, neurons=32, hidden_layers=4, epochs=300,
                                    batch_size=32, verbose=0),
                    output_dir=output_dir)

    test_mus = np.column_stack([rng.uniform(MU_MIN, MU_MAX, 10),
                                rng.uniform(ALPHA_MIN, ALPHA_MAX, 10)])
    predictor = EAPODNNPredictor.load(output_dir)
    errs = []
    for mu_new in test_mus:
        times, reference = solver.run(mu_new, run_id="reference")
        prediction = predictor.predict(mu_new, times)
        errs.append(np.linalg.norm(reference - prediction) / np.linalg.norm(reference))
    print(f"[{backend}] Relative error on {len(test_mus)} new parameters: "
          f"mean {np.mean(errs):.2%}, max {np.max(errs):.2%}")

    mu_new = np.array([4.0, 1.3])
    times, reference = solver.run(mu_new, run_id="reference")
    prediction = predictor.predict(mu_new, times)
    rel = np.linalg.norm(reference - prediction) / np.linalg.norm(reference)
    print(f"[{backend}] Relative error on mu={mu_new}: {rel:.2%}")
    plot_prediction(solver, mu_new, reference, prediction)
    print(f"Figures saved in {FIG_DIR}/")
