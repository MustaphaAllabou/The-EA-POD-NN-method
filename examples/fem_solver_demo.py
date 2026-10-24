#!/usr/bin/env python3
"""EA-POD-NN with a FEniCSx (dolfinx) finite element solver of the steady
advection-diffusion problem (same setting as the legacy FEniCS script):

    -div(Lambda(mu) grad u) + w . grad u = f      in (0,1)^2
    u = 1 on {x = 0},   homogeneous Neumann elsewhere
    Lambda(mu) = exp(mu0 (mu + 1)),  mu0 = 1.2,  mu in [1, 10]
    w = 1e3 * (1 + |cos(2 pi x)|, sin(2 pi x / 0.2))
    f = 1e3 * exp(-(|x - 0.5| + |y - 0.5|) / 0.1)
    P1 Lagrange + SUPG, delta = h / (2 |w|), h = MaxCellEdgeLength, mesh 35 x 35

    python examples/fenicsx_solver_demo.py            # TensorFlow / Keras
    python examples/fenicsx_solver_demo.py torch      # PyTorch
    python examples/fenicsx_solver_demo.py solver     # only test the FE solver

Written for dolfinx 0.7 -> 0.10.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

try:
    from ea_pod_nn import EAPODNNPredictor, HFSolver, NNConfig, train_ea_pod_nn  # noqa: E402
except ImportError:  # allows "solver" mode without the package
    HFSolver = object

# ------------------------------------------------------------- parameters
MU0 = 1.2
MU_MIN, MU_MAX = 1.0, 10.0
VEL_AMP = 1e3
NP = 35
RESULTS = "results_toy"
FIG_DIR = os.path.join(RESULTS, "figures_fenicsx")


def Lambda(mu):
    return np.exp(MU0 * (mu + 1.0))


class _FEProblem:
    """dolfinx objects (built once; only the diffusion constant changes)."""

    def __init__(self, n):
        import ufl
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import LinearProblem
        from mpi4py import MPI
        from petsc4py import PETSc

        self.domain = mesh.create_unit_square(MPI.COMM_SELF, n, n)
        try:                                         # dolfinx >= 0.8
            V = fem.functionspace(self.domain, ("Lagrange", 1))
            V_vec = fem.functionspace(self.domain, ("Lagrange", 1, (2,)))
        except AttributeError:                       # dolfinx 0.7
            V = fem.FunctionSpace(self.domain, ("Lagrange", 1))
            V_vec = fem.VectorFunctionSpace(self.domain, ("Lagrange", 1))
        self.V = V

        # velocity field and source, interpolated in P1 (as in legacy FEniCS)
        vel = fem.Function(V_vec)
        vel.interpolate(lambda x: np.vstack([
            VEL_AMP * (1.0 + np.abs(np.cos(2 * np.pi * x[0]))),
            VEL_AMP * np.sin(2 * np.pi / 0.2 * x[0])]))
        f = fem.Function(V)
        f.interpolate(lambda x: 1e3 * np.exp(-(np.abs(x[0] - 0.5) + np.abs(x[1] - 0.5)) / 0.1))
        self.vel = vel

        # diffusion coefficient as a Constant -> forms compiled only once
        self.diffus = fem.Constant(self.domain, PETSc.ScalarType(1.0))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        dx = ufl.dx
        F = (self.diffus * ufl.dot(ufl.grad(v), ufl.grad(u)) * dx
             + v * ufl.dot(vel, ufl.grad(u)) * dx - f * v * dx)
        # SUPG stabilisation
        r = -self.diffus * ufl.div(ufl.grad(u)) + ufl.dot(vel, ufl.grad(u)) - f
        vnorm = ufl.sqrt(ufl.dot(vel, vel))
        h = ufl.MaxCellEdgeLength(self.domain)
        delta = h / (2.0 * vnorm)
        F += delta * ufl.dot(vel, ufl.grad(v)) * r * dx
        a, L = ufl.lhs(F), ufl.rhs(F)

        # Dirichlet u = 1 on x = 0
        dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0, atol=1e-10))
        bc = fem.dirichletbc(PETSc.ScalarType(1.0), dofs, V)

        opts = {"ksp_type": "preonly", "pc_type": "lu"}
        try:                                         # dolfinx >= 0.10
            self.problem = LinearProblem(a, L, bcs=[bc], petsc_options=opts,
                                         petsc_options_prefix="advdiff_")
        except TypeError:                            # dolfinx < 0.10
            self.problem = LinearProblem(a, L, bcs=[bc], petsc_options=opts)

        xy = V.tabulate_dof_coordinates()[:, :2]
        self.x, self.y = xy[:, 0], xy[:, 1]
        self.n_dofs = len(self.x)

    def solve(self, mu):
        self.diffus.value = Lambda(mu)
        uh = self.problem.solve()
        if isinstance(uh, (tuple, list)):
            uh = uh[0]
        return uh.x.array.copy()


class FEniCSxAdvDiffSolver(HFSolver):
    """Steady FEniCSx advection-diffusion solver; mu = (mu,).

    Steady problem -> one "time" instant t = 0. The snapshot matrix has shape
    (n_dofs, 1). Values follow the dolfinx dof ordering (see ``coordinates``).

    The snapshots contain the HOMOGENEOUS part u0 = u - 1 (lifting u_g = 1 removed):
    u is ~1 everywhere, so with the raw field the constant would carry > 99.99 % of
    the POD energy and the truncation would keep only that mode. Use ``full(u0)``
    to get back u = 1 + u0 (the Dirichlet condition is then exactly satisfied).
    """

    LIFTING = 1.0

    variables = ("u",)

    def __init__(self, n=NP):
        self.n = n
        self._pb = None

    @property
    def pb(self):
        if self._pb is None:             # built lazily (dolfinx objects are not picklable)
            self._pb = _FEProblem(self.n)
        return self._pb

    def __getstate__(self):              # allow copies for parallel workers
        state = self.__dict__.copy()
        state["_pb"] = None
        return state

    def coordinates(self):
        return self.pb.x, self.pb.y

    def run(self, mu, run_id, worker_id=0):
        mu = float(np.atleast_1d(mu)[0])
        u = self.pb.solve(mu)
        return np.array([0.0]), (u - self.LIFTING)[:, None]

    def full(self, u0):
        """homogeneous part u0 -> full field u = 1 + u0."""
        return np.asarray(u0) + self.LIFTING


# ---------------------------------------------------------------- figures
def plot_field(ax, x, y, u, title, vmin=None, vmax=None, cmap="rainbow"):
    tc = ax.tricontourf(mtri.Triangulation(x, y), u, levels=40, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(tc, ax=ax, shrink=0.85)


def test_solver(solver):
    os.makedirs(FIG_DIR, exist_ok=True)
    x, y = solver.coordinates()
    print(f"  number of dofs NN = {len(x)}")
    vel = solver.pb.vel.x.array.reshape(-1, 2)
    wmax, wmin = np.linalg.norm(vel, axis=1).max(), np.linalg.norm(vel, axis=1).min()
    print(f"  Peclet |w|/Lambda: mu=1 -> [{wmin / Lambda(1.0):.3g}, {wmax / Lambda(1.0):.3g}],"
          f"  mu=10 -> [{wmin / Lambda(10.0):.3g}, {wmax / Lambda(10.0):.3g}]")

    cases = [1.0, 2.68, 5.0, 10.0]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, m in zip(axes, cases):
        _, U = solver.run(np.array([m]), run_id="test")
        u = solver.full(U[:, 0])
        assert np.all(np.isfinite(u))
        assert np.allclose(u[np.isclose(x, 0.0)], 1.0)      # Dirichlet BC
        print(f"  mu={m:5.2f}  Lambda={Lambda(m):10.3e}  min={u.min():.4f}  "
              f"max={u.max():.4f}  ||u||_2={np.linalg.norm(u):.3f}")
        plot_field(ax, x, y, u, rf"$\mu$={m:g}  ($\Lambda$={Lambda(m):.3g})")
    fig.suptitle("FEniCSx solution (P1 + SUPG) of the advection-diffusion problem")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fenicsx_solutions.png"), dpi=150)
    plt.close(fig)


def plot_prediction(solver, mu_new, reference, prediction):
    x, y = solver.coordinates()
    ref, pred = solver.full(reference[:, 0]), solver.full(prediction[:, 0])
    vmin, vmax = min(ref.min(), pred.min()), max(ref.max(), pred.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_field(axes[0], x, y, ref, "FE reference", vmin, vmax)
    plot_field(axes[1], x, y, pred, "EA-POD-NN prediction", vmin, vmax)
    plot_field(axes[2], x, y, np.abs(ref - pred), "|u_FE - u_pred|", cmap="magma")
    fig.suptitle(rf"New parameter $\mu$={mu_new[0]:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prediction_vs_reference.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "keras"
    solver = FEniCSxAdvDiffSolver(n=NP)

    print("== Testing the FEniCSx solver")
    test_solver(solver)
    if mode == "solver":
        print(f"Figures saved in {FIG_DIR}/")
        sys.exit(0)

    backend = mode
    output_dir = f"{RESULTS}/model_fenicsx_u0_{backend}"      # u0 = u - 1
    rng = np.random.default_rng(0)
    M = 100                                                     # number of snapshots
    mus = np.linspace(MU_MIN, MU_MAX, M)[:, None]              # as in the legacy script

    snapshots = solver.compute_snapshots(mus, f"{RESULTS}/snapshots_fenicsx_u0")
    train_ea_pod_nn(snapshots, epsilon_pod=1e-4, epsilon_err=1e-1,
                    config=NNConfig(backend=backend, neurons=32, hidden_layers=4, epochs=500,
                                    batch_size=32, verbose=0),
                    output_dir=output_dir)

    predictor = EAPODNNPredictor.load(output_dir)
    # Errors measured on u0 = u - 1 (the part that actually depends on mu).
    # Measured on u, they would look ~50x smaller only because u ~ 1 everywhere.
    # Test values in [1, 5]: for mu > 5, Lambda > 1e3 and u0 ~ 0.
    test_mus = rng.uniform(MU_MIN, 5.0, 10)[:, None]
    errs = []
    for mu_new in test_mus:
        times, reference = solver.run(mu_new, run_id="reference")
        prediction = predictor.predict(mu_new, times)
        errs.append(np.linalg.norm(reference - prediction) / np.linalg.norm(reference))
    print(f"[{backend}] Relative error on u-1, {len(test_mus)} new mu in [1,5]: "
          f"mean {np.mean(errs):.2%}, max {np.max(errs):.2%}")

    mu_new = np.array([2.68])                                  # online value of the legacy script
    times, reference = solver.run(mu_new, run_id="reference")
    prediction = predictor.predict(mu_new, times)
    rel0 = np.linalg.norm(reference - prediction) / np.linalg.norm(reference)
    u_ref, u_pred = solver.full(reference), solver.full(prediction)
    rel = np.linalg.norm(u_ref - u_pred) / np.linalg.norm(u_ref)
    print(f"[{backend}] mu={mu_new[0]}: relative error on u-1 = {rel0:.2%}, on u = {rel:.3%}")
    plot_prediction(solver, mu_new, reference, prediction)
    print(f"Figures saved in {FIG_DIR}/")
