#!/usr/bin/env python3
"""EA-POD-NN reduction method - Aude test case with DassFlow2D.

Usage:
    python main.py offline   # HF snapshots + POD + training
    python main.py online    # prediction + comparison with a HF reference
    python main.py all       # both

To use another simulation software, replace DassFlowSolver in
``make_solver`` by your own subclass of ea_pod_nn.solvers.HFSolver
(see ea_pod_nn/solvers/template.py). Nothing else needs to change.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from ea_pod_nn import EAPODNNPredictor, NNConfig, train_ea_pod_nn
from ea_pod_nn.hydrographs import TriangularHydrographs, parameter_grid, plot_parameter_space
from ea_pod_nn.postprocessing import (error_summary, plot_error_by_bins, plot_error_cdf,
                                      plot_mean_error_over_time)
from ea_pod_nn.solvers import split_variables

# --------------------------------------------------------------------------- #
# Configuration (adapt the paths to your machine)
# --------------------------------------------------------------------------- #
DASSFLOW_DIR = "/home-local/allabou/Bureau/dasshydro-main"
RESULTS_DIR = "results"

# Parameter space: mu = (t_peak1 [s], Q_max1 [m3/s], t_peak2 [s], Q_max2 [m3/s])
PARAMETERS = dict(
    t_peak1=np.linspace(75000, 100000, 5),
    q_max1=np.linspace(600, 1200, 5),
    t_peak2=np.linspace(50000, 80000, 5),
    q_max2=np.linspace(100, 400, 5),
    max_relative_shift=0.035,
)
N_SIMULATIONS = 10                 # None = whole grid
MU_ONLINE = np.array([31200. + 46800., 1100., 30200. + 46800., 300.])

HYDROGRAPH_TIMES = np.arange(3600., 172800. + 3600., 3600.)   # 48 values
T_RISE = 46800.

EPSILON_POD = 1e-3
EPSILON_ERROR = 1e-1
NN = NNConfig(neurons=20, epochs=10, batch_size=10, learning_rate=1e-3,
              val_fraction=0.2, test_fraction=0.1)
N_JOBS = 16


def make_solver():
    from ea_pod_nn.solvers.dassflow import DassFlowSolver
    return DassFlowSolver(
        case_dir=os.path.join(DASSFLOW_DIR, "cases/dev_case/Aude/built_bin"),
        work_dir=os.path.join(DASSFLOW_DIR, "code/ea_pod_nn_runs"),
        inflows=TriangularHydrographs(HYDROGRAPH_TIMES, T_RISE),
        final_time=216000., skip_first=12)


# --------------------------------------------------------------------------- #
def offline(solver):
    mus = parameter_grid(**PARAMETERS)[:N_SIMULATIONS]
    print(f"{len(mus)} parameter vectors")
    plot_parameter_space(mus, MU_ONLINE, os.path.join(RESULTS_DIR, "parameter_space.png"))

    snapshots = solver.compute_snapshots(mus, os.path.join(RESULTS_DIR, "snapshots"),
                                         n_jobs=N_JOBS)
    train_ea_pod_nn(snapshots, EPSILON_POD, EPSILON_ERROR, NN,
                    output_dir=os.path.join(RESULTS_DIR, "model"))


def online(solver):
    out = os.path.join(RESULTS_DIR, "online")
    os.makedirs(out, exist_ok=True)
    predictor = EAPODNNPredictor.load(os.path.join(RESULTS_DIR, "model"))

    times, reference = solver.run(MU_ONLINE, run_id="reference")
    prediction = predictor.predict(MU_ONLINE, times)
    np.savez(os.path.join(out, "online_solutions.npz"),
             times=times, reference=reference, prediction=prediction)

    ref = split_variables(reference, solver.variables)
    pred = split_variables(prediction, solver.variables)
    velocity = np.hypot(ref["u"], ref["v"])
    velocity_pred = np.hypot(pred["u"], pred["v"])
    errors = {"h": ref["h"] - pred["h"], "U": velocity - velocity_pred}

    error_summary(errors)
    plot_error_by_bins({"h": (ref["h"][:, -1], errors["h"][:, -1], np.linspace(0, 7, 7)),
                        "U": (velocity[:, -1], errors["U"][:, -1], np.linspace(0, 7, 6))},
                       os.path.join(out, "error_by_bins.png"))
    plot_error_cdf({k: e[:, -1] for k, e in errors.items()}, os.path.join(out, "error_cdf.png"))
    plot_mean_error_over_time(times, errors, os.path.join(out, "mean_error_vs_time.png"))

    try:  # maps on the mesh, if the solver supports it
        for values, title in ((velocity[:, -1], "U [m/s]"),
                              (velocity_pred[:, -1], "U EA-POD-NN [m/s]"),
                              (errors["U"][:, -1], "Error U [m/s]"),
                              (ref["h"][:, -1], "h [m]"),
                              (pred["h"][:, -1], "h EA-POD-NN [m]"),
                              (errors["h"][:, -1], "Error h [m]")):
            solver.plot_field(values, title).show()
    except NotImplementedError:
        pass
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["offline", "online", "all"])
    stage = parser.parse_args().stage
    os.makedirs(RESULTS_DIR, exist_ok=True)

    solver = make_solver()
    if stage in ("offline", "all"):
        offline(solver)
    if stage in ("online", "all"):
        online(solver)
