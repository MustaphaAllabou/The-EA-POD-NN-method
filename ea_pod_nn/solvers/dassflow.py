"""DassFlow2D implementation of the HF solver interface.

Requires the ``dassflow2d`` Python wrapper (installed with DassFlow), plus
``h5py`` and, for plotting, ``pyvista``.
"""
from __future__ import annotations

import os
import shutil
import time

import numpy as np

from .base import HFSolver


class DassFlowSolver(HFSolver):
    """Runs DassFlow2D for a given inflow parametrization.

    Parameters
    ----------
    case_dir : directory of the compiled case (e.g. ``.../cases/dev_case/Aude/built_bin``).
    work_dir : directory where each worker gets its own copy of the case.
    inflows : callable ``mu -> (q1, q2, ...)`` returning the discharge time
        series imposed on the hydrograph boundaries ``inflow_boundaries``.
    inflow_boundaries : indices of ``bc.hyd`` receiving the inflows (default (1, 2)).
    final_time : simulated time ``ts`` (s), as in the case input file.
    skip_first : number of initial outputs discarded (spin-up), default 12.
    run_type : DassFlow run type, default "direct".

    Output times are assumed to be written regularly from t = 0 to
    ``final_time``, which is DassFlow's behaviour with a constant ``dtw``.
    """

    variables = ("h", "u", "v")

    def __init__(self, case_dir, work_dir, inflows, inflow_boundaries=(1, 2),
                 final_time=216000.0, skip_first=12, run_type="direct"):
        self.case_dir = os.path.abspath(case_dir)
        self.work_dir = os.path.abspath(work_dir)
        self.inflows = inflows
        self.inflow_boundaries = tuple(inflow_boundaries)
        self.final_time = float(final_time)
        self.skip_first = int(skip_first)
        self.run_type = run_type

    # ------------------------------------------------------------------ #
    def _prepare_bin_dir(self, name):
        bin_dir = os.path.join(self.work_dir, name)
        if os.path.exists(bin_dir):
            shutil.rmtree(bin_dir)
        shutil.copytree(self.case_dir, bin_dir)
        os.makedirs(os.path.join(bin_dir, "res"), exist_ok=True)
        return bin_dir

    def _new_model(self, bin_dir):
        import dassflow2d as df2d
        model = df2d.dassflowmodel(bin_dir=bin_dir,
                                   hdf5_path=os.path.join(bin_dir, "res", "simu.hdf5"),
                                   run_type=self.run_type, clean=True)
        model.init_all()
        return df2d, model

    def run(self, mu, run_id, worker_id=0):
        import h5py

        bin_dir = self._prepare_bin_dir(f"worker_{worker_id}")
        previous_cwd = os.getcwd()
        os.chdir(bin_dir)
        try:
            df2d, model = self._new_model(bin_dir)
            bc = df2d.wrapping.m_model.get_bc()
            for boundary, q in zip(self.inflow_boundaries, self.inflows(mu)):
                bc.hyd[boundary].q = q
            start = time.time()
            model.run()
            print(f"[DassFlow] run {run_id}: {time.time() - start:.1f} s")
            model.save_all()
            df2d.wrapping.call_model.clean_model(model.kernel)

            with h5py.File(os.path.join(bin_dir, "res", "simu.hdf5"), "r") as f:
                result = f["output"]["result"]
                h, u, v = (np.array(result[name]) for name in self.variables)
        finally:
            os.chdir(previous_cwd)

        times = np.linspace(0.0, self.final_time, h.shape[1])[self.skip_first:]
        fields = np.concatenate([h, u, v])[:, self.skip_first:]
        return times, fields

    def plot_field(self, values, title=""):
        """Plot a nodal field on the DassFlow mesh (pyvista)."""
        bin_dir = self._prepare_bin_dir("plot")
        previous_cwd = os.getcwd()
        os.chdir(bin_dir)
        try:
            _, model = self._new_model(bin_dir)
            plotter = model.outputs.result.plot_field(
                my_mesh=model.meshing.mesh_pyvista, my_scalar=np.asarray(values),
                title_scale_bar=title, title_plot="", notebook=False)
            plotter.camera.zoom(1.47)
            return plotter
        finally:
            os.chdir(previous_cwd)
