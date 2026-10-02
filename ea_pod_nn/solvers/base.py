"""Common interface for high-fidelity (HF) solvers.

EA-POD-NN never talks to a simulation software directly: it only calls the
methods defined here. To use another software (TELEMAC, HEC-RAS, OpenFOAM,
an in-house code...), subclass :class:`HFSolver` and implement ``run``.
See ``template.py`` for a commented skeleton.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass
class SnapshotSet:
    """Snapshots and the matching network inputs.

    S      : (n_dofs, n_snapshots) snapshot matrix, variables stacked by rows.
    P      : (n_snapshots, n_params + 1) inputs, i.e. [mu_1, ..., mu_p, t].
    sim_id : (n_snapshots,) index of the HF simulation each column comes from.
    variables : names of the stacked variables, e.g. ("h", "u", "v").
    """
    S: np.ndarray
    P: np.ndarray
    sim_id: np.ndarray
    variables: tuple


class HFSolver(ABC):
    """Base class for any high-fidelity solver.

    A subclass must:

    * set ``variables``: names of the fields it returns, in stacking order;
    * implement :meth:`run`.

    Everything else (parallel runs, caching on disk, building the snapshot
    matrix and the parameter/time inputs) is handled here.
    """

    variables: tuple = ()

    @abstractmethod
    def run(self, mu, run_id, worker_id=0):
        """Run one HF simulation for the parameter vector ``mu``.

        Parameters
        ----------
        mu : 1D array, physical parameters of the simulation.
        run_id : identifier of the run (int or str), for logs / file names.
        worker_id : index of the parallel worker executing the run; use it
            to give each worker its own working directory if needed.

        Returns
        -------
        times : (n_t,) array of output times.
        fields : (n_vars * n_nodes, n_t) array; column k is the solution at
            ``times[k]`` with the variables stacked in ``self.variables`` order.
        """

    def plot_field(self, values, title=""):
        """Optional: plot a nodal field on the solver mesh."""
        raise NotImplementedError(f"{type(self).__name__} does not implement plot_field")

    # ------------------------------------------------------------------ #
    def compute_snapshots(self, mus, cache_dir, n_jobs=1, overwrite=False):
        """Run (or reload) the HF simulations for all parameter vectors.

        Each run is saved in ``cache_dir/run_XXXX.npz``, so an interrupted
        offline phase restarts where it stopped.
        """
        mus = np.atleast_2d(np.asarray(mus, dtype=float))
        os.makedirs(cache_dir, exist_ok=True)
        todo = [i for i in range(len(mus))
                if overwrite or not os.path.exists(_cache_path(cache_dir, i))]

        if todo:
            n_jobs = max(1, min(n_jobs, len(todo)))
            chunks = [todo[k::n_jobs] for k in range(n_jobs)]
            print(f"[HF] {len(todo)} simulation(s) to run on {n_jobs} worker(s)")
            if n_jobs == 1:
                _run_chunk(self, mus, chunks[0], cache_dir, 0)
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    futures = [pool.submit(_run_chunk, self, mus, c, cache_dir, w)
                               for w, c in enumerate(chunks)]
                    for f in futures:
                        f.result()

        S, P, sim_id = [], [], []
        for i, mu in enumerate(mus):
            data = np.load(_cache_path(cache_dir, i))
            if not np.allclose(data["mu"], mu):
                raise ValueError(f"{_cache_path(cache_dir, i)} was computed for other "
                                 "parameters; use overwrite=True or another cache_dir.")
            times, fields = data["times"], data["fields"]
            S.append(fields)
            P.append(np.column_stack([np.tile(mu, (len(times), 1)), times]))
            sim_id.append(np.full(len(times), i))
        snaps = SnapshotSet(np.hstack(S), np.vstack(P), np.concatenate(sim_id),
                            tuple(self.variables))
        print(f"[HF] snapshot matrix: {snaps.S.shape}, inputs: {snaps.P.shape}")
        return snaps


def split_variables(U, variables):
    """Split a stacked array (n_vars * n_nodes, ...) into {name: array}."""
    return dict(zip(variables, np.split(np.asarray(U), len(variables), axis=0)))


def _cache_path(cache_dir, i):
    return os.path.join(cache_dir, f"run_{i:04d}.npz")


def _run_chunk(solver, mus, indices, cache_dir, worker_id):
    for i in indices:
        times, fields = solver.run(mus[i], i, worker_id)
        times, fields = np.asarray(times, float), np.asarray(fields, float)
        if fields.ndim != 2 or fields.shape[1] != len(times):
            raise ValueError(f"run {i}: fields must be (n_dofs, n_t) with n_t = len(times), "
                             f"got {fields.shape} and {len(times)} times")
        if fields.shape[0] % len(solver.variables):
            raise ValueError(f"run {i}: {fields.shape[0]} rows is not a multiple of "
                             f"{len(solver.variables)} variables")
        np.savez(_cache_path(cache_dir, i), times=times, fields=fields, mu=mus[i])
        print(f"[HF] worker {worker_id}: run {i} done")
