"""Proper Orthogonal Decomposition (method of snapshots)."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PODResult:
    basis: np.ndarray        # (n_dofs, n_modes), orthonormal columns
    eigenvalues: np.ndarray  # all eigenvalues of the correlation matrix, decreasing
    n_modes: int


def pod(S, epsilon):
    """POD basis of ``S`` keeping a fraction ``1 - epsilon`` of the energy.

    Method of snapshots: eigen-decomposition of C = S^T S, then
    modes = S v_i / sqrt(lambda_i).
    """
    C = S.T @ S
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0.0, None), V[:, order]

    energy = np.cumsum(lam) / lam.sum()
    n_modes = int(min(np.searchsorted(energy, 1.0 - epsilon) + 1, np.count_nonzero(lam)))
    basis = S @ V[:, :n_modes] / np.sqrt(lam[:n_modes])
    return PODResult(basis, lam, n_modes)


def build_reduced_bases(S, epsilon_pod, epsilon_err):
    """Reduced basis Brb of the snapshots and Ber of the projection error.

    Returns (rb, er, Er) where Er = S - Brb Brb^T S.
    """
    rb = pod(S, epsilon_pod)
    Er = S - rb.basis @ (rb.basis.T @ S)
    er = pod(Er, epsilon_err)
    print(f"[POD] Nrb = {rb.n_modes} modes, Ner = {er.n_modes} error modes")
    return rb, er, Er


def plot_eigenvalues(eigenvalues, label="Normalized eigenvalues", filename=None):
    lam = np.asarray(eigenvalues)
    lam = lam[lam > 0] / lam.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(np.arange(1, len(lam) + 1), lam, "o-", ms=3, label=label)
    ax.set_xlabel("Mode index")
    ax.set_ylabel(r"$\lambda_i / \sum_j \lambda_j$")
    ax.legend()
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
    return fig
