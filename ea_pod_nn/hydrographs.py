"""Triangular inflow hydrographs and their parameter space (Aude test case).

A parameter vector is mu = (t_peak1, Q_max1, t_peak2, Q_max2): peak times (s)
and peak discharges (m3/s) of the two inflow hydrographs.
"""
from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np


class TriangularHydrographs:
    """Two triangular flood waves on top of constant base flows.

    Each wave starts at ``t_rise``, peaks at ``t_peak`` and ends at
    ``2 * t_peak - t_rise``. Outside the wave the discharge is the base flow.
    """

    def __init__(self, times, t_rise=46800.0, base_flows=(7.0, 2.0)):
        self.times = np.asarray(times, float)
        self.t_rise = float(t_rise)
        self.base_flows = base_flows

    def _wave(self, t_peak, q_max, q_base):
        t_end = 2.0 * t_peak - self.t_rise
        t = self.times
        q = np.full_like(t, q_base)
        rise = (t >= self.t_rise) & (t <= t_peak)
        fall = (t > t_peak) & (t <= t_end)
        q[rise] = q_base + (q_max - q_base) * (t[rise] - self.t_rise) / (t_peak - self.t_rise)
        q[fall] = q_max + (q_base - q_max) * (t[fall] - t_peak) / (t_end - t_peak)
        return q

    def __call__(self, mu):
        t_peak1, q_max1, t_peak2, q_max2 = mu
        return (self._wave(t_peak1, q_max1, self.base_flows[0]),
                self._wave(t_peak2, q_max2, self.base_flows[1]))

    def plot(self, mus, filename=None):
        fig, ax = plt.subplots()
        for mu in np.atleast_2d(mus):
            q1, q2 = self(mu)
            ax.plot(self.times / 3600.0, q1, "b", alpha=0.4)
            ax.plot(self.times / 3600.0, q2, "r", alpha=0.4)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(r"Discharge ($m^3.s^{-1}$)")
        if filename:
            fig.savefig(filename, dpi=150)
        return fig


def parameter_grid(t_peak1, q_max1, t_peak2, q_max2, max_relative_shift=None):
    """Tensor grid of parameters, optionally keeping only phase-shifted peaks
    with |t_peak1 - t_peak2| / t_peak1 <= max_relative_shift."""
    mus = np.array(list(itertools.product(t_peak1, q_max1, t_peak2, q_max2)), float)
    if max_relative_shift is not None:
        mus = mus[np.abs(mus[:, 0] - mus[:, 2]) / np.abs(mus[:, 0]) <= max_relative_shift]
    return mus


def plot_parameter_space(mus, reference=None, filename=None):
    mus = np.atleast_2d(mus)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = ((0, 2, r"$t_{peak,1}$ (h)", r"$t_{peak,2}$ (h)"),
             (0, 1, r"$t_{peak,1}$ (h)", r"$Q_{max,1}$ ($m^3.s^{-1}$)"),
             (2, 3, r"$t_{peak,2}$ (h)", r"$Q_{max,2}$ ($m^3.s^{-1}$)"))
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        sx = 3600.0 if i in (0, 2) else 1.0
        sy = 3600.0 if j in (0, 2) else 1.0
        ax.scatter(mus[:, i] / sx, mus[:, j] / sy, s=8)
        if reference is not None:
            ax.scatter(reference[i] / sx, reference[j] / sy, color="r", label="online")
            ax.legend()
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
    return fig
