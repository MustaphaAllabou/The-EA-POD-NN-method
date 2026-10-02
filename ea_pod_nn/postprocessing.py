"""Error analysis between the HF reference and the EA-POD-NN prediction."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def error_summary(errors):
    """Mean, median and max of |error| for each field of {name: error array}."""
    for name, e in errors.items():
        a = np.abs(e)
        print(f"[error] {name}: mean {a.mean():.3e}, median {np.median(a):.3e}, "
              f"max {a.max():.3e}")


def plot_error_by_bins(fields, filename=None):
    """Mean and std of |error| per range of the reference value.

    fields : {label: (reference_values, errors, bin_edges)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (values, errors, bins) in fields.items():
        bins = np.asarray(bins)
        centers, means, stds = [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = np.abs(errors[(values >= a) & (values <= b)])
            if sel.size:
                centers.append(0.5 * (a + b))
                means.append(sel.mean())
                stds.append(sel.std())
        ax.errorbar(centers, means, yerr=stds, fmt="o", capsize=5,
                    label=rf"$|\epsilon_{{{label}}}|$")
    ax.set_xlabel("Reference value")
    ax.set_ylabel("Mean and std of the absolute error")
    ax.legend()
    if filename:
        fig.savefig(filename, dpi=150)
    return fig


def plot_error_cdf(errors, filename=None):
    """Empirical CDF of |error| with a fitted normal CDF, for each field."""
    fig, ax = plt.subplots()
    for label, e in errors.items():
        data = np.sort(np.abs(np.ravel(e)))
        line, = ax.plot(data, np.arange(1, len(data) + 1) / len(data),
                        label=rf"Empirical $|\epsilon_{{{label}}}|$")
        mu, std = norm.fit(data)
        ax.plot(data, norm.cdf(data, mu, std), "--", color=line.get_color(),
                label=rf"Normal fit $|\epsilon_{{{label}}}|$")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("CDF")
    ax.legend()
    if filename:
        fig.savefig(filename, dpi=150)
    return fig


def plot_mean_error_over_time(times, errors, filename=None):
    """Spatial mean of |error| at each output time (times in seconds)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, e in errors.items():
        mean = np.mean(np.abs(e), axis=0)
        line, = ax.semilogy(np.asarray(times) / 3600.0, mean,
                            label=rf"$|\epsilon_{{{label}}}|$ (EA-POD-NN)")
        ax.axhline(mean.max(), color=line.get_color(), linestyle="--")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Spatial mean of the absolute error")
    ax.legend()
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
    return fig
