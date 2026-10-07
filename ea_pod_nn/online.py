"""EA-POD-NN online phase: fast prediction for new parameters.

No HF solver is needed here: only the files saved by the offline phase.
"""
from __future__ import annotations

import os

import numpy as np

from .networks import get_backend
from .offline import MinMax
from .solvers.base import split_variables


class EAPODNNPredictor:
    def __init__(self, network_rb, network_er, Brb, Ber, p_scale, rb_scale, er_scale,
                 variables, backend="keras"):
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.network_rb, self.network_er = network_rb, network_er
        self.Brb, self.Ber = Brb, Ber
        self.p_scale, self.rb_scale, self.er_scale = p_scale, rb_scale, er_scale
        self.variables = tuple(variables)

    @classmethod
    def load(cls, model_dir="results/model"):
        """Reload a reduced model; the backend used for training is detected."""
        d = np.load(os.path.join(model_dir, "reduced_model.npz"))
        backend = get_backend(str(d["backend"]) if "backend" in d else "keras")
        return cls(backend.load(os.path.join(model_dir, "network_rb")),
                   backend.load(os.path.join(model_dir, "network_er")),
                   d["Brb"], d["Ber"],
                   MinMax(d["p_min"], d["p_max"]), MinMax(d["rb_min"], d["rb_max"]),
                   MinMax(d["er_min"], d["er_max"]), [str(v) for v in d["variables"]],
                   backend)

    def predict(self, mu, times):
        """Reduced solution (n_dofs, n_t) for parameters ``mu`` at ``times``."""
        times = np.atleast_1d(np.asarray(times, float))
        X = np.column_stack([np.tile(np.asarray(mu, float), (len(times), 1)), times])
        X = self.p_scale.transform(X)
        urb = self.rb_scale.inverse(self.backend.predict(self.network_rb, X))
        uer = self.er_scale.inverse(self.backend.predict(self.network_er, X))
        return self.Brb @ urb.T + self.Ber @ uer.T

    def predict_fields(self, mu, times):
        """Same as :meth:`predict`, split into {variable: (n_nodes, n_t)}."""
        return split_variables(self.predict(mu, times), self.variables)
