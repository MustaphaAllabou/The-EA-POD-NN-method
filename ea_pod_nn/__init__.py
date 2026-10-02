"""EA-POD-NN: Error-Augmented POD-Neural Network reduced-order modelling.

The method is non-intrusive: it only needs snapshots produced by a
high-fidelity (HF) solver. Any solver can be plugged in by subclassing
:class:`ea_pod_nn.solvers.HFSolver`.
"""
from .solvers.base import HFSolver, SnapshotSet
from .pod import pod, build_reduced_bases
from .offline import NNConfig, train_ea_pod_nn
from .online import EAPODNNPredictor

__all__ = ["HFSolver", "SnapshotSet", "pod", "build_reduced_bases",
           "NNConfig", "train_ea_pod_nn", "EAPODNNPredictor"]
