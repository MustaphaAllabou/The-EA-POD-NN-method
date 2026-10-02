"""High-fidelity solvers usable by the EA-POD-NN method."""
from .base import HFSolver, SnapshotSet, split_variables

__all__ = ["HFSolver", "SnapshotSet", "split_variables"]
# DassFlowSolver is not imported here so that the package works without dassflow2d:
#     from ea_pod_nn.solvers.dassflow import DassFlowSolver
