"""Template to plug another HF software into EA-POD-NN.

Copy this file, rename the class and fill in ``run``. Nothing else in the
package needs to change: the offline and online phases only use the
``HFSolver`` interface.
"""
import numpy as np

from .base import HFSolver


class MySolver(HFSolver):
    # Names of the fields returned by run(), in stacking order.
    variables = ("h", "u", "v")

    def __init__(self, case_dir, work_dir):
        self.case_dir = case_dir
        self.work_dir = work_dir

    def run(self, mu, run_id, worker_id=0):
        # 1. Prepare a working directory for this worker, e.g.
        #    f"{self.work_dir}/worker_{worker_id}" (parallel runs must not
        #    share files).
        # 2. Translate the parameters ``mu`` into your software's inputs
        #    (boundary conditions, friction, ...) and write the input files.
        # 3. Launch the simulation (Python API, or subprocess.run([...])).
        # 4. Read the results and return them:
        #       times  : (n_t,) output times
        #       fields : (n_vars * n_nodes, n_t), e.g. np.concatenate([h, u, v])
        raise NotImplementedError("Fill in MySolver.run for your software")

    # Optional, used for maps in the online phase:
    # def plot_field(self, values, title=""):
    #     ...
