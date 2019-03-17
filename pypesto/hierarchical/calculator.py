import numpy as np
from ..objective.constants import(
    MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS)
from ..objective.amici_util import(
    log_simulation,
    add_sim_grad_to_opt_grad,
    add_sim_hess_to_opt_hess,
    sim_sres_to_opt_sres,
)

try:
    import amici
except ImportError:
    amici = None


class HierarchicalAmiciCalculator:

    def __init__(self, problem: "HierarchicalProblem"):
        self.problem = problem

    def __call__(self, obj, x, sensi_orders, mode):
        raise NotImplementedError()


class HierarchicalForwardAmiciCalculator(HierarchicalAmiciCalculator):

    def __call__(self, obj, x, sensi_orders, mode):
        pass


class HierarchicalAdjointAmiciCalculator(HierarchicalAmiciCalculator):

    def __call__(self, obj, x, sensi_orders, mode):
        raise NotImplementedError(
            "Combining adjoints and hierarchical optimization is not yet "
            "supported.")
