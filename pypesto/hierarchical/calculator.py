import numpy as np
from ..objective.constants import(
    MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS)
from ..objective.amici_util import(
    log_simulation,
    add_sim_grad_to_opt_grad,
    add_sim_hess_to_opt_hess,
    sim_sres_to_opt_sres,
)
from .parameter import HierarchicalParameter
from .problem import HierarchicalProblem

try:
    import amici
except ImportError:
    amici = None


class HierarchicalAmiciCalculator:

    def __init__(self, problem: "HierarchicalProblem"):
        self.problem = problem

    def __call__(self, obj, x, sensi_orders, mode):
        raise NotImplementedError(
            "This class is not intended to be called.")


class HierarchicalForwardAmiciCalculator(HierarchicalAmiciCalculator):

    def __call__(self, obj, x, sensi_order, mode):
        # prepare outputs
        nllh = 0.0
        snllh = np.zeros(obj.dim)
        s2nllh = np.zeros([obj.dim, obj.dim])

        res = np.zeros([0])
        sres = np.zeros([0, obj.dim])

        # set order in solver
        obj.amici_solver.setSensitivityOrder(sensi_order)

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            obj.amici_model,
            obj.amici_solver,
            obj.edatas,
            num_threads=min(obj.n_threads, len(obj.edatas)),
        )

        if any([rdata['status'] < 0.0 for rdata in rdatas]):
            return obj.get_error_output(rdatas)

        # edatas to numpy arrays
        # TODO cache
        edatas = [amici.numpy.edataToNumPyArrays(edata)
                  for edata in obj.edatas]

        # compute optimal parameters
        optimal_scalings = compute_optimal_scaling_matrix(self.problem, edatas, rdatas)
        
        # compute likelihood and derivatives
        for data_ix, rdata in enumerate(rdatas):
            log_simulation(data_ix, rdata)

            # compute objective
            if mode == MODE_FUN:
                nllh -= rdata['llh']
                if sensi_order > 0:
                    add_sim_grad_to_opt_grad(
                        obj.x_ids,
                        obj.mapping_par_opt_to_par_sim[data_ix],
                        rdata['sllh'],
                        snllh,
                        coefficient=-1.0
                    )
                    # TODO: Compute the full Hessian, and check here
                    add_sim_hess_to_opt_hess(
                        obj.x_ids,
                        obj.mapping_par_opt_to_par_sim[data_ix],
                        rdata['FIM'],
                        s2nllh,
                        coefficient=-1.0
                    )

            elif mode == MODE_RES:
                res = np.hstack([res, rdata['res']]) \
                    if res.size else rdata['res']
                if sensi_order > 0:
                    opt_sres = sim_sres_to_opt_sres(
                        obj.x_ids,
                        obj.mapping_par_opt_to_par_sim[data_ix],
                        rdata['sres'],
                        coefficient=1.0
                    )
                    sres = np.vstack([sres, opt_sres]) \
                        if sres.size else opt_sres

        return {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas
        }


class HierarchicalAdjointAmiciCalculator(HierarchicalAmiciCalculator):

    def __call__(self, obj, x, sensi_orders, mode):
        raise NotImplementedError(
            "Combining adjoints and hierarchical optimization is not yet "
            "supported.")


def compute_optimal_scaling_matrix(problem, edatas, rdatas):
    optimal_scalings = compute_optimal_scalings(problem, edatas, rdatas)
    matrix = matrix_like(rdatas, val=1.0)
    for x, opt_s in zip(problem.get_xs_for_type(HierarchicalParameter.SCALING), optimal_scalings):
        apply_optimal_scaling(x, opt_s, matrix)
    return matrix


def compute_optimal_scalings(problem, edatas, rdatas):
    optimal_scalings = []
    for x in problem.get_xs_for_type(HierarchicalParameter.SCALING):
        s_opt = compute_optimal_scaling(x, edatas, rdatas)
        optimal_scalings.append(s_opt)


def compute_optimal_scaling(x, edatas, rdatas):
    num = 0.0
    den = 0.0

    for ix in x.iterate():
        condition_ix, time_ix, observable_ix = ix
        y = edatas[condition_ix]['observedData'][time_ix, observable_ix]
        h = rdatas[condition_ix]['y'][time_ix, observable_ix]

        if np.isnan(y) or np.isnan(h):
            continue

        num += (y - h) * h
        den += h**2

    # TODO check for 0.0

    return num / den


def matrix_like(rdatas, val):
    matrix = []
    for rdata in rdatas:
        matrix.append(np.full(rdata['y'].shape, fill_value=val))
    return matrix


def apply_optimal_scalings(problem, optimal_scalings, matrix):
    for x, opt_s in zip(problem.get_xs_for_type(HierarchicalParameter.SCALING), optimal_scalings):
        apply_optimal_scaling(x, opt_s, matrix)


def apply_optimal_scaling(x, opt_s, matrix):
    for condition_ix, time_ix, observable_ix in x.iterate():
        matrix[condition_ix][time_ix, observable_ix] = opt_s
