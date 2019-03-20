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
    """
    A calculator is passed as `calculator` to the pypesto.AmiciObjective.
    While this class cannot be used directly, it has two subclasses
    which allow to use forward or adjoint sensitivity analysis to
    solve a `pypesto.HierarchicalProblem` efficiently in an inner loop,
    while the outer optimization is only concerned with variables not
    specified as `pypesto.HierarchicalParameter`s.
    """

    def __init__(self, problem: HierarchicalProblem):
        """
        Initialize the calculator from the given problem.
        """
        self.problem = problem

    def __call__(self, obj, x, sensi_order, mode):
        """
        This function is called inside `pypesto.AmiciObjective.__call__`
        after minor preprocessing,
        and is supposed to return the function value, derivatives and
        possibly residuals in a dict for the given input.
        """
        raise NotImplementedError(
            "This class is not intended to be called.")


class HierarchicalForwardAmiciCalculator(HierarchicalAmiciCalculator):
    """
    Use forward sensitivity analysis to compute derivatives for the
    hierarchical problem.
    """

    def __call__(self, obj, x, sensi_order, mode):
        # prepare outputs
        nllh = 0.0
        snllh = np.zeros(obj.dim)
        s2nllh = np.zeros([obj.dim, obj.dim])

        res = np.zeros([0])
        sres = np.zeros([0, obj.dim])

        # set order in solver
        #print("sensi_order: ", sensi_order)
        obj.amici_solver.setSensitivityOrder(sensi_order)

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            obj.amici_model,
            obj.amici_solver,
            obj.edatas,
            num_threads=min(obj.n_threads, len(obj.edatas)),
        )

        # check if any simulation failed
        if any([rdata['status'] < 0.0 for rdata in rdatas]):
            return obj.get_error_output(rdatas)

        # edatas to numpy arrays (TODO cache this?)
        edatas = [amici.numpy.edataToNumPyArrays(edata)
                  for edata in obj.edatas]

        # compute optimal parameters (code base same for fw+ad)

        # create matrices from optimal parameters (only fw)

        # compute objective and derivatives (only fw)

        # compute optimal parameters
        optimal_scalings = compute_optimal_scaling_matrix(self.problem, edatas, rdatas)

        nllh = compute_nllh(edatas, rdatas, optimal_scalings)
        if sensi_order > 0:
            snllh = compute_snllh(edatas, rdatas, optimal_scalings, obj.x_ids, obj.mapping_par_opt_to_par_sim, obj.dim)

        # TODO compute FIM or HESS
        # TODO RES, SRES should also be possible, right?

        return {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas
        }


class HierarchicalAdjointAmiciCalculator(HierarchicalAmiciCalculator):
    """
    Use adjoint sensitivity analysis to compute derivatives for the
    hierarchical problem.
    """

    def __call__(self, obj, x, sensi_orders, mode):
        raise NotImplementedError(
            "Combining adjoints and hierarchical optimization is not yet "
            "supported.")


def compute_optimal_scaling_matrix(problem, edatas, rdatas):
    optimal_scalings = compute_optimal_scalings(problem, edatas, rdatas)
    matrix = matrix_like(rdatas, val=1.0)
    for x, opt_s in zip(
			problem.get_xs_for_type(HierarchicalParameter.SCALING),
			optimal_scalings):
        apply_optimal_scaling(x, opt_s, matrix)
    return matrix


def compute_optimal_scalings(problem, edatas, rdatas):
    optimal_scalings = []
    for x in problem.get_xs_for_type(HierarchicalParameter.SCALING):
        s_opt = compute_optimal_scaling(x, edatas, rdatas)
        optimal_scalings.append(s_opt)
    return optimal_scalings


def compute_optimal_scaling(x, edatas, rdatas):
    num = 0.0
    den = 0.0

    for condition_ix, time_ix, observable_ix in x.iterate():
        y = edatas[condition_ix]['observedData'][time_ix, observable_ix]
        h = rdatas[condition_ix]['y'][time_ix, observable_ix]
        sigma = rdatas[condition_ix]['sigmay'][time_ix, observable_ix]

        if np.isnan(y) or np.isnan(h) or np.isnan(sigma):
            continue

        num += y * h / sigma**2
        den += h**2 / sigma**2

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


def compute_nllh(edatas, rdatas, optimal_scalings):
    nllh = 0.0
    for edata, rdata, optimal_s in zip(edatas, rdatas, optimal_scalings):
        #print(edata['observedData'], optimal_s, rdata['y'], rdata['sigmay'])
        nllh += 0.5 * np.nansum(np.log(2*np.pi*rdata['sigmay']**2))
        nllh += 0.5 * np.nansum((edata['observedData'] - optimal_s * rdata['y'])**2 / rdata['sigmay']**2)
    #print(nllh)
    return nllh


def compute_snllh(edatas, rdatas, optimal_scalings, x_ids, mapping_par_opt_to_par_sim, dim):
    snllh = np.zeros(dim)
    for condition_ix, (edata, rdata, optimal_s) in \
            enumerate(zip(edatas, rdatas, optimal_scalings)):
        # sy is n_obs x n_par x n_time
        sy_ = rdata['sy']
        sy = np.full((sy_.shape[1],sy_.shape[0],sy_.shape[2]), np.nan)
        for i in range(sy_.shape[1]):
            sy[i] = sy_[:,i,:]
        #print(sy)
        val = np.nansum((edata['observedData'] - optimal_s * rdata['y']) \
                / rdata['sigmay']**2 * optimal_s * sy, axis=(1, 2))
        #print(val)
        add_sim_grad_to_opt_grad(
            x_ids,
            mapping_par_opt_to_par_sim[condition_ix],
            val,
            snllh,
            coefficient=-1.0)
    return snllh
