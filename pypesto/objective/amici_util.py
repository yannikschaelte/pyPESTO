import numpy as np
import numbers
import logging

logger = logging.getLogger(__name__)


def log_simulation(data_ix, rdata):
    """
    Log the simulation results.
    """
    logger.debug(f"=== DATASET {data_ix} ===")
    logger.debug(f"status: {rdata['status']}")
    logger.debug(f"llh: {rdata['llh']}")

    t_steadystate = 't_steadystate'
    if t_steadystate in rdata and rdata[t_steadystate] != np.nan:
        logger.debug(f"t_steadystate: {rdata[t_steadystate]}")

    logger.debug(f"res: {rdata['res']}")


def map_par_opt_to_par_sim(mapping_par_opt_to_par_sim, par_opt_ids, x):
    """
    From the optimization vector `x`, create the simulation vector according
    to the mapping `mapping`.

    Parameters
    ----------

    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.
    par_opt_ids: array-like of str
        The optimization parameter ids. This vector is needed to know the
        order of the entries in x.
    x: array-like of float
        The optimization parameters vector.

    Returns
    -------

    y: array-like of float
        The simulation parameters vector corresponding to x under the
        specified mapping.
    """

    # number of simulation parameters
    n_par_sim = len(mapping_par_opt_to_par_sim)

    # prepare simulation parameter vector
    par_sim_vals = np.zeros(n_par_sim)

    # iterate over simulation parameter indices
    for j_par_sim in range(n_par_sim):
        # extract entry in mapping table for j_par_sim
        val = mapping_par_opt_to_par_sim[j_par_sim]

        if isinstance(val, numbers.Number):
            # fixed value assignment
            par_sim_vals[j_par_sim] = val
        else:
            # value is optimization parameter id
            par_sim_vals[j_par_sim] = x[par_opt_ids.index(val)]

    # return the created simulation parameter vector
    return par_sim_vals


def create_plist_from_par_opt_to_par_sim(mapping_par_opt_to_par_sim):
    """
    From the parameter mapping `mapping_par_opt_to_par_sim`, create the
    simulation plist according to the mapping `mapping`.

    Parameters
    ----------

    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.

    Returns
    -------

    plist: array-like of float
        List of parameter indices for which the sensitivity needs to be
        computed
    """
    plist = []

    # iterate over simulation parameter indices
    for j_par_sim, val in enumerate(mapping_par_opt_to_par_sim):
        if not isinstance(val, numbers.Number):
            plist.append(j_par_sim)

    # return the created simulation parameter vector
    return plist


def create_scale_mapping_from_model(amici_scales, n_edata):
    """
    Create parameter scaling mapping matrix from amici scaling
    vector.
    """
    scales = []
    amici_scales = list(amici_scales)

    for amici_scale in amici_scales:
        if amici_scale == amici.ParameterScaling_none:
            scale = 'lin'
        elif amici_scale == amici.ParameterScaling_ln:
            scale = 'log'
        elif amici_scale == amici.ParameterScaling_log10:
            scale = 'log10'
        else:
            raise Exception(
                f"Parameter scaling {amici_scale} in amici model not"
                f"recognized.")
        scales.append(scale)

    mapping_scale_opt_to_scale_sim = [scales for _ in range(n_edata)]

    return mapping_scale_opt_to_scale_sim


def add_sim_grad_to_opt_grad(par_opt_ids,
                             mapping_par_opt_to_par_sim,
                             sim_grad,
                             opt_grad,
                             coefficient: float = 1.0):
    """
    Sum simulation gradients to objective gradient according to the provided
    mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------

    par_opt_ids: array-like of str
        The optimization parameter ids. This vector is needed to know the
        order of the entries in x.
    mapping_par_opt_to_par_sim: array-like of str
        len == n_par_sim, the entries are either numeric, or
        optimization parameter ids.
    sim_grad: array-like of float
        Simulation gradient.
    opt_grad: array-like of float
        The optimization gradient. To which sim_grad is added.
        Will be changed in place.
    coefficient: float
        Coefficient for sim_grad when adding to opt_grad.
    """

    par_sim_idx = 0
    for par_opt_id in mapping_par_opt_to_par_sim:
        # we ignore non-string indices as a fixed value as been set for
        # those and they are not included in the condition specific nplist,
        # we do not only skip here, but also do not increase par_sim_idx!
        if not isinstance(par_opt_id, str):
            # this was a numeric override for which we ignore the gradient
            continue

        par_opt_idx = par_opt_ids.index(par_opt_id)
        opt_grad[par_opt_idx] += coefficient * sim_grad[par_sim_idx]
        par_sim_idx += 1


def add_sim_hess_to_opt_hess(par_opt_ids,
                             mapping_par_opt_to_par_sim,
                             sim_hess,
                             opt_hess,
                             coefficient: float = 1.0):
    """
    Sum simulation hessians to objective hessian according to the provided
    mapping `mapping_par_opt_to_par_sim`.
    Parameters
    ----------
    Same as for add_sim_grad_to_opt_grad, replacing the gradients by hessians.
    """

    # use enumerate for first axis as plist is not applied
    # https://github.com/ICB-DCM/AMICI/issues/274
    par_sim_idx = 0
    for par_opt_id in mapping_par_opt_to_par_sim:
        if not isinstance(par_opt_id, str):
            # this was a numeric override for which we ignore the hessian
            continue

        par_opt_idx = par_opt_ids.index(par_opt_id)

        # for second axis, plist was applied so we can skip over values with
        # numeric mapping
        par_sim_idx_2 = 0
        for par_opt_id_2 in mapping_par_opt_to_par_sim:
            # we ignore non-string indices as a fixed value as been set for
            # those and they are not included in the condition specific nplist,
            # we not only skip here, but also do not increase par_sim_idx_2!
            if not isinstance(par_opt_id_2, str):
                continue

            par_opt_idx_2 = par_opt_ids.index(par_opt_id_2)

            opt_hess[par_opt_idx, par_opt_idx_2] += \
                coefficient * sim_hess[par_sim_idx, par_sim_idx_2]
            par_sim_idx_2 += 1
    par_sim_idx += 1


def sim_sres_to_opt_sres(par_opt_ids,
                         mapping_par_opt_to_par_sim,
                         sim_sres,
                         coefficient: float = 1.0):
    """
    Sum simulation residual sensitivities to objective residual sensitivities
    according to the provided mapping `mapping_par_opt_to_par_sim`.

    Parameters
    ----------

    Same as for add_sim_grad_to_opt_grad, replacing the gradients by residual
    sensitivities.
    """
    opt_sres = np.zeros((sim_sres.shape[0], len(par_opt_ids)))

    par_sim_idx = 0
    for par_opt_id in mapping_par_opt_to_par_sim:
        if not isinstance(par_opt_id, str):
            # this was a numeric override for which we ignore the hessian
            continue

        par_opt_idx = par_opt_ids.index(par_opt_id)
        opt_sres[:, par_opt_idx] += \
            coefficient * sim_sres[:, par_sim_idx]
        par_sim_idx += 1

    return opt_sres
