import numpy as np
import copy
import logging
import numbers
from .objective import Objective
from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS

try:
    import amici
except ImportError:
    amici = None

logger = logging.getLogger(__name__)


class AmiciObjective(Objective):
    """
    This class allows to create an objective directly from an amici model.
    """

    def __init__(self,
                 amici_model, amici_solver, edatas,
                 max_sensi_order=None,
                 x_ids=None, x_names=None,
                 mapping_par_opt_to_par_sim=None,
                 mapping_scale_opt_to_scale_sim=None,
                 preprocess_edatas=True,
                 options=None):
        """
        Constructor.

        Parameters
        ----------

        amici_model: amici.Model
            The amici model.

        amici_solver: amici.Solver
            The solver to use for the numeric integration of the model.

        edatas: amici.ExpData or list of amici.ExpData
            The experimental data. If a list is passed, its entries correspond
            to multiple experimental conditions.

        max_sensi_order: int, optional
            Maximum sensitivity order supported by the model. Defaults to 2 if
            the model was compiled with o2mode, otherwise 1.

        x_ids: list of str, optional
            Ids of optimization parameters. In the simplest case, this will be
            the AMICI model parameters (default).

        x_names: list of str, optional
            See ``Objective.x_names``.

        mapping_par_opt_to_par_sim: optional
            Mapping of optimization parameters to model parameters. List array
            of size n_simulation_parameters * n_conditions.
            The default is just to assume that optimization and simulation
            parameters coincide. The default is to assume equality of both.

        mapping_scale_opt_to_scale_sim: optional
            Mapping of optimization parameter scales to simulation parameter
            scales. The default is to just use the scales specified in the
            `amici_model` already.

        preprocess_edatas: bool, optional (default = True)
            Whether to perform preprocessing, i.e. preequilibration, if that
            is specified in the model.

        options: pypesto.ObjectiveOptions, optional
            Further options.
        """
        if amici is None:
            raise ImportError(
                "This objective requires an installation of amici "
                "(https://github.com/icb-dcm/amici). "
                "Install via `pip3 install amici`.")

        if max_sensi_order is None:
            # 2 if model was compiled with second orders,
            # otherwise 1 can be guaranteed
            max_sensi_order = 2 if amici_model.o2mode else 1

        fun = self.get_bound_fun()

        if max_sensi_order > 0:
            grad = True
            hess = True
        else:
            grad = None
            hess = None

        res = self.get_bound_res()

        if max_sensi_order > 0:
            sres = True
        else:
            sres = None

        super().__init__(
            fun=fun, grad=grad, hess=hess, hessp=None,
            res=res, sres=sres,
            fun_accept_sensi_orders=True,
            res_accept_sensi_orders=True,
            options=options
        )

        self.amici_model = amici.ModelPtr(amici_model.clone())
        self.amici_solver = amici.SolverPtr(amici_solver.clone())

        # make sure the edatas are a list of edata objects
        if isinstance(edatas, amici.amici.ExpData):
            edatas = [edatas]

        self.preprocess_edatas = preprocess_edatas
        if preprocess_edatas:
            # preprocess the experimental data
            self.preequilibration_edatas = []
            self.init_preequilibration_edatas(edatas)
        else:
            self.preequilibration_edatas = None

        # set the experimental data container
        self.edatas = edatas

        # set the maximum sensitivity order
        self.max_sensi_order = max_sensi_order

        # optimization parameter ids
        if x_ids is None:
            # use model parameter ids as ids
            x_ids = list(self.amici_model.getParameterIds())
        self.x_ids = x_ids

        self.dim = len(self.x_ids)

        # mapping of parameters
        if mapping_par_opt_to_par_sim is None:
            # use identity mapping for each condition
            mapping_par_opt_to_par_sim = \
                [x_ids for _ in range(len(self.edatas))]
        self.mapping_par_opt_to_par_sim = mapping_par_opt_to_par_sim

        # mapping of parameter scales
        if mapping_scale_opt_to_scale_sim is None:
            # use scales from amici model
            mapping_scale_opt_to_scale_sim = \
                create_scale_mapping_from_model(
                    self.amici_model.getParameterScale(), len(self.edatas))
        self.mapping_scale_opt_to_scale_sim = mapping_scale_opt_to_scale_sim

        # optimization parameter names
        if x_names is None:
            # use ids as names
            x_names = x_ids
        self.x_names = x_names

    def get_bound_fun(self):
        """
        Generate a fun function that calls _call_amici with MODE_FUN. Defining
        a non-class function that references self as a local variable will bind
        the function to a copy of the current self object and will
        accordingly not take future changes to self into account.
        """
        def fun(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_FUN)

        return fun

    def get_bound_res(self):
        """
        Generate a res function that calls _call_amici with MODE_RES. Defining
        a non-class function that references self as a local variable will bind
        the function to a copy of the current self object and will
        accordingly not take future changes to self into account.
        """
        def res(x, sensi_orders):
            return self._call_amici(x, sensi_orders, MODE_RES)

        return res

    def rebind_fun(self):
        """
        Replace the current fun function with one that is bound to the current
        instance
        """
        self.fun = self.get_bound_fun()

    def rebind_res(self):
        """
        Replace the current res function with one that is bound to the current
        instance
        """
        self.res = self.get_bound_res()

    def __deepcopy__(self, memodict=None):
        model = amici.ModelPtr(self.amici_model.clone())
        solver = amici.SolverPtr(self.amici_solver.clone())
        edatas = [amici.ExpData(data) for data in self.edatas]
        other = AmiciObjective(model, solver, edatas)
        for attr in self.__dict__:
            if attr not in ['amici_solver', 'amici_model',
                            'edatas', 'preequilibration_edatas']:
                other.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return other

    def _update_from_problem(self,
                             dim_full,
                             x_free_indices,
                             x_fixed_indices,
                             x_fixed_vals):
        """
        Handle fixed parameters. Here we implement the amici exclusive
        initialization of ParameterLists and respectively replace the
        generic postprocess function

        TODO (see #99): Currently, this method is not used. Instead, the super
        fallback Objective.update_from_problem ist used, which in particular
        does not make use of pesto's ability to compute only compute required
        directional derivativs. If that is intended, the mapping between
        simulation and optimization parameters must be accounted for.

        Parameters
        ----------

        dim_full: int
            Dimension of the full vector including fixed parameters.

        x_free_indices: array_like of int
            Vector containing the indices (zero-based) of free parameters
            (complimentary to x_fixed_indices).

        x_fixed_indices: array_like of int, optional
            Vector containing the indices (zero-based) of parameter components
            that are not to be optimized.

        x_fixed_vals: array_like, optional
            Vector of the same length as x_fixed_indices, containing the values
            of the fixed parameters.
        """
        super(AmiciObjective, self).update_from_problem(
            dim_full,
            x_free_indices,
            x_fixed_indices,
            x_fixed_vals
        )

        # we subindex the existing plist in case there already is a user
        # specified plist
        plist = self.amici_model.getParameterList()
        plist = [plist[idx] for idx in x_free_indices]
        self.amici_model.setParameterList(plist)

        self.dim = len(plist)

        def postprocess(result):
            if HESS in result:
                hess = result[HESS]
                if hess.shape[0] == dim_full:
                    # see https://github.com/ICB-DCM/AMICI/issues/274
                    hess = hess[..., x_free_indices]
                    result[HESS] = hess
            return result

        self.postprocess = postprocess

        # now we need to rebind fun and res to this instance of AmiciObjective
        # for the changes to have an effect
        self.rebind_fun()
        self.rebind_res()

    def _call_amici(
            self,
            x,
            sensi_orders,
            mode
    ):
        # amici is built such that only the maximum sensitivity is required,
        # the lower orders are then automatically computed
        sensi_order = min(max(sensi_orders), 1)
        # order 2 currently not implemented, we are using the FIM

        # check if the requested sensitivities can be computed
        if sensi_order > self.max_sensi_order:
            raise Exception("Sensitivity order not allowed.")

        # prepare result objects

        rdatas = []

        nllh = 0.0
        snllh = np.zeros(self.dim)
        s2nllh = np.zeros([self.dim, self.dim])

        res = np.zeros([0])
        sres = np.zeros([0, self.dim])

        # set order in solver
        self.amici_solver.setSensitivityOrder(sensi_order)

        if self.preequilibration_edatas:
            preeq_status = self.run_preequilibration(x)
            if preeq_status is not None:
                return self.get_error_output(rdatas)

        # loop over experimental data
        for data_ix, edata in enumerate(self.edatas):

            # set model parameter scale for condition index
            self.set_parameter_scale(data_ix)

            # set parameters in model, according to mapping
            self.set_par_sim_for_condition(data_ix, x)

            if self.preequilibration_edatas:
                original_value_dict = self.preprocess_preequilibration(data_ix)
            else:
                original_value_dict = None

            # run amici simulation
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                edata)

            # append to result
            rdatas.append(rdata)

            # reset fixed preequilibration parameters and initial states
            if self.preequilibration_edatas:
                self.postprocess_preequilibration(edata, original_value_dict)

            # logging
            log_simulation(data_ix, rdata)

            # check if the computation failed
            if rdata['status'] < 0.0:
                return self.get_error_output(rdatas)

            # compute objective
            if mode == MODE_FUN:
                nllh -= rdata['llh']
                if sensi_order > 0:
                    add_sim_grad_to_opt_grad(
                        self.x_ids,
                        self.mapping_par_opt_to_par_sim[data_ix],
                        rdata['sllh'],
                        snllh,
                        coefficient=-1.0
                    )
                    # TODO: Compute the full Hessian, and check here
                    add_sim_hess_to_opt_hess(
                        self.x_ids,
                        self.mapping_par_opt_to_par_sim[data_ix],
                        rdata['FIM'],
                        s2nllh,
                        coefficient=-1.0
                    )

            elif mode == MODE_RES:
                res = np.hstack([res, rdata['res']]) \
                    if res.size else rdata['res']
                if sensi_order > 0:
                    sres = np.vstack([sres, rdata['sres']]) \
                        if sres.size else rdata['sres']

        return {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas
        }

    def init_preequilibration_edatas(self, edatas):
        """
        Extract information needed for doing preequilibration.
        """
        self.preequilibration_edatas = []

        for edata in edatas:
            # extract values of the fixed parameters
            fixed_parameters = list(edata.fixedParametersPreequilibration)

            # create edata object from model
            preeq_edata = amici.amici.ExpData(self.amici_model.get())

            # fill in fixed parameter values for preequilibration
            preeq_edata.fixedParametersPreequilibration = fixed_parameters

            # only preequilibration
            preeq_edata.setTimepoints([])

            # indicate whether preequilibration is required for this data set
            preequilibrate = len(fixed_parameters) > 0
            # TODO (see #100): Currently, len(fixed_parameters) == 0 is used
            # as an indicator of requiring no preequilibration. However, when
            # there are events (which amici cannot deal with yet in python),
            # the situation can occur that there are no fixed_parameters, but
            # events that are omitted in the preequilibration run.

            self.preequilibration_edatas.append(dict(
                edata=preeq_edata,
                preequilibrate=preequilibrate
            ))

    def run_preequilibration(self, x):
        """
        Run preequilibration.
        """

        for data_ix, preeq_dict in enumerate(self.preequilibration_edatas):

            if not preeq_dict['preequilibrate']:
                # no preequilibration required
                continue

            # set model parameter scales for condition index
            self.set_parameter_scale(data_ix)

            # map to simulation parameters
            self.set_par_sim_for_condition(data_ix, x)

            # TODO: Conditions might share preeq conditions and dynamic
            # parameters. In that case, we can save time here.

            # run amici simulation
            rdata = amici.runAmiciSimulation(
                self.amici_model,
                self.amici_solver,
                preeq_dict['edata'])

            # check if an error occurred
            if rdata['status'] < 0.0:
                return rdata['status']

            # fill state
            preeq_dict['x0'] = rdata['x0']
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                # fill state sensitivities
                # TODO check that these are always computed, i.e.
                # forward sensitivities used
                preeq_dict['sx0'] = rdata['sx0']

    def preprocess_preequilibration(self, edata_ix):
        """
        Update the model and data from the preequilibration states,
        before running the real simulation.
        """

        data = self.edatas[edata_ix]

        original_fixed_parameters_preequilibration = None
        original_initial_states = None
        original_initial_state_sensitivities = None

        # if this data set needed preequilibration, adapt the states
        # according to the previously run preequilibration
        if self.preequilibration_edatas[edata_ix]['preequilibrate']:

            # remember original states and sensitivities
            original_initial_states = self.amici_model.getInitialStates()
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                original_initial_state_sensitivities = \
                    self.amici_model.getInitialStateSensitivities()

            # remember original fixed parameters for preequilibration
            original_fixed_parameters_preequilibration \
                = data.fixedParametersPreequilibration

            # unset fixed preequilibration parameters in data
            # this prevents amici from doing preequilibration again
            data.fixedParametersPreequilibration = []

            # set initial state from preequilibration
            self.amici_model.setInitialStates(
                self.preequilibration_edatas[edata_ix]['x0']
            )

            # set initial sensitivities from preequilibration
            if self.amici_solver.getSensitivityOrder() > \
                    amici.SensitivityOrder_none:
                self.amici_model.setInitialStateSensitivities(
                    self.preequilibration_edatas[edata_ix]['sx0'].flatten()
                )

        # return the original values
        return {
            'k': original_fixed_parameters_preequilibration,
            'x0': original_initial_states,
            'sx0': original_initial_state_sensitivities
        }

    def postprocess_preequilibration(self, edata, original_value_dict):
        """
        Reset the model and edata to the true values, i.e. undo
        the temporary changes done in preprocess_preequilibration.
        """

        # reset values in edata from values in original_value_dict, if
        # the corresponding entry in original_value_dict is not None.

        if original_value_dict['k']:
            edata.fixedParametersPreequilibration = original_value_dict['k']

        if original_value_dict['x0']:
            self.amici_model.setInitialStates(original_value_dict['x0'])

        if original_value_dict['sx0']:
            self.amici_model.setInitialStateSensitivities(
                original_value_dict['sx0']
            )

    def get_error_output(self, rdatas):
        if not self.amici_model.nt():
            nt = sum([data.nt() for data in self.edatas])
        else:
            nt = sum([data.nt() if data.nt() else self.amici_model.nt()
                      for data in self.edatas])
        n_res = nt * self.amici_model.nytrue

        return {
            FVAL: np.inf,
            GRAD: np.nan * np.ones(self.dim),
            HESS: np.nan * np.ones([self.dim, self.dim]),
            RES: np.nan * np.ones(n_res),
            SRES: np.nan * np.ones([n_res, self.dim]),
            RDATAS: rdatas
        }

    def set_par_sim_for_condition(self, condition_ix, x):
        """
        Set the simulation parameters from the optimization parameters
        for the given condition.

        Parameters
        ----------

        condition_ix: int
            Index of the current experimental condition.

        x: array_like
            Optimization parameters.
        """
        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        x_sim = map_par_opt_to_par_sim(mapping, self.x_ids, x)
        self.amici_model.setParameters(x_sim)

    def set_parameter_scale(self, condition_ix):
        scale_list = self.mapping_scale_opt_to_scale_sim[condition_ix]
        amici_scale_vector = amici.ParameterScalingVector()

        for val in scale_list:

            if val == 'lin':
                scale = amici.ParameterScaling_none
            elif val == 'log10':
                scale = amici.ParameterScaling_log10
            elif val == 'log':
                scale = amici.ParameterScaling_ln
            else:
                raise ValueError(
                    f"Parameter scaling not recognized: {val}")

            # append to scale vector
            amici_scale_vector.append(scale)

        self.amici_model.setParameterScale(amici_scale_vector)


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

    for par_sim_idx, par_opt_id in enumerate(mapping_par_opt_to_par_sim):
        if not isinstance(par_opt_id, str):
            # this was a numeric override for which we ignore the gradient
            continue

        par_opt_idx = par_opt_ids.index(par_opt_id)
        opt_grad[par_opt_idx] += coefficient * sim_grad[par_sim_idx]


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

    for par_sim_idx, par_opt_id in enumerate(mapping_par_opt_to_par_sim):
        if not isinstance(par_opt_id, str):
            # this was a numeric override for which we ignore the hessian
            continue

        par_opt_idx = par_opt_ids.index(par_opt_id)

        for par_sim_idx_2, par_opt_id_2 in enumerate(
                mapping_par_opt_to_par_sim):
            if not isinstance(par_opt_id_2, str):
                continue

            par_opt_idx_2 = par_opt_ids.index(par_opt_id_2)

            opt_hess[par_opt_idx, par_opt_idx_2] += \
                coefficient * sim_hess[par_sim_idx, par_sim_idx_2]
