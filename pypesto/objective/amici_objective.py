import numpy as np
import copy
import logging
import numbers
from .objective import Objective
from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS
from .amici_calculator import simple_amici_calculate
from .amici_util import (
    map_par_opt_to_par_sim,
    create_plist_from_par_opt_to_par_sim,
    create_scale_mapping_from_model,
)

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
                 guess_steadystate=True,
                 n_threads=1,
                 calculator=None,
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

        guess_steadystate: bool, optional (default = True)
            Whether to guess steadystates based on previous steadystates and
            respective derivatives. This option may lead to unexpected
            results for models with conservation laws and should accordingly
            be deactivated for those models.

        n_threads: int, optional (default = 1)
            Number of threads that are used for parallelization over
            experimental conditions. If amici was not installed with openMP
            support this option will have no effect.

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

        # set the experimental data container
        self.edatas = edatas

        # set the maximum sensitivity order
        self.max_sensi_order = max_sensi_order

        self.guess_steadystate = guess_steadystate

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

        # preallocate guesses, construct a dict for every edata for which we
        # need to do preequilibration
        if self.guess_steadystate:
            if self.amici_model.ncl() > 0:
                raise ValueError('Steadystate prediciton is not supported for'
                                 'models with conservation laws!')

            if self.amici_model.getSteadyStateSensitivityMode() == \
                    amici.SteadyStateSensitivityMode_simulationFSA:
                raise ValueError('Steadystate guesses cannot be enabled when'
                                 ' `simulationFSA` as '
                                 'SteadyStateSensitivityMode!')
            self.steadystate_guesses = {
                'fval': np.inf,
                'data': {
                    iexp: dict()
                    for iexp, edata in enumerate(self.edatas)
                    if len(edata.fixedParametersPreequilibration) or
                    self.amici_solver.getNewtonPreequilibration()
                }
            }

        # optimization parameter names
        if x_names is None:
            # use ids as names
            x_names = x_ids
        self.x_names = x_names

        self.n_threads = n_threads

        if calculator is None:
            calculator = simple_amici_calculate
        self.calculator = calculator

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
        other = AmiciObjective(model, solver, edatas,
                               guess_steadystate=self.guess_steadystate)
        for attr in self.__dict__:
            if attr not in ['amici_solver', 'amici_model', 'edatas']:
                other.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return other

    def reset(self):
        """
        Resets the objective, including steadystate guesses
        """
        super(AmiciObjective, self).reset()
        self.reset_steadystate_guesses()

    def _call_amici(
            self,
            x,
            sensi_orders,
            mode
    ):
        return self.calculator(self, x, sensi_orders, mode)

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
            RES:  np.nan * np.ones(n_res),
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
        self.edatas[condition_ix].parameters = x_sim

    def set_plist_for_condition(self, condition_ix):
        """
        Set the plist according to the optimization parameters
        for the given condition.

        Parameters
        ----------

        condition_ix: int
            Index of the current experimental condition.

        x: array_like
            Optimization parameters.
        """
        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        plist = create_plist_from_par_opt_to_par_sim(mapping)
        self.edatas[condition_ix].plist = plist

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

        self.edatas[condition_ix].pscale = amici_scale_vector

    def apply_steadystate_guess(self, condition_ix, x):
        """
        Use the stored steadystate as well as the respective  sensitivity (
        if available) and parameter value to approximate the steadystate at
        the current parameters using a zeroth or first order taylor
        approximation:
        x_ss(x') = x_ss(x) [+ dx_ss/dx(x)*(x'-x)]
        """
        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        x_sim = map_par_opt_to_par_sim(mapping, self.x_ids, x)
        x_ss_guess = []  # resets initial state by default
        if condition_ix in self.steadystate_guesses['data']:
            guess_data = self.steadystate_guesses['data'][condition_ix]
            if guess_data['x_ss'] is not None:
                x_ss_guess = guess_data['x_ss']
            if guess_data['sx_ss'] is not None:
                linear_update = guess_data['sx_ss'].transpose().dot(
                    (x_sim - guess_data['x'])
                )
                # limit linear updates to max 20 % elementwise change
                if (x_ss_guess/linear_update).max() < 0.2:
                    x_ss_guess += linear_update

        self.edatas[condition_ix].x0 = tuple(x_ss_guess)

    def store_steadystate_guess(self, condition_ix, x, rdata):
        """
        Store condition parameter, steadystate and steadystate sensitivity in
        steadystate_guesses if steadystate guesses are enabled for this
        condition
        """

        if condition_ix not in self.steadystate_guesses['data']:
            return

        preeq_guesses = self.steadystate_guesses['data'][condition_ix]

        # update parameter

        mapping = self.mapping_par_opt_to_par_sim[condition_ix]
        x_sim = map_par_opt_to_par_sim(mapping, self.x_ids, x)
        preeq_guesses['x'] = x_sim

        # update steadystates
        preeq_guesses['x_ss'] = rdata['x_ss']
        preeq_guesses['sx_ss'] = rdata['sx_ss']

    def reset_steadystate_guesses(self):
        """
        Resets all steadystate guess data
        """
        if not self.guess_steadystate:
            return

        self.steadystate_guesses['fval'] = np.inf
        for condition in self.steadystate_guesses['data']:
            self.steadystate_guesses['data'][condition] = dict()