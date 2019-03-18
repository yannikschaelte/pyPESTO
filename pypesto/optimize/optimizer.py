import numpy as np
import scipy.optimize
import re
import abc
import time
import logging
from ..objective import res_to_chi2

try:
    import dlib
except ImportError:
    dlib = None


logger = logging.getLogger(__name__)


class OptimizerResult(dict):
    """
    The result of an optimizer run. Used as a standardized return value to
    map from the individual result objects returned by the employed
    optimizers to the format understood by pypesto.

    Can be used like a dict.

    Attributes
    ----------

    x: ndarray
        The best found parameters.

    fval: float
        The best found function value, fun(x).

    grad, hess: ndarray
        The gradient and Hessian at x.

    n_fval: int
        Number of function evaluations.

    n_grad: int
        Number of gradient evaluations.

    n_hess: int
        Number of Hessian evaluations.

    exitflag: int
        The exitflag of the optimizer.

    message: str
        Textual comment on the optimization result.

    Notes
    -----

    Any field not supported by the optimizer is filled with None. Some
    fields are filled by pypesto itself.
    """

    def __init__(self,
                 x=None,
                 fval=None,
                 grad=None,
                 hess=None,
                 n_fval=None,
                 n_grad=None,
                 n_hess=None,
                 n_res=None,
                 n_sres=None,
                 x0=None,
                 fval0=None,
                 trace=None,
                 exitflag=None,
                 time=None,
                 message=None):
        super().__init__()
        self.x = np.array(x)
        self.fval = fval
        self.grad = np.array(grad) if grad is not None else None
        self.hess = np.array(hess) if hess is not None else None
        self.n_fval = n_fval
        self.n_grad = n_grad
        self.n_hess = n_hess
        self.n_res = n_res
        self.n_sres = n_sres
        self.x0 = np.array(x0) if x0 is not None else None
        self.fval0 = fval0
        self.trace = trace
        self.exitflag = exitflag
        self.time = time
        self.message = message

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def objective_decorator(minimize):
    """
    Default decorator for the minimize() method to initialize and extract
    information stored in the objective.
    """

    def wrapped_minimize(self, problem, x0, index):
        problem.objective.reset_history(index=index)
        if hasattr(problem.objective, 'reset_steadystate_guesses'):
            problem.objective.reset_steadystate_guesses()
        result = minimize(self, problem, x0, index)
        problem.objective.finalize_history()
        result = fill_result_from_objective_history(
            result, problem.objective.history)
        logger.info("Final objective value: ", result.fval)
        return result
    return wrapped_minimize


def time_decorator(minimize):
    """
    Default decorator for the minimize() method to take time.
    Currently, the method time.time() is used, which measures
    the wall-clock time.
    """

    def wrapped_minimize(self, problem, x0, index):
        start_time = time.time()
        result = minimize(self, problem, x0, index)
        used_time = time.time() - start_time
        result.time = used_time
        return result
    return wrapped_minimize


def fix_decorator(minimize):
    """
    Default decorator for the minimize() method to include also fixed
    parameters in the result arrays (nans will be inserted in the
    derivatives).
    """

    def wrapped_minimize(self, problem, x0, index):
        result = minimize(self, problem, x0, index)
        result.x = problem.get_full_vector(result.x, problem.x_fixed_vals)
        result.grad = problem.get_full_vector(result.grad)
        result.hess = problem.get_full_matrix(result.hess)
        result.x0 = problem.get_full_vector(result.x0, problem.x_fixed_vals)
        return result
    return wrapped_minimize


def fill_result_from_objective_history(result, history):
    """
    Overwrite function values in the result object with the values recorded in
    the history.
    """

    # counters
    result.n_fval = history.n_fval
    result.n_grad = history.n_grad
    result.n_hess = history.n_hess
    result.n_res = history.n_res
    result.n_sres = history.n_sres

    # initial values
    result.x0 = history.x0
    result.fval0 = history.fval0

    # best found values
    result.x = history.x_min
    result.fval = history.fval_min

    # trace
    result.trace = history.trace

    return result


def recover_result(objective, startpoint, err):
    """
    Upon an error, recover from the objective history whatever available,
    and indicate in exitflag and message that an error occurred.
    """
    result = OptimizerResult(
        x0=startpoint,
        exitflag=-1,
        message='{0}'.format(err),
    )
    fill_result_from_objective_history(result, objective.history)

    return result


class Optimizer(abc.ABC):
    """
    This is the optimizer base class, not functional on its own.

    An optimizer takes a problem, and possibly a start point, and then
    performs an optimization. It returns an OptimizerResult.
    """

    def __init__(self):
        """
        Default constructor.
        """

    @abc.abstractmethod
    def minimize(self, problem, x0, index):
        """"
        Perform optimization.
        """

    @abc.abstractmethod
    def is_least_squares(self):
        return False

    @staticmethod
    def get_default_options():
        """
        Create default options specific for the optimizer.
        """
        return None


class ScipyOptimizer(Optimizer):
    """
    Use the SciPy optimizers.
    """

    def __init__(self, method='L-BFGS-B', tol=1e-9, options=None):
        super().__init__()

        self.method = method

        self.tol = tol

        self.options = options
        if self.options is None:
            self.options = ScipyOptimizer.get_default_options()

    @fix_decorator
    @time_decorator
    @objective_decorator
    def minimize(self, problem, x0, index):
        lb = problem.lb
        ub = problem.ub
        objective = problem.objective

        if self.is_least_squares():
            # is a residual based least squares method

            if not objective.has_res:
                raise Exception(
                    "For least squares optimization, the objective "
                    "must be able to compute residuals.")

            ls_method = self.method[3:]
            bounds = (lb, ub)

            fun = objective.get_res
            jac = objective.get_sres if objective.has_sres else '2-point'
            # TODO: pass jac computing methods in options

            # optimize
            res = scipy.optimize.least_squares(
                fun=fun,
                x0=x0,
                method=ls_method,
                jac=jac,
                bounds=bounds,
                ftol=self.tol,
                tr_solver='exact',
                loss='linear',
                verbose=2 if 'disp' in
                self.options.keys() and self.options['disp']
                else 0,
            )

        else:
            # is an fval based optimization method

            if not objective.has_fun:
                raise Exception("For this optimizer, the objective must "
                                "be able to compute function values")

            bounds = scipy.optimize.Bounds(lb, ub)

            # fun_may_return_tuple = self.method.lower() in \
            #    ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp',
            #        'dogleg', 'trust-ncg']
            # TODO: is it more efficient to have tuple as output of fun?
            method_supports_grad = self.method.lower() in \
                ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp',
                 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact',
                 'trust-constr']
            method_supports_hess = self.method.lower() in \
                ['newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov',
                 'trust-exact', 'trust-constr']
            method_supports_hessp = self.method.lower() in \
                ['newton-cg', 'trust-ncg', 'trust-krylov', 'trust-constr']

            fun = objective.get_fval
            jac = objective.get_grad \
                if objective.has_grad and method_supports_grad \
                else None
            hess = objective.get_hess \
                if objective.has_hess and method_supports_hess \
                else None
            hessp = objective.get_hessp \
                if objective.has_hessp and method_supports_hessp \
                else None
            # minimize will ignore hessp otherwise
            if hessp is not None:
                hess = None

            # optimize
            res = scipy.optimize.minimize(
                fun=fun,
                x0=x0,
                method=self.method,
                jac=jac,
                hess=hess,
                hessp=hessp,
                bounds=bounds,
                tol=self.tol,
                options=self.options,
            )

        # some fields are not filled by all optimizers, then fill in None
        grad = getattr(res, 'grad', None) if self.is_least_squares() \
            else getattr(res, 'jac', None)
        fval = res_to_chi2(res.fun) if self.is_least_squares() \
            else res.fun

        # fill in everything known, although some parts will be overwritten
        optimizer_result = OptimizerResult(
            x=res.x,
            fval=fval,
            grad=grad,
            hess=getattr(res, 'hess', None),
            n_fval=getattr(res, 'nfev', 0),
            n_grad=getattr(res, 'njev', 0),
            n_hess=getattr(res, 'nhev', 0),
            x0=x0,
            fval0=None,
            exitflag=res.status,
            message=res.message
        )

        return optimizer_result

    def is_least_squares(self):
        return re.match(r'(?i)^(ls_)', self.method)

    @staticmethod
    def get_default_options():
        options = {'maxiter': 1000, 'disp': False}
        return options


class DlibOptimizer(Optimizer):
    """
    Use the Dlib toolbox for optimization.
    """

    def __init__(self, method, options=None):
        super().__init__()

        self.method = method

        self.options = options
        if self.options is None:
            self.options = DlibOptimizer.get_default_options()

    @fix_decorator
    @time_decorator
    @objective_decorator
    def minimize(self, problem, x0, index):

        lb = problem.lb
        ub = problem.ub
        objective = problem.objective

        if dlib is None:
            raise ImportError(
                "This optimizer requires an installation of dlib."
            )

        if not objective.has_fun:
            raise Exception("For this optimizer, the objective must "
                            "be able to return function values.")

        # dlib requires variable length arguments
        def get_fval_vararg(*x):
            return objective.get_fval(x)

        dlib.find_min_global(
            get_fval_vararg,
            list(lb),
            list(ub),
            int(self.options['maxiter']),
            0.002,
        )

        optimizer_result = OptimizerResult(
            x0=x0
        )

        return optimizer_result

    def is_least_squares(self):
        return False

    @staticmethod
    def get_default_options():
        return {}
