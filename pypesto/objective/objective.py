import numpy as np
import copy
import pandas as pd
import logging

from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES
from .history import ObjectiveHistory
from .options import ObjectiveOptions
from .pre_post_process import PrePostProcessor, FixedParametersProcessor

logger = logging.getLogger(__name__)


class Objective:
    """
    The objective class is a simple wrapper around the objective function,
    giving a standardized way of calling. Apart from that, it manages several
    things including fixing of parameters and history.

    Parameters
    ----------

    fun: callable, optional
        The objective function to be minimized. If it only computes the
        objective function value, it should be of the form
            ``fun(x) -> float``
        where x is an 1-D array with shape (n,), and n is the parameter space
        dimension.

    grad: callable, bool, optional
        Method for computing the gradient vector. If it is a callable,
        it should be of the form
            ``grad(x) -> array_like, shape (n,).``
        If its value is True, then fun should return the gradient as a second
        output.

    hess: callable, optional
        Method for computing the Hessian matrix. If it is a callable,
        it should be of the form
            ``hess(x) -> array, shape (n,n).``
        If its value is True, then fun should return the gradient as a
        second, and the Hessian as a third output, and grad should be True as
        well.

    hessp: callable, optional
        Method for computing the Hessian vector product, i.e.
            ``hessp(x, v) -> array_like, shape (n,)``
        computes the product H*v of the Hessian of fun at x with v.

    res: {callable, bool}, optional
        Method for computing residuals, i.e.
            ``res(x) -> array_like, shape(m,).``

    sres: callable, optional
        Method for computing residual sensitivities. If its is a callable,
        it should be of the form
            ``sres(x) -> array, shape (m,n).``
        If its value is True, then res should return the residual
        sensitivities as a second output.

    fun_accept_sensi_orders: bool, optional
        Flag indicating whether fun takes sensi_orders as an argument.
        Default: False.

    res_accept_sensi_orders: bool, optional
        Flag indicating whether res takes sensi_orders as an argument.
        Default: False

    x_names: list of str
        Parameter names. None if no names provided, otherwise a list of str,
        length dim_full (as in the Problem class). Can be read by the
        problem.

    options: pypesto.ObjectiveOptions, optional
        Options as specified in pypesto.ObjectiveOptions.

    Attributes
    ----------

    history: pypesto.ObjectiveHistory
        For storing the call history. Initialized by the optimizer in
        reset_history().

    preprocess: callable
        Preprocess input values to __call__.

    postprocess: callable
        Postprocess output values from __call__.

    sensitivity_orders: tuple
        Temporary variable to store requested sensitivity orders

    Notes
    -----

    preprocess, postprocess are configured in update_from_problem()
    and can be reset using the reset() method.

    If fun_accept_sensi_orders resp. res_accept_sensi_orders is True,
    fun resp. res can also return dictionaries instead of tuples.
    In that case, they are expected to follow the naming conventions
    in ``constants.py``. This is of interest, because when __call__ is
    called with return_dict = True, the full dictionary is returned, which
    can contain e.g. also simulation data or debugging information.
    """

    def __init__(self,
                 fun=None, grad=None, hess=None, hessp=None,
                 res=None, sres=None,
                 fun_accept_sensi_orders=False,
                 res_accept_sensi_orders=False,
                 x_names=None,
                 options=None):
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.hessp = hessp
        self.res = res
        self.sres = sres
        self.fun_accept_sensi_orders = fun_accept_sensi_orders
        self.res_accept_sensi_orders = res_accept_sensi_orders

        if options is None:
            options = ObjectiveOptions()
        self.options = ObjectiveOptions.assert_instance(options)

        self.history = ObjectiveHistory(self.options)

        self.pre_post_processor = PrePostProcessor()

        self.x_names = x_names

    def __deepcopy__(self, memodict=None):
        other = Objective()
        for attr in self.__dict__:
            other.__dict__[attr] = copy.deepcopy(self.__dict__[attr])
        return other

    # The following has_ properties can be used to find out what values
    # the objective supports.

    @property
    def has_fun(self):
        return callable(self.fun)

    @property
    def has_grad(self):
        return callable(self.grad) or self.grad is True

    @property
    def has_hess(self):
        return callable(self.hess) or self.hess is True

    @property
    def has_hessp(self):
        # Not supported yet
        return False

    @property
    def has_res(self):
        return callable(self.res)

    @property
    def has_sres(self):
        return callable(self.sres) or self.sres is True

    def check_sensi_orders(self, sensi_orders, mode):
        """
        Check if the objective is able to compute the requested
        sensitivities. If not, throw an exception.
        """
        if (mode is MODE_FUN and
            (0 in sensi_orders and not self.has_fun
             or 1 in sensi_orders and not self.has_grad
             or 2 in sensi_orders and not self.has_hess)
            ) or (mode is MODE_RES and
                  (0 in sensi_orders and not self.has_res
                   or 1 in sensi_orders and not self.has_sres)
                  ):
            raise ValueError(
                f"Objective cannot be called with sensi_orders={sensi_orders}"
                f" and mode={mode}")

    def __call__(self,
                 x,
                 sensi_orders: tuple = (0, ),
                 mode=MODE_FUN,
                 return_dict=False):
        """
        Method to obtain arbitrary sensitivities. This is the central method
        which is always called, also by the get_* methods.

        There are different ways in which an optimizer calls the objective
        function, and in how the objective function provides information
        (e.g. derivatives via separate functions or along with the function
        values). The different calling modes increase efficiency in space
        and time and make the objective flexible.

        Parameters
        ----------

        x: array_like
            The parameters for which to evaluate the objective function.

        sensi_orders: tuple
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.

        mode: str
            Whether to compute function values or residuals.
        """

        # check input
        self.check_sensi_orders(sensi_orders, mode)

        # pre-process
        x = self.pre_post_processor.preprocess(x)

        # compute result
        result = self._call_unprocessed(x, sensi_orders, mode)

        # post-process
        result = self.pre_post_processor.postprocess(result)

        # update history
        self.history.update(x, sensi_orders, mode, result)

        # map to output format
        if not return_dict:
            result = Objective.output_to_tuple(sensi_orders, mode, **result)

        return result

    def _call_unprocessed(self, x, sensi_orders, mode):
        """
        Call objective function without pre- or post-processing and
        formatting.
        """
        if mode == MODE_FUN:
            result = self._call_mode_fun(x, sensi_orders)
        elif mode == MODE_RES:
            result = self._call_mode_res(x, sensi_orders)
        else:
            raise ValueError("This mode is not supported.")
        return result

    def _call_mode_fun(self, x, sensi_orders):
        """
        The method __call__ was called with mode MODE_FUN.
        """
        if self.fun_accept_sensi_orders:
            result = self.fun(x, sensi_orders)
            if not isinstance(result, dict):
                result = Objective.output_to_dict(
                    sensi_orders, MODE_FUN, result)
        elif sensi_orders == (0,):
            if self.grad is True:
                fval = self.fun(x)[0]
            else:
                fval = self.fun(x)
            result = {FVAL: fval}
        elif sensi_orders == (1,):
            if self.grad is True:
                grad = self.fun(x)[1]
            else:
                grad = self.grad(x)
            result = {GRAD: grad}
        elif sensi_orders == (2,):
            if self.hess is True:
                hess = self.fun(x)[2]
            else:
                hess = self.hess(x)
            result = {HESS: hess}
        elif sensi_orders == (0, 1):
            if self.grad is True:
                fval, grad = self.fun(x)[0:2]
            else:
                fval = self.fun(x)
                grad = self.grad(x)
            result = {FVAL: fval,
                      GRAD: grad}
        elif sensi_orders == (1, 2):
            if self.hess is True:
                grad, hess = self.fun(x)[1:3]
            else:
                hess = self.hess(x)
                if self.grad is True:
                    grad = self.fun(x)[1]
                else:
                    grad = self.grad(x)
            result = {GRAD: grad,
                      HESS: hess}
        elif sensi_orders == (0, 1, 2):
            if self.hess is True:
                fval, grad, hess = self.fun(x)[0:3]
            else:
                hess = self.hess(x)
                if self.grad is True:
                    fval, grad = self.fun(x)[0:2]
                else:
                    fval = self.fun(x)
                    grad = self.grad(x)
            result = {FVAL: fval,
                      GRAD: grad,
                      HESS: hess}
        else:
            raise ValueError("These sensitivity orders are not supported.")
        return result

    def _call_mode_res(self, x, sensi_orders):
        """
        The method __call__ was called with mode MODE_RES.
        """
        if self.res_accept_sensi_orders:
            result = self.res(x, sensi_orders)
            if not isinstance(result, dict):
                result = Objective.output_to_dict(
                    sensi_orders, MODE_RES, result)
        elif sensi_orders == (0,):
            if self.sres is True:
                res = self.res(x)[0]
            else:
                res = self.res(x)
            result = {RES: res}
        elif sensi_orders == (1,):
            if self.sres is True:
                sres = self.res(x)[1]
            else:
                sres = self.sres(x)
            result = {SRES: sres}
        elif sensi_orders == (0, 1):
            if self.sres is True:
                res, sres = self.res(x)
            else:
                res = self.res(x)
                sres = self.sres(x)
            result = {RES: res,
                      SRES: sres}
        else:
            raise ValueError("These sensitivity orders are not supported.")
        return result

    @staticmethod
    def output_to_dict(sensi_orders, mode, output_tuple):
        """
        Convert output tuple to dict.
        """
        output_dict = {}
        index = 0
        if not isinstance(output_tuple, tuple):
            output_tuple = (output_tuple,)
        if mode == MODE_FUN:
            if 0 in sensi_orders:
                output_dict[FVAL] = output_tuple[index]
                index += 1
            if 1 in sensi_orders:
                output_dict[GRAD] = output_tuple[index]
                index += 1
            if 2 in sensi_orders:
                output_dict[HESS] = output_tuple[index]
        elif mode == MODE_RES:
            if 0 in sensi_orders:
                output_dict[RES] = output_tuple[index]
                index += 1
            if 1 in sensi_orders:
                output_dict[SRES] = output_tuple[index]
        return output_dict

    @staticmethod
    def output_to_tuple(sensi_orders, mode, **kwargs):
        """
        Return values as requested by the caller, since usually only a subset
        is demanded. One output is returned as-is, more than one output are
        returned as a tuple in order (fval, grad, hess).
        """
        output = ()
        if mode == MODE_FUN:
            if 0 in sensi_orders:
                output += (kwargs[FVAL],)
            if 1 in sensi_orders:
                output += (kwargs[GRAD],)
            if 2 in sensi_orders:
                output += (kwargs[HESS],)
        elif mode == MODE_RES:
            if 0 in sensi_orders:
                output += (kwargs[RES],)
            if 1 in sensi_orders:
                output += (kwargs[SRES],)
        if len(output) == 1:
            output = output[0]
        return output

    # The following are convenience functions for getting specific outputs.

    def get_fval(self, x):
        """
        Get the function value at x.
        """
        fval = self(x, (0,), MODE_FUN)
        return fval

    def get_grad(self, x):
        """
        Get the gradient at x.
        """
        grad = self(x, (1,), MODE_FUN)
        return grad

    def get_hess(self, x):
        """
        Get the Hessian at x.
        """
        hess = self(x, (2,), MODE_FUN)
        return hess

    def get_res(self, x):
        """
        Get the residuals at x.
        """
        res = self(x, (0,), MODE_RES)
        return res

    def get_sres(self, x):
        """
        Get the residual sensitivities at x.
        """
        sres = self(x, (1,), MODE_RES)
        return sres

    # The following are functions that are called by other parts in
    # pypesto to modify the objective state, e.g. set its history, or
    # make it aware of fixed parameters.

    def reset_history(self, index=None):
        """
        Reset the objective history and specify temporary saving options.

        Parameters
        ----------

        index: As in ObjectiveHistory.index.
        """
        self.history.reset(index=index)

    def finalize_history(self):
        """
        Finalize the history object.
        """
        self.history.finalize()

    def reset(self):
        """
        Completely reset the objective, i.e. undo the modifications in
        update_from_problem().
        """
        self.history = ObjectiveHistory(self.options)

        def preprocess(x):
            return np.array(x)

        def postprocess(result):
            return result

        self.preprocess = preprocess
        self.postprocess = postprocess

    def update_from_problem(self,
                            dim_full,
                            x_free_indices,
                            x_fixed_indices,
                            x_fixed_vals):
        """
        Handle fixed parameters. Later, the objective will be given parameter
        vectors x of dimension dim, which have to be filled up with fixed
        parameter values to form a vector of dimension dim_full >= dim.
        This vector is then used to compute function value and derivaties.
        The derivatives must later be reduced again to dimension dim.

        This is so as to make the fixing of parameters transparent to the
        caller.

        The methods preprocess, postprocess are overwritten for the above
        functionality, respectively.

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

        pre_post_processor = FixedParametersProcessor(
            dim_full=dim_full,
            x_free_indices=x_free_indices,
            x_fixed_indices=x_fixed_indices,
            x_fixed_vals=x_fixed_vals)

        self.pre_post_processor = pre_post_processor

    def check_grad(self,
                   x,
                   x_indices=None,
                   eps=1e-5,
                   verbosity=1,
                   mode=MODE_FUN) -> pd.DataFrame:
        """
        Compare gradient evaluation: Firstly approximate via finite
        differences, and secondly use the objective gradient.

        Parameters
        ----------

        x: array_like
            The parameters for which to evaluate the gradient.

        x_indices: array_like, optional
            List of index values for which to compute gradients. Default: all.

        eps: float, optional
            Finite differences step size. Default: 1e-5.

        verbosity: int
            Level of verbosity for function output
                0: no output
                1: summary for all parameters
                2: summary for individual parameters
            Default: 1.

        mode: str
            Residual (MODE_RES) or objective function value
            (MODE_FUN, default) computation mode.

        Returns
        ----------

        result: pd.DataFrame
            gradient, finite difference approximations and error estimates.

        """

        if x_indices is None:
            x_indices = list(range(len(x)))

        tmp_trace_record = self.options.trace_record
        self.options.trace_record = False

        # function value and objective gradient
        fval, grad = self(x, (0, 1), mode)

        grad_list = []
        fd_f_list = []
        fd_b_list = []
        fd_c_list = []
        fd_err_list = []
        abs_err_list = []
        rel_err_list = []

        # loop over indices
        for ix in x_indices:
            # forward (plus) point
            x_p = copy.deepcopy(x)
            x_p[ix] += eps
            fval_p = self(x_p, (0,), mode)

            # backward (minus) point
            x_m = copy.deepcopy(x)
            x_m[ix] -= eps
            fval_m = self(x_m, (0,), mode)

            # finite differences
            fd_f_ix = (fval_p - fval) / eps
            fd_b_ix = (fval - fval_m) / eps
            fd_c_ix = (fval_p - fval_m) / (2 * eps)

            # gradient in direction ix
            grad_ix = grad[ix] if grad.ndim == 1 else grad[:, ix]

            # errors
            fd_err_ix = abs(fd_f_ix - fd_b_ix)
            abs_err_ix = abs(grad_ix - fd_c_ix)
            rel_err_ix = abs(abs_err_ix / (fd_c_ix + eps))

            # log for dimension ix
            if verbosity > 1:
                logger.info(
                    f'index:    {ix}\n'
                    f'grad:     {grad_ix}\n'
                    f'fd_f:     {fd_f_ix}\n'
                    f'fd_b:     {fd_b_ix}\n'
                    f'fd_c:     {fd_c_ix}\n'
                    f'fd_err:   {fd_err_ix}\n'
                    f'abs_err:  {abs_err_ix}\n'
                    f'rel_err:  {rel_err_ix}\n'
                )

            # append to lists
            grad_list.append(grad_ix)
            fd_f_list.append(fd_f_ix)
            fd_b_list.append(fd_b_ix)
            fd_c_list.append(fd_c_ix)
            fd_err_list.append(np.mean(fd_err_ix))
            abs_err_list.append(np.mean(abs_err_ix))
            rel_err_list.append(np.mean(rel_err_ix))

        # create dataframe
        result = pd.DataFrame(data={
            'grad': grad_list,
            'fd_f': fd_f_list,
            'fd_b': fd_b_list,
            'fd_c': fd_c_list,
            'fd_err': fd_err_list,
            'abs_err': abs_err_list,
            'rel_err': rel_err_list,
        })

        # log full result
        if verbosity > 0:
            logger.info(result)

        self.options.trace_record = tmp_trace_record

        return result

    def get_config(self) -> dict:
        # try to get the function name
        conf = {"name": self.__class__.__name__}
        try:
            conf["fun"] = self.fun.__name__
        except AttributeError:
            pass
        return conf

    def __repr__(self) -> str:
        return repr(self.get_config())
