import numpy as np
import pandas as pd
import time
import copy
import os
from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES
from .util import res_to_chi2, sres_to_schi2
from .options import ObjectiveOptions


class ObjectiveHistory:
    """
    Objective call history. Also handles saving of intermediate results.

    Parameteters
    ------------

    options: ObjectiveOptions, optional
        Values needed for creating a history are extracted.

    Attributes
    ----------

    n_fval, n_grad, n_hess, n_res, n_sres: int
        Counters of function values, gradients and hessians,
        residuals and residual sensitivities.

    trace: list
        List containing history of function values and parameters if
        options.tr_record is True.

    start_time: float
        Reference start time.

    fval0, fval_min: float
        Initial and best function value found.

    x0, x_min: np.ndarray
        Initial and best parameters found.

    index: str
        Id identifying the history object when called in a multistart
        setting.
    """

    def __init__(self, options=None):

        if options is None:
            options = ObjectiveOptions()
        self.options = options

        self.n_fval = None
        self.n_grad = None
        self.n_hess = None
        self.n_res = None
        self.n_sres = None

        self.trace = None
        self.start_time = None

        self.fval_min = None
        self.x_min = None
        self.fval0 = None
        self.x0 = None

        self.index = None

        self.reset()

    def reset(self, index=None):
        """
        Reset all counters, the trace, and start the timer, and create
        directory for trace file.
        """
        self.n_fval = 0
        self.n_grad = 0
        self.n_hess = 0
        self.n_res = 0
        self.n_sres = 0

        self.trace = None
        self.start_time = time.time()

        self.fval0 = None
        self.x0 = None
        self.fval_min = np.inf
        self.x_min = None

        self.index = index

        # create trace file dirs
        if self.options.trace_file is not None:
            dirname = os.path.dirname(self.options.trace_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    def update(self, x, sensi_orders, mode, result):
        """
        Update the history.

        Parameters
        ----------
        x: np.ndarray
            The current parameter.
        sensi_orders: tuple
            The sensitivity orders.
        mode: str
            As in constants.MODE_.
        result: dict
            The result for x.
        """
        self._update_counts(sensi_orders, mode)
        self._update_trace(x, sensi_orders, mode, result)
        self._update_vals(x, sensi_orders, mode, result)

    def finalize(self):
        """
        Save the trace to file if options.trace_save is True.
        """
        self._save_trace(finalize=True)

    def _update_counts(self, sensi_orders, mode):
        """
        Update the counters.
        """
        if mode == MODE_FUN:
            if 0 in sensi_orders:
                self.n_fval += 1
            if 1 in sensi_orders:
                self.n_grad += 1
            if 2 in sensi_orders:
                self.n_hess += 1
        elif mode == MODE_RES:
            if 0 in sensi_orders:
                self.n_res += 1
            if 1 in sensi_orders:
                self.n_sres += 1

    def _update_trace(self, x, sensi_orders, mode, result):
        """
        Update and possibly store the trace.
        """

        if not self.options.trace_record:
            return

        # init trace
        if self.trace is None:
            columns = ['time',
                       'n_fval', 'n_grad', 'n_hess', 'n_res', 'n_sres',
                       'fval', 'grad', 'hess', 'res', 'sres', 'chi2', 'schi2',
                       'x']
            self.trace = pd.DataFrame(columns=columns)

        # extract function values
        if mode == MODE_FUN:
            fval = None if 0 not in sensi_orders \
                else result.get(FVAL, None)
            grad = None if not self.options.trace_record_grad \
                or 1 not in sensi_orders \
                else result.get(GRAD, None)
            hess = None if not self.options.trace_record_hess \
                or 2 not in sensi_orders \
                else result.get(HESS, None)
            res = None
            sres = None
            chi2 = None
            schi2 = None
        else:  # mode == MODE_RES
            fval = None
            grad = None
            hess = None
            res_result = result.get(RES, None)
            sres_result = result.get(SRES, None)
            res = None if not self.options.trace_record_res \
                or 0 not in sensi_orders \
                else res_result
            sres = None if not self.options.trace_record_sres \
                or 1 not in sensi_orders \
                else sres_result
            chi2 = None if not self.options.trace_record_chi2 \
                or 0 not in sensi_orders \
                else res_to_chi2(res_result)
            schi2 = None if not self.options.trace_record_schi2 \
                or 1 not in sensi_orders \
                else sres_to_schi2(res_result, sres_result)

        # check whether to append to trace
        if not self.options.trace_all and fval >= self.fval_min:
            return

        used_time = time.time() - self.start_time

        # create table row
        values = [
            used_time,
            self.n_fval, self.n_grad, self.n_hess, self.n_res, self.n_sres,
            fval, grad, hess, res, sres, chi2, schi2,
            x
        ]

        # append to trace
        self.trace.loc[len(self.trace)] = values

        # save trace to file
        self._save_trace()

    def _save_trace(self, finalize=False):
        """
        Save to file via pd.DataFrame.to_csv() if options.trace_file is
        not None and other conditions apply.
        Format might be revised when storage is implemented.
        """
        if self.options.trace_file is None:
            return

        if finalize \
           or (len(self.trace) > 0 and len(self.trace) %
               self.options.trace_save_iter == 0):
            filename = self.options.trace_file
            if self.index is not None:
                filename = filename.replace("{index}", str(self.index))
            # save
            trace_copy = copy.deepcopy(self.trace)
            for field in ['grad', 'hess', 'res', 'sres', 'schi2', 'x']:
                trace_copy[field] = trace_copy[field].apply(
                    ndarray2string_full
                )
            trace_copy.to_csv(filename)

    def _update_vals(self, x, sensi_orders, mode, result):
        """
        Update initial and best function values. Must be called after
        update_trace().
        """

        # update initial point
        if self.fval0 is None and 0 in sensi_orders:
            if mode == MODE_FUN:
                self.fval0 = result[FVAL]
                self.x0 = x
            else:  # mode == MODE_RES:
                self.fval0 = res_to_chi2(result[RES])
                self.x0 = x

        # update best point
        fval = np.inf
        if 0 in sensi_orders:
            if mode == MODE_FUN:
                fval = result[FVAL]
            else:  # mode == MODE_RES:
                fval = res_to_chi2(result[RES])
        if fval < self.fval_min:
            self.fval_min = fval
            self.x_min = x

def ndarray2string_full(x):
    if x is None:
        return None
    return np.array2string(x, threshold=len(x), precision=16,
                           max_line_width=np.inf)
