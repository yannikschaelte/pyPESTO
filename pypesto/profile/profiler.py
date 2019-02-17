import numpy as np


class ProfilerResult(dict):
    """
    The result of a profiler run. The standardized return return value from
    pypesto.profile, which can either be initialized from an OptimizerResult
    or from an existing ProfilerResult (in order to extent the compputation).

    Can be used like a dict.

    Attributes
    ----------

    x_path: ndarray
        The path of the best found parameters along the profile
        (Dimension: n_par x n_profile_points)

    fval_path: ndarray
        The function values, fun(x), along the profile.

    ratio_path: ndarray
        The ratio of the posterior function along the profile.

    gradnorm_path: ndarray
        The gradient norm along the profile.

    exitflag_path: ndarray
        The exitflags of the optimizer along the profile.

    time_path: ndarray
        The computation time of the optimizer runs along the profile.

    time_total: ndarray
        The total computation time for the profile.

    n_fval: int
        Number of function evaluations.

    n_grad: int
        Number of gradient evaluations.

    n_hess: int
        Number of Hessian evaluations.

    message: str
        Textual comment on the profile result.

    Notes
    -----

    Any field not supported by the profiler or the profiling optimizer is
    filled with None. Some fields are filled by pypesto itself.
    """

    def __init__(self,
                 x_path,
                 fval_path,
                 ratio_path,
                 gradnorm_path=None,
                 exitflag_path=None,
                 time_path=None,
                 time_total=0.,
                 n_fval=0,
                 n_grad=0,
                 n_hess=0,
                 message=None):
        super().__init__()

        # initialize profile path
        x_shape = x_path.shape
        if len(x_shape) == 1:
            self.x_path = np.zeros((x_shape[0], 1))
            self.x_path[:, 0] = x_path[:]
        else:
            self.x_path = np.zeros((x_shape[0], x_shape[1]))
            self.x_path[:, :] = x_path[:, :]

        self.fval_path = np.array(fval_path)
        self.ratio_path = np.array(ratio_path)
        self.gradnorm_path = np.array(gradnorm_path) \
            if gradnorm_path is not None else None
        self.exitflag_path = np.array(exitflag_path)
        self.time_path = np.array(time_path)
        self.time_total = time_total
        self.n_fval = n_fval
        self.n_grad = n_grad
        self.n_hess = n_hess
        self.message = message

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def append_profile_point(self,
                             x,
                             fval,
                             ratio,
                             gradnorm=np.nan,
                             exitflag=np.nan,
                             time=np.nan,
                             n_fval=0,
                             n_grad=0,
                             n_hess=0):
        """
        This function appends a new OptimizerResult to an existing
        ProfilerResults
        """

        # short function to append to numpy vectors
        def append_to_vector(field_name, val):
            field_new = np.zeros(self[field_name].size + 1)
            field_new[0:-1] = self[field_name]
            field_new[-1] = val
            self[field_name] = field_new

        # write profile path
        x_new = np.zeros((self.x_path.shape[0], self.x_path.shape[1] + 1))
        x_new[:, 0:-1] = self.x_path
        x_new[:, -1] = x
        self.x_path = x_new

        # append to other paths
        append_to_vector("fval_path", fval)
        append_to_vector("ratio_path", ratio)
        append_to_vector("gradnorm_path", gradnorm)
        append_to_vector("exitflag_path", exitflag)
        append_to_vector("time_path", time)

        # increment the time and f_eval counters
        self.time_total += time
        self.n_fval += n_fval
        self.n_grad += n_grad
        self.n_hess += n_hess

    def flip_profile(self):
        """
        This function flips the profiling direction (left-right)
        Profiling direction needs to be changed once (if the profile is new)
        and twice, if we append to an existing profile
        """

        self.x_path = np.fliplr(self.x_path)
        self.fval_path = np.flip(self.fval_path)
        self.ratio_path = np.flip(self.ratio_path)
        self.gradnorm_path = np.flip(self.gradnorm_path)
        self.exitflag_path = np.flip(self.exitflag_path)
        self.time_path = np.flip(self.time_path)
