import logging
from ..result import Result, ProblemInfo
from ..startpoint import assign_startpoints, uniform
from .optimizer import OptimizerResult, recover_result, ScipyOptimizer
from ..engine import OptimizerTask, SingleCoreEngine


logger = logging.getLogger(__name__)


class OptimizeOptions(dict):
    """
    Options for the multistart optimization.

    Parameters
    ----------

    startpoint_resample: bool, optional
        Flag indicating whether initial points are supposed to be resampled if
        function evaluation fails at the initial point

    allow_failed_starts: bool, optional
        Flag indicating whether we tolerate that exceptions are thrown during
        the minimization process.
    """

    def __init__(self,
                 startpoint_resample=False,
                 allow_failed_starts=False):
        super().__init__()

        self.startpoint_resample = startpoint_resample
        self.allow_failed_starts = allow_failed_starts

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def assert_instance(maybe_options):
        """
        Returns a valid options object.

        Parameters
        ----------

        maybe_options: OptimizeOptions or dict
        """
        if isinstance(maybe_options, OptimizeOptions):
            return maybe_options
        options = OptimizeOptions(**maybe_options)
        return options


def minimize(
        problem,
        optimizer=None,
        n_starts=100,
        startpoint_method=None,
        result=None,
        engine=None,
        options=None) -> Result:
    """
    This is the main function to call to do multistart optimization.

    Parameters
    ----------

    problem: pypesto.Problem
        The problem to be solved.

    optimizer: pypesto.Optimizer
        The optimizer to be used n_starts times.

    n_starts: int
        Number of starts of the optimizer.

    startpoint_method: {callable, False}, optional
        Method for how to choose start points. False means the optimizer does
        not require start points

    result: pypesto.Result
        A result object to append the optimization results to. For example,
        one might append more runs to a previous optimization. If None,
        a new object is created.

    options: pypesto.OptimizeOptions, optional
        Various options applied to the multistart optimization.
    """

    # optimizer
    if optimizer is None:
        optimizer = ScipyOptimizer()

    # startpoint method
    if startpoint_method is None:
        startpoint_method = uniform

    # check options
    if options is None:
        options = OptimizeOptions()
    options = OptimizeOptions.assert_instance(options)

    # assign startpoints
    startpoints = assign_startpoints(n_starts, startpoint_method,
                                     problem, options)

    # prepare result
    if result is None:
        result = Result(problem_info=ProblemInfo.from_problem(problem))

    # engine
    if engine is None:
        engine = SingleCoreEngine()

    # define tasks
    tasks = []
    for j_start in range(0, n_starts):
        startpoint = startpoints[j_start, :]
        task = OptimizerTask(optimizer, problem, startpoint, j_start,
                             options, handle_exception)
        tasks.append(task)

    # do multistart optimization
    ret = engine.execute(tasks)

    # aggregate results
    for optimizer_result in ret:
        result.optimize_result.append(optimizer_result)

    # sort by best fval
    result.optimize_result.sort()

    return result


def handle_exception(
        objective, startpoint, j_start, err) -> OptimizerResult:
    """
    Handle exception by creating a dummy pypesto.OptimizerResult.
    """
    logger.error(('start ' + str(j_start) + ' failed: {0}').format(err))
    optimizer_result = recover_result(objective, startpoint, err)
    return optimizer_result
