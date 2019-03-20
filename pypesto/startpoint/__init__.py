"""
Startpoint
==========

Method for selecting points that can be used as start points
for multistart optimization. All methods have the form

    method(**kwargs) -> startpoints

where the kwargs can/should include the following parameters, which are
passed by pypesto:

n_starts: int
    Number of points to generate.

lb, ub: ndarray
    Lower and upper bound, may for most methods not contain nan or inf
    values.

x_guesses: ndarray, shape=(g, dim), optional
    Parameter guesses by the user, where g denotes the number of guesses.
    Note that these are only possibly taken as reference points to generate
    new start points (e.g. to maximize some distance) depending on the
    method, but regardless of g, there are always n_starts points generated
    and returned.

objective: pypesto.Objective, optional
    The objective can be used to evaluate the goodness of start points.

max_n_fval: int, optional
    The maximum number of evaluations of the objective function allowed.

"""

from .uniform import uniform
from .latin_hypercube import latin_hypercube
from .metaheuristic import metaheuristic
from .util import assign_startpoints

__all__ = ['uniform',
           'latin_hypercube',
           'metaheuristic',
           'assign_startpoints']
