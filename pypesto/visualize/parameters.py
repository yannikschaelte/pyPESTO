import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .clust_color import assign_clustered_colors


def parameters(result, ax=None, free_indices_only=True, lb=None, ub=None,
               size=None):
    """
    Plot parameter values.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'.

    ax: matplotlib.Axes, optional
        Axes object to use.

    free_indices_only: bool, optional
        If True, only free parameters are shown. If
        False, also the fixed parameters are shown.

    lb, ub: ndarray, optional
        If not None, override result.problem.lb, problem.problem.ub.
        Dimension either result.problem.dim or result.problem.dim_full.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    if lb is None:
        lb = result.problem.lb
    if ub is None:
        ub = result.problem.ub

    fvals = result.optimize_result.get_for_key('fval')
    xs = result.optimize_result.get_for_key('x')

    x_labels = result.problem.x_names

    if free_indices_only:
        for ix, x in enumerate(xs):
            xs[ix] = result.problem.get_reduced_vector(x)
        lb = result.problem.get_reduced_vector(lb)
        ub = result.problem.get_reduced_vector(ub)
        x_labels = [x_labels[int(i)] for i in result.problem.x_free_indices]
    else:
        lb = result.problem.get_full_vector(lb)
        ub = result.problem.get_full_vector(ub)

    return parameters_lowlevel(xs=xs, fvals=fvals, lb=lb, ub=ub,
                               x_labels=x_labels, ax=ax, size=size)


def parameters_lowlevel(xs, fvals, lb=None, ub=None, x_labels=None, ax=None,
                        size=None):
    """
    Plot parameters plot using list of parameters.

    Parameters
    ----------

    xs: nested list or array
        Including optimized parameters for each startpoint.
        Shape: (n_starts, dim).

    fvals: numeric list or array
        Function values. Needed to assign cluster colors.

    lb, ub: array_like, optional
        The lower and upper bounds.

    x_labels: array_like of str, optional
        Labels to be used for the parameters.

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see parameters

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    xs = np.array(xs)
    fvals = np.array(fvals)

    if size is None:
        # 0.5 inch height per parameter
        size = (18.5, xs.shape[1]/2)

    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # assign color
    colors = assign_clustered_colors(fvals)

    # parameter indices
    parameters_ind = range(1, xs.shape[1] + 1)

    # plot parameters
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j_x, x in reversed(list(enumerate(xs))):
        ax.plot(x, parameters_ind, color=colors[j_x], marker='o')

    plt.yticks(parameters_ind, x_labels)

    # draw bounds
    parameters_ind = np.array(parameters_ind).flatten()
    if lb is not None:
        ax.plot(lb.flatten(), parameters_ind, 'k--', marker='+')
    if ub is not None:
        ax.plot(ub.flatten(), parameters_ind, 'k--', marker='+')

    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Parameter index')
    ax.set_title('Estimated parameters')

    return ax
