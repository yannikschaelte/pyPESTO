import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .clust_color import assign_clustered_colors


def waterfall(result, ax=None, size=(18.5, 10.5)):
    """
    Plot waterfall plot.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # extract cost function values from result
    fvals = result.optimize_result.get_for_key('fval')

    return waterfall_lowlevel(fvals, ax, size)


def waterfall_lowlevel(fvals, ax=None, size=(18.5, 10.5)):
    """
    Plot waterfall plot using list of function values.

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see waterfall

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # parse input
    fvals = np.array(fvals)

    n_fvals = len(fvals)
    start_ind = range(n_fvals)

    # assign colors
    # note: this has to happen before sorting
    # to get the same colors in different plots
    colors = assign_clustered_colors(fvals)

    # sort
    indices = sorted(range(n_fvals),
                     key=lambda j: fvals[j])

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(start_ind, fvals)
    for j in range(n_fvals):
        j_fval = indices[j]
        color = colors[j_fval]
        fval = fvals[j_fval]
        ax.plot(j, fval, color=color, marker='o')

    # labels
    ax.set_xlabel('Ordered optimizer run')
    ax.set_ylabel('Function value')
    ax.set_title('Waterfall plot')

    return ax
