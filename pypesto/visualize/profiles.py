import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def profiles(result, ax=None, profile_indices=None, size=(18.5, 6.5)):
    """
    Plot classical 1D profile plot (using the posterior, e.g. Gaussian like
    profile)

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

    ax: matplotlib.Axes, optional
        Axes object to use.

    profile_indices: list of integer values
        list of integer values specifying which profiles should be plotted

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    if profile_indices is None:
        profile_indices = \
            [i for i in range(0, len(result.profile_result.list[0]))]

    # extract ratio values values from result
    fvals = []
    for i_par in range(0, len(result.profile_result.list[0])):
        if i_par in profile_indices:
            tmp = np.array(
                [result.profile_result.list[0][i_par].x_path[i_par, :],
                 result.profile_result.list[0][i_par].ratio_path[:]])
        else:
            tmp = None
        fvals.append(tmp)

    return profiles_lowlevel(fvals, ax)


def profiles_lowlevel(fvals, ax=None, size=(18.5, 6.5)):
    """
    Lowlevel routine for profile plotting, working with a list of arrays
    only, opening different axes objects in case

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

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

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    if isinstance(fvals, list):
        n_fvals = 0
        for fval in enumerate(fvals):
            if fval is not None:
                n_fvals += 1
    else:
        n_fvals = 1
        fvals = [fvals]

    columns = np.ceil(np.sqrt(n_fvals))
    if n_fvals > columns * (columns - 1):
        rows = columns
    else:
        rows = columns - 1

    counter = 1
    for i_plot, fval in enumerate(fvals):

        if fval is not None:
            ax = plt.subplot(rows, columns, counter)
            ax = profile_lowlevel(fval, ax)

            # labels
            ax.set_xlabel(f'Parameter {i_plot} value')
            if (counter-1) % columns == 0:
                ax.set_ylabel('Log-posterior ratio')
            else:
                ax.set_yticklabels([''])
            counter += 1


def profile_lowlevel(fvals, ax=None, size=(18.5, 6.5)):
    """
    Lowlevel routine for plotting one profile, working with a numpy array only

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

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

    # parse input
    fvals = np.array(fvals)

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        ax.set_xlabel('Parameter value')
        ax.set_ylabel('Log-posterior ratio')
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # plot
    if fvals.size != 0:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(fvals[0, :], fvals[1, :], color=[.9, .2, .2, 1.])

    return ax
