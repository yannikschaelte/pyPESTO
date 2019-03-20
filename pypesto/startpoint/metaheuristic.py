import numpy as np
import copy
from ..startpoint import latin_hypercube, uniform
from numpy.linalg import norm
from .util import rescale, resample_startpoints, assign_startpoints


class MetaheuristicPreSearch(dict):
    """
    Defines the system state for the scatter search pre optimization.

    Parameters
    ----------

    ndim: int
        problem dimension

    lb, ub: ndarray
        lower and upper bounds of the problem

    state_scale: ndarray
        vector of typical scale for each dimension of the problem (by
        default 1/50 * distance between bounds)

    xs: ndarray
        list of parameter vectors

    fvals: list
        list of function values for each parameter vector in the current xs

    div: list
        list of diversity values for each parameter guess

    sample_lb, sample_ub: ndarray, optional
        lower and upper bounds for sampling new parameters

    population_size: int, optional
        Size of the population in the pre-search phase

    n_generations: int, optional
        Number of generations in the pre-search phase

    n_fun_evals: int, optional
        Maximum number of function evaluations in metaheuristic pre-search

    weighting: tuple of float, optional
        two numbers between 0 and 1, which add up to less or equal than 1,
        which specify how the new population should be chosen among the
        created offspring:
        first number: fraction of fittest values that should survive
        second number: fraction of diverse values that should survive
        rest: will be chosen balanced among the remaining guesses
    """

    def __init__(self,
                 objective,
                 ndim,
                 lb,
                 ub,
                 population_size=None,
                 n_generations=None,
                 n_fun_evals=None,
                 weighting=None,
                 ):
        super().__init__()

        # set problem dimension and bounds
        self.ndim = ndim
        self.lb = lb
        self.ub = ub

        # create the state of the system
        self.xs = []
        self.fvals = []
        self.div = []

        # create the state of the system for next generation
        self.next_xs = []
        self.next_fvals = []
        self.next_divs = []

        # set objective function
        self.objective = objective

        # setting typical scale for each parameter
        self.state_scale = [0.02 * (self.ub[i] - self.lb[i])
                            for i in range(self.ndim)]

        # assign population size
        if population_size is not None:
            self.population_size = population_size
        else:
            self.population_size = int(np.ceil(np.min([0.5 * self.ndim,
                                                      np.sqrt(self.dim)])))

        # assign maximum number of function evaluations
        self.n_fun_evals = n_fun_evals

        # assign number of generations size
        if n_generations is not None:
            self.n_generations = int(n_generations)
        elif self.n_fun_evals is not None:
            self.n_generations = int(np.floor(self.n_fun_evals /
                                              self.population_size))
        else:
            self.n_generations = int(np.ceil(np.min([5, np.sqrt(self.ndim)])))

        # assign diversity
        if weighting is not None:
            self.weighting = weighting
        else:
            self.weighting = (0.4, 0.4)

        # compute proportions of new generation: How many offspring,
        # how many fit guesses and how many diverse guesses we will have
        self.offspring_pop = int(np.floor(0.4 * self.population_size))
        self.fit_pop = int(np.floor(0.3 * self.population_size))
        self.div_pop = self.population_size - self.offspring_pop - self.fit_pop

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def assert_instance(maybe_state):
        """
        Returns a valid options object.

        Parameters
        ----------

        maybe_state: PreSearchState or dict
        """
        if isinstance(maybe_state, MetaheuristicPreSearch):
            return maybe_state
        state = MetaheuristicPreSearch(**maybe_state)
        return state

    def get_next_generation(self):
        """
        Creates the next generation from the current state of the system

        """

        # sort according to quality of guesses
        self.sort_population()

        # create list of new guesses based on last state
        self.generate_new_list()

        # take the fittest 40%, create offspring from that, add to list
        self.generate_offspring()

        # take another 30%, sample points somewhere close by, add to list
        self.sample_fittest()

        # take the remaining 30%, sample diverse guesses, add to list
        self.sample_diversity()

        # choose the next generation among the created and the current guesses
        self.choose_guesses()

    def sort_population(self):
        """
        sort the current parameter vectors by objective value
        """

        population_order = np.argsort(self.fvals)
        self.fvals = [self.fvals[i] for i in population_order]
        self.xs = [self.xs[i].flatten() for i in population_order]

    def generate_new_list(self):
        """
        create pool of guesses for next generation
        """

        self.next_xs = copy.deepcopy(self.xs)
        self.next_fvals = copy.deepcopy(self.fvals)

    def generate_offspring(self):
        """
        finds parameter guesses which are close and generates the offspring
        from the found pairs
        """

        # create pairings of points
        pairs = []

        # loop until we have enough
        not_enough_pairs = True
        i_guess = 0
        while not_enough_pairs:
            # find closest points for creating offspring
            distances = get_distances(self.xs, self.xs[i_guess])
            dist_order = np.argsort(np.array(distances))

            # iterate to ensure that this pairing is really new
            j = 0
            while (i_guess, dist_order[j]) in pairs or \
                    (dist_order[j], i_guess) in pairs or \
                    j == i_guess:
                j += 1

            # append to pairs
            pairs.append((i_guess, dist_order[j]))

            # increase counter
            i_guess += 1

            # stop if enough pairs were created
            if i_guess == self.offspring_pop:
                not_enough_pairs = False

        for i_guess in range(self.offspring_pop):
            # generate offspring points
            point_not_evaluable = True
            while point_not_evaluable:
                x = self.create_offspring_from_pair(pairs[i_guess])

                # check whether guess could be evaluated
                fval = self.objective(x)
                if np.isfinite(fval):
                    point_not_evaluable = False

                # append
                self.next_xs.append(x)
                self.next_fvals.append(fval)

    def sample_fittest(self):
        """
        sample points close to fittest individuals
        """

        for i_guess in range(self.fit_pop):
            # sample point
            point_not_evaluable = True
            while point_not_evaluable:
                x = np.random.multivariate_normal(self.xs[i_guess],
                                                  np.diag(self.state_scale))

                # check whether guess could be evaluated
                fval = self.objective(x)
                if np.isfinite(fval):
                    point_not_evaluable = False

            # append
            self.next_xs.append(x)
            self.next_fvals.append(fval)

    def sample_diversity(self):
        # compute diversity scores for points
        for x in self.next_xs:
            self.next_divs.append(get_diversity(self.next_xs, x))

        # compute a median div, in order to make sure that new points show a
        # higher diversity than the meadian later on
        median_div = np.median(self.next_divs)

        for i_guess in range(self.div_pop):
            # sample point
            point_not_evaluable = True
            while point_not_evaluable:
                x = np.random.random((1, self.ndim))
                x = rescale(x, self.lb, self.ub)

                # check for diversity
                div = get_diversity(self.next_xs, x)
                if div < median_div:
                    continue

                # check whether guess could be evaluated
                fval = self.objective(x)
                if np.isfinite(fval):
                    point_not_evaluable = False

            # append
            self.next_xs.append(x)
            self.next_fvals.append(fval)


    def choose_guesses(self):
        # recompute diversity
        self.next_divs = []
        for x in self.next_xs:
            self.next_divs.append(get_diversity(self.next_xs, x))

        # recompute numbers of indiviuals, which will be chosen
        n_fittest = int(np.floor(self.weighting[0] * self.population_size))
        n_diverse = int(np.floor(self.weighting[1] * self.population_size))
        n_rest = self.population_size - n_fittest - n_diverse

        # get orderings accoring to fitness and to diversity
        ord_fit = np.argsort(self.next_fvals)
        ord_div = np.argsort(self.next_divs)

        # recreate new population
        new_xs = []
        new_fvals = []
        chosen = []
        # get fittest individuals
        for i in range(n_fittest):
            new_xs.append(self.next_xs[ord_fit[i]])
            new_fvals.append(self.next_fvals[ord_fit[i]])
            chosen.append(ord_fit[i])
        # get most diverse individuals
        for i in range(n_diverse):
            new_xs.append(self.next_xs[ord_div[i]])
            new_fvals.append(self.next_fvals[ord_div[i]])
            chosen.append(ord_div[i])

        possible_inds = [i for i in range(len(self.next_fvals))
                         if not i in chosen]

        # pick randomly the rest of the guesses
        inds = np.random.permutation(len(possible_inds))
        for i in range(n_rest):
            new_fvals.append(self.next_fvals[possible_inds[inds[i]]])
            new_xs.append(self.next_xs[possible_inds[inds[i]]].flatten())

        # update populations
        self.xs = new_xs
        self.fvals = new_fvals

        # clean up
        self.next_xs = []
        self.next_fvals = []
        self.next_divs = []


    def create_offspring_from_pair(self, pair):
        center = 0.5 * (self.xs[pair[0]] + self.xs[pair[1]])
        variance = 0.33 * np.absolute(self.xs[pair[0]] + self.xs[pair[1]])
        return np.random.multivariate_normal(center, np.diag(variance))


def metaheuristic(
        objective,
        lb,
        ub,
        x_guesses=None,
        n_generations=None,
        n_starts=None,
        options=None):
    """
    This is the main function to call to do multistart optimization.

    Parameters
    ----------

    problem: pypesto.Problem
        The problem to be solved.

    sample_lb, sample_ub: ndarray, optional
        lower and upper bounds for sampling new parameters

    population_size: int, optional
        Size of the population in the pre-search phase

    n_generations: int, optional
        Number of generations in the pre-search phase

    options: pypesto.PreSearchState, optional
        Various options for generating a presearch state.
    """

    # check options
    if options is None:
        options = MetaheuristicPreSearch(objective=objective,
                                         lb=lb,
                                         ub=ub,
                                         ndim=len(lb),
                                         population_size=n_starts,
                                         n_generations=n_generations)

    state = MetaheuristicPreSearch.assert_instance(options)

    # assign startpoints to presearch state
    for i_guess in range(state.population_size):
        # sample point
        point_not_evaluable = True
        while point_not_evaluable:
            x = np.random.random((1, state.ndim))
            x = rescale(x, state.lb, state.ub)

            # check whether guess could be evaluated
            fval = state.objective(x)
            if np.isfinite(fval):
                point_not_evaluable = False

        # append to states
        state.xs.append(x)
        state.fvals.append(fval)

    # iterate over the generations
    for i_gen in range(state.n_generations):
        # do evolution step
        state.get_next_generation()

    # create numpy array from guesses
    return np.array(state.xs)


def get_distances(xs, x):
    # create return value
    distances = []

    for ix in xs:
        # skip if x was part of xs
        if np.array_equal(x, ix):
            continue

        # compute distances
        distances.append(norm(x - ix))

    return distances

def get_diversity(xs, x):

    # count number of elemnts in xs which are nor x
    n_xs = len(xs) - sum([1 for ix in xs if np.array_equal(x, ix)])
    # compute distances
    dists = get_distances(xs, x)
    return sum([norm(dist) for dist in dists]) / n_xs