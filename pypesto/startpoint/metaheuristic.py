from ..startpoint import latin_hypercube, uniform, resample_startpoints
import copy
from np.linalg import norm


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

    weighting: tuple of float, optional
        two numbers between 0 and 1, which add up to less or equal than 1,
        which specify how the new population should be chosen among the
        created offspring:
        first number: fraction of fittest values that should survive
        second number: fraction of diverse values that should survive
        rest: will be chosen balanced among the remaining guesses
    """

    def __init__(self,
                 ndim,
                 lb,
                 ub,
                 sample_lb=None,
                 sample_ub=None,
                 population_size=None,
                 n_generations=None,
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

        # create sample bounds if needed
        if sample_lb is None and sample_ub is None:
            # create sample bounds
            self.sample_lb = np.zeros(self.ndim)
            self.sample_ub = np.zeros(self.ndim)

            # assign sample bounds
            for iPar in range(self.ndim):
                dist = self.ub[iPar] - self.ulb[iPar]
                self.sample_lb[iPar] = self.lb[iPar] + 0.1 * dist
                self.sample_ub[iPar] = self.ub[iPar] - 0.1 * dist
        else:
            # use user assigned sample bounds
            self.sample_lb = sample_lb
            self.sample_ub = sample_ub

            # setting typical scale for each parameter
            self.state_scale = [0.02 * (self.sample_ub[i] - self.sample_lb[
                i]) for i in range(ndims)]

        # assign population size
        if population_size is not None:
            self.population_size = population_size
        else:
            self.population_size = np.ceil(np.min([0.5 * self.ndim,
                                                   np.sqrt(self.dim)]))

        # assign number of generations size
        if n_generations is not None:
            self.n_generations = n_generations
        else:
            self.n_generations = np.ceil(np.min([5, np.sqrt(self.dim)]))

        # assign diversity
        if weighting is None:
            self.weighting = (0.4, 0.4)
        else:
            self.weighting = weighting

        # compute proportions of new generation: How many offspring,
        # how many fit guesses and how many diverse guesses we will have
        self.offspring_pop = np.floor(0.4 * self.population_size)
        self.fit_pop = np.floor(0.3 * self.population_size)
        self.div_pop = self.population_size - offspring_pop - fittest_pop

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

    def get_next_generation(self, problem):
        """
        Creates the next generation from the current state of the system

        """

        # sort according to quality of guesses
        self.sort_population()

        # create list of new guesses based on last state
        self.generate_new_list()

        # take the fittest 40%, create offspring from that, add to list
        self.generate_offspring(problem)

        # take another 30%, sample points somewhere close by, add to list
        self.sample_fittest(problem)

        # take the remaining 30%, sample diverse guesses, add to list
        self.sample_diversity(problem)

        # choose the next generation among the created and the current guesses
        self.choose_guesses()

    def sort_population(self):
        """
        sort the current parameter vectors by objective value
        """

        population_order = np.argsort(self.fvals)
        self.fvals = self.fvals[population_order]
        self.xs = self.xs[population_order]
        self.divs = self.divs[population_order]

    def generate_new_list(self):
        """
        create pool of guesses for next generation
        """

        self.next_xs = copy.deepcopy(self.xs)
        self.next_fvals = copy.deepcopy(self.fvals)

    def generate_offspring(self, problem):
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
                    (dist_order[j], i_guess) in pairs:
                j += 1

            # append to pairs
            pairs.append((i_guess, dist_order[0]))

            # increase counter
            i_guess += 1

            # stop if enough pairs were created
            if i_guess == self.offspring_pop:
                not_enough_pairs = False

        for i_guess in range(self.offspring_pop):
            # generate offspring points
            point_not_evaluable = True
            while point_not_evaluable:
                x = create_offspring_from_pair(pairs[i_guess])

                # check whether guess could be evaluated
                fval = problem.objective(x)
                if np.isfinite(fval):
                    point_not_evaluable = False

                # append
                self.next_xs.append(x)
                self.next_fvals.append(fval)

    def sample_fittest(self, problem):
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
                fval = problem.objective(x)
                if np.isfinite(fval):
                    point_not_evaluable = False

            # append
            self.next_xs.append(x)
            self.next_fvals.append(fval)

    def sample_diversity(self, problem):
        # compute diversity scores for points
        for x in next_xs:
            self.next_divs.append(get_diversity(next_xs, x))

        # compute a median div, in order to make sure that new points show a
        # higher diversity than the meadian later on
        median_div = np.median(self.next_divs)

        for i_guess in range(self.div_pop):
            # sample point
            point_not_evaluable = True
            while point_not_evaluable:
                x = np.random.random((1, self.ndim))
                x = rescale(x, self.sample_lb, self.sample_ub)

                # check for diversity
                div = get_diversity(next_xs, x)
                if div < median_div:
                    continue

                # check whether guess could be evaluated
                fval = problem.objective(x)
                if np.isfinite(fval):
                    point_not_evaluable = False

            # append
            self.next_xs.append(x)
            self.next_fvals.append(fval)


    def choose_guesses(self):
        # recompute diversity
        self.next_divs = []
        for x in next_xs:
            self.next_divs.append(get_diversity(next_xs, x))

        # recompute numbers of indiviuals, which will be chosen
        n_fittest = np.floor(self.weighting[0] * self.population_size)
        n_diverse = np.floor(self.weighting[1] * self.population_size)
        n_rest = self.population_size - n_fittest - n_diverse

        # get orderings accoring to fitness and to diversity
        ord_fit = np.argsort(self.next_fvals)
        ord_div = np.argsort(self.next_divs)

        # recreate new population
        new_xs = []
        new_fvals = []
        # get fittest individuals
        for i in range(n_fittest):
            new_xs.append(self.next_xs[ord_fit[i]])
            new_fvals.append(self.next_fvals[ord_fit[i]])
            self.next_xs[ord_fit[i]] = None
            self.next_fvals[ord_fit[i]] = None
        # get most diverse individuals
        for i in range(n_diverse):
            new_xs.append(self.next_xs[ord_div[i]])
            new_fvals.append(self.next_fvals[ord_div[i]])
            self.next_xs[ord_div[i]] = None
            self.next_fvals[ord_div[i]] = None

        # prune out entries which have already been taken
        next_xs = [x for x in self.next_xs if x is not None]
        next_fvals = [fval for fval in self.next_fvals if fval is not None]

        # pick randomly the rest of the guesses
        inds = np.random.permutation(len(next_fvals))
        for i in range(n_rest):
            new_fvals.append(next_fvals[inds[i]])
            new_xs.append(next_xs[inds[i]])

        # update populations
        self.xs = new_xs
        self.fvals = new_fvals


    def create_offspring_from_pair(self, pair):
        center = 0.5 * (pair[0] + pair[1])
        variance = 0.33 * np.absolute(pair[0] - pair[1])
        return np.random.multivariate_normal(center, np.diag(variance))


def metaheuristic(
        problem,
        sample_lb,
        sample_ub,
        population_size=None,
        n_generations=None,
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
        options = MetaheuristicPreSearch(problem.dim,
                                         problem.lb,
                                         problem.ub,
                                         sample_lb,
                                         sample_ub,
                                         population_size=population_size,
                                         n_generations=n_generations)

    state = MetaheuristicPreSearch.assert_instance(options)

    # assign startpoints to presearch state
    startpoints = assign_startpoints(population_size, uniform, problem)
    for i_guess in range(startpoints.size[1]):
        # assign first generation
        state.xs.append(list(startpoints[i_guess, :]).flatten())
        state.xs.fvals(problem.objective(startpoints[i_guess, :]))

    # iterate over the generations
    for i_gen in range(state.n_generations):
        # do evolution step
        state.get_next_generation(problem)

    # create numpy array from guesses
    startpoints = np.array(state.xs)

    return startpoints


def get_distances(xs, x):
    # create return value
    distances = []

    for ix in xs:
        # skip if x was part of xs
        if x == ix:
            continue

        # compute distances
        distances.append(norm(x - ix))

    return distances

def get_diversity(xs, x):
    dists = get_distances(xs, x)
    return sum([norm(dist) for dist in dists])