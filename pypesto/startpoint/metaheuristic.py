from ..startpoint import latin_hypercube, uniform, resample_startpoints


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
        numbers between 0 and 1, which add up to less or equal than 1,
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
                 diversity=None,
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
        if diversity is None:
            self.diversity = 0.3
        else:
            self.diversity = diversity

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

        # take the fittest 40%, create offspring from that
        self.generate_offspring(problem)

        # take another 30%, sample points somewhere close by
        self.sample_fittest(problem)

        # take the remaining 30%, sample diverse guesses
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
        self.div = self.div[population_order]


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
        state.xs.append(list(startpoints[i_guess,:]).flatten())
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
        distances.append(np.linalg.norm(x - ix))

    return distances