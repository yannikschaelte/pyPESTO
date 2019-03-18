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

    diversity: float, optional
        number between 0 and 1, which specifies how the new population
        should be chosen among the created offspring:
        high: Take more parameter with good values
        low: Take more parameters with diverse values
    """

    def __init__(self,
                 ndim,
                 lb,
                 ub,
                 sample_lb=None,
                 sample_ub=None,
                 population_size=None,
                 n_generations=None,
                 diversity=0.3,
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
            self.n_generations = np.ceil(np.min([10, np.sqrt(self.dim)]))

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

        # take the fittest 40%, create offspring from that
        self.generate_offspring()

        # take another 30%, sample points somewhere close by
        self.sample_fittest()

        # take the remaining 30%, sample diverse guesses
        self.sample_diversity()

        # choose the next generation among the created and the current guesses
        self.choose_guesses()



    def sort_population(self):
        # sort the current parameter vectors by objective value
        population_order = np.argsort(self.fvals)
        self.fvals = self.fvals[population_order]
        self.xs = self.xs[population_order]
        self.div = self.div[population_order]


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
        state.get_next_generation()

    # create numpy array from guesses
    startpoints = np.array(state.xs)

    return startpoints

