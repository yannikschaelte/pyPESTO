import logging


logger = logging.getLogger(__name__)


class HierarchicalParameter:

    SCALING = 'SCALING'
    OFFSET = 'OFFSET'
    SIGMA = 'SIGMA'

    def __init__(self, id_, ix_, type_, default_val_):
        """
        Parameters
        ----------

        id_: str
            Id of the parameter.
        ix_: int
            Index of the parameter in the optimization parameter list.
        type_: str
            Any of the supported parameter types (SCALING, OFFSET, SIGMA).
        default_val_: float
            Value to be used when the parameter is not present (in particular
            to simulate unscaled observables).
        """
        self.id = id_
        self.ix = ix_
        self.type = type_
        self.default_val = default_val_
        self.indices = []

        logger.info(
            f"Created HierarchicalParameter (id={self.id}, ix={self.ix}, "
            f"type={self.type}, default_val={self.default_val}).")

    def append(self, condition_ix, time_ix, observable_ix, time):
        """
        Append a data index to the parameter's list of associated
        indices.

        Arguments
        ---------

        condition_ix, time_ix, observable_ix: int
            Define a data point by condition, time, and observable.
        time: float
            Time value of the data point (required for adjoint methods).
        """
        self.indices.append((condition_ix, time_ix, observable_ix))
        # TODO also need to remember time for adjoints

    def iterate(self):
        """
        Iterate over the data indices associated with this parameter.

        Returns
        -------

        An iterator over the indices.
        """
        return (ix for ix in self.indices)
