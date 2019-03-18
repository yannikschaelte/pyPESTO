import logging


logger = logging.getLogger(__name__)


class HierarchicalParameter:

    SCALING = 'SCALING'
    OFFSET = 'OFFSET'
    SIGMA = 'SIGMA'

    def __init__(self, id_, ix_, type_, default_val_):
        self.id = id_
        self.ix = ix_
        self.type = type_
        self.default_val = default_val_
        self.indices = []

        logger.info(
            f"Created HierarchicalParameter (id={self.id}, ix={self.ix}, "
            f"type={self.type}, default_val={self.default_val}).")

    def append(self, condition_ix, time_ix, observable_ix):
        self.indices.append((condition_ix, time_ix, observable_ix))
        # TODO also need to remember time for adjoints

    def iterate(self):
        return (ix for ix in self.indices)
