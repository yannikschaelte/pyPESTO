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

    def append(self, condition_ix, time_ix, observable_ix):
        self.indices.append((condition_ix, time_ix, observable_ix))
        # TODO also need to remember time for adjoints

    def iterate(self):
        return (ix for ix in self.indices)

class HierarchicalProblem:

    def __init__(self, xs = None):
        """
        s_ids, b_ids, sigma_ids should all be estimated parameters.
        """
        if xs is None:
            xs = []
        self.xs = xs

    @staticmethod
    def from_parameter_df(df):
        """
        Create list of hierarchical parameters from parameter df
        based on name conventions.
        """
        # TODO need to make sure ordering is the same as in 
        # petab_problem.get_optimization_parameters

        xs = []

        for ix, x in enumerate(df.reset_index()['parameterId']):
            type_ = None
            default_val_ = None
            if x.startswith('scaling_'):
                type_ = HierarchicalParameter.SCALING
                default_val_ = 1.0
            elif x.startswith('offset_'):
                type_ = HierarchicalParameter.OFFSET
                default_val_ = 0.0
            elif x.startswith('sd_') or x.startswith('sigma_'):
                type_ = HierarchicalParameter.SIGMA
                default_val_ = 1.0
            if type_:
                x = HierarchicalParameter(
                    id_=x, ix_=ix, type_=type_, default_val_=default_val_)
                xs.append(x)

        return HierarchicalProblem(xs)

    def get_x_ids(self):
        return [x.id for x in self.xs]

    def get_x_by_id(self, id_):
        for x in self.xs:
            if x.id == id_:
                return x
        return None
    
    def get_xs_for_type(self, type_):
        return [x for x in self.xs if x.type == type_]

    def insert_for_id(self, id_, condition_ix, time_ix, observable_ix):
        x = self.get_x_by_id(id_)
        if x:
            x.append(condition_ix, time_ix, observable_ix)

    def is_empty(self):
        return len(self.xs) == 0

    def get_all_ixs_and_default_vals(self):
        ixs = self.s_ixs + self.b_ixs + self.sigma_ixs
        default_vals = [1.0] * len(self.s_ids) + [0.0] * len(self.b_ids) + [1.0] * len(self.sigma_ids)
        return ixs, default_vals
