import copy

try:
    import amici
except ImportError:
    amici = None

from ..objective import AmiciObjective


class PetabAmiciObjective(AmiciObjective):
    """
    This is a shallow wrapper around AmiciObjective to make it serializable.
    """

    def __init__(
            self,
            petab_importer,
            amici_model, amici_solver, edatas,
            x_ids, x_names,
            mapping_par_opt_to_par_sim,
            mapping_scale_opt_to_scale_sim):

        super().__init__(
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            x_ids=x_ids, x_names=x_names,
            mapping_par_opt_to_par_sim=mapping_par_opt_to_par_sim,
            mapping_scale_opt_to_scale_sim=mapping_scale_opt_to_scale_sim)

        self.petab_importer = petab_importer

    def __getstate__(self):
        state = {}
        for key in set(self.__dict__.keys()) - \
                set(['amici_model', 'amici_solver', 'edatas',
                     'preequilibration_edatas']):
            state[key] = self.__dict__[key]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        petab_importer = state['petab_importer']

        model = petab_importer.create_model()
        solver = petab_importer.create_solver(model)
        edatas = petab_importer.create_edatas(model)

        self.amici_model = model
        self.amici_solver = solver
        self.edatas = edatas

        if self.preprocess_edatas:
            self.init_preequilibration_edatas(edatas)
        else:
            self.preequilibration_edatas = None

    def __deepcopy__(self, memodict=None):
        other = self.__class__.__new__(self.__class__)

        for key in set(self.__dict__.keys()) - \
                set(['amici_model', 'amici_solver', 'edatas',
                     'preequilibration_edatas']):
            other.__dict__[key] = copy.deepcopy(self.__dict__[key])

        other.amici_model = amici.ModelPtr(self.amici_model.clone())
        other.amici_solver = amici.SolverPtr(self.amici_solver.clone())
        other.edatas = [amici.ExpData(data) for data in self.edatas]

        if self.preprocess_edatas:
            other.init_preequilibration_edatas(other.edatas)
        else:
            other.preequilibration_edatas = None

        return other
