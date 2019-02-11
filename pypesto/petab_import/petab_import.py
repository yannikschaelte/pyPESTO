import numpy as np
import pandas as pd
import os
import sys
import importlib
import numbers
import copy
import shutil
import yaml

import petab

from ..problem import Problem
from .petab_amici_objective import PetabAmiciObjective
from ..optimize import ScipyOptimizer

try:
    import amici
except ImportError:
    amici = None


class PetabImporter:

    def __init__(self, petab_problem, output_folder=None, config_file=None):
        """
        petab_problem: petab.Problem
            Managing access to the model and data.

        output_folder: str,  optional
            Folder to contain the amici model. Defaults to
            './tmp/petab_problem.name'.

        config_file: str, optional
            Path to a config file including options for amici and pypesto
            in yaml format.
        """
        self.petab_problem = petab_problem

        if output_folder is None:
            output_folder = os.path.abspath(
                os.path.join("tmp", self.petab_problem.model_name))
        self.output_folder = output_folder

        self.config_file = config_file

    @staticmethod
    def from_folder(folder, output_folder=None):
        """
        Simplified constructor exploiting the standardized petab folder
        structure.

        Parameters
        ----------

        folder: str
            Path to the base folder of the model, as in
            petab.Problem.from_folder.

        output_folder: see __init__.
        """
        petab_problem = petab.Problem.from_folder(folder)

        return PetabImporter(
            petab_problem=petab_problem,
            output_folder=output_folder)

    def create_model(self, force_compile=False):
        """
        Import amici model. If necessary or force_compile is True, compile
        first.

        Parameters
        ----------

        force_compile: str, optional
            If False, the model is compiled only if the output folder does not
            exist yet. If True, the output folder is deleted and the model
            (re-)compiled in either case.
        """
        # compile
        if force_compile or not os.path.exists(self.output_folder):
            self.compile_model()

        # add module to path
        if self.output_folder not in sys.path:
            sys.path.insert(0, self.output_folder)

        # load moduĺe
        model_module = importlib.import_module(self.petab_problem.model_name)

        # import model
        model = model_module.getModel()

        return model

    def compile_model(self):
        """
        Compile the model. If the output folder exists already, it is first
        deleted.
        """

        # check prerequisites
        if not petab.condition_table_is_parameter_free(
                self.petab_problem.condition_df):
            raise AssertionError(
                "Parameter dependent conditions in the condition file "
                "are not yet supported.")

        # delete output directory
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        # init sbml importer
        sbml_importer = amici.SbmlImporter(self.petab_problem.sbml_file)

        # constant parameters
        condition_columns = self.petab_problem.condition_df.columns.values
        constant_parameter_ids = list(
            set(condition_columns) - {'conditionId', 'conditionName'}
        )

        # observables
        observables = petab.get_observables(sbml_importer.sbml)

        # sigmas
        sigmas = petab.get_sigmas(sbml_importer.sbml)

        # convert
        sbml_importer.sbml2amici(
            modelName=self.petab_problem.model_name,
            output_dir=self.output_folder,
            observables=observables,
            constantParameters=constant_parameter_ids,
            sigmas=sigmas
        )

    def create_solver(self, model=None):
        """
        Return model solver.
        """
        # create model
        if model is None:
            model = self.create_model()

        solver = model.getSolver()

        # parse config
        config = self.config
        if config is not None and 'amici' in config:
            if 'sensitivity_method' in config['amici']:
                methods = {'adjoint': amici.SensitivityMethod_adjoint,
                           'forward': amici.SensitivityMethod_forward}
                config_method = config['amici']['sensitivity_method'].lower()
                if config_method in methods:
                    solver.setSensitivityMethod(methods[config_method])
                else:
                    raise ValueError(f"Sensitivity method must be one of {methods.keys()}.")
            if 'sensitivity_order' in config['amici']:
                orders = {'none': amici.SensitivityOrder_none,
                          'first': amici.SensitivityOrder_first,
                          'second': amici.SensitivityOrder_second}
                config_order = config['amici']['sensitivity_order'].lower()
                if config_order in orders:
                    solver.setSensitivityOrder(orders[config_order])
                else:
                    raise ValueError(f"Sensitivity order must be one of {orders.keys()}.")

        return solver
    
    @property
    def config(self):
        if self.config_file is None:
            return None
        with open(self.config_file, 'r') as f:
            return yaml.load(f)

    def create_edatas(self, model=None):
        """
        Create list of amici.ExpData objects.
        """
        # create model
        if model is None:
            model = self.create_model()

        condition_df = self.petab_problem.condition_df.reset_index()
        measurement_df = self.petab_problem.measurement_df

        # number of amici simulations will be number of unique
        # (preequilibrationConditionId, simulationConditionId) pairs.
        # Can be improved by checking for identical condition vectors.
        grouping_cols, simulation_conditions = \
            _get_simulation_conditions(condition_df, measurement_df)

        observable_ids = model.getObservableIds()

        fixed_parameter_ids = model.getFixedParameterIds()

        edatas = []
        for _, condition in simulation_conditions.iterrows():
            # amici.ExpData for each simulation

            # extract rows for condition
            cur_measurement_df = _get_rows_for_condition(
                measurement_df, grouping_cols, condition)

            # make list of all timepoints for which measurements exist
            timepoints = sorted(
                cur_measurement_df.time.unique().astype(float))

            # init edata object
            edata = amici.ExpData(model.get())
            edata.setTimepoints(timepoints)

            # handle fixed parameters
            _handle_fixed_parameters(
                edata, condition_df, fixed_parameter_ids, condition)

            # prepare measurement matrix
            y = np.full(shape=(edata.nt(), edata.nytrue()), fill_value=np.nan)
            # prepare sigma matrix
            sigma_y = np.full(
                shape=(edata.nt(), edata.nytrue()),
                fill_value=np.nan)

            # add measurements and sigmas
            for _, measurement in cur_measurement_df.iterrows():
                time_ix = timepoints.index(measurement.time)
                observable_ix = observable_ids.index(
                    f'observable_{measurement.observableId}')

                y[time_ix, observable_ix] = measurement.measurement
                if isinstance(measurement.noiseParameters, numbers.Number):
                    sigma_y[time_ix, observable_ix] = \
                        measurement.noiseParameters

            # fill measurements and sigmas into edata
            edata.setObservedData(y.flatten())
            edata.setObservedDataStdDev(sigma_y.flatten())

            # append edata to edatas list
            edatas.append(edata)

        return edatas

    def create_objective(self,
                         model=None,
                         solver=None,
                         edatas=None,
                         force_compile: bool = False):
        """
        Create a pypesto.PetabAmiciObjective.
        """
        # create model
        if model is None:
            model = self.create_model(force_compile=force_compile)
        # create solver
        if solver is None:
            solver = self.create_solver(model)
        # create conditions and edatas from measurement data
        if edatas is None:
            edatas = self.create_edatas(model=model)

        # simulation <-> optimization parameter mapping
        par_opt_ids = self.petab_problem.get_optimization_parameters()
        # take sim parameter vector from model to ensure correct order
        par_sim_ids = list(model.getParameterIds())

        parameter_mapping = \
            petab.core.get_optimization_to_simulation_parameter_mapping(
                condition_df=self.petab_problem.condition_df,
                measurement_df=self.petab_problem.measurement_df,
                parameter_df=self.petab_problem.parameter_df,
                sbml_model=self.petab_problem.sbml_model,
                par_opt_ids=par_opt_ids,
                par_sim_ids=par_sim_ids
            )

        scale_mapping = \
            petab.core.get_optimization_to_simulation_scale_mapping(
                parameter_df=self.petab_problem.parameter_df,
                mapping_par_opt_to_par_sim=parameter_mapping
            )

        # create objective
        obj = PetabAmiciObjective(
            petab_importer=self,
            amici_model=model, amici_solver=solver, edatas=edatas,
            x_ids=par_opt_ids, x_names=par_opt_ids,
            mapping_par_opt_to_par_sim=parameter_mapping,
            mapping_scale_opt_to_scale_sim=scale_mapping
        )

        return obj

    def create_problem(self, objective):
        problem = Problem(objective=objective,
                          lb=self.petab_problem.lb,
                          ub=self.petab_problem.ub,
                          x_fixed_indices=self.petab_problem.x_fixed_indices,
                          x_fixed_vals=self.petab_problem.x_fixed_vals,
                          x_names=self.petab_problem.x_ids)

        return problem
    
    def get_n_starts(self):
        config = self.config
        if config is not None and 'pypesto' in config:
            if 'n_starts' in config['pypesto']:
                return int(config['pypesto']['n_starts'])

    def create_optimizer(self):
        config = self.config
        if config is not None and 'pypesto' in config:
            if 'optimizer' not in config['pypesto']:
                return None
            optimizers = {'ScipyOptimizer': ScipyOptimizer}
            config_optimizer = config['pypesto']['optimizer']
            if config_optimizer not in optimizers:
                raise ValueError(f"Optimizer must be one of {optimizers}.")
            if config_optimizer == 'ScipyOptimizer':
                kwargs = {}
                if 'method' in config['pypesto']:
                    kwargs['method'] = config['pypesto']['method']
                if 'optimizer_options' in config['pypesto']:
                    kwargs['options'] = config['pypesto']['optimizer_options']
                return ScipyOptimizer(**kwargs)
            # TODO: add Dlib
        return None

    def rdatas_to_measurement_df(self, rdatas, model=None):
        """
        Create a measurement dataframe in the petab format from
        the passed `rdatas` and own information.

        Parameters
        ----------

        rdatas: list of amici.RData
            A list of rdatas as produced by
            pypesto.AmiciObjective.__call__(x, return_dict=True)['rdatas'].

        Returns
        -------

        df: pandas.DataFrame
            A dataframe built from the rdatas in the format as in
            self.petab_problem.measurement_df.
        """
        # create model
        if model is None:
            model = self.create_model()

        condition_df = self.petab_problem.condition_df.reset_index()
        measurement_df = self.petab_problem.measurement_df

        # initialize dataframe
        df = pd.DataFrame(
            columns=list(
                self.petab_problem.measurement_df.columns))

        # get simulation conditions
        grouping_cols, simulation_conditions = \
            _get_simulation_conditions(condition_df, measurement_df)

        # get observable ids
        observable_ids = model.getObservableIds()

        # iterate over conditions
        for data_idx, condition in simulation_conditions.iterrows():
            # current rdata
            rdata = rdatas[data_idx]
            # current simulation matrix
            y = rdata['y']
            # time array used in rdata
            t = list(rdata['t'])

            # extract rows for condition
            cur_measurement_df = _get_rows_for_condition(
                measurement_df, grouping_cols, condition)

            # iterate over entries for the given condition
            # note: this way we only generate a dataframe entry for every
            # row that existed in the original dataframe. if we want to
            # e.g. have also timepoints non-existent in the original file,
            # we need to instead iterate over the rdata['y'] entries
            for _, row in cur_measurement_df.iterrows():
                # copy row
                row_sim = copy.deepcopy(row)

                # extract simulated measurement value
                timepoint_idx = t.index(row.time)
                observable_idx = observable_ids.index(
                    "observable_" + row.observableId)
                measurement_sim = y[timepoint_idx, observable_idx]

                # change measurement entry
                row_sim.measurement = measurement_sim

                # append to dataframe
                df = df.append(row_sim, ignore_index=True)

        return df


def _get_simulation_conditions(condition_df, measurement_df):
    """
    Compute the conditions by which to group the measurements, so
    that for each group an amici.ExpData can be generated.

    Returns
    -------
    (grouping_cols, simulation_conditions): tuple
        Here, grouping_cols is the columns according to which the grouping
        was done, and simulation_conditions are the identified conditions.
    """
    # make sure index is reset
    condition_df = condition_df.reset_index()

    # find columns to group by (i.e. if not all nans).
    # number of amici simulations will be number of unique
    # (preequilibrationCondition, simulationCondition) pairs.
    # can be improved by checking for identical condition vectors.
    grouping_cols = petab.core.get_notnull_columns(
        measurement_df,
        ['simulationConditionId', 'preequilibrationConditionId'])

    # group by cols and return dataframe containing each combination
    # of those rows only once (and an additional counting row)
    simulation_conditions = measurement_df.groupby(
        grouping_cols).size().reset_index()

    return grouping_cols, simulation_conditions


def _get_rows_for_condition(measurement_df, grouping_cols, condition):
    """
    Extract rows in `measurement_df` according to `grouping_cols`
    for `condition`.

    Returns
    -------

    cur_measurement_df: pd.DataFrame
        The subselection for the condition.
    """
    # filter rows for condition
    row_filter = 1
    # check for equality in all grouping cols
    for col in grouping_cols:
        row_filter = (
            measurement_df[col] == condition[col]
        ) & row_filter

    # apply filter
    cur_measurement_df = measurement_df.loc[row_filter, :]
    return cur_measurement_df


def _handle_fixed_parameters(
        edata, condition_df, fixed_parameter_ids, condition):
    """
    Hande fixed parameters and update edata accordingly.

    Parameters
    ----------

    edata: amici.amici.ExpData
        Current edata.

    condition_df: pd.DataFrame
        The conditions table.

    fixed_parameter_ids: array_like
        Ids of parameters that are to be considered constant.

    condition:
        The current condition, as created by
        _get_simulation_conditions.
    """

    if len(fixed_parameter_ids) == 0:
        # nothing to be done
        return

    # find fixed parameter values
    fixed_parameter_vals = condition_df.loc[
        condition_df.conditionId ==
        condition.simulationConditionId,
        fixed_parameter_ids].values
    # fill into edata
    edata.fixedParameters = fixed_parameter_vals.astype(
        float).flatten()

    # same for preequilibration if necessary
    if 'preequilibrationConditionId' in condition \
            and condition.preequilibrationConditionId:
        fixed_preequilibration_parameter_vals = condition_df.loc[
            # TODO: preequilibrationConditionId might not exist
            condition_df.conditionId == \
            condition.preequilibrationConditionId,
            fixed_parameter_ids].values
        edata.fixedParametersPreequilibration = \
            fixed_preequilibration_parameter_vals.astype(float) \
                                                 .flatten()
