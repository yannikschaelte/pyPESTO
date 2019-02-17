import numpy as np
import pandas as pd
import os
import sys
import importlib
import numbers
import copy
import shutil
import re
import logging

import petab

from ..problem import Problem
from .amici_objective import AmiciObjective

try:
    import amici
except ImportError:
    amici = None


logger = logging.getLogger(__name__)


class PetabImporter:

    def __init__(self, petab_problem, output_folder=None):
        """
        petab_problem: petab.Problem
            Managing access to the model and data.

        output_folder: str,  optional
            Folder to contain the amici model. Defaults to
            './tmp/petab_problem.name'.
        """
        self.petab_problem = petab_problem

        if output_folder is None:
            output_folder = os.path.abspath(
                os.path.join("tmp", self.petab_problem.model_name))
        self.output_folder = output_folder

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
        return solver

    def create_edatas(self, model=None, simulation_conditions=None):
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
        if simulation_conditions is None:
            simulation_conditions = petab.core.get_simulation_conditions(
                measurement_df)

        observable_ids = model.getObservableIds()

        fixed_parameter_ids = model.getFixedParameterIds()

        edatas = []
        for _, condition in simulation_conditions.iterrows():
            # amici.ExpData for each simulation

            # extract rows for condition
            df_for_condition = petab.core.get_rows_for_condition(
                measurement_df, condition)

            # make list of all timepoints for which measurements exist
            timepoints = sorted(
                df_for_condition.time.unique().astype(float))

            # init edata object
            edata = amici.ExpData(model.get())

            # find rep numbers of time points
            timepoints_w_reps = []
            for time in timepoints:
                # subselect for time
                df_for_time = df_for_condition[df_for_condition.time == time]
                # rep number is maximum over rep numbers for observables
                n_reps = max(df_for_time.groupby(
                    ['observableId', 'time']).size())
                # append time point n_rep times
                timepoints_w_reps.extend([time] * n_reps)

            # set time points in edata
            edata.setTimepoints(timepoints_w_reps)

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
            # iterate over time points
            for time in timepoints:
                # subselect for time
                df_for_time = df_for_condition[df_for_condition.time == time]
                time_ix_0 = timepoints_w_reps.index(time)

                # remember used time indices for each observable
                time_ix_for_obs_ix = {}

                # iterate over measurements
                for _, measurement in df_for_time.iterrows():
                    # extract observable index
                    observable_ix = observable_ids.index(
                        f'observable_{measurement.observableId}')

                    # update time index for observable
                    if observable_ix in time_ix_for_obs_ix:
                        time_ix_for_obs_ix[observable_ix] += 1
                    else:
                        time_ix_for_obs_ix[observable_ix] = time_ix_0

                    # fill observable and possibly noise parameter
                    y[time_ix_for_obs_ix[observable_ix],
                        observable_ix] = measurement.measurement
                    if isinstance(measurement.noiseParameters, numbers.Number):
                        sigma_y[time_ix_for_obs_ix[observable_ix],
                                observable_ix] = measurement.noiseParameters

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
        # get simulation conditions
        simulation_conditions = petab.core.get_simulation_conditions(
            self.petab_problem.measurement_df)

        # create model
        if model is None:
            model = self.create_model(force_compile=force_compile)
        # create solver
        if solver is None:
            solver = self.create_solver(model)
        # create conditions and edatas from measurement data
        if edatas is None:
            edatas = self.create_edatas(
                model=model,
                simulation_conditions=simulation_conditions)

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
                par_sim_ids=par_sim_ids,
                simulation_conditions=simulation_conditions,
            )

        scale_mapping = \
            petab.core.get_optimization_to_simulation_scale_mapping(
                parameter_df=self.petab_problem.parameter_df,
                mapping_par_opt_to_par_sim=parameter_mapping
            )

        # check whether there is something suspicious in the mapping
        _check_parameter_mapping_ok(
            parameter_mapping, par_sim_ids, model, edatas)

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

        measurement_df = self.petab_problem.measurement_df

        # initialize dataframe
        df = pd.DataFrame(
            columns=list(
                self.petab_problem.measurement_df.columns))

        # get simulation conditions
        simulation_conditions = petab.core.get_simulation_conditions(
            measurement_df)

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
            cur_measurement_df = petab.core.get_rows_for_condition(
                measurement_df, condition)

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


def _check_parameter_mapping_ok(
        mapping_par_opt_to_par_sim, par_sim_ids, model, edatas):
    """
    Check whether there are suspicious parameter mappings and/or data points.

    Currently checks whether nan values in the parameter mapping table
    correspond to nan columns in the edatas, corresponding to missing
    data points.
    """
    # regular expression for noise and observable parameters
    pattern = "(noise|observable)Parameter[0-9]+_"
    rex = re.compile(pattern)

    # prepare output
    msg_data_notnan = ""
    msg_data_nan = ""

    # iterate over conditions
    for i_condition, (mapping_for_condition, edata_for_condition) in \
            enumerate(zip(mapping_par_opt_to_par_sim, edatas)):
        # turn amici.ExpData into pd.DataFrame
        df = amici.getDataObservablesAsDataFrame(model, [edata_for_condition])
        # iterate over simulation parameters indices and the mapped
        # optimization parameters
        for i_sim_id, par_sim_id in enumerate(par_sim_ids):
            # only continue if sim par is a noise or observable parameter
            if not rex.match(par_sim_id):
                continue
            # extract observable id
            obs_id = re.sub(pattern, "", par_sim_id)
            # extract mapped optimization parameter
            mapped_par = mapping_for_condition[i_sim_id]
            # check if opt par is nan, but not all corresponding data points
            if not isinstance(mapped_par, str) and np.isnan(mapped_par) \
                    and not df["observable_" + obs_id].isnull().all():
                msg_data_notnan += \
                    f"({i_condition, par_sim_id, obs_id})\n"
            # check if opt par is string, but all corresponding data points
            # are nan
            if isinstance(mapped_par, str) \
                    and df["observable_" + obs_id].isnull().all():
                msg_data_nan += f"({i_condition, par_sim_id, obs_id})\n"

    if not len(msg_data_notnan) + len(msg_data_nan):
        return

    logger.warn(
        "There are suspicious combinations of parameters and data "
        "points:\n"
        "For the following combinations of "
        "(condition_ix, par_sim_id, obs_id)"
        ", there are real-valued data points, but unmapped scaling or noise "
        "parameters: \n" + msg_data_notnan + "\n"
        "For the following combinations, scaling or noise parameters have "
        "been mapped, but all corresponding data points are nan: \n"
        + msg_data_nan + "\n")


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
