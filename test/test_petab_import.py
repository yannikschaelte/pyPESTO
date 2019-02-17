
"""
This is for testing the petab import.
"""

import unittest
import numpy as np

import petab
import pypesto
from test.petab_util import folder_base
import amici


class PetabImportTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.petab_problems = []
        cls.petab_importers = []
        cls.obj_edatas = []

    def test_0_import(self):
        for model_name in ["Zheng_PNAS2012", "Boehm_JProteomeRes2014"]:
            petab_problem = petab.Problem.from_folder(
                folder_base + model_name)
            self.petab_problems.append(petab_problem)

    def test_1_compile(self):
        for petab_problem in self.petab_problems:
            importer = pypesto.PetabImporter(petab_problem)
            self.petab_importers.append(importer)

            # check model
            model = importer.create_model(force_compile=True)

            # observable ids
            model_obs_ids = list(model.getObservableIds())
            problem_obs_ids = list(petab_problem.get_observables().keys())
            self.assertEqual(set(model_obs_ids),
                             set(problem_obs_ids))

            # also other checks would be possible here

    def test_2_simulate(self):
        for petab_importer in self.petab_importers:
            obj = petab_importer.create_objective()
            edatas = petab_importer.create_edatas()
            self.obj_edatas.append((obj, edatas))

            # run function
            x_nominal = petab_importer.petab_problem.x_nominal
            ret = obj(x_nominal)

            self.assertTrue(np.isfinite(ret))

    def test_3_optimize(self):
        # run optimization
        for obj_edatas, importer in \
                zip(self.obj_edatas, self.petab_importers):
            obj = obj_edatas[0]
            optimizer = pypesto.ScipyOptimizer()
            problem = importer.create_problem(obj)
            result = pypesto.minimize(
                problem=problem, optimizer=optimizer, n_starts=2)

            self.assertTrue(np.isfinite(
                result.optimize_result.get_for_key('fval')[0]))


class SpecialFeaturesTest(unittest.TestCase):

    def test_replicates(self):
        """
        Use a model that has replicates and check that all data points are
        inserted.
        """
        # import a model with replicates at some time points and observables
        importer = pypesto.PetabImporter.from_folder(
            folder_base + "Schwen_PONE2014")

        # create amici.ExpData list
        edatas = importer.create_edatas()

        # convert to dataframe
        amici_df = amici.getDataObservablesAsDataFrame(
            importer.create_model(), edatas)

        # extract original measurement df
        meas_df = importer.petab_problem.measurement_df

        # find time points
        amici_times = sorted(amici_df.time.unique().tolist())
        meas_times = sorted(meas_df.time.unique().tolist())

        # assert same time points
        for amici_time, meas_time in zip(amici_times, meas_times):
            self.assertTrue(np.isclose(amici_time, meas_time))

        # extract needed stuff from amici df
        amici_df = amici_df[[col for col in amici_df.columns
                             if col == 'time' or col.startswith('observable_')
                             and not col.endswith("_std")]]

        # find observable ids
        amici_obs_ids = [col for col in amici_df.columns if col != 'time']
        amici_obs_ids = [val.replace("observable_", "")
                         for val in amici_obs_ids]
        amici_obs_ids = sorted(amici_obs_ids)
        meas_obs_ids = sorted(meas_df.observableId.unique().tolist())
        for amici_obs_id, meas_obs_id in zip(amici_obs_ids, meas_obs_ids):
            self.assertEqual(amici_obs_id, meas_obs_id)

        # iterate over time points
        for time in meas_times:
            amici_df_for_time = amici_df[amici_df.time == time]
            amici_df_for_time = amici_df_for_time[
                [col for col in amici_df.columns if col != 'time']]

            meas_df_for_time = meas_df[meas_df.time == time]

            # iterate over observables
            for obs_id in meas_obs_ids:
                amici_df_for_obs = amici_df_for_time["observable_" + obs_id]
                meas_df_for_obs = meas_df_for_time[
                    meas_df_for_time.observableId == obs_id]

                # extract non-nans and sort
                amici_vals = amici_df_for_obs.values.flatten().tolist()
                amici_vals = sorted([val for val in amici_vals
                                     if np.isfinite(val)])

                meas_vals = meas_df_for_obs.measurement \
                    .values.flatten().tolist()
                meas_vals = sorted([val for val in meas_vals
                                    if np.isfinite(val)])

                # test if the measurement data coincide for the given time
                # point
                for amici_val, meas_val in zip(amici_vals, meas_vals):
                    self.assertTrue(np.isclose(amici_val, meas_val))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(PetabImportTest())
    suite.addTest(SpecialFeaturesTest())
    unittest.main()
