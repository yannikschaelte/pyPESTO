import pypesto
import pypesto.visualize
import numpy as np
import scipy as sp
import unittest


# define a pypesto problem
objective = pypesto.Objective(fun=sp.optimize.rosen,
                              grad=sp.optimize.rosen_der,
                              hess=sp.optimize.rosen_hess)
lb = -7 * np.ones((1, 2))
ub = 7 * np.ones((1, 2))
problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

# write some dummy results for optimization
result = pypesto.Result(problem=problem)
for j in range(0, 3):
    optimizer_result = pypesto.OptimizerResult(fval=j * 0.01,
                                               x=[j + 0.1, j + 1])
    result.optimize_result.append(optimizer_result=optimizer_result)
for j in range(0, 4):
    optimizer_result = pypesto.OptimizerResult(fval=10 + j * 0.01,
                                               x=[2.5 + j + 0.1, 2 + j + 1])
    result.optimize_result.append(optimizer_result=optimizer_result)

# write some dummy results for profiling
ratio_path_1 = [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]
ratio_path_2 = [0.1, 0.2, 0.7, 1., 0.8, 0.3, 0.1]
x_path_1 = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                     [1., 1.2, 1.4, 1.5, 1.6, 1.8, 2.]])
x_path_2 = np.array([[1., 1.1, 1.3, 1.5, 1.7, 1.9, 2.1],
                     [2.1, 2.2, 2.4, 2.5, 2.8, 2.9, 3.1]])
fval_path_1 = [4., 3., 1., 0., 1.5, 2.5, 5.]
fval_path_2 = [4.5, 3.5, 1.5, 0., 1.3, 2.3, 4.3]
tmp_result_1 = pypesto.ProfilerResult(x_path_1, fval_path_1, ratio_path_1)
tmp_result_2 = pypesto.ProfilerResult(x_path_2, fval_path_2, ratio_path_2)

# use pypesto function to write the numeric values into the results
result.profile_result.create_new_profile_list()
result.profile_result.create_new_profile(tmp_result_1)
result.profile_result.create_new_profile(tmp_result_2)


class TestVisualize(unittest.TestCase):

    @staticmethod
    def test_waterfall():
        pypesto.visualize.waterfall(result)

    @staticmethod
    def test_waterfall_lowlevel():
        # test empty input
        pypesto.visualize.waterfall_lowlevel([])

        # test if it runs at all
        fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
        pypesto.visualize.waterfall_lowlevel(fvals)
        fvals = np.array(fvals)
        pypesto.visualize.waterfall_lowlevel(fvals)

    def test_assign_clusters(self):
        # test empty input
        pypesto.visualize.assign_clusters([])

        # test if it runs at all
        fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11, 10]
        pypesto.visualize.assign_clusters(fvals)
        fvals = np.array(fvals)
        clust, clustsize, ind_clust = pypesto.visualize.assign_clusters(fvals)

        # test if clustering works as intended
        fvals = [0., 0.00001, 1., 2., 2.001]
        clust, clustsize, ind_clust = pypesto.visualize.assign_clusters(fvals)
        self.assertEqual(len(clust), 5)
        self.assertEqual(len(clustsize), 3)
        self.assertEqual(len(ind_clust), 5)

    @staticmethod
    def test_assign_clustered_colors():
        # test empty input
        pypesto.visualize.assign_clustered_colors([])

        # test if it runs at all
        fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
        pypesto.visualize.assign_clustered_colors(fvals)
        fvals = np.array(fvals)
        pypesto.visualize.assign_clustered_colors(fvals)

    @staticmethod
    def test_parameters():
        pypesto.visualize.parameters(result)

    @staticmethod
    def test_parameters_lowlevel():
        # test empty input
        xs = np.array([])
        xs.shape = (0, 0)  # we can assume in input that xs.ndim == 2
        fvals = np.array([])
        pypesto.visualize.parameters_lowlevel(xs, fvals)

        fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
        xs = [[0.1, 1], [1.2, 3], [2, 4], [1.2, 4.1], [1.1, 3.5],
              [4.2, 3.5], [1, 4], [6.2, 5], [4.3, 3], [3, 2]]

        # pass lists
        pypesto.visualize.parameters_lowlevel(xs, fvals, lb=lb, ub=ub)

        # pass numpy arrays
        fvals = np.array(fvals)
        xs = np.array(xs)
        pypesto.visualize.parameters_lowlevel(xs, fvals, lb=lb, ub=ub)

        # test no bounds
        pypesto.visualize.parameters_lowlevel(xs, fvals)

    @staticmethod
    def test_profiles():
        pypesto.visualize.profiles(result)

    @staticmethod
    def test_profiles_lowlevel():
        # test empty input
        pypesto.visualize.profiles_lowlevel([])

        # test if it runs at all
        p1 = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                       [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]])
        p2 = np.array([[1., 1.2, 1.4, 1.5, 1.6, 1.8, 2.],
                       [0.1, 0.2, 0.5, 1., 0.6, 0.4, 0.1]])
        fvals = [p1, p2]
        pypesto.visualize.profiles_lowlevel(fvals)

    @staticmethod
    def test_profile_lowlevel():
        # test empty input
        pypesto.visualize.profile_lowlevel(fvals=[])

        # test if it runs at all
        fvals = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                          [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]])
        pypesto.visualize.profile_lowlevel(fvals=fvals)


if __name__ == '__main__':
    unittest.main()
