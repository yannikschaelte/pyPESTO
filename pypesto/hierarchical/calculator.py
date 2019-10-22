import numpy as np
from ..objective.constants import(
    MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS)
from ..objective.amici_util import(
    log_simulation,
    add_sim_grad_to_opt_grad,
    add_sim_hess_to_opt_hess,
    sim_sres_to_opt_sres,
)
from .parameter import HierarchicalParameter
from .problem import HierarchicalProblem

from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

try:
    import amici
except ImportError:
    amici = None

try:
    import pypesto
except ImportError:
    amici = None


class HierarchicalAmiciCalculator:
    """
    A calculator is passed as `calculator` to the pypesto.AmiciObjective.
    While this class cannot be used directly, it has two subclasses
    which allow to use forward or adjoint sensitivity analysis to
    solve a `pypesto.HierarchicalProblem` efficiently in an inner loop,
    while the outer optimization is only concerned with variables not
    specified as `pypesto.HierarchicalParameter`s.
    """

    def __init__(self, problem: HierarchicalProblem):
        """
        Initialize the calculator from the given problem.
        """
        self.problem = problem

    def __call__(self, obj, x, sensi_order, mode):
        """
        This function is called inside `pypesto.AmiciObjective.__call__`
        after minor preprocessing,
        and is supposed to return the function value, derivatives and
        possibly residuals in a dict for the given input.
        """
        raise NotImplementedError(
            "This class is not intended to be called.")


class HierarchicalForwardAmiciCalculator(HierarchicalAmiciCalculator):
    """
    Use forward sensitivity analysis to compute derivatives for the
    hierarchical problem.
    """

    def __call__(self, obj, x, sensi_order, mode):
        # prepare outputs
        nllh = 0.0
        WLS = 0.0
        snllh = np.zeros(obj.dim)
        s2nllh = np.zeros([obj.dim, obj.dim])

        res = np.zeros([0])
        sres = np.zeros([0, obj.dim])

        # set order in solver
        #print("sensi_order: ", sensi_order)
        obj.amici_solver.setSensitivityOrder(sensi_order)

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            obj.amici_model,
            obj.amici_solver,
            obj.edatas,
            num_threads=min(obj.n_threads, len(obj.edatas)),
        )

        # check if any simulation failed
        if any([rdata['status'] < 0.0 for rdata in rdatas]):
            return obj.get_error_output(rdatas)

        # edatas to numpy arrays (TODO cache this?)
        edatas = [amici.numpy.ExpDataView(edata)
                  for edata in obj.edatas]

        # compute optimal parameters (code base same for fw+ad)

        # create matrices from optimal parameters (only fw)

        # compute objective and derivatives (only fw)

        # compute optimal parameters
        #optimal_scalings = compute_optimal_scaling_matrix(self.problem, edatas, rdatas)
        # compute optimal surrogate data
        if self.problem.optimalScalingOptions['timeInnerProblem']:
            import time
            from datetime import date
            num_repeat = 50
            #edatas = [amici.numpy.ExpDataView(edata)
            #          for edata in obj.edatas]
            start = time.perf_counter()
            for repeatIdx in range(num_repeat):
                if self.problem.optimalScalingOptions['method'] == 'reduced':
                    optimal_surrogate = compute_optimal_surrogates_edatas_reduced(self.problem, edatas, rdatas)
                    # plot_surrogate_fit_reduced(self.problem, edatas, rdatas, optimal_surrogate)
                elif self.problem.optimalScalingOptions['method'] == 'standard':
                    optimal_surrogate = compute_optimal_surrogates_edatas(self.problem, edatas, rdatas)
                else:
                    optimal_surrogate = compute_optimal_surrogates_edatas_reduced(self.problem, edatas, rdatas)

            end = time.perf_counter()
            comp_time = (end-start) / num_repeat

            # WLS = np.sum([optimal_surrogate[i]['fun'] for i in range(len(optimal_surrogate))])
            # WLS = np.sqrt(WLS)
            WLS = compute_WLS(optimal_surrogate, self.problem, edatas, rdatas)
            success = True
            if False in [optimal_surrogate[i]['success'] for i in range(len(optimal_surrogate))]:
                success = False
                WLS = np.nan

            with open('./results/timing_results_' + self.problem.optimalScalingOptions['modelName'] + '_' + str(date.today())
                      + '_' + self.problem.optimalScalingOptions['method'] + '.txt', 'a') as f:
                f.write(str(comp_time) + '\t' + str(success) + '\t' + str(WLS) + '\n')
            print('computation time inner problem: ' + str(comp_time))
        else:
            if self.problem.optimalScalingOptions['method'] == 'reduced':
                optimal_surrogate = compute_optimal_surrogates_edatas_reduced(self.problem, edatas, rdatas)
                #plot_surrogate_fit_reduced(self.problem, edatas, rdatas, optimal_surrogate)
            elif self.problem.optimalScalingOptions['method'] == 'standard':
                optimal_surrogate = compute_optimal_surrogates_edatas(self.problem, edatas, rdatas)
            else:
                optimal_surrogate = compute_optimal_surrogates_edatas_reduced(self.problem, edatas, rdatas)
            # nllh = compute_nllh(edatas, rdatas, optimal_scalings)


        if False in [optimal_surrogate[idx]['success'] for idx in range(len(optimal_surrogate))]:
            WLS = np.nan
            print('Inner optimization not succeeded')
        else:
            # WLS = np.sum([optimal_surrogate[i]['fun'] for i in range(len(optimal_surrogate))])
            # WLS = np.sqrt(WLS)
            WLS = compute_WLS(optimal_surrogate, self.problem, edatas, rdatas)

        print('cost function: ' + str(WLS))
        #TODO: gradient computation
        #if sensi_order > 0:
        #    snllh = compute_snllh(edatas, rdatas, optimal_scalings, obj.x_ids, obj.mapping_par_opt_to_par_sim, obj.dim)
        # TODO compute FIM or HESS
        # TODO RES, SRES should also be possible, right?

        return {
            FVAL: WLS,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas
        }


class HierarchicalAdjointAmiciCalculator(HierarchicalAmiciCalculator):
    """
    Use adjoint sensitivity analysis to compute derivatives for the
    hierarchical problem.
    """

    def __call__(self, obj, x, sensi_orders, mode):
        raise NotImplementedError(
            "Combining adjoints and hierarchical optimization is not yet "
            "supported.")


def plot_surrogate_fit_reduced(problem, edatas, rdatas, optimal_surrogate):
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy.stats import spearmanr
    num_groups = len(problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING))
    matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(24,8))
    for gr in problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING):
        optimal_scaling_bounds = optimal_surrogate[problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING).index(gr)]['x']
        xs = problem.get_xs_for_group(gr)
        y_sim_all, y_surrogate_all, x_lower, x_upper = get_surrogate_data_reduced(xs, optimal_scaling_bounds, edatas, rdatas, problem)

        ax = plt.subplot(1,num_groups, int(gr))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        plt.plot(y_sim_all, label='simulated data')
        #plt.legend('simulated data')
        plt.errorbar(np.linspace(0, len(y_surrogate_all)-1, len(y_surrogate_all)), y_surrogate_all, fmt='ro', ecolor='black',
                     label="surrogate data and bounds", capsize=9.0, yerr=[np.abs(x_lower - y_surrogate_all),
                                                                           np.abs(x_upper - y_surrogate_all)])
        plt.title('Spearman correlation: %.4f' %spearmanr(y_sim_all,y_surrogate_all)[0], fontsize=18)
        if int(gr) == 2:
            plt.legend(loc='upper right', fontsize=18)

        plt.ylabel('simulation [a.u.]', fontsize=18) #TODO
        plt.xlabel('timepoint [s]', fontsize=18) # TODO
        plt.draw()
    return fig

def plot_surrogate_fit(problem, edatas, rdatas, optimal_surrogate):
    import matplotlib.pyplot as plt
    num_groups = len(problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING))
    fig = plt.figure(figsize=(24, 8))
    for gr in problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING):
        optimal_scaling_bounds = optimal_surrogate[problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING).index(gr)]['x']
        xs = problem.get_xs_for_group(gr)
        y_sim_all, y_surrogate_all, x_lower, x_upper = get_surrogate_data(xs, optimal_scaling_bounds, edatas, rdatas)

        plt.subplot(1,num_groups, int(gr))
        plt.plot(y_sim_all, label='simulated data')
        #plt.legend('simulated data')
        plt.errorbar(np.linspace(0, len(y_surrogate_all)-1, len(y_surrogate_all)), y_surrogate_all, fmt='ro', ecolor='black',
                     label="surrogate data and bounds", capsize=7.5, yerr=[np.abs(x_lower - y_surrogate_all),
                                                                           np.abs(x_upper - y_surrogate_all)])
        if int(gr) == 2:
            plt.legend(loc='upper right', fontsize=18)

        plt.ylabel('simulation [a.u.]', fontsize=18)
        plt.xlabel('timepoint [s]', fontsize=18)
        plt.show()
    return fig


def compute_optimal_scaling_matrix(problem, edatas, rdatas):
    optimal_scalings = compute_optimal_scalings(problem, edatas, rdatas)
    matrix = matrix_like(rdatas, val=1.0)
    for x, opt_s in zip(
			problem.get_xs_for_type(HierarchicalParameter.SCALING),
			optimal_scalings):
        apply_optimal_scaling(x, opt_s, matrix)
    return matrix


def compute_optimal_scalings(problem, edatas, rdatas):
    optimal_scalings = []
    for x in problem.get_xs_for_type(HierarchicalParameter.SCALING):
        s_opt = compute_optimal_scaling(x, edatas, rdatas)
        optimal_scalings.append(s_opt)
    return optimal_scalings


def compute_optimal_scaling(x, edatas, rdatas):
    num = 0.0
    den = 0.0

    for condition_ix, time_ix, observable_ix in x.iterate():
        y = edatas[condition_ix]['observedData'][time_ix, observable_ix]
        h = rdatas[condition_ix]['y'][time_ix, observable_ix]
        sigma = rdatas[condition_ix]['sigmay'][time_ix, observable_ix]

        if np.isnan(y) or np.isnan(h) or np.isnan(sigma):
            continue

        num += y * h / sigma**2
        den += h**2 / sigma**2

    # TODO check for 0.0
    return num / den

def compute_optimal_surrogates_edatas(problem, edatas, rdatas):
    # compute optimal surrogate data and return as list of edatas
    optimal_surrogates = []
    for gr in problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING):
        xs = problem.get_xs_for_group(gr)
        surrogate_opt_results = optimize_surrogate_data(xs, edatas, rdatas, problem)
        write_surrogate_to_edatas(surrogate_opt_results, xs, edatas, rdatas)
        optimal_surrogates.append(surrogate_opt_results)
    return optimal_surrogates

def create_optimal_surrogate_problem(xs, edatas, rdatas, problem):
    # create paramter vector for optimal scaling bounds sorted like the category numbers with lower_bound, upper_bound
    # create constraints for gaps between intervals and bound size
    parameter_length = 2*len(xs)
    lb = -np.inf * np.ones((parameter_length, 1))
    ub = np.inf * np.ones((parameter_length, 1))

    constraints = get_constraints_for_optimization(xs, rdatas, problem)

    obj_surr = lambda x: obj_surrogate_data(xs, x,edatas, rdatas)
    obj = pypesto.Objective(fun=obj_surr, grad=False, hess=False)

    problem = pypesto.Problem(objective=obj, lb=lb, ub=ub, constraints=constraints) # TODO: Implement constraints in pyPESTO
    return problem


def compute_interval_constraints(xs, rdatas, problem):
    # compute constraints on interval size and interval gap size according to Pargett et al. (2014)
    eps = 1e-16
    max_simulation = 0.0
    for x in xs:
        for condition_ix, time_ix, observable_ix in x.iterate():
            max_simulation = np.max([max_simulation, rdatas[condition_ix]['y'][time_ix, observable_ix]])

    min_simulation = np.inf
    for x in xs:
        for condition_ix, time_ix, observable_ix in x.iterate():
            min_simulation = np.min([min_simulation, rdatas[condition_ix]['y'][time_ix, observable_ix]])
    if problem.optimalScalingOptions['intervalConstraints'] == 'max-min':
        interval_range = (max_simulation - min_simulation) / (2*len(xs)+1)
        interval_gap = (max_simulation - min_simulation) / (4*(len(xs)-1)+1)
    elif problem.optimalScalingOptions['intervalConstraints'] == 'max':
        interval_range = max_simulation / (2*len(xs)+1)
        interval_gap = max_simulation / (4*(len(xs)-1)+1)
    if interval_gap < eps:
        interval_gap = eps
    return interval_range, interval_gap

def optimize_surrogate_data(xs, edatas, rdatas, problem):
    from scipy.optimize import minimize
    parameter_length = 2 * len(xs)
    lb = -np.inf * np.ones((parameter_length, 1))
    ub = np.inf * np.ones((parameter_length, 1))

    w = get_weight_for_surrogate2(xs, rdatas, edatas)
    interval_range, _ = compute_interval_constraints(xs, rdatas, problem)
    constraints = get_constraints_for_optimization(xs, rdatas, problem)

    obj_surr = lambda x: obj_surrogate_data(xs, x, edatas, rdatas, problem, w)
    min_all , max_all = get_min_max(xs, rdatas)
    if problem.optimalScalingOptions['multistart']:
        results_all = []
        num_starts = 10
        for startIdx in range(num_starts):
            results = minimize(obj_surr, x0=np.sort(np.random.uniform(min_all, max_all, parameter_length)),
                                   method='SLSQP', constraints=constraints,
                                   options={'maxiter': 2000, 'ftol': 1e-10})
            results_all.append(results)

        fun_all = [results_all[i]['fun'] for i in range(len(results_all))]
        min_index = fun_all.index(np.min(fun_all))
        return results_all[min_index]
    else:
        #results = minimize(obj_surr, np.linspace(min_all, max_all, parameter_length),
        #                   method='trust-constr', constraints=constraints,
        #                   options={'maxiter': 2000})
        results = minimize(obj_surr, x0=np.linspace(0, max_all + interval_range, parameter_length),
                                   method='SLSQP', constraints=constraints,
                                   options={'maxiter': 2000, 'ftol': 1e-10})
        return results

def obj_surrogate_data(xs, optimal_scaling_bounds, edatas, rdatas, problem, w):
    # compute optimal scaling objective function
    obj = 0.0
    y_sim_all = []
    y_surrogate_all = []
    for x in xs:
        x_category =int(x.category)
        x_lower = optimal_scaling_bounds[2*x_category-2]
        x_upper = optimal_scaling_bounds[2*x_category-1]
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            if x_lower > y_sim:
                y_surrogate = x_lower
            elif y_sim > x_upper:
                y_surrogate = x_upper
            else:
                y_surrogate = y_sim
            obj += (y_surrogate - y_sim)**2
            y_sim_all.append(y_sim)
            y_surrogate_all.append(y_surrogate)
    # w = get_weight_for_surrogate(y_surrogate_all, y_sim_all, problem)
    obj = np.divide(obj,w)
    return obj

def get_qualitative_data_indices(problem, edatas):
    qualitative_data_indices_all = []
    for gr in problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING):
        xs = problem.get_xs_for_group(gr)
        for x in xs:
            for condition_ix, time_ix, observable_ix in x.iterate():
                qualitative_data_indices_all.append([condition_ix, time_ix, observable_ix])
    return qualitative_data_indices_all


def get_surrogate_data(xs, optimal_scaling_bounds, edatas, rdatas):
    y_sim_all = []
    y_surrogate_all = []
    x_lower_all = []
    x_upper_all = []
    time_idx = []
    for x in xs:
        x_category =int(x.category)
        x_lower = optimal_scaling_bounds[2*x_category-2]
        x_upper =optimal_scaling_bounds[2*x_category-1]
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            if x_lower > y_sim:
                y_surrogate = x_lower
            elif y_sim > x_upper:
                y_surrogate = x_upper
            else:
                y_surrogate = y_sim
            y_sim_all.append(y_sim)
            y_surrogate_all.append(y_surrogate)
            x_lower_all.append(x_lower)
            x_upper_all.append(x_upper)
            time_idx.append(time_ix)
    return sort_results(np.array(y_sim_all), time_idx), sort_results(np.array(y_surrogate_all), time_idx),\
           sort_results(np.array(x_lower_all), time_idx),\
           sort_results(np.array(x_upper_all), time_idx)

def get_surrogate_data_reduced(xs, optimal_scaling_bounds, edatas, rdatas, problem):
    y_sim_all = []
    y_surrogate_all = []
    x_lower_all = []
    x_upper_all = []
    time_idx = []
    interval_range, interval_gap = compute_interval_constraints(xs, rdatas, problem)
    for x in xs:
        x_category =int(x.category)
        x_upper = optimal_scaling_bounds[x_category-1]
        if x_category==1:
            x_lower = 0
        elif x_category > 1:
            x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            if x_lower > y_sim:
                y_surrogate = x_lower
            elif y_sim > x_upper:
                y_surrogate = x_upper
            else:
                y_surrogate = y_sim
            y_sim_all.append(y_sim)
            y_surrogate_all.append(y_surrogate)
            x_lower_all.append(x_lower)
            x_upper_all.append(x_upper)
            time_idx.append(time_ix)
    return sort_results(np.array(y_sim_all), time_idx), sort_results(np.array(y_surrogate_all), time_idx),\
           sort_results(np.array(x_lower_all), time_idx),\
           sort_results(np.array(x_upper_all), time_idx)

def sort_results(unsorted_results, index_list):
    sorted_results = np.zeros(shape=np.shape(unsorted_results))
    for i in range(len(index_list)):
        sorted_results[index_list[i]] = unsorted_results[i]
    return sorted_results

def get_constraints_for_optimization(xs, rdatas, problem):
    num_categories = len(xs)
    interval_range, interval_gap = compute_interval_constraints(xs, rdatas, problem)
    # A*x-b>=0, with A[i,i] == -1, A[i,i+1] == 1, A[i,j] ==0 otherwise, b= [interval_range, interval_gap, interval_range, interval_gap, ...]
    '''
    A = np.diag(-np.ones(2*num_categories)) + np.diag(np.ones(2*num_categories-1),1)
    A = A[:-1, :]
    b = np.empty((2*num_categories-1,))
    b[0::2] = interval_range
    b[1::2] = interval_gap
    '''
    A = np.diag(-np.ones(2*num_categories),-1)+ np.diag(np.ones(2*num_categories+1))
    A = A[:-1, :]
    A = A[:, :-1]
    b = np.empty((2*num_categories,))
    b[0] = 0
    b[1::2] = interval_range
    b[2::2] = interval_gap
    ineq_cons = {'type': 'ineq','fun': lambda x: A.dot(x)-b}
    #from scipy.optimize import LinearConstraint
    #linear_constraint = LinearConstraint(A, b, [np.inf]*len(b))
    return ineq_cons
    #return linear_constraint

def compute_optimal_surrogates_edatas_reduced(problem, edatas, rdatas):
    # compute optimal surrogate data and return as list of edatas
    optimal_surrogates = []
    for gr in problem.get_groups_for_xs(HierarchicalParameter.OPTIMALSCALING):
        xs = problem.get_xs_for_group(gr)
        surrogate_opt_results = optimize_surrogate_data_reduced(xs, edatas, rdatas, problem)
        write_surrogate_to_edatas_reduced(surrogate_opt_results, xs, edatas, rdatas, problem)
        optimal_surrogates.append(surrogate_opt_results)
    return optimal_surrogates

def optimize_surrogate_data_reduced(xs, edatas, rdatas, problem):
    from scipy.optimize import minimize
    from scipy.optimize import Bounds
    parameter_length = len(xs)
    lb = -np.inf * np.ones((parameter_length, 1))
    ub = np.inf * np.ones((parameter_length, 1))

    interval_range, interval_gap = compute_interval_constraints(xs, rdatas, problem)
    w = get_weight_for_surrogate2(xs, rdatas, edatas)

    min_all, max_all = get_min_max(xs, rdatas)

    if problem.optimalScalingOptions['reparameterized']:
        obj_surr = lambda x: obj_surrogate_data_reduced_reparameterized(xs, x, edatas, rdatas, interval_gap, interval_range, problem, w)
        if problem.optimalScalingOptions['multistart']:
            results_all = []
            num_starts = 100
            for startIdx in range(num_starts):
                results = minimize(obj_surr, x0=np.random.uniform(min_all, max_all, parameter_length),
                                   method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-10})
                results_all.append(results)
            fun_all = [results_all[i]['fun'] for i in range(len(results_all))]
            min_index = fun_all.index(np.min(fun_all))
            return results_all
        else:
            x0 = y2xi(np.linspace(np.max([min_all, interval_range]), max_all + interval_range, parameter_length), xs, interval_gap, interval_range)
            parameter_bounds = Bounds([0] * parameter_length, [max_all] * parameter_length)
            results = minimize(obj_surr, x0=x0,
                                   method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-10}, bounds=parameter_bounds)
        return results
    else:
        constraints = get_constraints_for_optimization_reduced(xs, rdatas, problem)
        obj_surr = lambda x: obj_surrogate_data_reduced(xs, x, edatas, rdatas, interval_gap, problem, w)
        if problem.optimalScalingOptions['multistart']:
            results_all = []
            num_starts = 100
            for startIdx in range(num_starts):
                results = minimize(obj_surr, x0=np.sort(np.random.uniform(min_all, max_all, parameter_length)),
                                   method='SLSQP', constraints=constraints,
                                   options={'maxiter': 2000, 'ftol': 1e-10})
                results_all.append(results)
            fun_all = [results_all[i]['fun'] for i in range(len(results_all))]
            min_index = fun_all.index(np.min(fun_all))
            return results_all
        else:
            results = minimize(obj_surr, x0=np.linspace(np.max([min_all, interval_range]), max_all + interval_range, parameter_length),
                               method='SLSQP', constraints=constraints,
                               options={'maxiter': 2000, 'ftol': 1e-10})
            #results = minimize(obj_surr, x0=np.linspace(min_all, max_all, parameter_length), method='trust-constr',
            #                   constraints=constraints, options={'maxiter': 2000})

        return results

def get_min_max(xs, rdatas):
    y_sim_all = []
    for x in xs:
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            y_sim_all.append(y_sim)
    return np.min(y_sim_all), np.max(y_sim_all)

def obj_surrogate_data_reduced(xs, optimal_scaling_bounds, edatas, rdatas, interval_gap, problem, w):
    # compute optimal scaling objective function
    obj = 0.0
    y_sim_all = []
    y_surrogate_all = []
    for x in xs:
        x_category =int(x.category)
        x_upper = optimal_scaling_bounds[x_category-1]
        if x_category==1:
            x_lower = 0
        elif x_category > 1:
            x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            if x_lower > y_sim:
                y_surrogate = x_lower
            elif y_sim > x_upper:
                y_surrogate = x_upper
            else:
                y_surrogate = y_sim
            obj += (y_surrogate - y_sim)**2
            # obj += np.abs(y_surrogate - y_sim)
            y_sim_all.append(y_sim)
            y_surrogate_all.append(y_surrogate)
    # w = get_weight_for_surrogate(y_surrogate_all, y_sim_all, problem)
    obj = np.divide(obj,w)
    return obj

def obj_surrogate_data_reduced_reparameterized(xs, optimal_scaling_bounds_reparameterized, edatas, rdatas, interval_gap, interval_range, problem, w):
    # compute optimal scaling objective function
    obj = 0.0
    y_sim_all = []
    y_surrogate_all = []
    optimal_scaling_bounds = xi2y(optimal_scaling_bounds_reparameterized, xs,interval_gap, interval_range)
    for x in xs:
        x_category =int(x.category)
        x_upper = optimal_scaling_bounds[x_category-1]
        if x_category==1:
            x_lower = 0
        elif x_category > 1:
            x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            if x_lower > y_sim:
                y_surrogate = x_lower
            elif y_sim > x_upper:
                y_surrogate = x_upper
            else:
                y_surrogate = y_sim
            obj += (y_surrogate - y_sim)**2
            y_sim_all.append(y_sim)
            y_surrogate_all.append(y_surrogate)
    # w = get_weight_for_surrogate(y_surrogate_all, y_sim_all, problem)
    obj = np.divide(obj,w)
    return obj

def xi2y(optimal_scaling_bounds_reparameterized, xs, interval_gap, interval_range):
    #TODO: optimal scaling parameters in parameter sheet have to be ordered at the moment
    optimal_scaling_bounds = np.full(shape=(np.shape(optimal_scaling_bounds_reparameterized)), fill_value=np.nan)
    for x in xs:
        x_category = int(x.category)
        if x_category == 1:
            #optimal_scaling_bounds[x_category - 1] = interval_range + np.exp(optimal_scaling_bounds_reparameterized[x_category-1])
            optimal_scaling_bounds[x_category - 1] = interval_range + optimal_scaling_bounds_reparameterized[x_category - 1]
        else:
            #optimal_scaling_bounds[x_category - 1] = np.exp(optimal_scaling_bounds_reparameterized[x_category-1]) + \
            #                                         interval_gap + interval_range + optimal_scaling_bounds[x_category-2]
            optimal_scaling_bounds[x_category - 1] = optimal_scaling_bounds_reparameterized[x_category-1] + \
                                                     interval_gap + interval_range + optimal_scaling_bounds[x_category-2]
    return optimal_scaling_bounds


def y2xi(optimal_scaling_bounds, xs, interval_gap, interval_range):
    optimal_scaling_bounds_reparameterized = np.full(shape=(np.shape(optimal_scaling_bounds)), fill_value=np.nan)

    for x in xs:
        x_category = int(x.category)
        if x_category == 1:
            #optimal_scaling_bounds_reparameterized[x_category - 1] = np.log(optimal_scaling_bounds[x_category - 1]
            #                                                                - interval_range)
            optimal_scaling_bounds_reparameterized[x_category - 1] = optimal_scaling_bounds[x_category - 1] \
                                                                     - interval_range
        else:
            #optimal_scaling_bounds_reparameterized[x_category - 1] = np.log(optimal_scaling_bounds[x_category - 1]
            #                                                                - optimal_scaling_bounds[x_category - 2]
            #                                                                - interval_gap - interval_range)
            optimal_scaling_bounds_reparameterized[x_category - 1] = optimal_scaling_bounds[x_category - 1]\
                                                                     - optimal_scaling_bounds[x_category - 2]\
                                                                     - interval_gap - interval_range

    return optimal_scaling_bounds_reparameterized


def write_surrogate_to_edatas_reduced(surrogate_opt_results, xs,  edatas, rdatas, problem):
    optimal_scaling_bounds = surrogate_opt_results['x']
    y_sim_all = []
    y_surrogate_all = []
    interval_range, interval_gap = compute_interval_constraints(xs, rdatas, problem)
    for x in xs:
        x_category =int(x.category)
        x_upper = optimal_scaling_bounds[x_category-1]
        if x_category==1:
            x_lower = 0
        elif x_category > 1:
            x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            if x_lower > y_sim:
                y_surrogate = x_lower
            elif y_sim > x_upper:
                y_surrogate = x_upper
            else:
                y_surrogate = y_sim
            edatas[condition_ix]['observedData'][time_ix, observable_ix] = y_surrogate
    return True

def get_constraints_for_optimization_reduced(xs, rdatas, problem):
    #TODO
    num_categories = len(xs)
    interval_range, interval_gap = compute_interval_constraints(xs, rdatas, problem)
    #A = np.diag(-np.ones(num_categories)) + np.diag(np.ones(num_categories-1),1)
    #A = A[:-1, :]

    A = np.diag(-np.ones(num_categories), -1) + np.diag(np.ones(num_categories + 1))
    A = A[:-1, :-1]
    b = np.empty((num_categories,))
    b[0] = interval_range
    b[1:] = interval_range + interval_gap
    ineq_cons = {'type': 'ineq','fun': lambda x: A.dot(x)-b}

    #from scipy.optimize import LinearConstraint
    #linear_constraint = LinearConstraint(A, b, [np.inf]*len(b))
    return ineq_cons
    #return linear_constraint

def get_weight_for_surrogate(y_surrogate_all,y_sim_all, problem):
    #TODO: Good choice of weights
    w=0.0
    eps = 1e-10
    v_net = 0
    for idx in range(len(y_sim_all)-1):
        v_net += np.abs(y_sim_all[idx+1] - y_sim_all[idx])
    if problem.optimalScalingOptions['weightsWithSurrogate']:
        w = 0.5*np.sum(np.abs(y_sim_all)+ np.abs(y_surrogate_all)) + v_net + eps# + np.abs(y_surrogate_all))
    else:
        w = 0.5 * np.sum(np.abs(y_sim_all)) + v_net + eps
    return w**2

def get_weight_for_surrogate2(xs, rdatas, edatas):
    y_sim_all = []
    for x in xs:
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            y_sim_all.append(y_sim)
    w=0.0
    eps = 1e-10
    v_net = 0
    for idx in range(len(y_sim_all)-1):
        v_net += np.abs(y_sim_all[idx+1] - y_sim_all[idx])
    #w = 0.5*np.sum(np.abs(y_sim_all)+ np.abs(y_surrogate_all)) + v_net + eps# + np.abs(y_surrogate_all))
    w = 0.5 * np.sum(np.abs(y_sim_all)) + v_net + eps
    return w**2


def write_surrogate_to_edatas(surrogate_opt_results, xs,  edatas, rdatas):
    optimal_scaling_bounds = surrogate_opt_results['x']
    for x in xs:
        x_category =int(x.category)
        x_lower = optimal_scaling_bounds[2*x_category-2]
        x_upper = optimal_scaling_bounds[2*x_category-1]
        for condition_ix, time_ix, observable_ix in x.iterate():
            y_sim = rdatas[condition_ix]['y'][time_ix, observable_ix]
            if x_lower > y_sim:
                y_surrogate = x_lower
            elif y_sim > x_upper:
                y_surrogate = x_upper
            else:
                y_surrogate = y_sim
            edatas[condition_ix]['observedData'][time_ix, observable_ix] = y_surrogate
    return True

def matrix_like(rdatas, val):
    matrix = []
    for rdata in rdatas:
        matrix.append(np.full(rdata['y'].shape, fill_value=val))
    return matrix


def apply_optimal_scalings(problem, optimal_scalings, matrix):
    for x, opt_s in zip(problem.get_xs_for_type(HierarchicalParameter.SCALING), optimal_scalings):
        apply_optimal_scaling(x, opt_s, matrix)


def apply_optimal_scaling(x, opt_s, matrix):
    for condition_ix, time_ix, observable_ix in x.iterate():
        matrix[condition_ix][time_ix, observable_ix] = opt_s


def compute_nllh(edatas, rdatas, optimal_scalings):
    nllh = 0.0
    for edata, rdata, optimal_s in zip(edatas, rdatas, optimal_scalings):
        #print(edata['observedData'], optimal_s, rdata['y'], rdata['sigmay'])
        nllh += 0.5 * np.nansum(np.log(2*np.pi*rdata['sigmay']**2))
        nllh += 0.5 * np.nansum((edata['observedData'] - optimal_s * rdata['y'])**2 / rdata['sigmay']**2)
    #print(nllh)
    return nllh

def compute_WLS(optimal_surrogate, problem, edatas, rdatas):
    qualitative_data_indices = get_qualitative_data_indices(problem, edatas)
    WLS_qualitative = np.sum([optimal_surrogate[i]['fun'] for i in range(len(optimal_surrogate))])

    WLS_quantitative = 0.0
    for condition_ix in range(len(edatas)):
        for observable_ix in range(np.shape(edatas[condition_ix]['observedData'])[1]):
            y_sim_obs = []
            y_mes_obs = []
            for time_ix in range(np.shape(edatas[condition_ix]['observedData'])[0]):
                if not [condition_ix, time_ix, observable_ix] in qualitative_data_indices:
                    y_sim_obs.append(rdatas[condition_ix]['y'][time_ix, observable_ix])
                    y_mes_obs.append(edatas[condition_ix]['observedData'][time_ix, observable_ix])
            squared_res = np.nansum((np.array(y_sim_obs) - np.array(y_mes_obs))**2)
            v_net = 0
            eps = 1e-10
            for idx in range(len(y_sim_obs) - 1):
                v_net += np.abs(y_sim_obs[idx + 1] - y_sim_obs[idx])
            w = v_net + 0.5 * np.sum(np.abs(y_sim_obs)) + eps
            w = w ** 2
            WLS_quantitative +=  np.divide(squared_res, w)
    # print('quantitative obj function: ' + str(np.sqrt(WLS_quantitative)))
    WLS = np.sqrt(WLS_qualitative) + np.sqrt(WLS_quantitative)
    return WLS

def get_weights_for_LS(edata, rdata):
    num_observables = edata['observedData'].shape[1]
    num_timepoints = edata['observedData'].shape[0]
    eps = 1e-10

    w = np.zeros((num_observables,))
    for observable_ix in range(num_observables):
        v_net = 0
        for time_ix in range(num_timepoints-1):
            v_net += np.abs(rdata['y'][time_ix+1,observable_ix] -rdata['y'][time_ix,observable_ix])
        w[observable_ix] = eps + v_net + 0.5*np.sum(np.abs(rdata['y'][:,observable_ix]))
    return w**2


def compute_snllh(edatas, rdatas, optimal_scalings, x_ids, mapping_par_opt_to_par_sim, dim):
    snllh = np.zeros(dim)
    for condition_ix, (edata, rdata, optimal_s) in \
            enumerate(zip(edatas, rdatas, optimal_scalings)):
        # sy is n_obs x n_par x n_time
        sy_ = rdata['sy']
        sy = np.full((sy_.shape[1],sy_.shape[0],sy_.shape[2]), np.nan)
        for i in range(sy_.shape[1]):
            sy[i] = sy_[:,i,:]
        #print(sy)
        val = np.nansum((edata['observedData'] - optimal_s * rdata['y']) \
                / rdata['sigmay']**2 * optimal_s * sy, axis=(1, 2))
        #print(val)
        add_sim_grad_to_opt_grad(
            x_ids,
            mapping_par_opt_to_par_sim[condition_ix],
            val,
            snllh,
            coefficient=-1.0)
    return snllh
