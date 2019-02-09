import pypesto
import pypesto.storage.history
import numpy as np
import scipy as sp
import sys


sys.path.insert(0, '/home/yannik/pypesto')


objective = pypesto.Objective(fun=sp.optimize.rosen,
                              grad=sp.optimize.rosen_der,
                              hess=sp.optimize.rosen_hess)

dim_full = 10
lb = -2 * np.ones((dim_full, 1))
ub = 2 * np.ones((dim_full, 1))

problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

optimizer = pypesto.ScipyOptimizer()
n_starts = 20

result = pypesto.minimize(problem=problem, optimizer=optimizer, n_starts=n_starts)

history = pypesto.storage.history.History("sqlite:///db.db")
history.save_result(result)

result2 = history.load_result()


print(result.optimize_result.as_dataframe(['fval', 'grad']))
print(result2.optimize_result.as_dataframe(['fval', 'grad']))
