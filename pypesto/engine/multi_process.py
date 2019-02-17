from multiprocessing import Pool
import cloudpickle as pickle
import os
import logging

from .base import Engine


logger = logging.getLogger(__name__)


def work(pickled_task):
    task = pickle.loads(pickled_task)
    return task.execute()


class MultiProcessEngine(Engine):
    """
    Parallelize the task execution using the `multiprocessing.Pool`
    environment.

    Attributes
    ----------

    n_procs: int, optional
        The number of cores to use. Defaults to the number of cpus available
        on the system according to ``os.cpu_count()``.
        The effectively used number of cores will be the minimum of n_procs
        and the number of tasks submitted (and the number of CPUs available).
    """

    def __init__(self, n_procs: int = None):
        if n_procs is None:
            n_procs = os.cpu_count()
            logger.warning(
                f"Engine set up to use up to {n_procs} processes in total. "
                f"The number was automatically determined and might not be "
                f"appropriate on some systems.")
        self.n_procs = n_procs

    def execute(self, tasks):
        n_tasks = len(tasks)

        pickled_tasks = [pickle.dumps(task) for task in tasks]

        n_procs = min(self.n_procs, n_tasks)
        logger.info(f"Performing parallel task execution on {n_procs} "
                    f"processes.")

        with Pool(processes=n_procs) as pool:
            results = pool.map(work, pickled_tasks)

        return results
