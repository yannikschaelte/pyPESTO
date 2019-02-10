import logging

from .base import Engine


logger = logging.getLogger(__name__)


class SingleCoreEngine(Engine):
    """
    Dummy engine for sequential execution on one core. Note that the
    objective itself may be multithreaded.
    """

    def __init__(self):
        pass

    def execute(self, tasks):
        """
        Execute all tasks in a simple for loop sequentially.
        """
        results = []
        for j_task, task in enumerate(tasks):
            logger.info(f"Starting task {j_task + 1} of {len(tasks)}.")
            results.append(task.execute())

        return results
