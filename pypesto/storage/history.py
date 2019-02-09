from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, subqueryload

from .db_model import (Base, Result, OptimizeResult, OptimizerResult)
from ..result import Result as PyResult
from ..optimize import OptimizerResult as PyOptimizerResult


def with_session(f):
    @wraps(f)
    def f_wrapper(self: "History", *args, **kwargs):
        no_session = self._session is None and self._engine is None
        if no_session:
            self._make_session()
        res = f(self, *args, **kwargs)
        if no_session:
            self._close_session()
        return res
    return f_wrapper


class History:

    DB_TIMEOUT = 120

    def __init__(self, db_identifier: str):
        self.db_identifier = db_identifier

        self._session = None
        self._engine = None

        self.id = self._pre_calculate_id()

    @property
    def in_memory(self):
        return (self._engine is not None
                and str(self._engine.url) == "sqlite://")

    def _make_session(self):
        engine = create_engine(self.db_identifier,
                               connect_args={'timeout': self.DB_TIMEOUT})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        self._session = session
        self._engine = engine
        return session

    def _close_session(self):
        # don't close in-memory database
        if self.in_memory:
            return

        self._session.close()
        self._engine.dispose()
        self._session = None
        self._engine = None

    @with_session
    def _pre_calculate_id(self):
        results = self._session.query(Result).all()
        if len(results) == 1:
            return results[-1].id
        return None

    @with_session
    def save_result(self, py_result):
        result = Result()
        optimize_result = OptimizeResult(result=result)

        for py_optimizer_result in py_result.optimize_result.as_list():
            optimizer_result = OptimizerResult(
                x=py_optimizer_result.x,
                fval=py_optimizer_result.fval,
                grad=py_optimizer_result.grad,
                hess=py_optimizer_result.hess,
                n_fval=py_optimizer_result.n_fval,
                n_grad=py_optimizer_result.n_grad,
                n_hess=py_optimizer_result.n_hess,
                n_res=py_optimizer_result.n_res,
                n_sres=py_optimizer_result.n_sres,
                x0=py_optimizer_result.x0,
                fval0=py_optimizer_result.fval0,
                exitflag=py_optimizer_result.exitflag,
                time=py_optimizer_result.time,
                message=py_optimizer_result.message)
            optimize_result.optimizer_results.append(optimizer_result)

        self._session.add(result)
        self._session.commit()
        self.id = result.id

    @with_session
    def load_result(self):
        result = (self._session.query(Result)
                  .options(
                      subqueryload(Result.optimize_result)
                      .subqueryload(OptimizeResult.optimizer_results))
                  .filter(Result.id == self.id)
                  .one())

        py_result = PyResult()

        for optimizer_result in result.optimize_result.optimizer_results:

            py_optimizer_result = PyOptimizerResult(
                x=optimizer_result.x,
                fval=optimizer_result.fval,
                grad=optimizer_result.grad,
                hess=optimizer_result.hess,
                n_fval=optimizer_result.n_fval,
                n_grad=optimizer_result.n_grad,
                n_hess=optimizer_result.n_hess,
                n_res=optimizer_result.n_res,
                n_sres=optimizer_result.n_sres,
                x0=optimizer_result.x0,
                fval0=optimizer_result.fval0,
                exitflag=optimizer_result.exitflag,
                time=optimizer_result.time,
                message=optimizer_result.message)

            py_result.optimize_result.append(py_optimizer_result)

        return py_result
