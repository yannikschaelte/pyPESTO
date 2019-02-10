import sqlalchemy.types as types
from sqlalchemy import (
    Column, ForeignKey,
    Integer, Float, String, LargeBinary)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from .bytes_storage import from_bytes, to_bytes


Base = declarative_base()


class BytesStorage(types.TypeDecorator):
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        return to_bytes(value)

    def process_result_value(self, value, dialect):
        return from_bytes(value)


class Result(Base):
    __tablename__ = 'result'
    id = Column(Integer, primary_key=True)
    problem_info = relationship(
        'ProblemInfo', uselist=False, back_populates='result')
    optimize_result = relationship(
        'OptimizeResult', uselist=False, back_populates='result')


class ProblemInfo(Base):
    __tablename__ = 'problem_info'
    id = Column(Integer, primary_key=True)

    result_id = Column(Integer, ForeignKey('result.id'))
    result = relationship('Result', back_populates='problem_info')

    objective_info = Column(String(5000))
    lb = Column(BytesStorage)
    ub = Column(BytesStorage)
    dim = Column(Integer)
    lb_full = Column(BytesStorage)
    ub_full = Column(BytesStorage)
    dim_full = Column(Integer)
    x_fixed_indices = Column(BytesStorage)
    x_fixed_vals = Column(BytesStorage)
    x_free_indices = Column(BytesStorage)
    x_guesses = Column(BytesStorage)
    x_names = Column(BytesStorage)


class OptimizeResult(Base):
    __tablename__ = 'optimize_result'
    id = Column(Integer, primary_key=True)

    result_id = Column(Integer, ForeignKey('result.id'))
    result = relationship('Result', back_populates='optimize_result')

    optimizer_results = relationship('OptimizerResult')


class OptimizerResult(Base):
    __tablename__ = 'optimizer_result'
    id = Column(Integer, primary_key=True)

    optimize_result_id = Column(Integer, ForeignKey('optimize_result.id'))

    x = Column(BytesStorage)
    fval = Column(Float)
    grad = Column(BytesStorage)
    hess = Column(BytesStorage)
    n_fval = Column(Integer)
    n_grad = Column(Integer)
    n_hess = Column(Integer)
    n_res = Column(Integer)
    n_sres = Column(Integer)
    x0 = Column(BytesStorage)
    fval0 = Column(Float)
    # trace = Column(BytesStorage)
    exitflag = Column(Integer)
    time = Column(Float)
    message = Column(String(5000))
