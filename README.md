# pyPESTO - Parameter EStimation TOolbox for python

**pyPESTO** is a widely applicable and highly customizable toolbox for
parameter estimation.

[![PyPI version](https://badge.fury.io/py/pypesto.svg)](https://badge.fury.io/py/pypesto)
[![Build Status](https://travis-ci.com/ICB-DCM/pyPESTO.svg?branch=master)](https://travis-ci.com/ICB-DCM/pyPESTO)
[![Code coverage](https://codecov.io/gh/ICB-DCM/pyPESTO/branch/master/graph/badge.svg)](https://codecov.io/gh/ICB-DCM/pyPESTO) [![Code quality](https://api.codacy.com/project/badge/Grade/134432ddad0e464b8494587ff370f661)](https://www.codacy.com/app/dweindl/pyPESTO?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ICB-DCM/pyPESTO&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/pypesto/badge/?version=latest)](https://pypesto.readthedocs.io)

## Feature overview

pyPESTO features include:

* Multi-start local optimization
* Profile computation
* Result visualization
* Interface to [AMICI](https://github.com/ICB-DCM/AMICI/) for efficient simulation and sensitivity analysis of ordinary differential equation (ODE) models
* Parameter estimation pipeline for systems biology problems specified in [SBML](http://sbml.org/) and [PEtab](https://github.com/ICB-DCM/PEtab)
* Parameter estimation with qualitative data as described in [Schmiester et al. (2019)](https://www.biorxiv.org/content/10.1101/848648v1).

## Quick install

The simplest way to install **pyPESTO** is via pip:

```shell
pip3 install pypesto
```

More information is available here:
https://pypesto.readthedocs.io/en/latest/install.html

## Documentation

The documentation is hosted on readthedocs.io:
<https://pypesto.readthedocs.io>

## Examples

Multiple use cases are discussed in the documentation. In particular, there are
jupyter notebooks in the [doc/example](doc/example) directory.

## Contributing

We are happy about any contributions. For more information on how to contribute
to pyPESTO check out
<https://pypesto.readthedocs.io/en/latest/contribute.html>

## References

[**PESTO**](https://github.com/ICB-DCM/PESTO/):
Parameter estimation toolbox for MATLAB. Development is discontinued, but PESTO
comes with additional features waiting to be ported to pyPESTO.
