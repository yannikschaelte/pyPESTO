Release notes
=============


0.0 series
..........


0.0.7 (2019-03-21)
------------------

* Support noise models in Petab and Amici.
* Minor Petab update bug fixes.


0.0.6 (2019-03-13)
------------------

* Several minor error fixes, in particular on tests and steady state.


0.0.5 (2019-03-11)
------------------

* Introduce AggregatedObjective to use multiple objectives at once.
* Estimate steady state in AmiciObjective.
* Check amici model build version in PetabImporter.
* Use Amici multithreading in AmiciObjective.
* Allow to sort multistarts by initial value.
* Show usage of visualization routines in notebooks.
* Various fixes, in particular to visualization.


0.0.4 (2019-02-25)
------------------

* Implement multi process parallelization engine for optimization.
* Introduce PrePostProcessor to more reliably handle pre- and
  post-processing.
* Fix problems with simulating for multiple conditions.
* Add more visualization routines and options for those (colors, 
  reference points, plotting of lists of result obejcts)


0.0.3 (2019-01-30)
------------------

* Import amici models and the petab data format automatically using
  pypesto.PetabImporter.
* Basic profiling routines.


0.0.2 (2018-10-18)
------------------

* Fix parameter values
* Record trace of function values
* Amici objective to directly handle amici models


0.0.1 (2018-07-25)
------------------

* Basic framework and implementation of the optimization
