API Reference
=============

The AMEP **A**\ ctive **M**\ atter **E**\ valuation **P**\ ackage
Python library is a powerful tool for analyzing data
from molecular-dynamics (MD), Brownian-dynamics (BD),
and continuum simulations.
It comprises various methods to analyze structural and
dynamical properties of condensed matter systems in
general and active matter systems in particular.
AMEP is exclusively built on Python, and therefore,
it is easy to modify and allows
to easily add user-defined methods.
AMEP provides an efficient data format for saving both
simulation data and analysis results based on the HDF5
file format.
To be fast and usable on modern
HPC (**H**\ igh **P**\ erformance **C**\ omputing) hardware,
the methods are optimized to run also in parallel.)

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst

    amep.base
    amep.cluster
    amep.continuum
    amep.evaluate
    amep.functions
    amep.load
    amep.order
    amep.pbc
    amep.plot
    amep.reader
    amep.spatialcor
    amep.statistics
    amep.thermo
    amep.timecor
    amep.trajectory
    amep.utils
