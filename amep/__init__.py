# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) 2023-2025 Lukas Hecht and the AMEP development team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Contact: Lukas Hecht (lukas.hecht@pkm.tu-darmstadt.de)
# =============================================================================
"""
AMEP
====

The AMEP (**A**ctive **M**atter **E**valuation **P**ackage) Python library is
a powerful tool for analyzing data from molecular-dynamics (MD),
Brownian-dynamics (BD), and continuum simulations. It comprises various methods
to analyze structural and dynamical properties of condensed matter systems in
general and active matter systems in particular. AMEP is exclusively built on
Python, and therefore, it is easy to modify and allows to easily add
user-defined methods. AMEP provides an efficient data format for saving both
simulation data and analysis results based on the HDF5 file format. To be fast
and usable on modern HPC (**H**igh **P**erformance **C**omputing) hardware, the
methods are optimized to run also in parallel.

"""
from . import utils
from . import functions
from . import pbc
from . import load
from . import spatialcor
from . import continuum
from . import order
from . import statistics
from . import cluster
from . import thermo
from . import evaluate
from . import plot
from . import timecor
from . import trajectory
from . import reader
from . import base
from ._version import __version__

__author__  = 'Lukas Hecht'
__email__   = 'lukas.hecht@pkm.tu-darmstadt.de'

__all__ = [
    'utils',
    'functions',
    'pbc',
    'load',
    'spatialcor',
    'continuum',
    'order',
    'statistics',
    'cluster',
    'thermo',
    'evaluate',
    'plot',
    'timecor',
    'trajectory',
    'reader',
    'base',
]