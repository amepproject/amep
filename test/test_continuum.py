# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) 2024 Lukas Hecht and the AMEP development team.
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
"""Test units for the continuum module.

Including it's main class, readers and methods."""
import unittest
from pathlib import Path
from matplotlib import use
from amep.load import traj
from amep import plot
from amep.continuum import identify_clusters
use("Agg")
DATA_DIR = Path("../examples/data/")
N_LINES = 100
COORDS = ("X", "Y", "Z")
FIELDS = ("c", "rho", "alpha", "omega")
TIMES = (0, 0.4, 0.5, 0.9, 2.3)

FIELD_DIR: Path = DATA_DIR/"continuum"


class TestFieldMethods(unittest.TestCase):
    """A test case for all field and continuum data methods"""

    def test_coords_to_density(self):
        """Test coords to density.
            TO BE IMPLEMENTED
        """
        pass

    def test_density_estimators(self):
        """Test density estimators.
            TO BE IMPLEMENTED
        """
        pass

    def test_sf2d(self):
        """Test 2d structure factor.
            TO BE IMPLEMENTED
        """
        pass

    def test_cluster_methods(self):
        """Test the cluster detection with both methods.
            TO BE IMPLEMENTED when I know how to check cluster
            detection in a sane way.
            Maybe combine it with coords to density.
        """
        pass
