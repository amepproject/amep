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
"""Test units for the continuum class.

Including it's main class, readers and methods."""
import unittest
from zipfile import ZipFile
from pathlib import Path
from random import random
from requests import get
import numpy as np
from matplotlib import use
from amep.load import traj
from amep.evaluate import ClusterGrowth, ClusterSizeDist
use("Agg")
DATA_DIR = Path("../examples/data/")

FIELD_DIR: Path = DATA_DIR/"continuum"
PARTICLE_DIR: Path = DATA_DIR/"lammps"
PLOT_DIR: Path = DATA_DIR/"plots"


class TestEvaluateMethods(unittest.TestCase):
    """A test case for all field and continuum data methods"""
    @classmethod
    def setUpClass(cls):
        """Set up needed data"""
        cls.field_traj = traj(DATA_DIR/"continuum.h5amep")
        cls.particle_traj = traj(DATA_DIR/"lammps.h5amep")

    def test_cluster_growth(self):
        """Test the cluster growth methods.

        Due to their weighting they have a given order they take
        for arbitrary trajectories.
        This order gets tested here.
        Since ClusterGrowth calls ClusterSizeDist we don't
        have to check it separately.
        """
        self.assertTrue((ClusterGrowth(self.field_traj, scale=1.5,
                                       cutoff=0.8,
                                       ftype="c", mode="mean").frames.sum() <=
                        ClusterGrowth(self.field_traj, scale=1.5,
                                      cutoff=0.8, ftype="c",
                                      mode="weighted mean").frames.sum()))
        self.assertTrue((ClusterGrowth(self.field_traj,
                                       ftype="c",
                                       mode="largest").frames >= 0).all())

    def test_energy_methods(self):
        """Test the energy methods.
        TO BE IMPLEMENTED
        """
        pass

    def test_function(self):
        """Test arbitray function evaluation.
        TO BE IMPLEMENTED
        """
        pass

    def test_order_evaluations(self):
        """Test order parameter evaluation.
        TO BE IMPLEMENTED
        """
        pass

    def test_correlation(self):
        """Test order parameter evaluation.
        TO BE IMPLEMENTED
        """
        pass

    def test_transforms(self):
        """Test order parameter evaluation.
        TO BE IMPLEMENTED
        """
        pass
