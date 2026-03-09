# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) 2023-2024 Lukas Hecht and the AMEP development team.
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
Test units for the amep.spatial module.
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import numpy as np
import unittest
import amep


# =============================================================================
# MAIN TIMECOR TEST
# =============================================================================
class TestSpatialcor(unittest.TestCase):
    """Test case for spatial correlation functions.
    """

    @classmethod
    def setUpClass(cls):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        # initialize random number generator with seed 0
        rng = np.random.default_rng(0)
        numberofparticles = 100
        # generate random coordinates
        cls.coords = np.zeros((numberofparticles, 3))
        cls.coords[:, :2] = rng.uniform(
            low=-10,
            high=10,
            size=(numberofparticles, 2)
        )
        # generate random orientations
        cls.orientations = np.zeros((numberofparticles, 3))
        cls.orientations[:, :2] = rng.uniform(
            low=-1,
            high=1,
            size=(numberofparticles, 2)
        )
        cls.orientations=cls.orientations/np.linalg.norm(cls.orientations, axis=1)[:,None]
        # create box
        cls.box = np.array(
            [[-10, 10],
             [-10, 10],
             [-0.5, 0.5]]
        )

    def test_rdf(self):
        """Calculate and compare radial distribution functions.

        Calculate rdf by different means and compare the results."""
        # calculate rdf with mode diff
        rdf_diff, _ = amep.spatialcor.rdf(
            self.coords,
            self.box,
            nbins=50,
            pbc=True,
            rmax=5.0,
            mode='diff'
        )
        # calculate rdf with mode kdtree
        rdf_kdtree, _ = amep.spatialcor.rdf(
            self.coords,
            self.box,
            nbins=50,
            pbc=True,
            rmax=5.0,
            mode='kdtree'
        )
        # compare results
        compare = rdf_diff.round(3) == rdf_kdtree.round(3)
        self.assertTrue(
            compare.all(),
            'The rdf calculation with mode diff and mode kdtree are not the '
            'same. Got a summed difference of '
            f'{np.abs(rdf_diff-rdf_kdtree).sum()}'
        )


    def test_pcf(self):
        """Calculate spatial correlation functions."""
        # calculate angular pcf with respect to particle orientations
        grt, r, t = amep.spatialcor.pcf_angle(
                self.coords,
                self.box,
                psi = None,
                e=self.orientations,
                nabins=10,
                rmax=3,
                ndbins=20
            )
        # calculate angular pcf with respect to x-axis (=default e)
        grt_x, r_x, t_x = amep.spatialcor.pcf_angle(
                self.coords,
                self.box,
                nabins=10,
                rmax=3,
                ndbins=20
            )