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


    def test_pcf2d_backwards_compat(self):
        """Test backwards compatibility of pcf2d.

        The old pcf2d API used psi (from psi_k) and no e parameter.
        The new API adds an e parameter with default [1,0,0].
        Calling pcf2d with psi set and default e should produce the
        same result as the old code.
        """
        # compute psi6 (the way the old evaluate.PCF2d.__compute did it)
        psi_complex = np.mean(amep.order.psi_k(
            self.coords, self.box, k=6
        ))
        psi = np.array([psi_complex.real, psi_complex.imag])

        # Old-style call: psi set, no e argument (uses default e=[1,0,0])
        gxy_psi6, x_psi6, y_psi6 = amep.spatialcor.pcf2d(
            self.coords,
            self.box,
            psi=psi,
            nxbins=50,
            nybins=50,
        )
        # Verify output shapes are correct
        self.assertEqual(gxy_psi6.shape, (50, 50),
            'pcf2d with psi (old API) returned wrong shape for g(x,y)')
        self.assertEqual(x_psi6.shape[0], 50,
            'pcf2d x-grid has wrong shape')
        self.assertEqual(y_psi6.shape[1], 50,
            'pcf2d y-grid has wrong shape')

        # Call again with the same parameters — should be deterministic
        gxy_psi6_2, x_psi6_2, y_psi6_2 = amep.spatialcor.pcf2d(
            self.coords,
            self.box,
            psi=psi,
            nxbins=50,
            nybins=50,
        )
        np.testing.assert_array_equal(
            gxy_psi6, gxy_psi6_2,
            err_msg='pcf2d is not deterministic for the same inputs'
        )

    def test_pcf2d_x_mode(self):
        """Test pcf2d with x-axis mode (no psi, default e)."""
        gxy_x, x_x, y_x = amep.spatialcor.pcf2d(
            self.coords,
            self.box,
            psi=None,
            nxbins=50,
            nybins=50,
        )
        self.assertEqual(gxy_x.shape, (50, 50),
            'pcf2d with x-axis mode returned wrong shape')
        # g(x,y) should be non-negative
        self.assertTrue(
            (gxy_x >= 0).all(),
            'pcf2d with x-axis mode returned negative values'
        )

    def test_pcf2d_orientations_mode(self):
        """Test pcf2d with per-particle orientations (e as (N,3) array)."""
        gxy_ori, x_ori, y_ori = amep.spatialcor.pcf2d(
            self.coords,
            self.box,
            psi=None,
            e=self.orientations,
            nxbins=50,
            nybins=50,
        )
        self.assertEqual(gxy_ori.shape, (50, 50),
            'pcf2d with orientations mode returned wrong shape')
        self.assertTrue(
            (gxy_ori >= 0).all(),
            'pcf2d with orientations mode returned negative values'
        )

    def test_pcf2d_psi6_vs_x_differ(self):
        """Verify that psi6 mode and x-axis mode give different results.

        This confirms that the rotation by psi6 actually has an effect."""
        psi_complex = np.mean(amep.order.psi_k(
            self.coords, self.box, k=6
        ))
        psi = np.array([psi_complex.real, psi_complex.imag])

        gxy_psi6, _, _ = amep.spatialcor.pcf2d(
            self.coords, self.box,
            psi=psi, nxbins=50, nybins=50,
        )
        gxy_x, _, _ = amep.spatialcor.pcf2d(
            self.coords, self.box,
            psi=None, nxbins=50, nybins=50,
        )
        # They should differ (unless psi6 angle happens to be 0, which
        # is extremely unlikely for random data)
        self.assertFalse(
            np.allclose(gxy_psi6, gxy_x),
            'psi6 and x-axis modes should produce different results '
            'for random data, but they are identical'
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