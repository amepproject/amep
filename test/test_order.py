# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) 2023 Lukas Hecht and the AMEP development team.
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
Test units for the amep.order module.
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import numpy as np
import unittest
import sys
sys.path.append('/Lukas/Documents/17_Promotion/10_scripts/development/amep-dev')
import amep

# =============================================================================
# MAIN PBC TEST
# =============================================================================
class TestOrder(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        self.coords = np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-2,0]])
        self.box = np.array([[-4,4],[-4,4],[-0.5,0.5]])
        self.center = np.array([0,0,0])

    def test_local_number_density(self):
        """
        Tests the `local_number_density` function of `amep.order`.

        Returns
        -------
        None.

        """
        # calculate local number density with fixed particle radius
        ld = amep.order.local_number_density(
            self.coords, self.box, 0.5, rmax = 2.0, pbc = False, enforce_nd = 2
        )
        # expected results for the given coords
        expected = np.array(
            [0.35809862, 0.29952419, 0.31830989, 0.29952419, 0.16137229]
        )
        # comparison
        compare = ld.round(3) == expected.round(3)
        # check result
        self.assertTrue(
            compare.all(),
            'Incorrect local number density for calculation with fixed '\
            f'particle radius. Got {ld} instead of {expected}.'
        )
        
        # calculate local number density with different particle radii
        radius = np.array([0.5,0.5,0.5,0.5,1.5])
        ld = amep.order.local_number_density(
            self.coords, self.box, radius,
            rmax = 2.0, pbc = False, enforce_nd = 2
        )
        # expected results for the given coords
        expected = np.array(
            [0.35809862, 0.31204799, 0.3315728 , 0.31204799, 0.16137229]
        )
        # comparison
        compare = ld.round(3) == expected.round(3)
        # check result
        self.assertTrue(
            compare.all(),
            'Incorrect local number density for calculation with different '\
            f'particle radii. Got {ld} instead of {expected}.'
        )

    def test_local_mass_density(self):
        """
        Tests the `local_mass_density` function of `amep.order`.

        Returns
        -------
        None.

        """
        # calculate local mass density with fixed mass
        ld1 = amep.order.local_mass_density(
            self.coords, self.box, 0.5, 1.0,
            rmax = 2.0, pbc = False, enforce_nd = 2
        )
        ld2 = amep.order.local_mass_density(
            self.coords, self.box, 0.5, 2.0,
            rmax = 2.0, pbc = False, enforce_nd = 2
        )
        # expected results for the given coords
        expected = np.array(
            [0.35809862, 0.29952419, 0.31830989, 0.29952419, 0.16137229]
        )
        # comparison
        compare = ld1.round(3) == expected.round(3)
        # check result
        self.assertTrue(
            compare.all(),
            'Incorrect local mass density for calculation with fixed '\
            f'mass. Got {ld1} instead of {expected}.'
        )
        # check if twice the mass leads to twice the mass density
        compare = ld2.round(3) == (2*ld1).round(3)
        # check result
        self.assertTrue(
            compare.all(),
            'Doubling the mass does not lead to twice the local mass density.'\
            f' Got {ld2} instead of {2*ld1}.'
        )
        # calculate local number density with different masses
        mass = np.array([5.0,1.5,0.5,0.5,1.5])
        radius = np.array([1.0,0.5,0.5,0.5,1.5])
        ld = amep.order.local_mass_density(
            self.coords, self.box, radius, mass,
            rmax = 2.0, pbc = False, enforce_nd = 2
        )
        # expected results for the given coords
        expected = np.array(
            [0.65651414, 0.62722693, 0.6167254 , 0.58743819, 0.36031597]
        )
        # comparison
        compare = ld.round(3) == expected.round(3)
        # check result
        self.assertTrue(
            compare.all(),
            'Incorrect local mass density for calculation with different '\
            f'particle radii and masses. Got {ld} instead of {expected}.'
        )
            
    def test_local_packing_fraction(self):
        """
        Tests the `local_packing_fraction` function of `amep.order`.

        Returns
        -------
        None.

        """
        # calculate local packing fraction with fixed particle radius
        ld = amep.order.local_packing_fraction(
            self.coords, self.box, 0.5, rmax = 2.0, pbc = False, enforce_nd = 2
        )
        # expected results for the given coords
        expected = np.array(
            [0.28125, 0.23524575, 0.25, 0.23524575, 0.1267415]
        )
        # comparison
        compare = ld.round(3) == expected.round(3)
        # check result
        self.assertTrue(
            compare.all(),
            'Incorrect local packing fraction for calculation with fixed '\
            f'particle radius. Got {ld} instead of {expected}.'
        )
        # result should be pi/4 times local number density
        lnd = amep.order.local_number_density(
            self.coords, self.box, 0.5, rmax = 2.0, pbc = False, enforce_nd = 2
        )
        compare = ld.round(3) == (lnd*np.pi/4).round(3)
        self.assertTrue(
            compare.all(),
            'Local packing fraction and pi/4 times local number density do '\
            f'not match. Got phi={ld} instead of rho*pi/4={lnd*np.pi/4}.'
        )
        # calculate local packing fraction with different particle radii
        radius = np.array([0.5,0.5,0.5,0.5,1.0])
        ld = amep.order.local_packing_fraction(
            self.coords, self.box, radius,
            rmax = 2.0, pbc = False, enforce_nd = 2
        )
        # expected results for the given coords
        expected = np.array(
            [0.375, 0.3142415, 0.25, 0.3142415, 0.3142415]
        )
        # comparison
        compare = ld.round(3) == expected.round(3)
        # check result
        self.assertTrue(
            compare.all(),
            'Incorrect local packing fraction for calculation with different '\
            f'particle radii. Got {ld} instead of {expected}.'
        )


if __name__ == '__main__':
    unittest.main()
