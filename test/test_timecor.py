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
Test units for the amep.timecor module.
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
# MAIN TIMECOR TEST
# =============================================================================
class TestTimecor(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        self.start_coords = np.array([
            [1.0, 2.0, 3.0],
            [-3.0, 0.0, 5.0],
            [0.0, -2.0, 5.0]
        ])
        self.end_coords = np.array([
            [2.0, 4.0, 3.0],
            [-1.0, 1.0, 5.0],
            [4.0, -2.0, 4.0]
        ])

    def test_msd(self):
        msd = amep.timecor.msd(
            self.start_coords,
            self.end_coords
        )
        self.assertTrue(
            np.isclose(msd, 9.0),
            f'Incorrect MSD value. Got {msd} instead of 9.0.'
        )
    
    def test_acf(self):
        # 1d test
        acf = amep.timecor.acf(
            self.start_coords[:,0],
            self.end_coords[:,0]
        )
        self.assertTrue(
            np.isclose(acf, 5/3),
            f'Incorrect ACF value for 1D data. Got {acf} instead of {5/3}.'
        )
        # 2d test
        acf = amep.timecor.acf(
            self.start_coords[:,0:2],
            self.end_coords[:,0:2]
        )
        self.assertTrue(
            np.isclose(acf, 4+5/3),
            f'Incorrect ACF value for 2D data. Got {acf} instead of {4+5/3}.'
        )
        # 3d test
        acf = amep.timecor.acf(
            self.start_coords,
            self.end_coords
        )
        self.assertTrue(
            np.isclose(acf, 4+5/3+54/3),
            f'''Incorrect ACF value for 3D data. Got {acf}
            instead of {4+5/3+54/3}.'''
        )
    
    def test_isf(self):
        isf = amep.timecor.isf(self.start_coords, self.end_coords, 2*np.pi)
        self.assertTrue(
            np.isclose(isf, 0.05625849532694646),
            f'''Incorrect ISF value for k={2*np.pi}.
            Got {isf} instead of 0.05625849532694646.'''
        )
        isf = amep.timecor.isf(self.start_coords, self.end_coords, 3*np.pi)
        self.assertTrue(
            np.isclose(isf, 0.032969797157540344),
            f'''Incorrect ISF value for k={3*np.pi}.
            Got {isf} instead of 0.05625849532694646.'''
        )        
        
        
if __name__ == '__main__':
    unittest.main()