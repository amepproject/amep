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
Test units for the amep.functions module.
"""
import unittest
from pathlib import Path
from numpy import linspace, average
from numpy.random import normal
from amep.functions import Gaussian, gaussian
from amep.plot import new

DATA_DIR = Path("../examples/data/")


class TestMathFunctions(unittest.TestCase):
    '''A test case for all mathematical functions like gauss, etc.'''
    @classmethod
    def setUpClass(cls):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def test_gaussian(self):
        """Test the gaussian function and the Gaussian fit class."""
        x_vec = linspace(-10, 10, 500)
        y_vec = gaussian(x_vec, mu=0.0, sig=1.0, offset=0.0, A=10)
        mean = average(x_vec, weights=y_vec)
        sig = average((x_vec - mean)**2, weights=y_vec)
        amp = max(y_vec)
        gauss = Gaussian()
        gauss.fit(x_vec, y_vec, p0=[mean, sig, amp])
        # Check if fit and generated data are close.
