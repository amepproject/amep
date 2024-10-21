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
"""Test all the read and evaluate methods for particle based simulations"""
from pathlib import Path
import unittest
import numpy as np
from amep.load import traj
URL_END: str = "&dl=1"
DATA_DIR: Path = Path("../examples/data/")
PARTICLE_DIR: Path = DATA_DIR/"lammps"


class TestParticleMethods(unittest.TestCase):
    """A test case for methods that use particle methods"""

    @classmethod
    def setUpClass(cls):
        """Provides test case with instance of loaded particle data.
        """
        cls.traj = traj(PARTICLE_DIR)

    def test_class_methods(self):
        """Testing all the class methods of the particle trajectory"""
        first_frame = self.traj[0]
        first_frames = self.traj[:3]
        self.assertEqual(first_frame, first_frames[0], "indexing went wrong")
        self.traj.add_param("T", 0.65)
        self.traj.add_author_info("KLS", "name", "Kai Luca Spanheimer")
        self.traj.add_author_info("KLS", "email",
                                  "kai.luca@pkm.tu-darmstadt.de")
        self.traj.add_script(PARTICLE_DIR/"abps.run")
        self.traj.delete_param("T")
        self.traj.nojump()

    def test_frame_methods(self):
        """Test all the relevant frame methods."""
        first_frame = self.traj[0]
        second_frame = self.traj[1]
        third_frame = self.traj[2]
        self.assertNotEqual(first_frame.step, second_frame.step)
        self.assertNotEqual(first_frame.time, second_frame.time)
        self.assertEqual(first_frame.dim, second_frame.dim)
        self.assertEqual(first_frame.volume, second_frame.volume)
        self.assertEqual(third_frame.keys, second_frame.keys)
        np.testing.assert_equal(first_frame.center, second_frame.center)
        np.testing.assert_equal(first_frame.box-second_frame.box,
                                np.zeros_like(second_frame.box))
        first_frame.density()
        first_frame.n()
        first_frame.coords()
        # with self.assertRaises(KeyError, msg=""):
        #     first_frame.unwrapped_coords()
        #     first_frame.nojump_coords()
        first_frame.velocities()
        first_frame.forces()
        first_frame.orientations()
        first_frame.omegas()
        first_frame.torque()
        first_frame.radius()
        first_frame.ids()
        first_frame.data("x", "vx", "omegay", "id")
        for ptype in first_frame.ptypes:
            ptypes = [ptype,]
            first_frame.density(ptype=ptypes)
            first_frame.n(ptype=ptypes)
            first_frame.coords(ptype=ptypes)
            # first_frame.unwrapped_coords(ptype=ptypes)
            # first_frame.nojump_coords(ptype=ptypes)
            first_frame.velocities(ptype=ptypes)
            first_frame.forces(ptype=ptypes)
            first_frame.orientations(ptype=ptypes)
            first_frame.omegas(ptype=ptypes)
            first_frame.torque(ptype=ptypes)
            first_frame.radius(ptype=ptypes)
            first_frame.ids(ptype=ptypes)
            first_frame.data("x", "vx",
                             "omegay", "id", ptype=ptypes)
