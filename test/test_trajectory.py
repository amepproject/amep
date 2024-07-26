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
Test units for the amep.trajectory module.
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
from pathlib import Path
import unittest
import os
import numpy as np
import amep

# =============================================================================
# GLOBAL CONFIG.
# =============================================================================
DATADIR = Path("../examples/data")
FIELDDIR = DATADIR/'continuum'
LAMMPSDIR = DATADIR/'lammps'

RNG = np.random.default_rng(1234)

# for field data creation
BOX = np.array([[-10,10], [-20,20], [-0.5,0.5]])
SHAPE = np.array([40,80,2])
X,Y,Z = np.meshgrid(
    np.linspace(BOX[0,0], BOX[0,1], SHAPE[0]),
    np.linspace(BOX[1,0], BOX[1,1], SHAPE[1]),
    np.linspace(BOX[2,0], BOX[2,1], SHAPE[2]),
    indexing = 'ij'
)
COORDS = [X.flatten(), Y.flatten(), Z.flatten()]
TIMESTEPS = [0, 1000, 2000, 3000]
FIELDS = ['rho', 'c', 'alpha', 'beta']
DATA = ['id', 'x', 'y', 'z', 'vx', 'vy', 'fx', 'fy', 'mass', 'radius']
NATOMS = 100


# =============================================================================
# PARTICLETRAJECTORY TESTS
# =============================================================================
class TestParticleTrajectory(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        # generate some test LAMMPS data
        
        # load the data
        cls.traj = amep.load.traj(
            LAMMPSDIR,
            mode='lammps',
            reload=True
        )
        
        # add some particle information
        cls.traj.add_particle_info(1, 'name', 'atom A')
        cls.traj.add_particle_info(1, 'type', 'carbon')

    def test_nojump(self):
        pass
    
    def test_add_particle_info(self):
        # add particle info
        self.traj.add_particle_info(1, 'test', 10)
        # check if particle info has been added correctly
        self.assertIn(
            'test',
            self.traj.get_particle_info(1),
            f'''Particle info "test" has not been added correctly. Got
            {self.traj.get_particle_info(1)}.'''
        )
        # delete added info
        self.traj.delete_particle_info(1, key='test')
        
    def test_get_particle_info(self):
        self.assertEqual(
            self.traj.get_particle_info(1)['name'],
            'atom A',
            f'''Invalid particle info. Got
            {self.traj.get_particle_info(1)['name']} instead of "atom A".'''
        )
        self.assertTrue(
            isinstance(self.traj.get_particle_info(1), dict),
            f'''Invalid type. Got {type(self.traj.get_particle_info(1))}
            instead of dict.'''
        )

    def test_ptype_type(self):
        self.traj.add_particle_info(1, "key", "value")
        self.traj.add_particle_info("asdf", "key2", "value2")
        self.assertTrue(
            isinstance(self.traj.get_particle_info(), dict),
            f'''Invalid type. Got {type(self.traj.get_particle_info(1))}
            instead of dict.'''
        )
        self.assertTrue(
            isinstance(self.traj.get_particle_info("asdf"), dict),
            f'''Invalid type. Got {type(self.traj.get_particle_info(1))}
            instead of dict.'''
        )

    def test_delete_particle_info(self):
        # delete particle info
        self.traj.delete_particle_info(None)
        # check if particle info has been deleted correctly
        self.assertEqual(
            self.traj.get_particle_info(),
            {},
            f'''Particle info has not been deleted correctly. Got
            {self.traj.get_particle_info()} instead of an empty dictionary.'''
        )
        # add deleted information again
        self.traj.add_particle_info(1, 'name', 'atom A')
        self.traj.add_particle_info(1, 'type', 'carbon')

# =============================================================================
# FIELDTRAJECTORY TESTS
# =============================================================================
class TestFieldTrajectory(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        # load the data
        cls.traj = amep.load.traj(
            FIELDDIR,
            mode='field',
            delimiter=' ',
            dumps='field_*.txt',
            reload=True
        )

        # add some field information
        cls.traj.add_field_info('c', 'name', 'chemicals')
        cls.traj.add_field_info('rho', 'name', 'bacterial density')

    def test_add_field_info(self):
        # add field info
        self.traj.add_field_info('rho', 'test', 10)
        # check if particle info has been added correctly
        self.assertIn(
            'test',
            self.traj.get_field_info('rho'),
            f'''Field info "test" has not been added correctly. Got
            {self.traj.get_field_info('rho')}.'''
        )
        # delete added info
        self.traj.delete_field_info('rho', key='test')
    
    def test_get_field_info(self):
        self.assertEqual(
            self.traj.get_field_info('c')['name'],
            'chemicals',
            f'''Invalid field info. Got
            {self.traj.get_field_info('c')['name']} instead of "chemicals".'''
        )
        self.assertTrue(
            isinstance(self.traj.get_field_info('c'), dict),
            f'''Invalid type. Got {type(self.traj.get_field_info('c'))}
            instead of dict.'''
        )
    
    def test_delete_field_info(self):
        # delete particle info
        self.traj.delete_field_info(None)
        # check if particle info has been deleted correctly
        self.assertEqual(
            self.traj.get_field_info(),
            {},
            f'''Field info has not been deleted correctly. Got
            {self.traj.get_field_info()} instead of an empty dictionary.'''
        )
        # add deleted field information again
        self.traj.add_field_info('c', 'name', 'chemicals')
        self.traj.add_field_info('rho', 'name', 'bacterial density')
