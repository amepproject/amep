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
"""
Test units for the amep.load module.
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import unittest
from zipfile import ZipFile
from requests import get
from pathlib import Path
import amep

DATA_DIR: Path = Path("../examples/data/")
PARTICLE_DIR: Path = DATA_DIR/"lammps"
FIELD_DIR: Path = DATA_DIR/"continuum"
FILE_NAMES: tuple[str, ...] = ("particles",
                               "fields"
                               )


# =============================================================================
# MAIN READER TEST
# =============================================================================
class TestLoad(unittest.TestCase):
    '''A test case for all load functions for AMEP save files.'''

    def test_traj(self):
        # test lammps
        traj = amep.load.traj(
            PARTICLE_DIR,
            mode='lammps',
            dumps='dump*.txt'
        )
        # test fields
        traj = amep.load.traj(
            FIELD_DIR,
            mode='field',
            dumps='field_*.txt',
            delimiter=' ',
        )
        allframes = amep.load.traj(
            FIELD_DIR,
            mode='field',
            dumps='field_*.txt',
            trajfile='traj.h5amep',
            delimiter=' ',
            reload=True,
            timestep=1
        ).nframes
        halfframes = amep.load.traj(
            FIELD_DIR,
            mode='field',
            dumps='field_*.txt',
            trajfile='traj.h5amep',
            delimiter=' ',
            reload=True,
            start=0.5,
            timestep=1
        ).nframes
        quarterframes = amep.load.traj(
            FIELD_DIR,
            mode='field',
            dumps='field_*.txt',
            trajfile='traj.h5amep',
            delimiter=' ',
            reload=True,
            nth=4,
            timestep=1
        ).nframes
        onefifthframes = amep.load.traj(
            FIELD_DIR,
            mode='field',
            dumps='field_*.txt',
            trajfile='traj.h5amep',
            delimiter=' ',
            reload=True,
            stop=0.2,
            timestep=1
        ).nframes
        self.assertEqual(
            halfframes,
            int(allframes/2)+1,
            "Invalid number of frames with start=0.5. "
            f"Got {halfframes} instead of {int(allframes/2)+1}."
        )
        self.assertEqual(
            quarterframes,
            int(allframes/4)+1,
            "Invalid number of frames with nth=4. "
            f"Got {halfframes} instead of {int(allframes/4)+1}."
        )
        self.assertEqual(
            onefifthframes,
            int(allframes/5),
            "Invalid number of frames with stop=0.2. "
            f"Got {halfframes} instead of {int(allframes/5)}."
        )

    def test_evaluation(self):
        pass

    def test_database(self):
        pass
