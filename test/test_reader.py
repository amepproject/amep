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
Test units for the amep.reader module.
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
from pathlib import Path
import unittest
import numpy as np
import amep
import os

# =============================================================================
# GLOBAL CONFIG.
# =============================================================================
DATADIR = Path('../examples/data/')
LAMMPSDIR = DATADIR/'lammps'
FIELDDIR = DATADIR/'continuum'
EMPTYDIR = DATADIR/'empty'
SAVEDIR = DATADIR/'trajs'
INVALIDFIELDDIR = DATADIR/'invalid'

RNG = np.random.default_rng(1234)

# for field data creation
BOX = np.array([[-10, 10], [-20, 20], [-0.5, 0.5]])
SHAPE = np.array([40, 80, 2])
X, Y, Z = np.meshgrid(
        np.linspace(BOX[0, 0], BOX[0, 1], SHAPE[0]),
        np.linspace(BOX[1, 0], BOX[1, 1], SHAPE[1]),
        np.linspace(BOX[2, 0], BOX[2, 1], SHAPE[2]),
        indexing='ij'
)
COORDS = [X.flatten(), Y.flatten(), Z.flatten()]
TIMESTEPS = [0, 1000, 2000, 3000]
FIELDS = ['rho', 'c', 'alpha', 'beta']
DATA = ['id', 'x', 'y', 'z', 'vx', 'vy', 'fx', 'fy', 'mass', 'radius']
NATOMS = 100


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================
def create_invalid_field_data():
    """Generate a set of invalid field data."""

    # INVALID BOX
    # generate grid.txt file
    with open(
            os.path.join(INVALIDFIELDDIR, "grid-box.txt"),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write(f'BOX:\n{BOX[0, 0]}\t{BOX[0, 1]}\n{BOX[1, 0]}\t{BOX[1, 1]}\n')
        wfile.write('SHAPE:\n' + '\t'.join(str(s) for s in SHAPE) + '\n')
        wfile.write('COORDINATES:\tX\tY\tZ\n')
        wfile.write('\n'.join('\t'.join(
            str(COORDS[i][j]) for i in range(3)
        ) for j in range(len(COORDS[0]))))
    # generate one dump file with random data
    with open(
            os.path.join(INVALIDFIELDDIR, 'field-box_0.txt'),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write('TIMESTEP:\n0\nDATA:\t'+'\t'.join(FIELDS)+'\n')
        wfile.write('\n'.join('\t'.join(
            str(RNG.random()) for _ in FIELDS
        ) for _ in range(len(COORDS[0]))))

    # INVALID SHAPE
    # generate grid.txt file
    with open(
            os.path.join(INVALIDFIELDDIR, "grid-shape.txt"),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write(f'BOX:\n{BOX[0, 0]}\t{BOX[0, 1]}\n{BOX[1, 0]}\t{BOX[1, 1]}\n{BOX[2, 0]}\t{BOX[2, 1]}\n')
        wfile.write('COORDINATES:\tX\tY\tZ\n')
        wfile.write('\n'.join('\t'.join(
            str(COORDS[i][j]) for i in range(3)
        ) for j in range(len(COORDS[0]))))
    # generate one dump file with random data
    with open(
            os.path.join(INVALIDFIELDDIR, 'field-shape_0.txt'),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write('TIMESTEP:\n0\nDATA:\t'+'\t'.join(FIELDS)+'\n')
        wfile.write('\n'.join('\t'.join(
            str(RNG.random()) for _ in FIELDS
        ) for _ in range(len(COORDS[0]))))

    # INVALID COORDINATES
    # generate grid.txt file
    with open(
            os.path.join(INVALIDFIELDDIR, "grid-coords.txt"),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write(f'BOX:\n{BOX[0, 0]}\t{BOX[0, 1]}\n{BOX[1, 0]}\t{BOX[1, 1]}\n{BOX[2, 0]}\t{BOX[2, 1]}\n')
        wfile.write('SHAPE:\n' + '\t'.join(str(s) for s in SHAPE) + '\n')
        wfile.write('COORDINATES:X\tY\tZ\n')
        wfile.write('\n'.join('\t'.join(
            str(COORDS[i][j]) for i in range(3)
        ) for j in range(len(COORDS[0]))))
    # generate one dump file with random data
    with open(
            os.path.join(INVALIDFIELDDIR, 'field-coords_0.txt'),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write('TIMESTEP:\n0\nDATA:\t'+'\t'.join(FIELDS)+'\n')
        wfile.write('\n'.join('\t'.join(
            str(RNG.random()) for _ in FIELDS
        ) for _ in range(len(COORDS[0]))))

    # INVALID TIMESTEP
    # generate grid.txt file
    with open(
            os.path.join(INVALIDFIELDDIR, "grid-step.txt"),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write(f'BOX:\n{BOX[0, 0]}\t{BOX[0, 1]}\n{BOX[1, 0]}\t{BOX[1, 1]}\n{BOX[2, 0]}\t{BOX[2, 1]}\n')
        wfile.write('SHAPE:\n' + '\t'.join(str(s) for s in SHAPE) + '\n')
        wfile.write('COORDINATES:\tX\tY\tZ\n')
        wfile.write('\n'.join('\t'.join(
            str(COORDS[i][j]) for i in range(3)
        ) for j in range(len(COORDS[0]))))
    # generate one dump file with random data
    with open(
            os.path.join(INVALIDFIELDDIR, 'field-step_0.txt'),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write('\nDATA:\t'+'\t'.join(FIELDS)+'\n')
        wfile.write('\n'.join('\t'.join(
            str(RNG.random()) for _ in FIELDS
        ) for _ in range(len(COORDS[0]))))

    # INVALID DATA
    # generate grid.txt file
    with open(
            os.path.join(INVALIDFIELDDIR, "grid-data.txt"),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write(f'BOX:\n{BOX[0, 0]}\t{BOX[0, 1]}\n{BOX[1, 0]}\t{BOX[1, 1]}\n{BOX[2, 0]}\t{BOX[2, 1]}\n')
        wfile.write('SHAPE:\n' + '\t'.join(str(s) for s in SHAPE) + '\n')
        wfile.write('COORDINATES:\tX\tY\tZ\n')
        wfile.write('\n'.join('\t'.join(
            str(COORDS[i][j]) for i in range(3)
        ) for j in range(len(COORDS[0]))))
    # generate one dump file with random data
    with open(
            os.path.join(INVALIDFIELDDIR, 'field-data_0.txt'),
            "w",
            encoding="utf-8"
    ) as wfile:
        wfile.write('TIMESTEP:\n0\nDATA:'+'\t'.join(FIELDS)+'\n')
        wfile.write('\n'.join('\t'.join(
            str(RNG.random()) for _ in FIELDS
        ) for _ in range(len(COORDS[0]))))


# =============================================================================
# MAIN READER TESTS
# =============================================================================
class TestContinuumReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        INVALIDFIELDDIR.mkdir(exist_ok=True)
        SAVEDIR.mkdir(exist_ok=True)
        EMPTYDIR.mkdir(exist_ok=True)
        create_invalid_field_data()

    def test_init(self):
        """Test constructor method."""
        # read data
        (FIELDDIR/"#temp#traj.h5amep").touch()
        _ = amep.reader.ContinuumReader(
            FIELDDIR,
            FIELDDIR,
            trajfile='#temp#traj.h5amep',
            deleteold=False,
            dumps='field_*.txt',
            gridfile='grid.txt',
            delimiter=' ',
            timestep=1.0,
            nth=None
        )
        # test exceptions for invalid/corrupted data
        with self.assertRaises(
                Exception,
                msg='No exception raised for invalid box.'
        ):
            _ = amep.reader.ContinuumReader(
                INVALIDFIELDDIR,
                INVALIDFIELDDIR,
                trajfile='field-box.h5amep',
                deleteold=False,
                dumps='field-box_*.txt',
                gridfile='grid-box.txt',
                delimiter=' ',
                timestep=0.5
            )
        with self.assertRaises(
                Exception,
                msg='No exception raised for invalid shape.'
        ):
            _ = amep.reader.ContinuumReader(
                INVALIDFIELDDIR,
                INVALIDFIELDDIR,
                trajfile='field-shape.h5amep',
                deleteold=False,
                dumps='field-shape_*.txt',
                gridfile='grid-shape.txt',
                delimiter=' ',
                timestep=0.5
            )
        with self.assertRaises(
                Exception,
                msg='No exception raised for invalid coordinates.'
        ):
            _ = amep.reader.ContinuumReader(
                INVALIDFIELDDIR,
                INVALIDFIELDDIR,
                trajfile='field-coords.h5amep',
                deleteold=False,
                dumps='field-coords_*.txt',
                gridfile='grid-coords.txt',
                delimiter=' ',
                timestep=0.5
            )
        with self.assertRaises(
                Exception,
                msg='No exception raised for invalid timestep.'
        ):
            _ = amep.reader.ContinuumReader(
                INVALIDFIELDDIR,
                INVALIDFIELDDIR,
                trajfile='field-step.h5amep',
                deleteold=False,
                dumps='field-step_*.txt',
                gridfile='grid-step.txt',
                delimiter=' ',
                timestep=0.5
            )
        with self.assertRaises(
                Exception,
                msg='No exception raised for invalid data.'
        ):
            _ = amep.reader.ContinuumReader(
                INVALIDFIELDDIR,
                INVALIDFIELDDIR,
                trajfile='field-data.h5amep',
                deleteold=False,
                dumps='field-data_*.txt',
                gridfile='grid-data.txt',
                delimiter=' ',
                timestep=0.5
            )

    def test_savedir(self):
        """Test saving method"""
        _ = amep.reader.ContinuumReader(
            FIELDDIR,
            SAVEDIR,
            trajfile='traj.h5amep',
            deleteold=False,
            dumps='field_*.txt',
            gridfile='grid.txt',
            delimiter=' ',
            timestep=0.5
        )
        self.assertTrue(
            os.path.exists(os.path.join(SAVEDIR, 'traj.h5amep')),
            f'''Savedir error: {os.path.join(SAVEDIR, 'traj.h5amep')}
            does not exist.'''
        )

    def test_trajfile(self):
        _ = amep.reader.ContinuumReader(
            FIELDDIR,
            FIELDDIR,
            trajfile='traj.h5amep',
            deleteold=False,
            dumps='field_*.txt',
            gridfile='grid.txt',
            delimiter=' ',
            timestep=0.5
        )
        self.assertTrue(
            os.path.exists(os.path.join(FIELDDIR, 'traj.h5amep')),
            f'''trajfile error: {os.path.join(FIELDDIR, 'traj.h5amep')}
            does not exist.'''
        )

    def test_deleteold(self):
        """Test backup during creation."""
        # create trajfile
        _ = amep.reader.ContinuumReader(
            FIELDDIR,
            SAVEDIR,
            trajfile='field2.h5amep',
            deleteold=False,
            dumps='field_*.txt',
            gridfile='grid.txt',
            delimiter=' ',
            timestep=0.5
        )
        # recreate trajfile and create backup
        _ = amep.reader.ContinuumReader(
            FIELDDIR,
            SAVEDIR,
            trajfile='field2.h5amep',
            deleteold=False,
            dumps='field_*.txt',
            gridfile='grid.txt',
            delimiter=' ',
            timestep=0.5
        )
        self.assertTrue(
            os.path.exists(os.path.join(SAVEDIR, '#field2.h5amep')),
            f'''Backup error: backup file
            {os.path.join(SAVEDIR, '#field2.h5amep')} does not exist.'''
        )
        # recreate traj file and delete backup
        _ = amep.reader.ContinuumReader(
            FIELDDIR,
            SAVEDIR,
            trajfile='field2.h5amep',
            deleteold=True,
            dumps='field_*.txt',
            gridfile='grid.txt',
            delimiter=' ',
            timestep=0.5
        )
        self.assertFalse(
            os.path.exists(os.path.join(SAVEDIR, '#field2.h5amep')),
            f'''Backup error: backup file
            {os.path.join(SAVEDIR, 'field2.h5amep')} has not been deleted.'''
        )

    def test_dumps(self):
        """Test dump methods."""
        with self.assertRaises(
                Exception,
                msg=f'''No exception raised for empty directory {EMPTYDIR}.'''
                ):
            _ = amep.reader.ContinuumReader(
                EMPTYDIR,
                EMPTYDIR,
                trajfile='traj.h5amep',
                deleteold=False,
                dumps='field_*.txt',
                gridfile='grid.txt',
                delimiter=' ',
                timestep=0.5
                )
        self.assertFalse(
            os.path.exists(os.path.join(EMPTYDIR, 'traj.h5amep')),
            f'''Trajectory file {os.path.join(EMPTYDIR, 'traj.h5amep')}
            has been created although there is no data in {EMPTYDIR}.'''
        )

    def test_gridfile(self):
        with self.assertRaises(
                FileNotFoundError,
                msg='No exception raised for wrong grid file name.'
        ):
            _ = amep.reader.ContinuumReader(
                FIELDDIR,
                FIELDDIR,
                trajfile='traj.h5amep',
                deleteold=False,
                dumps='field_*.txt',
                gridfile='test.txt',
                delimiter=' ',
                timestep=1.0
            )

    def test_delimiter(self):
        with self.assertRaises(
                ValueError,
                msg='No ValueError raised for wrong delimiter.'
        ):
            _ = amep.reader.ContinuumReader(
                FIELDDIR,
                FIELDDIR,
                trajfile='traj.h5amep',
                deleteold=False,
                dumps='field_*.txt',
                gridfile='grid.txt',
                delimiter=',',
                timestep=0.5
            )

    def test_timestep(self):
        # test warning
        with self.assertLogs(level="WARNING"):
            reader = amep.reader.ContinuumReader(
                FIELDDIR,
                FIELDDIR,
                trajfile='traj.h5amep',
                deleteold=False,
                dumps='field_*.txt',
                gridfile='grid.txt',
                delimiter=' ',
                timestep=None
            )
        # test value
        reader = amep.reader.ContinuumReader(
            FIELDDIR,
            FIELDDIR,
            trajfile='traj.h5amep',
            deleteold=False,
            dumps='field_*.txt',
            gridfile='grid.txt',
            delimiter=' ',
            timestep=0.5
        )
        self.assertEqual(
            reader.dt,
            0.5,
            f'''Got dt={reader.dt} instead of 0.5.'''
        )


class TestLammpsReader(unittest.TestCase):

    def test_savedir(self):
        _ = amep.reader.LammpsReader(
            LAMMPSDIR,
            SAVEDIR,
            trajfile='traj.h5amep',
            deleteold=False,
            dumps='dump*.txt',
        )
        self.assertTrue(
            os.path.exists(os.path.join(SAVEDIR, 'traj.h5amep')),
            f'''Savedir error: {os.path.join(SAVEDIR, 'traj.h5amep')}
            does not exist.'''
        )

    def test_dumps(self):
        with self.assertRaises(
                Exception,
                msg=f'''No exception raised for empty directory {EMPTYDIR}.'''
        ):
            _ = amep.reader.LammpsReader(
                EMPTYDIR,
                EMPTYDIR,
                trajfile='traj.h5amep',
                deleteold=False,
                dumps='dump*.txt'
            )
        self.assertFalse(
            os.path.exists(os.path.join(EMPTYDIR, 'traj.h5amep')),
            f'''Trajectory file {os.path.join(EMPTYDIR, 'traj.h5amep')}
            has been created although there is no data in {EMPTYDIR}.'''
        )

    def test_trajfile(self):
        _ = amep.reader.LammpsReader(
            LAMMPSDIR,
            LAMMPSDIR,
            trajfile='traj.h5amep',
            deleteold=False,
            dumps='dump*.txt'
        )
        self.assertTrue(
            os.path.exists(os.path.join(LAMMPSDIR, 'traj.h5amep')),
            f'''trajfile error: {os.path.join(LAMMPSDIR, 'traj.h5amep')}
            does not exist.'''
        )

    def test_deleteold(self):
        # create trajfile
        _ = amep.reader.LammpsReader(
            LAMMPSDIR,
            SAVEDIR,
            trajfile='lammps2.h5amep',
            deleteold=False,
            dumps='dump*.txt'
        )
        # recreate trajfile and create backup
        _ = amep.reader.LammpsReader(
            LAMMPSDIR,
            SAVEDIR,
            trajfile='lammps2.h5amep',
            deleteold=False,
            dumps='dump*.txt'
        )
        self.assertTrue(
            os.path.exists(os.path.join(SAVEDIR, '#lammps2.h5amep')),
            f'''Backup error: backup file
            {os.path.join(SAVEDIR, '#lammps2.h5amep')} does not exist.'''
        )
        # recreate traj file and delete backup
        _ = amep.reader.LammpsReader(
            LAMMPSDIR,
            SAVEDIR,
            trajfile='lammps2.h5amep',
            deleteold=True,
            dumps='dump*.txt'
        )
        self.assertFalse(
            os.path.exists(os.path.join(SAVEDIR, '#lammps2.h5amep')),
            f'''Backup error: backup file 
            {os.path.join(SAVEDIR, 'lammps2.h5amep')} has not been deleted.'''
        )


class TestH5amepReader(unittest.TestCase):
    """Testcase for the standard reader.

    TO BE IMPLEMENTED.
    """

    @classmethod
    def setUpClass(cls):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        pass

    def test_trajfile(self):
        pass


class TestGromacsReader(unittest.TestCase):
    """Testcase for GROMACS data.

    TO BE IMPLEMENTED.
    """
    @classmethod
    def setUpClass(self):
        """
        Basic setup. Generating test data.

        TO BE IMPLEMENTED.
        Returns
        -------
        None.

        """
        pass

    def test_trajfile(self):
        pass
