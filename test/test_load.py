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

SERVER_URL: str = "https://kuno.fkp.physik.tu-darmstadt.de/d/a3d9887b8a5747e0a56e/files/?p=/"
URL_END: str = "&dl=1"
DATA_DIR: Path = Path("./data")
PARTICLE_DIR: Path = DATA_DIR/"particles"
FIELD_DIR: Path = DATA_DIR/"fields"
TRAJ_DIR: Path = DATA_DIR/"trajs"
FILE_NAMES: tuple[str, ...] = ("particles",
                               "fields"
                               )
FILE_END: str = ".zip"


def get_test_files():
    """Gets and extracts needed Simulation data in order to make it accessible for the tests"""
    for file_name in FILE_NAMES:
        sim_dir = DATA_DIR
        zip_path = sim_dir/Path(f"{file_name}{FILE_END}")
        file_url = f"{SERVER_URL}{file_name}{FILE_END}{URL_END}"
        if not sim_dir.is_dir():
            sim_dir.mkdir()
        if (sim_dir/file_name).is_dir():
            print("Directory already there.")
        else:
            try:
                print(f"Downloading {file_name}")
                with open(zip_path, "wb") as ofile:
                    ofile.write(get(file_url, allow_redirects=True).content)
                print(f"unziping {file_name}")
                with ZipFile(zip_path, "r") as z_file:
                    z_file.extractall(sim_dir)
                zip_path.unlink()
            except:
                print(f"Downloading and Unziping {file_name} failed!")


# =============================================================================
# MAIN READER TEST
# =============================================================================
class TestLoad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        DATA_DIR.mkdir(exist_ok=True)
        TRAJ_DIR.mkdir(exist_ok=True)
        if not PARTICLE_DIR.is_dir() or not FIELD_DIR.is_dir():
            get_test_files()

    def test_traj(self):
        # test lammps
        traj = amep.load.traj(
            str(PARTICLE_DIR/"different-sized-particles"),
            mode='lammps',
            dumps='dump*.txt'
        )
        traj = amep.load.traj(
            str(PARTICLE_DIR/"different-sized-particles"),
            savedir=TRAJ_DIR,
            mode='lammps',
            dumps='dump*.txt',
            trajfile='lammps.h5amep'
        )

        # test fields
        traj = amep.load.traj(
            str(FIELD_DIR/'cahn-hilliard-model'),
            mode='field',
            dumps='dump_*.txt',
            delimiter='\t',
        )
        traj = amep.load.traj(
            str(FIELD_DIR/'cahn-hilliard-model'),
            savedir='./data/trajs',
            mode='field',
            dumps='dump_*.txt',
            trajfile='field.h5amep',
            delimiter=' ',
            reload = True
        )
        allframes = traj.nframes
        traj = amep.load.traj(
            str(FIELD_DIR/'cahn-hilliard-model'),
            savedir='./data/trajs',
            mode='field',
            dumps='dump_*.txt',
            trajfile='field.h5amep',
            delimiter=' ',
            reload = True,
            start = 0.5
        )
        halfframes = traj.nframes
        traj = amep.load.traj(
            str(FIELD_DIR/'cahn-hilliard-model'),
            savedir='./data/trajs',
            mode='field',
            dumps='dump_*.txt',
            trajfile='field.h5amep',
            delimiter=' ',
            reload = True,
            nth = 4
        )
        quarterframes = traj.nframes
        traj = amep.load.traj(
            str(FIELD_DIR/'cahn-hilliard-model'),
            savedir='./data/trajs',
            mode='field',
            dumps='dump_*.txt',
            trajfile='field.h5amep',
            delimiter=' ',
            reload = True,
            stop = 0.2
        )
        onefifthframes = traj.nframes
        self.assertEqual(
            halfframes,
            int(allframes/2)+1,
            "Invalid number of frames with start=0.5. "\
            f"Got {halfframes} instead of {int(allframes/2)+1}."
        )
        self.assertEqual(
            quarterframes,
            int(allframes/4)+1,
            "Invalid number of frames with nth=4. "\
            f"Got {halfframes} instead of {int(allframes/4)+1}."
        )
        self.assertEqual(
            onefifthframes,
            int(allframes/5),
            "Invalid number of frames with stop=0.2. "\
            f"Got {halfframes} instead of {int(allframes/5)}."
        )

    def test_evaluation(self):
        pass

    def test_database(self):
        pass

if __name__ == '__main__':
    unittest.main()
