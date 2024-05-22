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
import unittest
from pathlib import Path
from zipfile import ZipFile
from requests import get
from amep.load import traj
SERVER_URL: str = "https://kuno.fkp.physik.tu-darmstadt.de/d/a3d9887b8a5747e0a56e/files/?p=/"
URL_END: str = "&dl=1"
DATA_DIR: Path = Path("./data/")
PARTICLE_DIR: Path = DATA_DIR/"particles"
FILE_NAMES: tuple[str, ...] = ("particles",
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


class TestParticleMethods(unittest.TestCase):
    """A test case for methods that use particle methods"""

    @classmethod
    def setUpClass(cls):
        """Checks whether data needed for tests is already there.

        And gets it if not.
        """
        DATA_DIR.mkdir(exist_ok=True)
        if not PARTICLE_DIR.is_dir():
            get_test_files()
        cls.trajs = [traj(str(file_name)) for file_name in PARTICLE_DIR.iterdir()]

    def test_class_methods(self):
        """Testing all the class methods of the particle trajectory"""
        for trace in self.trajs:
            first_frame = trace[0]
            first_frames = trace[:3]
            self.assertEqual(first_frame, first_frames[0],
                             "indexing went wrong")
        print([trace.reader.savedir for trace in self.trajs])
        test_traj = [trace for trace in self.trajs if trace.reader.savedir ==
                     str(PARTICLE_DIR/"kob-andersen-liquid")][0]
        test_traj.add_param("T", 0.65)
        test_traj.add_author_info("KLS", "name", "Kai Luca Spanheimer")
        test_traj.add_author_info("KLS", "email",
                                  "kai.luca@pkm.tu-darmstadt.de")
        test_traj.add_script(PARTICLE_DIR/Path("kob-andersen-liquid/production.run"))
        test_traj.delete_param("T")
        test_traj.nojump()

    def test_frame_methods(self):
        for trace in self.trajs:
            first_frame = trace[0]
            second_frame = trace[1]
            third_frame = trace[2]
            self.assertNotEqual(first_frame.step, second_frame.step)
            self.assertNotEqual(first_frame.time, second_frame.time)
            self.assertEqual(first_frame.dim, second_frame.dim)
            self.assertEqual(first_frame.volume, second_frame.volume)
            self.assertEqual(third_frame.keys, second_frame.keys)
            # print(trace.info)
            # self.assertEqual(first_frame.center,second_frame.center)
            # self.assertEqual(first_frame.box-second_frame.box,np.zeros_like(second_frame.box))
            first_frame.density()
            first_frame.n()
            first_frame.coords()
            """
            with self.assertRaises(KeyError, msg=""):
                first_frame.unwrapped_coords()
                first_frame.nojump_coords()
            """
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

if __name__ == '__main__':
    unittest.main()