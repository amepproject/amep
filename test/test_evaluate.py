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
"""Test units for the continuum class.

Including it's main class, readers and methods."""
import unittest
from zipfile import ZipFile
from pathlib import Path
from random import random
from requests import get
import numpy as np
from matplotlib import use
from amep.load import traj
from amep.evaluate import ClusterGrowth, ClusterSizeDist
use("Agg")
DATA_DIR = Path("./data")

SERVER_URL: str = "https://kuno.fkp.physik.tu-darmstadt.de/d/a3d9887b8a5747e0a56e/files/?p=/"
URL_END: str = "&dl=1"
FIELD_DIR: Path = DATA_DIR/"fields"
PARTICLE_DIR: Path = DATA_DIR/"particles"
PLOT_DIR: Path = DATA_DIR/"plots"
FILE_NAMES: tuple[str, ...] = ("fields", "particles")
FILE_END: str = ".zip"


def get_test_files():
    """Gets and extracts needed Simulation data.

    In order to make it accessible for the tests.
    """
    for file_name in FILE_NAMES:
        sim_dir = DATA_DIR
        zip_path = sim_dir/Path(f"{file_name}{FILE_END}")
        file_url = f"{SERVER_URL}{file_name}{FILE_END}{URL_END}"
        sim_dir.mkdir(exist_ok=True)
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


class TestEvaluateMethods(unittest.TestCase):
    """A test case for all field and continuum data methods"""
    @classmethod
    def setUpClass(cls):
        DATA_DIR.mkdir(exist_ok=True)
        PLOT_DIR.mkdir(exist_ok=True)
        if not FIELD_DIR.is_dir() or not PARTICLE_DIR.is_dir():
            get_test_files()
        else:
            print("Already here")
        cls.field_trajs = [traj(field) for field in FIELD_DIR.iterdir()]
        cls.particle_trajs = [traj(field) for field in PARTICLE_DIR.iterdir()]

    # def test_clustersize_dist(self):
    #     print(ClusterSizeDist(self.field_trajs[0], ftype="c").frames)

    def test_cluster_growth(self):
        self.assertTrue((ClusterGrowth(self.field_trajs[1], scale=1.5, cutoff=0.8,
                                       ftype="c", mode="mean").frames <=
                        ClusterGrowth(self.field_trajs[1], scale=1.5, cutoff=0.8,
                                      ftype="c",
                                      mode="weighted mean").frames).all())
        self.assertTrue((ClusterGrowth(self.field_trajs[1],
                                       ftype="c",
                                       mode="largest").frames >= 0).all())

if __name__ == '__main__':
    unittest.main()