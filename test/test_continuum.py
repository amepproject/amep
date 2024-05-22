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
"""Test units for the continuum module.

Including it's main class, readers and methods."""
import unittest
from zipfile import ZipFile
from pathlib import Path
from random import random
from requests import get
from matplotlib import use
from amep.load import traj
from amep import plot
from amep.continuum import identify_clusters
use("Agg")
DATA_DIR = Path("./data")
N_LINES = 100
COORDS = ("X", "Y", "Z")
FIELDS = ("c", "rho", "alpha", "omega")
TIMES = (0, 0.4, 0.5, 0.9, 2.3)

SERVER_URL: str = "https://kuno.fkp.physik.tu-darmstadt.de/d/a3d9887b8a5747e0a56e/files/?p=/"
URL_END: str = "&dl=1"
FIELD_DIR: Path = DATA_DIR/"fields"
PLOT_DIR: Path = DATA_DIR/"plots"
FILE_NAMES: tuple[str, ...] = ("fields",)
FILE_END: str = ".zip"


def get_test_files():
    """Gets and extracts needed Simulation data.

    In order to make it accessible for the tests.
    """
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


def create_test_files():
    """Creates grid and some field data files with random data"""
    with open(DATA_DIR/Path("grid.txt"), "w", encoding="utf-8") as wfile:
        wfile.write("BOX:\n10\t10\n10\t10\n0.5\t0.5\nCOORDINATES:\t" +
                    "\t".join(COORDS)+"\n")
        wfile.write("\n".join("\t".join(str(random())for _ in COORDS)
                              for _ in range(N_LINES)))
    for index, time in enumerate(TIMES):
        with open(DATA_DIR/Path(f"field_{index}.txt"), "w",
                  encoding="utf-8") as wfile:
            wfile.write(f"TIMESTEP:\n{time}\nDATA:\t"+"\t".join(FIELDS)+"\n")
            wfile.write("\n".join("\t".join(str(random()) for _ in FIELDS)
                                  for _ in range(N_LINES)))


class TestFieldMethods(unittest.TestCase):
    """A test case for all field and continuum data methods"""
    @classmethod
    def setUpClass(cls):
        DATA_DIR.mkdir(exist_ok=True)
        PLOT_DIR.mkdir(exist_ok=True)
        if not FIELD_DIR.is_dir():
            get_test_files()
        else:
            print("Already here")
        # if not all(path.isfile(filius) for filius in
        #            (path.join(DATA_DIR, "grid.txt"),
        #             *(f"field_{index}" for index, _ in enumerate(TIMES)))):
        #     DATA_DIR.mkdir(parents=True, exist_ok=True)
        #     create_test_files()
        # else:
        #     print("All needed  Field-Data is here")

    def test_creation(self):
        """Tests the creation of AMEP file from field data"""
        # reader=ContinuumReader("../data/continuum_data")
        # field=BaseField(reader,0)
        trajectory = traj(str(FIELD_DIR/"cahn-hilliard-model"),
                          mode="field", reload=True,
                          start=0.0, stop=1.0, nth=1,
                          delimiter=" ",
                          dumps="dump_*.txt")
        print(trajectory.nframes)
        field = trajectory[1]
        print("Box:", field.box)
        print("Center:", field.center)
        print("Dimension:", field.dim)
        print("Grid:", field.grid)
        print("Step: ", field.step)
        print("Time: ", field.time)
        print("Volume", field.volume)
        print(field.data())

    def test_plottability(self):
        """Try plotting a field"""
        trajectory = traj(str(FIELD_DIR/"cahn-hilliard-model"),
                          mode="field")
        field = trajectory[-1]
        fig, axe = plot.new()
        print(field.grid)
        axe.pcolormesh(field.grid[0],
                       field.grid[1],
                       field.data("c"))
        fig.savefig(PLOT_DIR/"Field_test.pdf")

    def test_cluster(self):
        """Test the cluster detection with both methods."""
        trajectory = traj(str(FIELD_DIR/"cahn-hilliard-model"),
                          mode="field", reload=False)
        fig, axe = plot.new(ncols=3)
        plot.field(axe[0], trajectory[-1].data("c"), *trajectory[-1].grid)
        _, labels = identify_clusters(trajectory[-1].data("c"))
        plot.field(axe[1], labels, *trajectory[-1].grid)
        _, labels = identify_clusters(
            trajectory[-1].data("c"),
            cutoff=0.7, method="threshold"
        )
        plot.field(axe[2], labels, *trajectory[-1].grid)
        fig.savefig(PLOT_DIR/"Cluster_test.pdf")


if __name__ == '__main__':
    unittest.main()