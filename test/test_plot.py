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
"""Test units for all functions in the plot part of amep"""
import unittest
from requests import get
from pathlib import Path
from zipfile import ZipFile
from random import choice
from numpy import linspace, exp
import numpy as np
from matplotlib.pyplot import close
from amep import plot, load
from amep.trajectory import FieldTrajectory, ParticleTrajectory

DATA_DIR = Path("./data")
FIELD_DIR = DATA_DIR/"fields"
PLOT_DIR = DATA_DIR/"plots"
LATEX_TESCASES = {"<": r"\textless ",
                  ">": r"\textgreater ",
                  "&": r"\&",
                  "^": r"\^{}",
                  "_": r"\_",
                  "3^5>9^2": r"3\^{}5\textgreater 9\^{}2"
                  }
SERVER_URL: str = "https://kuno.fkp.physik.tu-darmstadt.de"
WEB_FOLDER: str = "d/a3d9887b8a5747e0a56e/files/?p=%2F"
FILE_NAMES: tuple[str] = ("fields", "particles")
FILE_END: str = ".zip"
URL_END: str = "&dl=1"


def get_test_files():
    for file_name in FILE_NAMES:
        zip_path = DATA_DIR/f"{file_name}{FILE_END}"
        file_url = f"{SERVER_URL}/{WEB_FOLDER}{file_name}{FILE_END}{URL_END}"
        # try:
        print(f"Downloading {file_name}")
        with open(zip_path, "wb") as ofile:
            ofile.write(get(file_url, allow_redirects=True).content)
        print(f"unziping {file_name}")
        print(zip_path)
        with ZipFile(zip_path, "r") as z_file:
            z_file.extractall(DATA_DIR)
            zip_path.unlink()
        # except Exception as exc:
        #     print(f"Something failed while getting {file_name}.\nNamely{exc}")


class TestPlotMethods(unittest.TestCase):
    """A test Case for the plot functions"""
    @classmethod
    def setUpClass(cls):
        DATA_DIR.mkdir(exist_ok=True)
        PLOT_DIR.mkdir(exist_ok=True)

        if not all((DATA_DIR/file_name).is_dir() for file_name in FILE_NAMES):
            get_test_files()
        cls.trajs = [*(load.traj(str(directory)) for directory
                       in (DATA_DIR/"particles").iterdir()),
                     *(load.traj(str(directory), mode="field") for directory
                       in (DATA_DIR/"fields").iterdir())
                     ]

    def test_to_latex(self):
        """Tests some cases for the to_latex string conversion"""
        for key, value in LATEX_TESCASES.items():
            self.assertEqual(plot.to_latex(key), value)

    def test_plot_and_save(self):
        """Plot and save test"""
        fig, _ = plot.new()
        fig.savefig(PLOT_DIR/"test.pdf")
        close(fig)

    def test_set_locators(self):
        """New Plot, set locators, save away"""
        fig, axes = plot.new(nrows=2, ncols=2)
        plot.set_locators(axes)
        close(fig)
        fig2, axe = plot.new()
        plot.set_locators(axe)
        fig.savefig(PLOT_DIR/"test_locators.pdf")
        close(fig2)

    def test_format_axis(self):
        """Testing the format axis function"""
        fig, axes = plot.new(nrows=2, ncols=2)
        plot.format_axis(axes, axiscolor="blue")
        close(fig)
        fig2, axe = plot.new()
        plot.format_axis(axe)
        fig.savefig(PLOT_DIR/"test_axe_format.pdf")
        close(fig2)

    def test_add_colorbar(self):
        """Testing the add_colorbar and linear_mappable functions."""
        mappy = plot.linear_mappable("viridis", 0, 1)
        fig, axes = plot.new(nrows=2, ncols=2)
        plot.add_colorbar(fig, axes[1], mappy, (0, 1, 1, 1))
        close(fig)
        fig2, axes2 = plot.new()
        plot.add_colorbar(fig2, axes2, mappy, (0, 1, 1, 1))
        close(fig2)

    def test_colored_line(self):
        """Testing the colored_line function"""
        fig, axe = plot.new()
        x_vals = linspace(0, 1)
        y_vals = x_vals**2
        plot.colored_line(axe, x_vals, y_vals)
        close(fig)

    def test_add_inset(self):
        """Testing the add_inset function"""
        x_data = linspace(-2, 2)
        y_data = exp(-x_data**2)
        sub_extent = (-1, 0.4, 0.2, 0.2)
        # start plot stuff
        fig, axe = plot.new()
        axe.plot(x_data, y_data)
        plot.add_inset(axe, (0, 0, 0.1, 0.1), (0, 1, 0, 1))
        # all the inset configuration
        m_inset = plot.add_inset(axe, (-2, 0.5, 0.4, 0.4),
                                 sub_extent)
        m_inset.plot(x_data, y_data)
        m_inset.set_xlim(sub_extent[0], sub_extent[0]+sub_extent[2])
        m_inset.set_ylim(sub_extent[1], sub_extent[1]+sub_extent[3])
        m_inset.set_xticklabels([])
        m_inset.set_yticklabels([])
        plot.format_axis(m_inset, axiscolor="orange")
        fig.savefig(PLOT_DIR/"test_inset.pdf")
        close(fig)

    def test_animation(self):
        """Test the animation facilities of AMEP."""
        f_trajectory = choice([traj for traj in self.trajs
                               if isinstance(traj, FieldTrajectory)])
        OUT_FIELD = PLOT_DIR/"field_vid.gif"
        p_trajectory = choice([traj for traj in self.trajs
                               if isinstance(traj, ParticleTrajectory)])
        OUT_PARTICLES = PLOT_DIR/"particle_vid.gif"
        plot.animate_trajectory(f_trajectory, OUT_FIELD)
        plot.animate_trajectory(p_trajectory, OUT_PARTICLES)

    def test_ll_video(self):
        trajectory = choice([traj for traj in self.trajs
                             if isinstance(traj, FieldTrajectory)])
        fig, axie = plot.new((1, 1))
        image = axie.imshow(trajectory[0].data("c"))
        fig.colorbar(image)

        def make_frame(frame: np.ndarray) -> list:
            now = frame.data("c")
            nowmin = np.min(now)
            nowmax = np.max(now)
            image.set_array(now)
            image.set_clim(nowmin, nowmax)
            return [image,]

        plot.create_video(fig, make_frame, trajectory,
                          PLOT_DIR/"ll_vid.gif",
                          output_codec="libx264", bitrate=500000)
        close(fig)

if __name__ == '__main__':
    unittest.main()
