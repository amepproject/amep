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
"""Test units for the evaluate module."""
import unittest
from pathlib import Path
from matplotlib import use
from amep.load import traj
from amep.evaluate import ClusterGrowth, ClusterSizeDist, Function, SpatialVelCor, RDF, PCF2d, PCFangle, SF2d
from amep.evaluate import VelDist, Dist, EkinRot, EkinTrans, EkinTot
import amep

use("Agg")
DATA_DIR = Path("../examples/data/")

FIELD_DIR: Path = DATA_DIR/"continuum"
PARTICLE_DIR: Path = DATA_DIR/"lammps"
RESULT_DIR: Path = DATA_DIR/"results"
PLOT_DIR: Path = DATA_DIR/"plots"


class TestEvaluateMethods(unittest.TestCase):
    """A test case for all field and continuum data methods"""
    @classmethod
    def setUpClass(cls):
        """Set up needed data"""
        RESULT_DIR.mkdir(exist_ok=True)
        cls.field_traj = traj(DATA_DIR/"continuum.h5amep")
        cls.particle_traj = traj(DATA_DIR/"lammps.h5amep")
    
    def test_cluster_growth(self):
        """Test the cluster growth methods.

        Due to their weighting they have a given order they take
        for arbitrary trajectories.
        This order gets tested here.
        Since ClusterGrowth calls ClusterSizeDist we don't
        have to check it separately.
        """
        self.assertTrue((ClusterGrowth(self.field_traj, scale=1.5,
                                       cutoff=0.8,
                                       ftype="c", mode="mean").frames.sum() <=
                        ClusterGrowth(self.field_traj, scale=1.5,
                                      cutoff=0.8, ftype="c",
                                      mode="weighted mean").frames.sum()))
        self.assertTrue((ClusterGrowth(self.field_traj,
                                       ftype="c",
                                       mode="largest").frames >= 0).all())

    def test_energy_methods(self):
        """Test the energy methods.
        """
        traj = self.particle_traj
        # Ekintrans
        ekintrans = EkinTrans(traj, mass=0.05, skip=0.9, nav=2)
        ekintrans.save(RESULT_DIR/"ekintrans_eval.h5", database=True, name="particles")
        # Ekinrot
        ekinrot = EkinRot(traj, inertia=0.005, skip=0.9, nav=2)
        ekinrot.save(RESULT_DIR/"ekinrot_eval.h5", database=True, name="particles")
        # Ekinrot
        ekintot = EkinRot(traj, mass=0.05, inertia=0.005, skip=0.9, nav=2)
        ekintot.save(RESULT_DIR/"ekintot_eval.h5", database=True, name="particles")

    def test_function(self):
        """Test arbitray function evaluation.
        """
        # MSD
        def msd(frame, start=None):
            vec = start.unwrapped_coords() - frame.unwrapped_coords()
            return (vec ** 2).sum(axis=1).mean()
        msd_eval = Function(
            self.particle_traj, msd,
            nav=self.particle_traj.nframes,
            start=self.particle_traj[0]
            )
        msd_eval.name = "msd"
        msd_eval.save(RESULT_DIR/"msd_eval.h5")


    def test_correlations(self):
        """Test correlation function evaluations.
        """
        svc = SpatialVelCor(self.particle_traj, skip=0.9, nav=2, njobs=4)

        svc.save(RESULT_DIR/"svc.h5")
        rdfcalc = RDF(
                self.particle_traj,
                nav=2, nbins=1000,
                skip=0.9, njobs=4)
        rdfcalc.save(RESULT_DIR/'rdf.h5')

        traj = self.particle_traj
        ftraj = self.field_traj
        # PCF2d
        pcf2d = PCF2d(traj,
                      nav=2, nxbins=50, nybins=50,
                      njobs=4, skip=0.9
                      )
        pcf2d.save(RESULT_DIR/"pcf2d.h5")
        # PCFangle
        pcfangle = PCFangle(
            traj, nav=2, ndbins=50, nabins=50,
            njobs=4, rmax=8.0, skip=0.9
            )
        pcfangle.save(RESULT_DIR/"pcfangle.h5")
        pcfangle = PCFangle(
            traj, nav=2, ndbins=50, nabins=50,
            njobs=4, rmax=3.0, skip=0.9,
            mode="psi6"
            )
        pcfangle = PCFangle(
            traj, nav=2, ndbins=50, nabins=50,
            njobs=4, rmax=3.0, skip=0.9,
            mode="orientations"
            )
        pcfangle = PCFangle(
            traj, nav=2, ndbins=50, nabins=50,
            njobs=4, rmax=3.0, skip=0.9,
            mode="x"
            )
        # SF2d
        psf2d = SF2d(traj, skip=0.9, nav=2)
        psf2d.save(RESULT_DIR/"sf2d_eval.h5", database=True, name="particles")
        fsf2d = SF2d(ftraj, skip=0.9, nav=2, ftype="c")
        fsf2d.save(RESULT_DIR/"sf2d_eval.h5", database=True, name="field")

    def test_distributions(self):
        """Test distribution functions.
        """
        # VelDist
        veldist = VelDist(self.particle_traj, skip=0.9, nav=2)
        veldist.save(RESULT_DIR/"veldist_eval.h5", database=True, name="particles")
        # VelDist
        import numpy as np
        dist = Dist(self.particle_traj, "v*",func=np.linalg.norm, axis=1, skip=0.9, nav=2)
        dist.save(RESULT_DIR/"distv_eval.h5", database=True, name="particles")
        dist = Dist(self.particle_traj, "vx", skip=0.9, nav=2)
        dist.save(RESULT_DIR/"distvx_eval.h5", database=True, name="particles")

    def test_correlation(self):
        """Test spatial correlation evaluation.
        """
        svc = SpatialVelCor(self.particle_traj, skip=0.9, nav=2, njobs=4)

        svc.save(RESULT_DIR/"svc.h5")
        rdfcalc = RDF(
                self.particle_traj,
                nav=2, nbins=1000,
                skip=0.9, njobs=4)
        rdfcalc.save(RESULT_DIR/'rdf.h5')

    def test_parallel(self):
        """Test parallel computation of evaluation method(s).
        """
        import os
        import numpy as np
        print("available threads:", len(os.sched_getaffinity(0))) # on GitHub ~4
        msd_1 = amep.evaluate.MSD(self.particle_traj, nav=20, max_workers=4)
        msd_2 = amep.evaluate.MSD(self.particle_traj, nav=20, max_workers=-1)
        msd_3 = amep.evaluate.MSD(self.particle_traj, nav=20, max_workers=None)
        msd_4 = amep.evaluate.MSD(self.particle_traj, nav=20, max_workers=1)
        self.assertTrue(np.all(msd_1.avg==msd_2.avg),
            '4 thread result differs from -1 thread result'
        )
        self.assertTrue(np.all(msd_2.avg==msd_3.avg),
            '-1 thread result differs from `None` thread result'
        )
        self.assertTrue(np.all(msd_3.avg==msd_4.avg),
            '`None` thread result differs from 1 thread result'
        )
