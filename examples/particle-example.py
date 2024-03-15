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
This file contains a basic example on how to use AMEP to analyze 
particle-based simulation data such as obtained from a molecular or Brownian 
dynamics simulations with LAMMPS for example. It is based on the example data 
available at https://github.com/amepproject/amep/tree/main/examples/data.
"""
# load required modules
import amep
import numpy as np

# load simulation data
traj = amep.load.traj(
    './data/lammps',
    mode = 'lammps',
    dumps = 'dump*.txt',
    savedir = './data',
    trajfile = 'lammps.h5amep'
)

# visualize the trajectories of the particles
traj.animate('./particles.mp4', xlabel=r'$x$', ylabel=r'$y$')

# calculate the mean-square displacement of the particles
msd = amep.evaluate.MSD(traj)

# calculate the orientational autocorrelation function
oacf = amep.evaluate.OACF(traj)

# calculate the radial distribution function averaged over 10 frames
# (here we skip the first 80 % of the trajectory and do the calculation
# in parallel with 4 jobs)
rdf = amep.evaluate.RDF(
    traj, nav = 10, skip = 0.8, njobs = 4
)

# save all analysis results in separate HDF5 files
msd.save('./msd.h5')
oacf.save('./oacf.h5')
rdf.save('./rdf.h5')

# save all analysis results in one database file
msd.save('./results-db.h5', database = True)
oacf.save('./results-db.h5', database = True)
rdf.save('./results-db.h5', database = True)

# load all analysis data
results = amep.load.evaluation(
    './results-db.h5',
    database = True
)

# check which data is available within the loaded file
print(results.keys())

# fit the OACF results
def f(t, tau=1.0):
    return np.exp(-t/tau)

fit = amep.functions.Fit(f)
fit.fit(results.oacf.times, results.oacf.frames)

print(f"Fit result: tau = {fit.params[0]} +/- {fit.errors[0]}")

# create a figure object
fig, axs = amep.plot.new(
    (6.5,2),
    ncols = 3,
    wspace = 0.1
)

# plot the MSD in a log-log plot
axs[0].plot(
    results.msd.times,
    results.msd.frames,
    label="data",
    marker=''
)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("MSD")
axs[0].loglog()

# plot the OACF and the fit with logarithmic x axis
axs[1].plot(
    results.oacf.times,
    results.oacf.frames,
    label="data",
    marker=''
)
axs[1].plot(
    results.oacf.times,
    fit.generate(results.oacf.times),
    label="fit",
    marker='',
    color='orange',
    linestyle='--'
)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("OACF")
axs[1].semilogx()
axs[1].legend()

# plot the RDF
axs[2].plot(
    results.rdf.r,
    results.rdf.avg,
    marker=''
)
axs[2].set_xlabel("Distance")
axs[2].set_ylabel("RDF")

# save the figure as a pdf file
fig.savefig("particle-example.pdf")