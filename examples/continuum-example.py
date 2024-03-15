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
continuum simulation data such as obtained from the numerical solution of 
partial differential equations such as the Keller-Segel model for example. It 
is based on the example data available at https://github.com/amepproject/amep/tree/main/examples/data.
"""
import amep

# load simulation data
traj = amep.load.traj(
    './data/continuum',
    mode = 'field',
    dumps = 'field_*.txt',
    timestep = 0.01,
    savedir = './data',
    trajfile = 'continuum.h5amep'
)

# check which data is included within each frame
print(traj[0].keys)

# visualize the time evolution of the bacterial density p
traj.animate('./field.mp4', ftype='c', xlabel=r'$x$', ylabel=r'$y$', cbar_label=r'$c(x,y)$')

# calculate the local density distribution
ldd = amep.evaluate.LDdist(
    traj, nav = traj.nframes, ftype = 'p'
)

# create a new figure object
fig, axs = amep.plot.new()

# plot the results for three different frames
axs.plot(
    ldd.ld, ldd.frames[0,0],
    label = traj.times[0]
)
axs.plot(
    ldd.ld, ldd.frames[5,0],
    label = traj.times[5]
)
axs.plot(
    ldd.ld, ldd.frames[10,0],
    label = traj.times[10]
)

# add legends and labels
axs.legend(title = 'Time')
axs.set_xlabel(r'$\rho$')
axs.set_ylabel(r'$p(\rho)$')

# save the plot as a pdf file
fig.savefig('./continuum-example.pdf')

# save analysis result in HDF5 file
ldd.save('./ldd.h5')