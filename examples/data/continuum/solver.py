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
This code solves the Keller-Segel model [1]_ for chemotaxis, which describes 
bacteria that interact with a chemical field:

    ρ̇(x, t) = D_ρ ∇² ρ(x, t)− B ∇ ⋅(ρ(x, t) ∇ c(x, t))
    
    ċ(x, t) = D_c ∇² c(x, t) + ks ρ(x, t) − kd c(x, t)
    
Here, ρ denotes the bacterial density and c the chemical concentration. This 
code uses the FiPy Python library to solve the coupled partial differential 
equations (PDEs) using the finite-volume method.

To run this code, ensure that you have installed FiPy. This can be done via

pip install fipy

or

conda install conda-forge::fipy

for example.

The simulation data is stored in .txt files following the continuum data format
required for AMEP.


References
----------

    .. [1] G. Arumugam and J. Tyagi, Keller-Segel Chemotaxis Models: A Review, 
           Acta Appl. Math. 171, 6 (2021). 
           http://dx.doi.org/10.1007/s10440-020-00374-2

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import fipy

import numpy as np

from tqdm import tqdm

# =============================================================================
# DEFINE SIMULATION PARAMETERS
# =============================================================================

Lx = 50.0 # length of the domain in x direction
Ly = 50.0 # length of the domain in y direction

nx = 50 # number of cells in x direction
ny = 50 # number of cells in y direction

dx = Lx / nx # cell size in x direction
dy = Ly / ny # cell size in y direction

D_p = 1.0 # diffusion coefficient of p
D_c = 1.0 # diffusion coefficient of c

dt = 0.01 # time step size
time_steps = 10000 # number of time steps to take
write_every = 1000 # write output every write_every time steps

B = 2.0 # chemotaxis coefficient
ks = 1.0 # production rate of c
kd = 1.0 # decay rate of c
ini = 1.0 # initial value of p and c

# =============================================================================
# INITIALIZE SIMULATION
# =============================================================================
# Create a 2D grid periodic grid
mesh = fipy.PeriodicGrid2D(
    nx = nx,
    ny = ny,
    dx = dx,
    dy = dy
)
# Define and initialize bacterial (p) and chemical (c) fields
p = fipy.CellVariable(
    name = "p",
    mesh = mesh,
    value = ini
)
c = fipy.CellVariable(
    name = "c",
    mesh = mesh,
    value = ini
)
# Add small fluctuations to the initial conditions
noise = fipy.UniformNoiseVariable(
    mesh = mesh,
    minimum = -0.01,
    maximum = 0.01
)
p[:] = ini + noise
c[:] = ini + noise

# Define the coupled PDEs
eq_p = (
    fipy.TransientTerm(var = p)
    == fipy.DiffusionTerm(coeff = D_p, var = p)
    - fipy.PowerLawConvectionTerm(coeff = B * c.faceGrad, var = p)
)
eq_c = (
    fipy.TransientTerm(var = c)
    == fipy.DiffusionTerm(coeff = D_c, var = c)
    + ks*p
    - fipy.ImplicitSourceTerm(kd, var=c)
)
eq = eq_p & eq_c


# =============================================================================
# SAVE GRID INFO
# =============================================================================
X = mesh.x.value.reshape((nx,ny))
Y = mesh.y.value.reshape((nx,ny))

# Define the min and max values for X, Y, and Z
X_min, X_max = X.min(), X.max()
Y_min, Y_max = Y.min(), Y.max()
Z_min, Z_max = 0.0, 0.0

# Write grid data to grid.txt
with open('grid.txt', 'w') as f:
    
    # Write box information
    f.write("BOX:\n")
    f.write("%s %s\n" %(X_min,X_max))
    f.write("%s %s\n" %(Y_min,Y_max))
    f.write("%s %s\n" %(Z_min,Z_max))
    
    # Write information of the grid shape
    f.write("SHAPE:\n")
    f.write("%s %s %s\n" %(nx,ny,0))
    
    f.write("COORDINATES: X Y Z\n")
    
    # Create coordinate arrays for X, Y, and Z
    X_coords = X.flatten()
    Y_coords = Y.flatten()
    Z_coords = np.zeros_like(X_coords)
    
    # Combine coordinates into a single array
    coordinates = np.column_stack((X_coords, Y_coords, Z_coords))
    
    # Write coordinates to the file
    np.savetxt(f, coordinates, fmt="%.4f %.4f %.4f")

# =============================================================================
# SOLVE COUPLED PDEs
# =============================================================================
for i in tqdm(range(time_steps+1)):  
    # Solve for the next time step
    eq.solve(dt=dt)
    
    # Write data to file
    if (i%write_every == 0):
        # Create the output filename
        output_filename = 'field_%s.txt' %i
        
        # Open the file for writing
        with open(output_filename, 'w') as f:
            # Write simulation time step
            f.write('TIMESTEP:\n')
            f.write('%s\n' %i)
            
            # Write simulation time
            f.write('TIME:\n')
            f.write('%s\n' %(i*dt))
            
            # Write field names
            f.write('DATA: ' + ' '.join(['p', 'c']) + '\n')
            
            # Collect data
            data = np.column_stack((p.value, c.value))
            
            # Write field values       
            np.savetxt(f, data, fmt="%.16f")
