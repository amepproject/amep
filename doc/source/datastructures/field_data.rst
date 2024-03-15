=================
Field data format
=================

**AMEP** is compatible with multiple data formats. The current version can load 
particle-based simulation data obtained from LAMMPS (https://www.lammps.org) 
and continuum simulation data with the following format: The main directory 
should contain one file with data that stays constant throughout the entire 
simulation such as the boundaries of the simulation box, the shape of the 
underlying grid and the grid coordinates. It's standard name is `grid.txt` and 
it should have the following form::

   BOX:
   <X_min> <X_max>
   <Y_min> <Y_max>
   <Z_min> <Z_max>
   SHAPE:
   <nx> <ny> <nz>
   COORDINATES: X Y Z
   <X_0> <Y_0> <Z_0>
   <X_1> <Y_1> <Z_1>
   ...

All data that varies in time is to be put into files named `field_<index>.txt`. 
The index should increase with time, i.e., the file `field_1000.txt` should 
contain the data of the continuum simulation at timestep 1000. The data files 
should have the folowing form::

   TIMESTEP:
   <Simulation timestep>
   TIME:
   <Physical time>
   DATA: <fieldname 0> <fieldname 1> <fieldname 2> <fieldname 3>
   <field 0 0> <field 1 0> <field 2 0> <field 3 0>
   <field 0 1> <field 1 1> <field 2 1> <field 3 1>
   <field 0 2> <field 1 2> <field 2 2> <field 3 2>
   ...

An exemplary dataset can be found `here <https://github.com/amepproject/amep/tree/main/examples/data>`_.