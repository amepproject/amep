# AMEP example data
In this directory, we provide the simulation data which we have used for all 
**AMEP** examples. This data can either be downloaded directly from the 
repository or with the following download link:

**add the download link here...**

For **AMEP** beginners, we strongly recommend to download this data and run 
the examples to get familiar with the **AMEP** workflow.


## The data directory
All data is contained in the `data` directory. It contains the following 
subdirectories:

- `continuum`: Raw data of a numerical solution of the Keller-Segel model.
- `lammps`: Raw data of a [LAMMPS](https://www.lammps.org) simulation of 
overdamped interacting active Brownian particles.

The file `lammps.h5amep` has been created by running

```python
import amep
traj = amep.load.traj(
    './data/lammps',
    mode = 'lammps',
    dumps = 'dump*.txt',
    savedir = './data',
    trajfile = 'lammps.h5amep'
)
```

and the file `continuum.h5amep` by running

```python
import amep
traj = amep.load.traj(
    './data/continuum',
    mode = 'field',
    dumps = 'field_*.txt',
    timestep = 0.01,
    savedir = './data',
    trajfile = 'continuum.h5amep'
)
```

which are the corresponding **AMEP** data files.

### Run the continuum simulation
To run the continuum simulation by yourself, you have to install the 
[FiPy](https://www.ctcms.nist.gov/fipy/) Python library, which is a 
finite-volume solver for partial differential equations, via

```
pip install fipy
```

or alternatively via

```
conda install conda-forge::fipy
```

Then, you can simply run

```
python /data/continuum/solver.py
```

### Run the LAMMPS simulation
To run the LAMMPS simulation by yourself, you have to install the latest 
stable LAMMPS release. Please download the respective version for your 
operating system from https://www.lammps.org and install it on your machine. 
The example provided by **AMEP** requires the `BROWNIAN` and the `DIPOLE` 
packages. To compile it under Linux, you might use the following commands:

```
cd /path/to/lammps/src
make clean-all
make yes-BROWNIAN
make yes-DIPOLE
make serial
```

Then, you can run the simulation with the following command:

```
/path/to/lammps/src/lmp_serial -in abps.run
```

For further details about LAMMPS, please visit their documentation available
at https://docs.lammps.org/Manual.html.

## Examples
We provide two basic examples, one for the particle-based simulation data 
obtained from the LAMMPS simulation (`particle-example.py`) and one for the 
continuum data (`continuum-data.py`). The Jupyter notebook 
`amep-examples.ipynb` contains the same examples. These examples serve as a 
starting point for everyone who uses **AMEP** for the first time.