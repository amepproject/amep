[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub Discussions](https://img.shields.io/github/discussions/amepproject/amep)](https://github.com/amepproject/amep/discussions)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Famepproject%2Famep%2Fmain%2Fpyproject.toml)
[![Static Badge](https://img.shields.io/badge/documentation-amepproject.de-blue)](https://amepproject.de)
[![Pepy Total Downlods](https://img.shields.io/pepy/dt/amep?label=pypi%7Cdownloads)](https://pypi.org/project/amep/)
[![Conda Downloads](https://img.shields.io/conda/d/conda-forge/amep)](https://anaconda.org/conda-forge/amep)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/amepproject/amep/test.yml?label=pipeline)](https://github.com/amepproject/amep/actions)
[![Static Badge](https://img.shields.io/badge/arXiv-2404.16533-brown)](https://doi.org/10.48550/arXiv.2404.16533)


<center><img src="https://raw.githubusercontent.com/amepproject/amep/main/doc/source/_static/images/amep-logo_v2.png" alt="amep logo" width="200" height="200"/></center>

The **AMEP** (**A**ctive **M**atter **E**valuation **P**ackage) Python library 
is a powerful tool for analyzing data from molecular-dynamics (MD), 
Brownian-dynamics (BD), and continuum simulations. It comprises various 
methods to analyze structural and dynamical properties of condensed matter 
systems in general and active matter systems in particular. **AMEP** is 
exclusively built on Python, and therefore, it is easy to modify and allows to 
easily add user-defined functions. **AMEP** provides an efficient data format 
for saving both simulation data and analysis results based on the HDF5 file 
format. To be fast and usable on modern HPC (**H**igh **P**erformance 
**C**omputing) hardware, the methods are optimized to run also in parallel.


# How to cite AMEP

If you use **AMEP** for a project that leads to a scientific publication, please acknowledge 
the use of **AMEP** within the body of your publication for example by copying or adapting 
the following formulation:

*Data analysis for this publication utilized the AMEP library [1].*

> [1] Hecht L., Dormann, K.-R., Spanheimer, K. L., Ebrahimi, M., Cordts, M., Mandal, S., 
>     Mukhopadhyay, A. K. & Liebchen, B. (2025). "AMEP: The Active Matter Evaluation Package for Python", 
>     *Comput. Phys. Commun., 309*, 109483. https://doi.org/10.1016/j.cpc.2024.109483


The publication is available as open access at [Comput. Phys. Commun.](https://doi.org/10.1016/j.cpc.2024.109483).
To cite this reference, you can use the following BibTeX entry:

```bibtex
@article{hecht2025amep,
    title = {AMEP: The Active Matter Evaluation Package for Python}, 
    author = {Lukas Hecht and 
                Kay-Robert Dormann and 
                Kai Luca Spanheimer and 
                Mahdieh Ebrahimi and 
                Malte Cordts and 
                Suvendu Mandal and 
                Aritra K. Mukhopadhyay and 
                Benno Liebchen},
    journal = {Computer Physics Communications},
    year = {2025},
    volume = {309},
    pages = {109483},
    doi = {https://doi.org/10.1016/j.cpc.2024.109483}
}
```


# Installation

The **AMEP** library can be installed via `pip`, `conda`, or by manually adding 
the `amep` directory to your Python path. Installation via `pip` or `conda` is 
recommended. To use all plot animation features, please additionally install 
FFmpeg (https://ffmpeg.org/) on your machine (see below).

## Installation via pip

**AMEP** can be simply installed from [PyPI](https://pypi.org/project/amep/) 
via 

```bash
pip install amep
```

## Installation via conda

**AMEP** can be simply installed from 
[conda-forge](https://anaconda.org/conda-forge/amep) via 

```bash
conda install conda-forge::amep
```

## Manual installation

Before installing **AMEP** manually, ensure that your Python environment 
fulfills the required specifications as published together with each release. 
If your Python environment is set up, download the latest version from 
[https://github.com/amepproject/amep](https://github.com/amepproject/amep) 
and extract the zipped file. Then, add the path to your Python path and 
import `amep`:

```python
import sys
sys.path.append('/path/to/amep-<version>')
import amep
```

Alternatively, you can add the path permanently to your Python path by adding 
the line

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/amep-<version>"
```

to the `.bash_profile` file (Linux only). If you use the Anaconda distribution, 
you can alternatively add the `amep` directory to `Lib/site-packages` in the 
Anaconda installation path.

## FFmpeg

**AMEP** provides the possibility to animate plots and trajectories. 
**To enable all animation features, FFmpeg must be installed on the device on** 
**which you run AMEP**. FFmpeg is not automatically installed when you install 
**AMEP**. Please visit [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) 
to download FFmpeg and to get further information on how to install FFmpeg on your machine.


# Getting started

The following example briefly demonstrates the **AMEP** workflow. A typical 
task is to calculate the average of an observable over several frames of the 
simulation (time average). In the example below, we first load LAMMPS 
simulation data stored as individual `dump*.txt` files for each frame, and 
second, we calculate and plot the time-averaged radial pair distribution 
function (RDF).

```python
# import the amep library
import amep

# load simulation data (creates an h5amep file and returns a trajectory object)
traj = amep.load.traj('./examples/data/lammps')

# calculate the radial pair-distribution function, skip the first half of the
# trajectory, and average over 10 frames with equal distance in time
rdf = amep.evaluate.RDF(traj, skip=0.5, nav=10, nbins=1000)

# save result in file
rdf.save('./rdf.h5')

# plot the result
fig, axs = amep.plot.new()
axs.plot(rdf.r, rdf.avg)
axs.set_title(amep.plot.to_latex(rdf.name))
axs.set_xlim(0,10)
axs.set_xlabel(r'$r$')
axs.set_ylabel(r'$g(r)$')
amep.plot.set_locators(axs, which='both', major=1, minor=0.2)
fig.savefig(rdf.name + '.png')
fig.savefig(rdf.name + '.pdf')
```

For more detailed examples, check the [examples](https://github.com/amepproject/amep/tree/main/examples) directory.


# Project description

The **AMEP** Python library provides a unified framework for handling 
both particle-based and continuum simulation data. It is made for the analysis 
of molecular-dynamics (MD), Brownian-dynamics (BD), and continuum simulation 
data of condensed matter systems and active matter systems in particular. 
**AMEP** provides a huge variety of analysis methods for both data types that 
allow to evaluate various dynamic and static observables based on the 
trajectories of the particles or the time evolution of continuum fields. For 
fast and efficient data handling, **AMEP** provides a unified framework for 
loading and storing simulation data and analysis results in a compressed, 
HDF5-based data format. **AMEP** is written purely in Python and uses powerful 
libraries such as NumPy, SciPy, Matplotlib, and scikit-image commonly used in 
computational physics. Therefore, understanding, modifying, and building up on 
the provided framework is comparatively easy. All evaluation functions are 
optimized to run efficiently on HPC hardware to provide fast computations. To 
plot and visualize simulation data and analysis results, **AMEP** provides an 
optimized plotting framework based on the Matplotlib Python library, which 
allows to easily plot and animate particles, fields, and lines. Compared to 
other analysis libraries, the huge variety of analysis methods combined with 
the possibility to handle both most common data types used in soft-matter 
physics and in the active matter community in particular, enables the analysis 
of a much broader class of simulation data including not only classical 
molecular-dynamics or Brownian-dynamics simulations but also any kind of 
numerical solutions of partial differential equations. The following table 
gives an overview on the observables provided by **AMEP** and on their 
capability of processing particle-based and continuum 
simulation data.


| Observable | Particles | Fields |
|:-----------|:---------:|:------:|
| **Spatial Correlation Functions:** |||
| RDF (radial pair distribution function) | ✔ | ➖ |
| PCF2d (2d pair correlation function) | ✔ | ➖ |
| PCFangle (angular pair correlation function) | ✔ | ➖ |
| SFiso (isotropic static structure factor) | ✔ | ✔ |
| SF2d (2d static structure factor) | ✔ | ✔ |
| SpatialVelCor (spatial velocity correlation function) | ✔ | ➖ |
| PosOrderCor (positional order correlation function) | ✔ | ➖ |
| HexOrderCor (hexagonal order correlation function) | ✔ | ➖ |
| **Local Order:** |||
| Voronoi tesselation | ✔ | ➖ |
| Local density | ✔ | ✔ |
| Local packing fraction | ✔ | ➖ |
| k-atic bond order parameter | ✔ | ➖ |
| Next/nearest neighbor search | ✔ | ➖ |
| **Time Correlation Functions:** |||
| MSD (mean square displacement) | ✔ | ➖ |
| VACF (velocity autocorrelation function) | ✔ | ➖ |
| OACF (orientation autocorrelation function) | ✔ | ➖ |
| **Cluster Analysis:** |||
| Clustersize distribution | ✔ | ✔ |
| Cluster growth | ✔ | ✔ |
| Radius of gyration | ✔ | ✔ |
| Linear extension | ✔ | ✔ |
| Center of mass | ✔ | ✔ |
| Gyration tensor | ✔ | ✔ |
| Inertia tensor | ✔ | ✔ |
| **Miscellaneous:** |||
| Translational/rotational kinetic energy | ✔ | ➖ |
| Kinetic temperature | ✔ | ➖ |


# Module descriptions

In the following, we provide a list of all **AMEP** modules together with a 
short description.

| **Module:** | **Description:** |
|:------------|:-----------------|
| base.py | base classes (backend) |
| cluster.py | cluster analysis for particle-based data |
| continuum.py | coarse-graining and continuum field analysis |
| evaluate.py | trajectory analysis |
| functions.py | mathematical functions and fitting |
| load.py | loading simulation data and analysis results |
| order.py | spatial order analysis |
| pbc.py | handling of periodic boundary conditions |
| plot.py | visualization and animation |
| reader.py | simulation data reader (backend) |
| spatialcor.py | spatial correlation functions |
| statistics.py | statistical analysis |
| thermo.py | thermodynamic observables |
| trajectory.py | trajectory classes (backend) |
| utils.py | collection of utility functions |


# Data Formats

**AMEP** is compatible with multiple data formats. The current version can load 
particle-based simulation data obtained from LAMMPS (https://www.lammps.org), 
HOOMD-blue, GROMACS and continuum simulation data with the following format: 
The main directory should contain one file with data that stays constant 
throughout the entire simulation such as the boundaries of the simulation 
box, the shape of the underlying grid and the grid coordinates. It's standard 
name is `grid.txt` and it should have the following form:
```
BOX:
<X_min>	<X_max>
<Y_min>	<Y_max>
<Z_min>	<Z_max>
SHAPE:
<nx> <ny> <nz>
COORDINATES: X Y Z
<X_0> <Y_0> <Z_0>
<X_1> <Y_1> <Z_1>
...
```
All data that varies in time is to be put into files named `dump<index>.txt`. 
The index should increase with time, i.e., the file `dump1000.txt` should 
contain the data of the continuum simulation at timestep 1000, and the prefix 
`dump` is user-defined and can be changed (if it is changed, the new naming 
convention has to be specified with the keyword `dumps` in `amep.load.traj`, 
e.g., for files named `field_100.txt`, `field_200.txt`, ..., use 
`dumps='field_*.txt'`). The data files should have the following form:
```
TIMESTEP:
<Simulation timestep>
TIME:
<Physical time>
DATA: <fieldname 0> <fieldname 1> <fieldname 2> <fieldname 3>
<field 0 0> <field 1 0> <field 2 0> <field 3 0>
<field 0 1> <field 1 1> <field 2 1> <field 3 1>
<field 0 2> <field 1 2> <field 2 2> <field 3 2>
...
```

# Support
If you need support for using **AMEP**, we recommend to use our [GitHub discussions](https://github.com/amepproject/amep/discussions) page. If you find a bug, please create an [issue](https://github.com/amepproject/amep/issues).

## Creating issues
To create an issue, go to [https://github.com/amepproject/amep/issues](https://github.com/amepproject/amep/issues) and
click on `New issue`. Then, continue with the following steps:

1. Add a short and clear title.
2. Write a precise description of the bug which you found. If you got an error message, add it to the description together with a short code snippet with which you can reproduce the error.
3. If it is already known how the bug can be fixed, please add a short to-do list to the description.

When creating issues, text is written as markdown, which allows formatting text, code, or 
tables for example. A useful guide can be found [here](https://www.markdownguide.org/).


# Roadmap
Planned new features for future releases are listed as issues in the [issue list](https://github.com/amepproject/amep/issues).

# Contributing
If you want to contribute to this project, please check the file [CONTRIBUTING.md](https://github.com/amepproject/amep/blob/main/CONTRIBUTING.md).

# Contributors/Authors
The following people contributed to **AMEP**:

- Lukas Hecht (creator and lead developer)
- Kay-Robert Dormann (lead developer)
- Kai Luca Spanheimer (developer)
- Aritra Mukhopadhyay (developer)
- Mahdieh Ebrahimi (developer)
- Suvendu Mandal (developer)
- Benno Liebchen (planning)
- Lukas Walter (former developer)
- Malte Cordts (former developer)

# Acknowledgments
Many thanks to the whole 
[group of Benno Liebchen](https://www.ipkm.tu-darmstadt.de/research_ipkm/liebchen_group/index.en.jsp) 
at the Institute for Condensed Matter Physics at Technical University of 
Darmstadt for testing and supporting **AMEP**, for fruitful discussions, and 
for very helpful feedback. Additionally, the authors gratefully acknowledge the 
computing time provided to them at the NHR Center NHR4CES at TU Darmstadt 
(project number p0020259). This is funded by the Federal Ministry of Education 
and Research, and the state governments participating on the basis of the 
resolutions of the GWK for national high performance computing at universities 
([https://www.nhr-verein.de/unsere-partner](https://www.nhr-verein.de/unsere-partner)).

# License
The **AMEP** library is published under the GNU General Public License, 
version 3 or any later version. Please see the file [LICENSE](https://github.com/amepproject/amep/blob/main/LICENSE) for more 
information.
