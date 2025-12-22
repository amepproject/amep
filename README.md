[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/amepproject/amep/main/pyproject.toml&color=yellow?style=flat-square)
[![Docs](https://img.shields.io/badge/documentation-amepproject.de-blue?style=flat-square)](https://amepproject.de)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.cpc.2024.109483-orange?style=flat-square)](https://doi.org/10.1016/j.cpc.2024.109483)


<!-- <p align="center"><img src="https://raw.githubusercontent.com/amepproject/amep/main/doc/source/_static/images/amep-logo_v2.png" alt="amep logo" width="200" height="200"/></p> -->
<img src="https://raw.githubusercontent.com/amepproject/amep/main/doc/source/_static/images/amep-logo_v2.png"
     alt="amep logo"
     width="200"
     height="200"
     align="left"
     style="margin-right: 20px; margin-bottom: 20px;" />

**AMEP** is a Python library that focuses on the fast and user-friendly analysis 
of active and soft matter simulations. It can natively analyze data from molecular 
dynamics, Brownian dynamics, and continuum simulations from software such as LAMMPS, 
HOOMD-blue, and GROMACS. 

With a plethora of methods for calculating observables 
and visualizing results, AMEP is suitable for calculating complex observables 
equally for advanced studies of active and soft matter, as well as for beginners 
in the field. Computationally intensive methods are parallelized to run on systems 
ranging from laptops and workstations to high-performance computing clusters.

AMEP utilizes the simplicity of NumPy for users to extract data from the 
internal functions, which allows for easy extension and individualization of 
analyses and handling of results and data with existing Python workflows. Additionally, 
AMEP provides an efficient data format for saving both simulation data and analysis 
results in a binary file based on the well-established [HDF5](https://www.hdfgroup.org/solutions/hdf5/) 
file format.

The methods range from correlation functions and order parameters to cluster detection 
and coarse-graining methods. Examples and the documentation can be found on our homepage 
[amepproject.de](https://amepproject.de). AMEP can be installed via pip and conda.



# How to cite AMEP

If you use AMEP for a project that leads to a scientific publication, please acknowledge 
the use of AMEP within the body of your publication for example by copying
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

The AMEP Python library can be installed via `pip`, `conda`, or by manually adding 
the `amep` directory to your Python path. Installation via `pip` or `conda` is 
recommended. To use all plot animation features, please additionally install 
FFmpeg (https://ffmpeg.org/) on your machine (see below).

## Installation via pip

AMEP can be simply installed from [PyPI](https://pypi.org/project/amep/) 
via 

```bash
pip install amep
```

## Installation via conda

AMEP can be simply installed from 
[conda-forge](https://anaconda.org/conda-forge/amep) via 

```bash
conda install conda-forge::amep
```

## Manual installation

Before installing AMEP manually, ensure that your Python environment 
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

## Python environment

For system cleanliness and easy dependency management, we recommend to use
virtual environments as a good practice using Python. You can create and 
activate one by following the [official Python instructions](https://docs.python.org/3/library/venv.html). 
Here are the instructions for Linux or macOS (for Microsoft Windows you 
may adapt the path formatting to the Windows specific style).

```bash
python3 -m venv amepenv
source amepenv/bin/activate
```

Depending on you Python installation, you may need to use `python3` or `python`.
The virtual environment `amepenv` will be created in the directory you have 
your terminal running. Follow the official instructions linked above for more
details.


## FFmpeg

AMEP provides the possibility to animate plots and trajectories. 
To enable the animation features, _FFmpeg must be installed on the device on which you run AMEP_.
FFmpeg is not automatically installed when you install 
AMEP. Please visit [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) 
to download FFmpeg and to get further information on how to install FFmpeg on your machine.


# Getting started

The following example briefly demonstrates the AMEP workflow. A typical 
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

For more detailed examples, check the [examples](https://github.com/amepproject/amep/tree/main/examples)
directory and check out our homepage [amepproject.de](https://amepproject.de).


# Feature overview

The following table is a subset of functions and methods implemented in AMEP.
It does not cover all analysis methods and shall only be seen as a exemplary list.
Please go to our [API Reference](https://amepproject.de/stable/api.html) for the
complete documentation of AMEP and its implemented features.

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

In the following, we provide a list of all AMEP modules together with a 
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

For details on the supported data formats such as LAMMPS, GROMACS, HOOMD-blue,
or continuum simulations, please refer to the documentation section 
[Supported data formats](https://amepproject.de/stable/user_guide/data_formats/index.html)
on our homepage or the corresponding files 
[here on GitHub](https://github.com/amepproject/amep/tree/main/doc/source/user_guide/data_formats).

# Support
If you need support for using AMEP, please feel free to contact us either 
directly on our [GitHub discussions](https://github.com/amepproject/amep/discussions) 
page. If you find a bug or inconsistency, please create an [issue](https://github.com/amepproject/amep/issues).

## Creating issues
To create an issue, go to [https://github.com/amepproject/amep/issues](https://github.com/amepproject/amep/issues) and
click on `New issue`. Then, continue with the following steps:

1. Add a short and clear title.
2. Write a precise description of the bug which you found. If you got an error message, add it to the description together with a short code snippet with which you can reproduce the error.
3. If it is already known how the bug can be fixed, please add a short to-do list to the description.

When creating issues, text is written as markdown, which allows formatting text, code, or 
tables for example. A useful guide can be found [here](https://www.markdownguide.org/).


# Contributing
Contributions are always welcome! If you want to contribute to this project, 
please check the file [CONTRIBUTING.md](https://github.com/amepproject/amep/blob/main/CONTRIBUTING.md).
You can also send us a message or post in our [GitHub discussions](https://github.com/amepproject/amep/discussions).

# Authors and Contributors
The list of contributors can be found in the file [CONTRIBUTORS.md](https://github.com/amepproject/amep/blob/main/CONTRIBUTORS.md)


# Acknowledgments
Many thanks to the whole 
[group of Benno Liebchen](https://www.ipkm.tu-darmstadt.de/research_ipkm/liebchen_group/index.en.jsp) 
at the Institute for Condensed Matter Physics at Technical University of 
Darmstadt for testing and supporting AMEP, for fruitful discussions, and 
for very helpful feedback. Additionally, the authors gratefully acknowledge the 
computing time provided to them at the NHR Center NHR4CES at TU Darmstadt 
(project number p0020259). This is funded by the Federal Ministry of Education 
and Research, and the state governments participating on the basis of the 
resolutions of the GWK for national high performance computing at universities 
([https://www.nhr-verein.de/unsere-partner](https://www.nhr-verein.de/unsere-partner)).

# License
The AMEP library is published under the GNU General Public License, 
version 3 or any later version. Please see the file [LICENSE](https://github.com/amepproject/amep/blob/main/LICENSE) for more 
information.
