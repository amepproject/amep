.. image:: /_static/images/amep-logo_v2.png
  :width: 200
  :align: center


AMEP |release| documentation
============================

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
results in a binary file based on the well-established `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
file format.

The methods range from correlation functions and order parameters to cluster detection 
and coarse-graining methods. For examples check out our :doc:`Getting Started <gettingstarted/index>` section 
and the documentation can be found in the :doc:`API Reference <api>`. 
AMEP can be installed via pip and conda.

.. .. include:: ../../README.md
..    :parser: myst_parser.sphinx_
..    :start-line: 11

Table of Contents
=================

.. toctree::
   :maxdepth: 1

   gettingstarted/index
   howto_cite
   user_guide/index
   changelog
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

  ..
    * :ref:`search`
