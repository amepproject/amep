.. _data_formats_label:

Supported data formats
======================

**AMEP** is compatible with multiple data formats.
An overview can be found in the table below. We are continuously
improving the supported data formats to increase usability.

If you are missing a specific reader, you are welcome to create
an issue (see :ref:`support_label`) or write your own reader and
help us by contributing it to the **AMEP** project
(see :ref:`contribute_label`).

.. Alternatively, you can directly save your simulation data to a ``h5amep``
.. file by using a writer that can be downloaded from our GitHub. 
.. They will be available for Python, C++ and Rust.


--------
Overview
--------

.. list-table:: Overview of supported data formats
   :widths: 25 75
   :header-rows: 1

   * - Software
     - Support
   * - LAMMPS
     - ✔ (individual dump files)
   * - LAMMPS binary
     - not yet
   * - Field data
     - ✔
   * - HOOMD-blue
     - ✔
   * - GROMACS
     - ✔


-----------------
Data format pages
-----------------

Below, you can find the individual data formats and a more detailed
description of their supported styles.
More simulation software can be supported by writing your own
**AMEP** reader or modifying an existing one.

.. toctree::
    
    h5amep_data
    lammps_data
    field_data
    hoomd_data
    gromacs_data
