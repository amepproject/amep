``h5amep`` file format
----------------------

Based on the `Hierarchical Data Format version 5 <https://www.hdfgroup.org/solutions/hdf5/>`_ (HDF5), `<https://github.com/HDFGroup/hdf5>`_, 
**AMEP** introduces a new file format ``h5amep`` to store simulation data and additional metadata. 
This format is used in the backend of **AMEP**. The HDF5 files are structured into groups and datasets. 
The ``h5amep`` format has the following groups, subgroups, and attributes:

.. code-block:: none

    h5amep root
    \amep
    \info
        \authors
        \software
    \scripts
    \params
    \type
    \particles or \fields
    \frames
        \steps
        \times
        \[frame0]
            \coords
            \velocities
            \...
        \[frame1]
            \...
        \...

The group ``amep`` contains information about the **AMEP** version that has been used to create the 
``h5amep`` file. The group ``info`` contains the saved information about authors and software. 
The ``scripts`` group gives the possibility to save text files such as simulation scripts and 
log files that correspond to the simulation data. In the ``params`` group, **AMEP** stores 
parameters such as the simulation timestep for example. Additional simulation parameters can be added. 
The attribute ``type`` contains a flag about the type of data stored in the ``h5amep`` file. 
This can be either ``"particle"`` or ``"field"``. The groups ``particles`` and ``fields`` 
contain user-defined information about the particles and the fields used in the simulation, respectively. 
Finally, the group ``frames`` contains multiple datasets and subgroups with the simulation data. 
The dataset ``steps`` stores a list of all the frame numbers (i.e., the number of timesteps for each frame) 
and the dataset ``times`` the corresponding (physical) time, while the individual frames of the simulation 
are stored in subgroups of ``frames`` named by their simulation step. Within an individual frame, the 
simulation data is stored, e.g., coordinates and velocities for particle-based simulations or density 
and concentrations for continuum simulations, as separate datasets.


This section is an excerpt of the **AMEP** `publication <https://doi.org/10.1016/j.cpc.2024.109483>`_.
