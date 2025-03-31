LAMMPS data
-----------

The simulation data from particle simulation suite 
`Large-scale Atomic/Molecular Massively Parallel Simulator <https://www.lammps.org/>`_ 
(LAMMPS) is the original cause **AMEP** was developed for.
LAMMPS is a favoured tool of active matter researchers because of its fast computations,
open source license and modifyability.

The currently supported LAMMPS output format are the individual text files.
An example of a dump command for 2D simulations could be:

.. code-block:: none

    dump        myDump all custom ${savesnap} dump*.txt id type mass radius x y vx vy mux muy mu omegaz tqz
    dump_modify myDump sort id

Or for reduced informations in the dump files, only positions and velocities:

.. code-block:: none

    dump myDump all custom ${savesnap} dump*.txt id x y vx vy
    dump_modify myDump sort id

We recommend to always sort by id. Unsorted data causes many analyses to return wrong results.
Currently, we also recommend to save the id, type, mass and radius, especially if the particles
are not identical and have different properties.
Missing information in the dump files can also be added in **AMEP** with the following lines:

.. code-block:: python

    for frame in traj:
        frame.add_data("radius", 0.5*np.ones(len(frame.n())))

The keys to add data with ``frame.add_data`` should be the same as the LAMMPS keys for **AMEP**
to work properly.

More information about the dump styles from LAMMPS can be found 
`here <https://docs.lammps.org/dump.html>`_.

An exemplary dataset can be found in the examples on our GitHub `<https://github.com/amepproject/amep/tree/main/examples/data>`_.
