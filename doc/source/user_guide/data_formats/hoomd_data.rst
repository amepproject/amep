HOOMD-blue data
---------------

GSD files from the simulation software `HOOMD-blue <https://glotzerlab.engin.umich.edu/hoomd-blue/>`_ 
can be loaded with ``amep.traj.load`` and the keyword argument ``mode='hoomd'``.
The instructions on how to store the simulation data with HOOMD-blue, can be found 
on their `documentation <https://hoomd-blue.readthedocs.io/>`_ or directly in their
`examples <https://hoomd-blue.readthedocs.io/en/v5.0.1/tutorial/00-Introducing-HOOMD-blue/06-Equilibrating-the-System.html>`_.
At the bottom of this page, we show a brief excerpt of the HOOMD-blue examples, showing
the the necessary Python code to create a ``gsd_writer``.

AMEP uses the ``hoomd`` module of the python package ``gsd``, the documentation of 
which can be found `here <https://gsd.readthedocs.io/>`_.

Both, HOOMD-blue and the ``gsd`` python package, are developed by the 
`Glotzer Group <https://glotzerlab.engin.umich.edu/home/>`_. More information can also 
be found on their GitHub `<https://github.com/glotzerlab>`_.

.. code-block:: python

    import hoomd

    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=12)

    # set up simulation

    gsd_writer = hoomd.write.GSD(
        filename="trajectory.gsd", trigger=hoomd.trigger.Periodic(1000), mode="xb"
    )
    simulation.operations.writers.append(gsd_writer)

    # run simulation

    gsd_writer.flush()
