.. _getting_started_label:

Getting Started
===============

Below you can find everything that is needed to start using AMEP for 
analyzing your simulation data. The installation guide explains how AMEP 
can be installed and the examples serve as an easy starting point. 


============
Installation
============

The AMEP library can be installed via ``pip``, ``conda``, or by manually 
adding the ``amep`` directory to your Python path. Installation via ``pip`` or 
``conda`` is recommended. To use all plot animation features, please 
additionally install FFmpeg (https://ffmpeg.org/) on your machine (see below).

--------------------
Installation via pip
--------------------

AMEP can be simply installed from `PyPI <https://pypi.org/project/amep/>`_ 
via 

.. code-block:: bash

    pip install amep


----------------------
Installation via conda
----------------------

AMEP can be simply installed from 
`conda-forge <https://anaconda.org/conda-forge/amep>`_ via 

.. code-block:: bash

    conda install conda-forge::amep


-------------------
Manual installation
-------------------

Before installing AMEP manually, ensure that your Python environment fulfills 
the required specifications as published together with each release.
If your Python environment is set up, download the latest version from
https://github.com/amepproject/amep and extract 
the zipped file. Then, add the path to your Python path and import ``amep``:

.. code-block:: python
   
   import sys
   sys.path.append('/path/to/amep-<version>')
   import amep

Alternatively, you can add the path permanently to your Python path by adding the line

.. code-block:: bash

   export PYTHONPATH="${PYTHONPATH}:/path/to/amep-<version>"

to the ``.bash_profile file`` (Linux only). If you use the Anaconda distribution,
you can alternatively add the ``amep`` directory to ``Lib/site-packages`` in the Anaconda installation path.


------------------
Python environment
------------------

For system cleanliness and easy dependency management, we recommend to use
virtual environments as a good practice using Python. You can create and 
activate one by following the `official Python instructions <https://docs.python.org/3/library/venv.html>`_.
Here are the instructions for Linux or macOS (for Microsoft Windows you 
may adapt the path formatting to the Windows specific style).

.. code-block:: bash

   python3 -m venv amepenv
   source amepenv/bin/activate

Depending on you Python installation, you may need to use ``python3`` or ``python``.
The virtual environment ``amepenv`` will be created in the directory you have 
your terminal running. Follow the official instructions linked above for more
details.


------
FFmpeg
------

AMEP provides the possibility to animate plots and trajectories. 
To enable all animation features, *FFmpeg must be installed on the device on which you run AMEP*.
FFmpeg is not automatically installed when you install 
AMEP. Please visit the `FFmpeg download <https://ffmpeg.org/download.html>`_ 
page to download FFmpeg and to get further information on how to install 
FFmpeg on your machine.



========
Examples
========

The following examples serve as a starting point for everyone who uses AMEP for the first time. The examples are based on the data available at https://github.com/amepproject/amep/tree/main/examples/data:

`Download zip <https://download-directory.github.io/?url=https://github.com/amepproject/amep/tree/main/examples/data>`_

In the first example, we analyze simulation data from a LAMMPS simulation of active Brownian particles, in the second example of a continuum simulation of the Keller-Segel model for chemotaxis.

----------------------------------------------------------
Example 1: Particle-based data (active Brownian particles)
----------------------------------------------------------
First, we import AMEP and NumPy:

.. code-block:: python

    import amep
    import numpy as np

Next, we load the simulation data and animate it:

.. code-block:: python

    # load simulation data (returns a ParticleTrajectory object)
    traj = amep.load.traj(
        './data/lammps',
        mode = 'lammps',
        dumps = 'dump*.txt',
        savedir = './data',
        trajfile = 'lammps.h5amep'
    )
    # visualize the trajectories of the particles
    traj.animate('./particles.mp4', xlabel=r'$x$', ylabel=r'$y$')

.. image:: /_static/images/examples/examples-particles.gif
  :width: 400
  :align: center

Next, we calculate three observables: the mean-square displacement (MSD), the orientational autocorrelation function (OACF), and the radial distribution function (RDF).

.. code-block:: python

    # calculate the mean-square displacement of the particles
    msd = amep.evaluate.MSD(traj)

    # calculate the orientational autocorrelation function
    oacf = amep.evaluate.OACF(traj)

    # calculate the radial distribution function averaged over 10 frames
    # (here we skip the first 80 % of the trajectory and do the calculation
    # in parallel with 4 jobs)
    rdf = amep.evaluate.RDF(
        traj, nav = 10, skip = 0.8, njobs = 4
    )

Let us now save the results in a file:

.. code-block:: python

    # save all analysis results in separate HDF5 files
    msd.save('./msd.h5')
    oacf.save('./oacf.h5')
    rdf.save('./rdf.h5')

Alternatively, you can save all results in one HDF5 file using AMEP's evaluation database feature:

.. code-block:: python

    # save all analysis results in one database file
    msd.save('./results-db.h5', database = True)
    oacf.save('./results-db.h5', database = True)
    rdf.save('./results-db.h5', database = True)

The results can later be loaded using the amep.load.evaluation function for further processing.

Finally, we will exemplarily fit the orientational correlation function to extract the correlation time and plot all results using AMEP's Matplotlib wrapper. For that, we will first load the previously stored analysis results from the database file. Second, we will define the fit function and plot the results.

.. code-block:: python

    # load all analysis data
    results = amep.load.evaluation(
        './results-db.h5',
        database = True
    )
    # check which data is available within the loaded file
    print(results.keys())

    # fit the OACF results
    def f(t, tau=1.0):
        return np.exp(-t/tau)

    fit = amep.functions.Fit(f)
    fit.fit(results.oacf.times, results.oacf.frames)

    print(f"Fit result: tau = {fit.params[0]} +/- {fit.errors[0]}")

    # create a figure object
    fig, axs = amep.plot.new(
        (6.5,2),
        ncols = 3,
        wspace = 0.1
    )
    # plot the MSD in a log-log plot
    axs[0].plot(
        results.msd.times,
        results.msd.frames,
        label="data",
        marker=''
    )
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("MSD")
    axs[0].loglog()

    # plot the OACF and the fit with logarithmic x axis
    axs[1].plot(
        results.oacf.times,
        results.oacf.frames,
        label="data",
        marker=''
    )
    axs[1].plot(
        results.oacf.times,
        fit.generate(results.oacf.times),
        label="fit",
        marker='',
        color='orange',
        linestyle='--'
    )
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("OACF")
    axs[1].semilogx()
    axs[1].legend()
    
    # plot the RDF
    axs[2].plot(
        results.rdf.r,
        results.rdf.avg,
        marker=''
    )
    axs[2].set_xlabel("Distance")
    axs[2].set_ylabel("RDF")
    
    # save the figure as a pdf file
    fig.savefig("particle-example.pdf")
    
.. image:: /_static/images/examples/particle-example.png
  :width: 600
  :align: center


----------------------------------------------
Example 2: Continuum data (Keller-Segel model)
----------------------------------------------
First, we load the simulation data:

.. code-block:: python

    # load simulation data (returns a FieldTrajectory object)
    traj = amep.load.traj(
        './data/continuum',
        mode = 'field',
        dumps = 'field_*.txt',
        timestep = 0.01,
        savedir = './data',
        trajfile = 'continuum.h5amep'
    )
    
Next, let us check which data is included within each frame of the trajectory file:

.. code-block:: python

    print(traj[0].keys)
    
Here, 'c' denotes the chemical field and 'p' the bacterial density. In the following, we will analyze the latter. Let us first animate it:

.. code-block:: python

    # visualize the time evolution of the bacterial density p
    traj.animate('./field.mp4', ftype='c', xlabel=r'$x$', ylabel=r'$y$', cbar_label=r'$c(x,y)$')
    
.. image:: /_static/images/examples/examples-field.gif
  :width: 400
  :align: center
  
Next, we calculate and plot the local density distribution. Note that the following line is calculating the local density distribution for each frame within the trajectory. It is then averaging over all the results, i.e., it is performing a time average (ldd.avg). If the simulation is not in a steady state, one has be careful. Here, clearly not all frames are in the steady state. However, the results for each individual frame are still accessible (ldd.frames). We will use them here to plot the local density distribution for three different frames.

.. code-block:: python

    # calculate the local density distribution
    ldd = amep.evaluate.LDdist(
        traj, nav = traj.nframes, ftype = 'p'
    )
    # create a new figure object
    fig, axs = amep.plot.new()
    
    # plot the results for three different frames
    axs.plot(
        ldd.ld, ldd.frames[0,0],
        label = traj.times[0]
    )
    axs.plot(
        ldd.ld, ldd.frames[5,0],
        label = traj.times[5]
    )
    axs.plot(
        ldd.ld, ldd.frames[10,0],
        label = traj.times[10]
    )
    
    # add legends and labels
    axs.legend(title = 'Time')
    axs.set_xlabel(r'$\rho$')
    axs.set_ylabel(r'$p(\rho)$')
    
    # save the plot as a pdf file
    fig.savefig('./continuum-example.pdf')
    
.. image:: /_static/images/examples/continuum-example.png
  :width: 400
  :align: center
  
Finally, let us save the analysis results in an HDF5 file:

.. code-block:: python

    ldd.save('./ldd.h5')

