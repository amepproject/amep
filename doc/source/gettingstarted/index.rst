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
``conda`` is recommended. To use all animation features, you need to install 
FFmpeg (https://ffmpeg.org/) on your machine (see below).

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



|
|
|
|

========
Examples
========

.. include:: amep-examples.md
   :parser: myst_parser.sphinx_
