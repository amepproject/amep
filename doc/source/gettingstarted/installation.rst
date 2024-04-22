============
Installation
============

The **AMEP** library can be installed via ``pip``, ``conda``, or by manually 
adding the ``amep`` directory to your Python path. Installation via ``pip`` or 
``conda`` is recommended. To use all plot animation features, please 
additionally install FFmpeg (https://ffmpeg.org/) on your machine (see below).

--------------------
Installation via pip
--------------------

**AMEP** can be simply installed from `PyPI <https://pypi.org/project/amep/>`_ 
via 

.. code-block:: bash

    pip install amep


----------------------
Installation via conda
----------------------

**AMEP** can be simply installed from 
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


------
FFmpeg
------
**AMEP** provides the possibility to animate plots and trajectories. 
**To enable all animation features, FFmpeg must be installed on the device on** 
**which you run AMEP**. FFmpeg is not automatically installed when you install 
**AMEP**. Please visit https://ffmpeg.org/download.html to download FFmpeg and 
to get further information on how to install FFmpeg on your machine.