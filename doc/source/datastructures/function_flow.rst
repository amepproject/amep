========
Workflow
========

The following flowchart demonstrates the design of **AMEP**. The three layers “Data
Storage”, “Data Analysis”, and “Result Storage” outline the unification of storing and
analyzing simulation data and analysis results. The simulation data is stored in HDF5
files together with various metadata such as simulation parameters, simulation scripts,
software information, and author information. For the data analysis, this data is loaded
into Python as a Trajectory object which is then either used as an input to the evaluate
module or to directly access individual frames of the trajectory and the contained data
as NumPy arrays, which can then be analyzed using **AMEP**’s various submodules. Analysis
results are stored in an HDF5-based data format as well. This format can be used as a
container for multiple analysis results and can be loaded with **AMEP** as a DataBase object,
from which individual observables can be returned as EvalData object. The latter again
allows to return the data as NumPy arrays for further processing and visualization. The
red arrows show a typical workflow of first loading the trajectory, second, calculating an 
observable, third, storing the result in an HDF5 file, and finally visualizing the results.

.. image:: /_static/images/amep-flowchart.png
  :width: 600
  :align: center