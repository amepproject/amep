# Change Log

All notable changes to **AMEP** will be documented in this file. **AMEP** 
adheres to [semantic versioning](https://semver.org/).

## AMEP 1.0.2 (22 Mai 2024)

### Bug fixes:

* `fps` can now be set by the user in `amep.plot.animate_trajectory`
* bug in `amep.evaluate.MSD` related to incorrect data availability checks fixed

### Contributors:

* Lukas Hecht



## AMEP 1.0.1 (22 Apr 2024)

### Bug fixes:

* bug related to physical times not getting updated when timestep of trajectory object was changed fixed (LH)
* incompatibility with Python 3.12 related to distutils fixed (LH)
* bug in watershed cluster detection related to bubble detection fixed (KS)
* some minor issues in documentation fixed (LH)

### Contributors:

* Lukas Hecht (LH)
* Kai Luca Spanheimer (KS)


## AMEP 1.0.0  (21 Mar 2024)

This is the first public version of **AMEP**. It fully integrates the analysis 
of continuum simulation data, fixes various bugs of version 0.5.0, and provides 
useful new features. It is also the first version that can be installed via 
conda and pip.

### New features:

* gyration tensor and inertia tensor added to `amep.continuum.cluster_properties` (LH)
* new modes for `amep.evaluate.ClusterGrowth` (KS)
* `amep.plot.draw_box` with new text features (KD)
* new cluster detection method for continuum data (KS)
* `pbc` keyword now consitently available in all evaluate classes (LH)
* property time added to `amep.base.BaseField` (KS)
* physical time added to continuum data format (KS)
* parallelized calculation of distance matrix (ME, LH)

### Bug fixes:

* `traj.animate` colorbar bug fixed (LH)
* loading trajectories bug when loading process interrupted fixed (LH, KD)
* bug in `amep.statistics.distribution` for large datasets fixed (ME)
* `amep.evaluate.LDdist` bug fixed related to keyword forwarding (KD)
* 2d data detection for constant z coordinates (KD)
* unclear error message in particle cluster method fixed (LH)
* incorrect normalization in `amep.evaluate.OACF` fixed (LH)
* bug in `amep.evaluate.ClusterSizeDist` related to negative density fields fixed (KS)
* incorrect return in `amep.order.local_density` fixed (LH)
* `amep.plot.add_colorbar` now returns the axis object (LH)
* bug in `amep.spatialcor.spatialcor` fixed related to incorrect calculation for `pbc=True` (LH)
* incorrect backup behavior in `amep.base.BaseEvaluation.save` fixed (LH)
* bug in `amep.load.traj` related to dots in path names fixed (LH, KD)
* bug in `amep.order.next_neighbors` fixed related to incorrect ids from the voronoi tesselation (KD)
* evaluation HDF5 data format unified (LH)
* cluster methods unified (AM)
* bug related to very slow cluster method fixed (LH)
* not working `save` method of `amep.evaluate.HexOrderCor` fixed (LH)
* incorrect directory return of `amep.base.check_path` fixed (LH)
* bug related to multiple warnings printed when loading continuum data fixed (LH)
* bug in `amep.reader.ContinuumReader` related to missing keywords `nth`, `start`, and `stop` fixed (LH)
* error for continuum data with `reload=True` in `amep.load.traj` fixed (LH)
* bug in `amep.base.BaseFrame.data` related to wildcard characters fixed (LH, KD)
* incorrect normalization in `amep.cluster.sf2d` fixed (AM)
* minor bugs in `amep.plot.particles` fixed (KD)
* bug in `amep.order.voronoi_density` related to QhullError fixed (KD)
* problem with repeated indices in `amep.utils.average_func` fixed (LH)
* missing integration limits in `amep.utils.domain_length´ added (LH)
* incorrect calculation of moments in `amep.statistics.binder_cumulant` fixed (LH, LW)
* bug in `amep.pbc.kdtree` related to an error occured when particles are exactly at the border of the simulation box fixed (LH)
* new plot styles are now installed when installing **AMEP** via pip (KS)
* `amep.plot.format_axis` improved (LH, ME)

### Deprecation and removals:

* `local_density` in module `order` replaced by the three functions `local_number_density`, `local_mass_density`, and `local_packing_fraction` (LH)
* `amep.plot.savefig` removed (LH, KD)
* `amep.cluster.cluster` replaced by `amep.cluster.identify` (AM)
* `amep.cluster.csf2d` renamed to `amep.cluster.sf2d` (LH)
* all functions now take the box boundary as an input instead of the box length (LH)
* `amep.continuum.cluster` replaced by `amep.continuum.identify_clusters` and `amep.continuum.cluster_properties` (AM)

### Contributors:

* Lukas Hecht (LH)
* Kay-Robert Dormann (KD)
* Kai Luca Spanheimer (KS)
* Mahdieh Ebrahimi (ME)
* Aritra Mukhopadhyay (AM)
* Lukas Walter (LW)



## AMEP 0.5.0 (26 Oct 2023)

This version includes many new features and fixes important bugs of version 
0.4.0. Additionally, the usability has been improved and the syntax has been 
simplified. This version also has various visualization features and improved 
storage methods for evaluation results. Furthermore, the parallelization of 
methods has been made robust and has been successfully tested on different HPC 
hardware.

### New features:

* progress bar (LH)
* cluster algorithm for particles of different sizes (LH)
* Gaussian kernel density estimation (LH)
* method to plot the simulation box (LH)
* loading continuum data (KS)
* analyzing continuum data with evaluate objects (LH, AM)
* domain length function (KS)
* weighted running mean (KS, KD)
* segmented mean (KS, KD)
* cluster detection for continuum fields (AM, KS)
* Voronoi tesselation (KD, SM)
* cluster properties: center of mass, radius of gyration, end-to-end distance, inertia tensor, linear extension (AM)
* save evaluate results in HDF5 file (LH)
* HDF5-based database for evaluation results (LH)
* plot particles with correct size (ME, AM)
* nearest neighbors and k nearest neighbors (LH)
* general bond order parameter (AM)
* general time correlation function (ME)
* video creation (ME, KD, KS)
* 2d histogram (ME)
* general fit class (KS, AM, MC)
* AMEP plot styles (ME, LH)
* local density calculation from Voronoi diagrams (KD)
* number of next neighbors from Voronoi diagrams (KD)
* plot fields (AM)

### Bug fixes:

* `amep.plot.format_axis` improved (LH)
* hexagonal order parameter calculation improved (SM)
* small bugs in `amep.plot.add_inset` fixed (LH)
* bug in `amep.load.traj` fixed related to wrong error message for `reload=False` (LH)
* `amep.utils.time_average` renamed to `amep.utils.average_func` and returns correct number of outputs (MC)
* wrong values of the structure factor for `q=0` are now excluded from the result (MC)
* bug related to accuracy value in `amep.spatialcor.sf2d` fixed (LH)
* bug in `amep.base.BaseFrame.data` fixed (MC)
* small bugs in evaluate module fixed (LH)
* kdtree bug with coords at box border fixed (LH)
* `amep.pbc.pbc_points` twod keyword simplified (LH)
* storing `h5amep` files in different directory (LW, LH)
* bug related to cancelling parallelized methods (LH)
* parallelization improved (LH)
* `amep.base.BaseFrame.data` bug related to fnmatch (wildcard characters ignored) fixed (LH)
* metadata handling in trajectory files improved and simplified (LH)
* bug fixed in `amep.base.BaseReader` related to temporary `h5amep` files that are not deleted (KD)
* `amep.base.BaseFrame` "get_" removed for better usability (LH)
* Gaussian fit function improved (MC, KS, AM)
* bug in `amep.thermo.kintemp` fixed (ME, KD)
* `amep.spatialcor.pcf_angle` problems if `other_coords` contains only one particle fixed (KS, MC)
* evaluate objects now also have times as property (MC)
* getting data from evaluate objects improved (MC, LH)
* local density calculation now possible for particles of different size (LH)

### Contributors:

* Lukas Hecht (LH)
* Kay-Robert Dormann (KD)
* Kai Luca Spanheimer (KS)
* Aritra Mukhopadhyay (AM)
* Suvendu Mandal (SM)
* Malte Coordts (MC)
* Mahdieh Ebrahimi (ME)
* Lukas Walter (LW)



## AMEP 0.4.0 (22 Feb 2023)

This version fixes many important bugs of version 2.0.0 and now allows to 
efficiently use the new HDF5-based data format. The loading of the data has 
been improved significantly and is robust against data loss. With this version, 
it is also possible to install **AMEP** via `pip`.

### New features:

* `pip` support added (KS)
* possibility to add insets to a plot implemented (KS)
* `axiscolor` option added to `format_axis` (KS)
* `.h5amep` files now also store AMEP version for backwards compatibility (LH)
* non-LaTeX mode for plots implemented (KD)
* `get_data` method improved; allows individual keys and wildcard characters now (LH)
* energy functions added for energy calculations (KD)
* sort particles by id during loading (KD)
* general evaluation function added (KD)
* `frame.get_torque` implemented (LH)

### Bug fixes:

* bug in adding scripts to trajectory objects fixed (LH)
* `time_average` improved (KD)
* bug in `spatial_cor` fixed (LH)
* bug in reader objects related to uncomplete dump files fixed (LH)
* bug in `MSD` fixed - now checks if required data available (LH)
* data loading backend improved (KD, LH)

### Contributors:

* Lukas Hecht (LH)
* Kay-Robert Dormann (KD)
* Kai Luca Spanheimer (KS)



## AMEP 0.3.0 (09 Nov 2022)

This is the new **AMEP** version which includes full HDF5 support and uses 
HDF5 files in the backend.

### New features:

* new backend with full HDF5 support
* velocity autocorrelation function added
* orientational autocorrelation function added

### Bug fixes:

* improved calculation of structure factors

### Deprecation and removals:

* loading of previous `.pkl` trajectory file format not supported anymore

### Contributors:

* Lukas Hecht



## AMEP 0.2.0 (21 Oct 2022)

### New features:

* fast mode for 2d structure factor implemented
* `replace_frame` added to BaseTrajectory
* cluster-resolved kinetic temperature added
* real fft and spectrum added
* axis line width and tick width added to `amep.plot.format_axis`
* cluster-resolved msd added
* `linear_mappable` added
* radius of gyration added
* `get_ids()` added to BaseFrame
* `in_circle` added
* cluster-resolved calculation of the number of next neighbors added
* `ClusterFraction` added

### Bug fixes:

* small bug fixed in `pbc.pbc_points`
* bug in `traj.get_forces()` fixed
* bug in loading LAMMPS data fixed
* saving results of evaluate objects now possible with user-defined file name

### Contributors:

* Lukas Hecht



## AMEP 0.1.1 (27 Jun 2022)

This version includes small bug fixes and improvements.

### Contributors:

* Lukas Hecht



## AMEP 0.1.0 (30 Mar 2022)

This is the first complete **AMEP** version ready to be shared with our group 
members.

### Contributors
* Lukas Hecht