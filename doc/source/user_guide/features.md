# Feature overview

The **AMEP** Python library provides a unified framework for handling 
both particle-based and continuum simulation data. It is made for the analysis 
of molecular-dynamics (MD), Brownian-dynamics (BD), and continuum simulation 
data of condensed matter systems and active matter systems in particular. 
**AMEP** provides a huge variety of analysis methods for both data types that 
allow to evaluate various dynamic and static observables based on the 
trajectories of the particles or the time evolution of continuum fields. For 
fast and efficient data handling, **AMEP** provides a unified framework for 
loading and storing simulation data and analysis results in a compressed, 
HDF5-based data format. **AMEP** is written purely in Python and uses powerful 
libraries such as NumPy, SciPy, Matplotlib, and scikit-image commonly used in 
computational physics. Therefore, understanding, modifying, and building up on 
the provided framework is comparatively easy. All evaluation functions are 
optimized to run efficiently on HPC hardware to provide fast computations. To 
plot and visualize simulation data and analysis results, **AMEP** provides an 
optimized plotting framework based on the Matplotlib Python library, which 
allows to easily plot and animate particles, fields, and lines. Compared to 
other analysis libraries, the huge variety of analysis methods combined with 
the possibility to handle both most common data types used in soft-matter 
physics and in the active matter community in particular, enables the analysis 
of a much broader class of simulation data including not only classical 
molecular-dynamics or Brownian-dynamics simulations but also any kind of 
numerical solutions of partial differential equations. The following table 
gives an overview on the observables provided by **AMEP** and on their 
capability of processing particle-based and continuum 
simulation data.

We try to keep the following table up to date, but please check the **API Reference** 
for the full documentation and all features of **AMEP**

| Observable | Particles | Fields |
|:-----------|:---------:|:------:|
| **Spatial Correlation Functions:** |||
| RDF (radial pair distribution function) | ✔ | ➖ |
| PCF2d (2d pair correlation function) | ✔ | ➖ |
| PCFangle (angular pair correlation function) | ✔ | ➖ |
| SFiso (isotropic static structure factor) | ✔ | ✔ |
| SF2d (2d static structure factor) | ✔ | ✔ |
| SpatialVelCor (spatial velocity correlation function) | ✔ | ➖ |
| PosOrderCor (positional order correlation function) | ✔ | ➖ |
| HexOrderCor (hexagonal order correlation function) | ✔ | ➖ |
| **Local Order:** |||
| Voronoi tesselation | ✔ | ➖ |
| Local density | ✔ | ✔ |
| Local packing fraction | ✔ | ➖ |
| k-atic bond order parameter | ✔ | ➖ |
| Next/nearest neighbor search | ✔ | ➖ |
| **Time Correlation Functions:** |||
| MSD (mean square displacement) | ✔ | ➖ |
| VACF (velocity autocorrelation function) | ✔ | ➖ |
| OACF (orientation autocorrelation function) | ✔ | ➖ |
| **Cluster Analysis:** |||
| Clustersize distribution | ✔ | ✔ |
| Cluster growth | ✔ | ✔ |
| Radius of gyration | ✔ | ✔ |
| Linear extension | ✔ | ✔ |
| Center of mass | ✔ | ✔ |
| Gyration tensor | ✔ | ✔ |
| Inertia tensor | ✔ | ✔ |
| **Miscellaneous:** |||
| Translational/rotational kinetic energy | ✔ | ➖ |
| Kinetic temperature | ✔ | ➖ |
