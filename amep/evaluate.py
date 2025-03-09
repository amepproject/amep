# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) 2023-2025 Lukas Hecht and the AMEP development team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Contact: Lukas Hecht (lukas.hecht@pkm.tu-darmstadt.de)
# =============================================================================
"""
Trajectory Analysis
===================

.. module:: amep.evaluate

The AMEP module :mod:`amep.evaluate` contains the main analysis classes for
analyzing whole trajectories, performing time averages, and calculating
observables from simulation data of particle-based and continuum simulations.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
from packaging.version import Version
from collections.abc import Callable
import warnings
import numpy as np

from .utils import average_func, kpeaks, rotate_coords, in_box, sq_from_sf2d
from .base import BaseEvaluation
from .order import psi_k, local_number_density, local_mass_density
from .order import local_packing_fraction, voronoi_density
from .spatialcor import spatialcor, rdf, pcf2d, pcf_angle, sf2d, sfiso
from .statistics import distribution
from .cluster import identify
from .cluster import sizes as cluster_sizes
from .timecor import msd, acf
from .continuum import sf2d as csf2d
from .continuum import identify_clusters, cluster_properties

from .pbc import pbc_points
from .trajectory import ParticleTrajectory, FieldTrajectory
from . import thermo as thermo

from typing import Callable

# =============================================================================
# GENERAL FUNCTION
# =============================================================================
class Function(BaseEvaluation):
    """Apply a user-defined function to a trajectory.
    """

    def __init__(self, traj: ParticleTrajectory | FieldTrajectory, 
        func: Callable, skip: float = 0.0, nav: int = 10, **kwargs):
        r'''Calculate a given function for a trajectory.

        Parameters
        ----------
        traj: BaseTrajectory
            Trajectory object with simulation data.
        func: Function
            Function to be used for analysis.
            Its signature should be `func(frame, **kwargs)->float`
        skip: float, default=0.0
            Skip this fraction at the beginning of
            the trajectory.
        nav: int, optional
            Number of frames to consider for the time average.
            The default is 10.
        **kwargs: Keyword Arguments
            General python keyword arguments to be
            forwarded to the function f.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> def msd(frame, start=None):
        ...     vec = start.unwrapped_coords() - frame.unwrapped_coords()
        ...     return (vec ** 2).sum(axis=1).mean()
        >>> msd_eval = amep.evaluate.Function(
        ...     traj, msd, nav=traj.nframes, start=traj[0]
        ... )
        >>> msd_eval.name = "msd"
        >>> msd_eval.save("./eval/msd_eval.h5")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(traj.times, msd_eval.frames)
        >>> axs.set_xlabel(r"$t$")
        >>> axs.set_ylabel(r"MSD")
        >>> axs.loglog()
        >>> fig.savefig("./figures/evaluate/evaluate-Function.png")

        .. image:: /_static/images/evaluate/evaluate-Function.png
          :width: 400
          :align: center
        
        >>> def polarization(frame, start=None):
        ...     mean_orientation = frame.orientations().mean(axis=0)
        ...     return np.sqrt(np.sum(mean_orientation**2))
        >>> pol_eval = amep.evaluate.Function(
        ...     traj, polarization, nav=traj.nframes
        ... )
        >>> pol_eval.name = "global polarization"
        >>> pol_eval.save("./eval/global_polarization.h5")
        >>> print("Time average: ", pol_eval.avg)
        Time average:  0.0091200555
        >>>

        '''
        super().__init__()
        
        self.name = 'function'
        
        self.__traj   = traj
        self.__func   = func
        self.__skip   = skip
        self.__nav    = nav
        self.__kwargs = kwargs
        
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute, np.arange(self.__traj.nframes), skip=self.__skip,
            nr=self.__nav, indices=True, **kwargs)
        
        self.__times = self.__traj.times[self.__indices]
    def __compute(self, ind: int, **kwargs):
        r'''
        Calculation for a single frame,

        Parameters
        ----------
        ind : int
            Frame index.
        **kwargs : Keyword Arguments
            Keyword arguments to be passed to the
            analysis function.

        Returns
        -------
        res : np.ndarray
            result.
        '''
        return self.__func(self.__traj[ind], **kwargs)

    @property
    def frames(self):
        r'''
        Function value for each frame.

        Returns
        -------
        np.ndarray
            Function value for each frame.

        '''
        return self.__frames

    @property
    def times(self):
        r'''
        Times at which the function is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the function is evaluated.

        '''
        return self.__times

    @property
    def avg(self):
        r'''
        Time-averaged function value (averaged over the given number
        of frames).

        Returns
        -------
        various
            Time-averaged function value.

        '''
        return self.__avg

    @property
    def indices(self):
        r'''
        Indices of all frames for which the function has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices

# =============================================================================
# SPATIAL CORRELATION FUNCTIONS
# =============================================================================
class SpatialVelCor(BaseEvaluation):
    """Spatial velocity correlation function.
    """

    def __init__(
            self, traj: ParticleTrajectory, skip: float = 0.0, nav: int = 10,
            ptype: int | None = None, other: int | None = None, **kwargs
            ) -> None:
        r'''
        Calculates the spatial velocity correlation function.
        
        Notes
        -----
        The spatial velocity correlation function is defined by

        .. math::
                    C_v(r)=\frac{\langle\vec{v}(r)\cdot\vec{v}(0)\rangle}{\langle\vec{v}(0)^2\rangle}

        See Ref. [1]_ for further information.

        .. [1] Caprini, L., & Marini Bettolo Marconi, U. (2021).
           Spatial velocity correlations in inertial systems
           of active Brownian particles. Soft Matter, 17(15),
           4109–4121. https://doi.org/10.1039/D0SM02273J

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with simulation data.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float, optional
            Particle type. The default is None.
        other : float, optional
            Other particle type (to calculate the correlation between
            different particle types). The default is None.
        **kwargs
            Other keyword arguments are forwarded to
            `amep.spatialcor.spatialcor`.
              
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> svc = amep.evaluate.SpatialVelCor(traj, skip=0.9, nav=5, njobs=4)
        >>> svc.save("./eval/svc.h5")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(svc.r, svc.avg, marker='.')
        >>> axs.semilogx()
        >>> axs.set_xlabel(r"$r$")
        >>> axs.set_ylabel(r"$C_v(r)$")
        >>> fig.savefig("./figures/evaluate/evaluate-SpatialVelCor.png")

        .. image:: /_static/images/evaluate/evaluate-SpatialVelCor.png
          :width: 400
          :align: center

        '''
        super(SpatialVelCor, self).__init__()

        self.name = 'svc'
        
        self.__traj   = traj
        self.__skip   = skip
        self.__nav    = nav
        self.__ptype  = ptype
        self.__other  = other
        self.__kwargs = kwargs
        
        self.__frames, res, self.__indices = average_func(
            self.__compute,
            self.__traj,
            skip = self.__skip,
            nr = self.__nav,
            indices = True
        )
        
        self.__times = self.__traj.times[self.__indices]
        self.__avg   = res[0]
        self.__r     = res[1]
        
    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
            A single frame of particle-based simulation data.
              
        Returns
        -------
        c : np.ndarray
            Correlation function.
        r : np.ndarray
            Distances.
        '''
        c, r = spatialcor(
            frame.coords(ptype = self.__ptype),
            frame.box,
            frame.velocities(ptype = self.__ptype),
            other_coords = frame.coords(ptype = self.__other),
            other_values = frame.velocities(ptype = self.__other),
            **self.__kwargs
        )
        return c, r
    
    @property
    def r(self):
        r'''
        Distances.

        Returns
        -------
        np.ndarray
            Distances.

        '''
        return self.__r
    
    @property
    def frames(self):
        r'''
        Spatial velocity correlation function for each frame.

        Returns
        -------
        np.ndarray
            Function value for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the spatial velocity correlation
        function is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the function is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged spatial velocity correlation function 
        (averaged over the given number of frames).

        Returns
        -------
        np.ndarray
            Time-averaged spatial velocity correlation function.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the spatial velocity 
        correlation function has been evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class RDF(BaseEvaluation):
    """Radial pair distribution function.
    """

    def __init__(
            self, traj: ParticleTrajectory, skip: float = 0.0, nav: int = 10,
            ptype: int | None = None, other: int | None = None,
            **kwargs) -> None:
        r'''
        Calculate the radial pair-distribution function [1]_.

        Works for a one-component system.
        Takes the time average over several time steps.

        Notes
        -----
        This version only works for one-component systems in 2D. The radial
        pair-distribution function is defined by (see Refs. [2]_ [3]_)

        .. math::
                g(r) = \frac{1}{\rho N}\sum\limits_{k}\sum\limits_{l\neq k}
                       \left\langle\frac{\delta\left(r-\left|\vec{r}_k(t)
                       -\vec{r}_l(t)\right|\right)}{2\pi r}\right\rangle_t

        References
        ----------
        .. [1] Santos, A. (2016). A Concise Course on the Theory of Classical
           Liquids (Vol. 923). Springer International Publishing. (p. 101).
           https://doi.org/10.1007/978-3-319-29668-5

        .. [2] Abraham, M. J., Hess, B., Spoel, D. van der, Lindahl, E.,
           Apostolov, R., Berendsen, H. J. C., Buuren, A. van,
           Bjelkmar, P., Drunen, R. van, Feenstra, A., Fritsch, S.,
           Groenhof, G., Junghans, C., Hub, J., Kasson, P., Kutzner, C.,
           Lambeth, B., Larsson, P., Lemkul, J. A., … Maarten, W. (2018).
           GROMACS User Manual version 2018.3. 258. https://www.gromacs.org/

        .. [3] Hecht, L., Horstmann, R., Liebchen, B., & Vogel, M. (2021).
           MD simulations of charged binary mixtures reveal a generic relation
           between high- and low-temperature behavior.
           The Journal of Chemical Physics, 154(2), 024501.
           https://doi.org/10.1063/5.0031417

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object of particle-based simulation data.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory. The default
            is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float, optional
            Particle type. The default is None.
        other : float, optional
            Other particle type (to calculate the correlation between
            different particle types). The default is None.
        **kwargs
            Other keyword arguments are forwarded to `amep.spatialcor.rdf`.
              
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> rdfcalc = amep.evaluate.RDF(
        ...     traj, nav=2, nbins=1000, skip=0.9, njobs=4
        ... )
        >>> rdfcalc.save('./eval/rdf.h5')
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(rdfcalc.r, rdfcalc.avg)
        >>> axs.set_xlim(0,10)
        >>> axs.set_xlabel(r'$r$')
        >>> axs.set_ylabel(r'$g(r)$')
        >>> fig.savefig("./figures/evaluate/evaluate-RDF.png")

        .. image:: /_static/images/evaluate/evaluate-RDF.png
          :width: 400
          :align: center

        '''
        super(RDF, self).__init__()
        
        self.name = 'rdf'
        
        self.__traj = traj
        self.__skip = skip
        self.__nav = nav
        self.__ptype = ptype
        self.__other = other
        self.__kwargs = kwargs
        
        self.__frames, res, self.__indices = average_func(
            self.__compute, self.__traj, skip=self.__skip,
            nr=self.__nav, indices=True
        )
        
        self.__times = self.__traj.times[self.__indices]
        self.__avg   = res[0]
        self.__r     = res[1]
        
    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
            Frame object of particle-based simulation data.
              
        Returns
        -------
        gr : np.ndarray
            g(r)
        r : np.ndarray
            distances
        '''
        if self.__other is None:
            gr,r = rdf(
                frame.coords(ptype = self.__ptype),
                frame.box,
                **self.__kwargs
            )
        else:
            gr,r = rdf(
                frame.coords(ptype = self.__ptype),
                frame.box,
                other_coords = frame.coords(ptype = self.__other),
                **self.__kwargs
            )
        return gr,r

    @property
    def r(self):
        r'''
        Distances.

        Returns
        -------
        np.ndarray
            Distances.

        '''
        return self.__r
    
    @property
    def frames(self):
        r'''
        RDF for each frame.

        Returns
        -------
        np.ndarray
            Function value for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the RDF is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the RDF is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged RDF (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged RDF.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the RDF has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class PCF2d(BaseEvaluation):
    """2d pair correlation function.
    """

    def __init__(
            self, traj: ParticleTrajectory, skip: float = 0.0, nav: int = 10,
            ptype: int | None = None, other: int | None = None,
            **kwargs) -> None:
        r'''
        Calculate the two-dimensional pair correlation function g(x,y).
        
        Implemented for a 2D system.
        Takes the time average over several time steps.

        To allow for averaging the result (either with respect to time or to
        make an ensemble average), the coordinates are rotated such that the
        mean orientation points along the x axis (see Ref. [1]_ for details).

        Notes
        -----
        The 2D pair correlation function is defined by

        .. math::
                   g(x,y) = \frac{1}{\rho N}\sum\limits_{i=1}^{N}
                   \sum\limits_{j\neq i}^{N}\delta(x-x_{ij})\delta(y-y_{ij})

        References:

        .. [1] Bernard, E. P., & Krauth, W. (2011). Two-Step Melting in Two
           Dimensions: First-Order Liquid-Hexatic Transition.
           Physical Review Letters, 107(15), 155704.
           https://doi.org/10.1103/PhysRevLett.107.155704

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with particle-based simulation data.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory. The default
            is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float or None, optional
            Particle type. The default is None.
        other : float or None, optional
            Other particle type (to calculate the correlation between
            different particle types). The default is None.
        **kwargs
            All other keyword arguments are forwarded to
            `amep.spatialcor.pcf2d`.
              
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> pcf2d = amep.evaluate.PCF2d(
        ...     traj, nav=2, nxbins=2000, nybins=2000, njobs=4, skip=0.9
        ... )
        >>> pcf2d.save("./eval/pcf2d.h5")
        >>> fig, axs = amep.plot.new(figsize=(3.6,3))
        >>> mp = amep.plot.field(axs, pcf2d.avg, pcf2d.x, pcf2d.y)
        >>> cax = amep.plot.add_colorbar(
        ...     fig, axs, mp, label=r"$g(\Delta x, \Delta y)$"
        ... )
        >>> axs.set_xlim(-5,5)
        >>> axs.set_ylim(-5,5)
        >>> axs.set_xlabel(r"$\Delta x$")
        >>> axs.set_ylabel(r"$\Delta y$")
        >>> fig.savefig("./figures/evaluate/evaluate-PCF2d.png")
        >>> 
        
        '''
        super(PCF2d, self).__init__()
        
        self.name = 'pcf2d'
        
        self.__traj   = traj
        self.__skip   = skip
        self.__nav    = nav
        self.__ptype  = ptype
        self.__other  = other
        self.__kwargs = kwargs
        
        self.__frames, res, self.__indices = average_func(
            self.__compute, self.__traj, skip=self.__skip,
            nr=self.__nav, indices=True)
        
        self.__times = self.__traj.times[self.__indices]
        self.__avg   = res[0]
        self.__x     = res[1]
        self.__y     = res[2]
        
    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
            One frame of particle-based simulation data.
              
        Returns
        -------
        gxy : np.ndarray
            g(x,y)
        x : np.ndarray
            x values
        y : np.ndarray
            y values
        '''
        # hexagonal order parameter to specify mean orientation
        psi = np.mean(psi_k(
            frame.coords(),
            frame.box,
            k = 6
        ))
            
        if self.__other is None:
            gxy, x, y = pcf2d(
                frame.coords(ptype = self.__ptype),
                frame.box,
                psi = np.array([psi.real, psi.imag]),
                **self.__kwargs
            )
        else:
            gxy, x, y = pcf2d(
                frame.coords(ptype = self.__ptype),
                frame.box,
                psi = np.array([psi.real, psi.imag]),
                other_coords = frame.coords(ptype = self.__other),
                **self.__kwargs) 
        return gxy, x, y
    
    @property
    def x(self):
        r'''
        x values.

        Returns
        -------
        np.ndarray
            x values.

        '''
        return self.__x
    
    @property
    def y(self):
        r'''
        y values.

        Returns
        -------
        np.ndarray
            y values.

        '''
        return self.__y
    
    @property
    def frames(self):
        r'''
        PCF2d for each frame.

        Returns
        -------
        np.ndarray
            PCF2d for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the PCF2d is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the PCF2d is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged PCF2d (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged PCF2d.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the PCF2d has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class PCFangle(BaseEvaluation):
    """2d angular pair correlation function.
    """

    def __init__(
            self, traj: ParticleTrajectory, skip: float = 0.0, nav: int = 10,
            ptype: int | None = None, other: int | None = None,
            **kwargs) -> None:
        r'''
        Calculate the two-dimensional pair correlation function g(r,theta).

        Implemented for a 2D system.
        Takes the time average over several time steps.

        To allow for averaging the result (either with respect to time or to
        make an ensemble average), the coordinates are rotated such that the
        mean orientation points along the :math:`x`-axis
        (see Ref. [1]_ for details).

        Notes
        -----
        The angle-dependent pair correlation function is defined by
        (see Ref. [2]_)

        .. math::
           g(r,\theta) = \frac{1}{\langle \rho\rangle_{local,\theta} N}
           \sum\limits_{i=1}^{N} \sum\limits_{j\neq i}^{N}
           \frac{\delta(r_{ij} -r)
           \delta(\theta_{ij}-\theta)}{2\pi r^2 \sin(\theta)}

        The angle :math:`\theta` is defined with respect to a certain
        axis :math:`\vec{e}` and is given by

        .. math::
           \cos(\theta)=\frac{{\vec{r}}_{ij}\cdot\vec{e}}{r_{ij}e}

        Here, we choose :math:`\vec{e}=\hat{e}_x`.

        References
        ----------
        .. [1] Bernard, E. P., & Krauth, W. (2011). Two-Step Melting in Two
           Dimensions: First-Order Liquid-Hexatic Transition.
           Physical Review Letters, 107(15), 155704.
           https://doi.org/10.1103/PhysRevLett.107.155704

        .. [2] Abraham, M. J., Hess, B., Spoel, D. van der, Lindahl, E.,
            Apostolov, R., Berendsen, H. J. C., Buuren, A. van, Bjelkmar, P.,
            Drunen, R. van, Feenstra, A., Fritsch, S., Groenhof, G.,
            Junghans, C., Hub, J., Kasson, P., Kutzner, C., Lambeth, B.,
            Larsson, P., Lemkul, J. A., … Maarten, W. (2018).
            GROMACS User Manual version 2018.3. 258. www.gromacs.org


        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with particle-based simulation data.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory. The default
            is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float or None, optional
            Particle type. The default is None.
        other : float or None, optional
            Other particle type (to calculate the correlation between
            different particle types). The default is None.
        **kwargs
            All other keyword arguments are forwarded to
            `amep.spatialcor.pcf_angle`.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> pcfangle = amep.evaluate.PCFangle(
        ...     traj, nav=2, ndbins=1000, nabins=1000,
        ...     njobs=4, rmax=8.0, skip=0.9
        ... )
        >>> pcfangle.save("./eval/pcfangle.h5")
        >>> r = pcfangle.r
        >>> theta = pcfangle.theta
        >>> X = r*np.cos(theta)
        >>> Y = r*np.sin(theta)
        >>> fig, axs = amep.plot.new(figsize=(3.6,3))
        >>> mp = amep.plot.field(
        ...     axs, pcfangle.avg, X, Y
        ... )
        >>> cax = amep.plot.add_colorbar(
        ... fig, axs, mp, label=r"$g(\Delta x, \Delta y)$"
        ... )
        >>> axs.set_xlim(-5, 5)
        >>> axs.set_ylim(-5, 5)
        >>> axs.set_xlabel(r"$\Delta x$")
        >>> axs.set_ylabel(r"$\Delta y$")
        >>> fig.savefig("./figures/evaluate/evaluate-PCFangle.png")
        >>> 

        '''
        super(PCFangle, self).__init__()
        
        self.name = 'pcfa'
        
        self.__traj   = traj
        self.__skip   = skip
        self.__nav    = nav
        self.__ptype  = ptype
        self.__other  = other
        self.__kwargs = kwargs
        
        self.__frames, res, self.__indices = average_func(
            self.__compute, self.__traj, skip = self.__skip,
            nr = self.__nav, indices = True
        )
            
        self.__times = self.__traj.times[self.__indices]
        self.__avg   = res[0]
        self.__r     = res[1]
        self.__theta = res[2]
        
    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
            One frame of particle-based simulation data.
              
        Returns
        -------
        grt : np.ndarray
            g(x,y)
        r : np.ndarray
            r values
        t : np.ndarray
            theta values
        '''
        # hexagonal order parameter to specify mean orientation
        psi = np.mean(psi_k(
            frame.coords(),
            frame.box,
            k = 6
        ))
        if self.__other is None:
            grt, r, t = pcf_angle(
                frame.coords(ptype = self.__ptype),
                frame.box,
                psi = np.array([psi.real, psi.imag]),
                **self.__kwargs
            )
        else:
            grt, r, t = pcf_angle(
                frame.coords(ptype = self.__ptype),
                frame.box,
                psi = np.array([psi.real, psi.imag]),
                other_coords = frame.coords(ptype = self.__other),
                **self.__kwargs
            )
        return grt, r, t
    
    @property
    def r(self):
        r'''
        Distances.

        Returns
        -------
        np.ndarray
            Distances.

        '''
        return self.__r
    
    @property
    def theta(self):
        r'''
        Angles.

        Returns
        -------
        np.ndarray
            Angles.

        '''
        return self.__theta
    
    @property
    def frames(self):
        r'''
        PCFangle for each frame.

        Returns
        -------
        np.ndarray
            PCFangle for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the PCFangle is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the PCFangle is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged PCFangle (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged function value.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the PCFangle has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices    


class SF2d(BaseEvaluation):
    """2d static structure factor.
    """

    def __init__(
            self, traj: ParticleTrajectory | FieldTrajectory,
            skip: float = 0.0, nav: int = 10, ptype: int = None,
            other: int = None, rotate: bool = True,
            ftype: str | list | None = None, **kwargs) -> None:
        r'''
        Calculate the 2d static structure factor.

        Notes
        -----
        The static structure factor is defined by
        
        .. math::
            
            S(\vec{q}) = \frac{1}{N} \left\langle\sum_{j=1}^{N}\sum_{k=1}^{N}\exp\left\lbrace-i\vec{q}\cdot(\vec{r}_j-\vec{r}_k)\right\rbrace\right\rangle
                       = \frac{1}{N} \left\langle\left|\sum_{j=1}^{N}\exp\left\lbrace-i\vec{q}\cdot\vec{r}_j\right\rbrace\right|^2\right\rangle,
            
        (see Ref. [1]_ for further information).

        References
        ----------

        .. [1] Hansen, J.-P., & McDonald, I. R. (2006). Theory of Simple Liquids 
           (3rd ed.). Elsevier. https://doi.org/10.1016/B978-0-12-370535-8.X5000-9

        Parameters
        ----------
        traj : ParticleTrajectory or FieldTrajectory
            Trajectory object.
        skip : float, optional
            Skip this fraction at the beginning of
            the trajectory. The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float, optional
            Particle type. The default is None.
        other : float, optional
            Other particle type (to calculate the correlation between
            different particle types). The default is None.
        rotate : bool, optional
            If True, the whole system is rotated according to its psi6
            parameter (mean orientation along x axis). The default is True.
        ftype : str or list or None, optional
            Allows to specify for which field(s) in a given FieldTrajectory
            the 2d structure factor should be calculated. If None, the
            2d structure factor is calculated for all fields.
            The default is None.
        **kwargs
            Other keyword arguments are forwarded to 
            `amep.spatialcor.sf2d` in the case of particle-based simulation
            data and to `amep.continuum.csf2d` in the case of continuum 
            simulation data.

        Returns
        -------
        None.
        
        Examples
        --------
        >>> import amep
        >>> ptraj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> psf2d = amep.evaluate.SF2d(ptraj, skip=0.9, nav=2)
        >>> psf2d.save("./eval/sf2d_eval.h5", database=True, name="particles")
        >>> ftraj = amep.load.traj("../examples/data/continuum.h5amep")
        >>> fsf2d = amep.evaluate.SF2d(ftraj, skip=0.9, nav=2, ftype="p")
        >>> fsf2d.save("./eval/sf2d_eval.h5", database=True, name="field")
        >>> fig, axs = amep.plot.new(ncols=2, figsize=(7.5,3))
        >>> mp1 = amep.plot.field(
        ...     axs[0], psf2d.avg, psf2d.qx, psf2d.qy,
        ...     cscale="log", vmin=1e-1
        ... )
        >>> cax1 = amep.plot.add_colorbar(
        ...     fig, axs[0], mp1, label=r"$S(q_x, q_y)$"
        ... )
        >>> axs[0].set_xlabel(r"$q_x$")
        >>> axs[0].set_ylabel(r"$q_y$")
        >>> axs[0].set_title("active Brownian particles")
        >>> mp2 = amep.plot.field(
        ...     axs[1], fsf2d.avg, fsf2d.qx, fsf2d.qy,
        ...     cscale="log", vmin=1e0
        ... )
        >>> cax2 = amep.plot.add_colorbar(
        ...     fig, axs[1], mp2, label=r"$S(q_x, q_y)$"
        ... )
        >>> axs[1].set_xlabel(r"$q_x$")
        >>> axs[1].set_ylabel(r"$q_y$")
        >>> axs[1].set_title("Keller-Segel model")
        >>> fig.savefig("./figures/evaluate/evaluate-SF2d.png")

        .. image:: /_static/images/evaluate/evaluate-SF2d.png
          :width: 600
          :align: center

        '''
        
        super(SF2d, self).__init__()
        
        self.name = 'sf2d'
        
        self.__traj     = traj
        self.__skip     = skip
        self.__nav      = nav
        self.__ptype    = ptype
        self.__other    = other
        self.__rotate   = rotate
        self.__ftype    = ftype
        self.__kwargs   = kwargs
        
        if type(self.__traj)==ParticleTrajectory:
            # calculation for particles
            self.__frames, res, self.__indices = average_func(
                self.__compute_particles,
                self.__traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        elif type(self.__traj)==FieldTrajectory:
            # calculation for fields
            self.__frames, res, self.__indices = average_func(
                self.__compute_fields,
                self.__traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        else:
            raise TypeError(f'Invalid type of traj: {type(self.__traj)}.')
            
        self.__times = self.__traj.times[self.__indices]
        self.__avg = res[0]
        self.__qx  = res[1]
        self.__qy  = res[2]

        
    def __compute_particles(self, frame):
        r'''
        Calculation for a single frame of particle-based simulation data.
        
        Parameters
        ----------
        frame : BaseFrame
            Frame object of particle-based simulation data.
              
        Returns
        -------
        S : np.ndarray
            Structure factor.
        qx : np.ndarray
            Wave vector's x component.
        qy : np.ndarray
            Wave vector's y component.
        '''
        if self.__rotate:
            
            # hexagonal order parameter to specify mean orientation
            psi = np.mean(psi_k(
                frame.coords(),
                frame.box,
                k = 6
            ))
            psi = np.array([psi.real, psi.imag])  
            
            # orient x-axis along mean sample orientation
            ex = np.array([1,0])
            theta = np.arccos(np.dot(ex, psi)/np.sqrt(psi[0]**2+psi[1]**2))
            if psi[1] < 0:
                theta = 2*np.pi - theta

            # rotate all coordinates (include periodic images)
            coords = rotate_coords(pbc_points(
                frame.coords(ptype=self.__ptype),
                frame.box
            ), -theta, frame.center)
            other_coords = rotate_coords(pbc_points(
                frame.coords(ptype=self.__other),
                frame.box
            ), -theta, frame.center)
            
            # only get back those coordinates inside the original box
            coords = in_box(coords, frame.box)
            other_coords = in_box(other_coords, frame.box)
            
            # calculate structure factor
            S, qx, qy = sf2d(
                coords,
                frame.box,
                other_coords = other_coords,
                **self.__kwargs
            )
                
        else:
            # calculate structure factor
            S, qx, qy = sf2d(
                frame.coords(ptype=self.__ptype), 
                frame.box, 
                other_coords = frame.coords(ptype=self.__other),
                **self.__kwargs
            )
        return S, qx, qy
    
    def __compute_fields(self, frame):
        r'''
        Calculation for a single frame of continuum simulation data.
        
        Parameters
        ----------
        frame : BaseField
            Field object of continuum simulation data.
              
        Returns
        -------
        S : np.ndarray
            Structure factor.
        qx : np.ndarray
            Wave vector's x component.
        qy : np.ndarray
            Wave vector's y component.
        '''
        # get grid coordinates
        X, Y = frame.grid
        
        if self.__ftype is None and len(frame.keys)>1:
            # calculate for all fields
            S = []
            for key in frame.keys:
                Sxy, qx, qy = csf2d(frame.data(key), X, Y, **self.__kwargs)
                S.append(Sxy)
            return *S, qx, qy
        elif type(self.__ftype)==list:
            # calculate for all given fields
            S = []
            for key in self.__ftype:
                Sxy, qx, qy = csf2d(frame.data(key), X, Y, **self.__kwargs)
                S.append(Sxy)
            return *S, qx, qy
        else:
            # calculate for single field (either given or the only available)
            if self.__ftype is None:
                S, qx, qy = csf2d(
                    frame.data(frame.keys[0]), X, Y, **self.__kwargs
                )
            else:
                S, qx, qy = csf2d(
                    frame.data(self.__ftype), X, Y, **self.__kwargs
                )
            return S, qx, qy
    
    @property
    def qx(self):
        r'''
        x-components of the scattering vectors.

        Returns
        -------
        np.ndarray
            x-components of the scattering vectors.

        '''
        return self.__qx
    
    @property
    def qy(self):
        r'''
        y-components of the scattering vectors.

        Returns
        -------
        np.ndarray
            y-components of the scattering vectors.

        '''
        return self.__qy
    
    @property
    def frames(self):
        r'''
        SF2d for each frame.

        Returns
        -------
        np.ndarray
            SF2d for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the SF2d is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the SF2d is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged SF2d (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged SF2d.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the SF2d has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices    


class SFiso(BaseEvaluation):
    """Isotropic static structure factor.
    """

    def __init__(
            self, traj: ParticleTrajectory | FieldTrajectory,
            skip: float = 0.0, nav: int = 10, qmax: float = 20.0,
            twod: bool = True, njobs: int = 1, chunksize: int = 1000,
            mode: str = 'fast', ptype: int | None = None,
            other: int | None = None, accuracy: float = 0.5,
            num: int = 8, ftype: str | list | None = None) -> None:
        r'''
        Calculate the isotropic static structure.

        Average over several frames.

        Notes
        -----
        The isotropic static structure factor is defined by

        .. math::
            S_{3D}(q) = \frac{1}{N}\left\langle\sum_{m,l=1}^N\frac{\sin(qr_{ml})}{qr_{ml}}\right\rangle

        .. math::
            S_{2D}(q) = \frac{1}{N}\left\langle\sum_{m,l=1}^N J_0(qr_{ml}\right\rangle

        with :math:`r_{ml}=|\vec{r}_m-\vec{r}_l|` and the Bessel function
        of the first kind :math:`J_0(x)`.
        See also Ref. [1]_ for further information on the
        static structure factor.

        References
        ----------
        .. [1] Hansen, J.-P., & McDonald, I. R. (2006).
           Theory of Simple Liquids (3rd ed.). Elsevier. p. 78.
           https://doi.org/10.1016/B978-0-12-370535-8.X5000-9

        Parameters
        ----------
        traj : ParticleTrajectory or FieldTrajectory
            Trajectory object.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        qmax : float, optional
            Maximum wave number to consider. This value is ignored if a
            FieldTrajectory is provided as input data. The default is 20.0.
        twod : bool, optional
            If True, the 2D form is used. The default is True.
        njobs : int, optional
            Number of jobs for multiprocessing. The default is 1.
        chunksize : int, optional
            Divide calculation into chunks of this size. The default is 1000.
        mode : str, optional
            One of ['std', 'fast', 'fft']. The 'fft' mode only works
            if `twod=True`. The default is 'fast'.
        ptype : int or None, optional
            Particle type. The default is None.
        other : int or None, optional
            Other particle type (to calculate the correlation between
            different particle types). The default is None.
        accuracy : float, optional
            Accuracy for `mode='fft'`. 0.0 means least accuracy, 1.0 best
            accuracy. The default is 0.5. Note that a higher accuracy needs
            more memory for the computation. The accuracy must be in (0,1].
        num : int, optional
            Number of q vectors to average over in `mode='fast'`. If twod is
            False, the number of q vectors is equal to num^2. The default is 8.
        ftype : str or list or None, optional
            Allows to specify for which field(s) in a given FieldTrajectory
            the isotropic structure factor should be calculated. If None, the
            isotropic structure factor is calculated for all field.
            The default is None.

        Examples
        --------
        >>> import amep
        >>> ptraj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> ftraj = amep.load.traj("../examples/data/continuum.h5amep")
        >>> psfiso = amep.evaluate.SFiso(ptraj, skip=0.9, nav=2, qmax=20)
        >>> fsfiso = amep.evaluate.SFiso(ftraj, skip=0.9, nav=2, ftype="c")
        >>> psfiso.save("./eval/sfiso_eval.h5", database=True, name="particles")
        >>> fsfiso.save("./eval/sfiso_eval.h5", database=True, name="field")
        >>> L = amep.utils.domain_length(psfiso.avg, psfiso.q, qmax=1.0)
        >>> print(L)
        57.88815991321122
        >>> fig, axs = amep.plot.new(figsize=(7,3), ncols=2)
        >>> axs[0].plot(psfiso.q, psfiso.avg)
        >>> axs[0].set_xlabel(r"$q$")
        >>> axs[0].set_ylabel(r"$S(q)$")
        >>> axs[0].set_title("active Brownian particles")
        >>> axs[0].set_ylim(-0.5,50)
        >>> axs[1].plot(fsfiso.q, fsfiso.avg)
        >>> axs[1].set_xlabel(r"$q$")
        >>> axs[1].set_ylabel(r"$S(q)$")
        >>> axs[1].set_title("Keller-Segel model")
        >>> fig.savefig("./figures/evaluate/evaluate-SFiso.png")

        .. image:: /_static/images/evaluate/evaluate-SFiso.png
          :width: 600
          :align: center
        
        '''
        super(SFiso, self).__init__()
        
        self.name = 'sfiso'
        
        self.__traj      = traj
        self.__skip      = skip
        self.__nav       = nav
        self.__qmax      = qmax
        self.__twod      = twod
        self.__njobs     = njobs
        self.__chunksize = chunksize
        self.__mode      = mode
        self.__ptype     = ptype
        self.__other     = other
        self.__num       = num
        self.__accuracy  = accuracy
        self.__ftype     = ftype        
        
        if type(self.__traj)==ParticleTrajectory:
            # calculation for particles
            self.__frames, res, self.__indices = average_func(
                self.__compute_particles,
                self.__traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        elif type(self.__traj)==FieldTrajectory:
            # calculation for fields
            self.__frames, res, self.__indices = average_func(
                self.__compute_fields,
                self.__traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        else:
            raise TypeError(f'Invalid type of traj: {type(self.__traj)}.')
        
        self.__times = self.__traj.times[self.__indices]
        self.__avg = res[0]
        self.__q   = res[1]
        
    def __compute_particles(self, frame):
        r'''
        Calculation for a single frame of particle-based simulation data.
        
        Parameters
        ----------
        frame : BaseFrame
            Frame object of particle-based simulation data.
              
        Returns
        -------
        Sq : np.ndarray
            S(q)
        q : np.ndarray
            q values.
        '''
        if self.__other is None:
            Sq, q = sfiso(
                frame.coords(ptype=self.__ptype),
                frame.box,
                frame.n(),
                qmax = self.__qmax,
                twod = self.__twod,
                njobs = self.__njobs, 
                chunksize = self.__chunksize,
                mode = self.__mode,
                accuracy = self.__accuracy,
                num = self.__num
            )
        else:
            Sq, q = sfiso(
                frame.coords(ptype=self.__ptype),
                frame.box,
                frame.n(),
                qmax = self.__qmax,
                twod = self.__twod,
                njobs = self.__njobs,
                chunksize = self.__chunksize,
                mode = self.__mode,
                accuracy = self.__accuracy,
                num = self.__num,
                other_coords = frame.coords(ptype=self.__other)
            )
            
        return Sq, q
    
    def __compute_fields(self, frame):
        r'''
        Calculation for a single frame of continuum simulation data.
        
        Parameters
        ----------
        frame : BaseField
            Field object of continuum simulation data.
              
        Returns
        -------
        Sq : np.ndarray
            S(q)
        q : np.ndarray
            q values.
        '''
        # get grid coordinates
        X, Y = frame.grid
        
        if self.__ftype is None and len(frame.keys)>1:
            # calculate for all fields
            Sq = []
            for key in frame.keys:
                Sxy, kx, ky = csf2d(frame.data(key), X, Y)
                s, q = sq_from_sf2d(Sxy, kx, ky)
                Sq.append(s)
            return *Sq, q
        elif type(self.__ftype)==list:
            # calculate for all given fields
            Sq = []
            for key in self.__ftype:
                Sxy, kx, ky = csf2d(frame.data(key), X, Y)
                s, q = sq_from_sf2d(Sxy, kx, ky)
                Sq.append(s)
            return *Sq, q
        else:
            # calculate for single field (either given or the only available)
            if self.__ftype is None:
                Sxy, kx, ky = csf2d(frame.data(frame.keys[0]), X, Y)
                s, q = sq_from_sf2d(Sxy, kx, ky)
            else:
                Sxy, kx, ky = csf2d(frame.data(self.__ftype), X, Y)
                s, q = sq_from_sf2d(Sxy, kx, ky)
            return s, q
    
    @property
    def q(self):
        r'''
        Magnitude of the scattering vectors.

        Returns
        -------
        np.ndarray
            Magnitude of the scattering vectors.

        '''
        return self.__q
    
    @property
    def frames(self):
        r'''
        SFiso for each frame.

        Returns
        -------
        np.ndarray
            SFiso for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the SFiso is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the SFiso is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged SFiso (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged SFiso.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the SFiso has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class PosOrderCor(BaseEvaluation):
    """Positional order correlation function.
    """

    def __init__(
            self, traj: ParticleTrajectory, 
            grtdata: BaseEvaluation | None = None,
            sxydata: BaseEvaluation | None = None, 
            skip: float = 0.0, nav: int = 10,
            k0: list[np.ndarray, ...] | None = None, 
            order: str = 'hexagonal', 
            dk: float = 4.0, ptype: int | None = None, 
            other: int | None = None,
            **kwargs) -> None:
        r'''
        Calculate the positional order correlation function.

        Based on the pair correlation function g(r,theta).

        Notes
        -----
        The positional order correlation function is defined as

        .. math::
            C_{\vec{k}_0}(r) = <\exp(i\vec{k}_0\cdot (\vec{r}_j-\vec{r}_l))>

        with :math:`r=|\vec{r}_j-\vec{r}_l|` (see Ref. [1]_ for further information).
        As shown in Ref. [2]_, this can be rewritten as

        .. math::
            C_{\vec{k}_0}(r) = \int\text{d}\theta\,g(r,\theta)
            \exp(i\vec{k}_0\cdot\vec{r}) / \int\text{d}\theta\,g(r,\theta).

        References
        ----------
        .. [1] Digregorio, P., Levis, D., Suma, A., Cugliandolo,
            L. F., Gonnella, G., & Pagonabarraga, I. (2018). Full
            Phase Diagram of Active Brownian Disks: From Melting
            to Motility-Induced Phase Separation.
            Physical Review Letters, 121(9), 098003.
            https://doi.org/10.1103/PhysRevLett.121.098003

        .. [2] Bernard, E. P., & Krauth, W. (2011).
            Two-Step Melting in Two Dimensions:
            First-Order Liquid-Hexatic Transition.
            Physical Review Letters, 107(15), 155704.
            https://doi.org/10.1103/PhysRevLett.107.155704

        Parameters
        ----------
        traj : Traj
            Trajectory object with simulation data.
        grtdata : BaseEvaluation, optional
            Pair correlation function g(r,theta) as obtained from
            amep.evaluate.PCFangle. If None, the angular pair
            correlation function is calculated within this method.
            The default is None.
        sxydata : BaseEvaluation, optional
            2d structure factor as obtained from amep.evaluate.SF2d.
            If None, the 2d structure factor is calculated within this
            method. The default is None.
        skip : float, optional
            Skip this fraction at the beginning of
            the trajectory. The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        k0 : list, optional
            list of k vectors (each has to be a
            np.ndarray of shape (1,3); the z component
            is ignored since this version only works
            for 2D systems). The result will be averaged over these values.
            The default is None.
        order : str, optional
            Specifies the order of the system to determine the
            k vectors if not given by k0. Possible orders: 'hexagonal'.
            The default is 'hexagonal'.
        dk : float, optional
            Size of the area in k-space around each estimate used
            for the Gaussian fit to determine the k0 vectors.
            The default is 4.0.
        **kwargs
            Other keyword arguments are forwarded to `amep.evaluate.PCFangle`.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> poscor = amep.evaluate.PosOrderCor(
        ...     traj, skip=0.9, nav=2
        ... )
        >>> poscor.save("./eval/poscor.h5")
        >>> y,x = amep.utils.envelope(poscor.avg, x=poscor.r)
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(poscor.r, poscor.avg, label="data")
        >>> axs.plot(x, y, marker="", ls="--", label="envelope", c="orange")
        >>> axs.loglog()
        >>> axs.set_xlabel(r"$r$")
        >>> axs.set_ylabel(r"$C(r)$")
        >>> axs.legend()
        >>> fig.savefig("./figures/evaluate/evaluate-PosOrderCor.png")

        .. image:: /_static/images/evaluate/evaluate-PosOrderCor.png
          :width: 400
          :align: center

        '''
        super(PosOrderCor, self).__init__()
        
        self.name = 'poc'
        
        self.__traj = traj
        self.__grtdata = grtdata
        self.__sxydata = sxydata
        self.__skip = skip
        self.__nav = nav
        self.__k0 = k0
        self.__order = order
        self.__dk = dk
        self.__ptype = ptype
        self.__other = other
        self.__kwargs = kwargs
        
        # check if pair correlation function is available
        if self.__grtdata is None:
            self.__grtdata = PCFangle(
                self.__traj, skip=self.__skip,
                nav=self.__nav, ptype=self.__ptype,
                other=self.__other, **self.__kwargs
            )

        self.__grt   = self.__grtdata['avg']
        self.__r     = np.unique(self.__grtdata['r'].flatten())
        self.__theta = np.unique(self.__grtdata['theta'].flatten())
        
        # determine k0 vectors
        if self.__k0 is None:
            self.__getk()
        
        c = self.__compute()
        
        self.__avg = c
        self.__r   = np.unique(self.__r.flatten())
        
        
    def __compute(self):
        r'''
        Calculation.
              
        Returns
        -------
        np.ndarray
            c(r).
        '''
        # average over all k0 vectors
        ck = 0
        for i in range(len(self.__k0)):
            integrand = self.__grt * np.exp(1j*self.__r[:,None]*\
                                              (self.__k0[i][0]*np.cos(self.__theta) +\
                                              self.__k0[i][1]*np.sin(self.__theta)))

            if Version(np.__version__) < Version("2.0.0"):
                ck += np.trapz(integrand, x=self.__theta, axis=1)
            else:
                ck += np.trapezoid(integrand, x=self.__theta, axis=1)

        ck = np.abs(ck/len(self.__k0))

        # normalization
        if Version(np.__version__) < Version("2.0.0"):
            norm = np.trapz(self.__grt, x=self.__theta, axis=1)
        else:
            norm = np.trapezoid(self.__grt, x=self.__theta, axis=1)

        return ck/norm


    def __getk(self):
        r'''
        Calculates the k vectors corresponding to the first peaks of 
        the structure factor.

        Returns
        -------
        None.
        '''
        if self.__sxydata is None:
            self.__sxydata = SF2d(
                self.__traj, skip=self.__skip, nav=self.__nav)
        
        Sxy = self.__sxydata['avg']
        kx = self.__sxydata['qx']
        ky = self.__sxydata['qy']
        
        self.__k0 = kpeaks(Sxy, kx, ky, mode=self.__order, dk=self.__dk)
        
    @property
    def k0(self):
        r'''
        Reciprocal lattive vectors over which the average has been calculated.

        Returns
        -------
        list of np.ndarrays of shape (1,3)
            Reciprocal lattices vectors over which to average.

        '''
        return self.__k0
    
    @property
    def r(self):
        r'''
        Distances.

        Returns
        -------
        np.ndarray
            Distances.

        '''
        return self.__r
    
    @property
    def avg(self):
        r'''
        Time-averaged PosOrderCor (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged PosOrderCor.

        '''
        return self.__avg


class HexOrderCor(BaseEvaluation):
    """Spatial correlation function of the hexagonal order parameter.
    """

    def __init__(
            self, traj: ParticleTrajectory, skip: float = 0.0, nav: int = 10,
            ptype: int | None = None, other: int | None = None,
            **kwargs) -> None:
        r'''
        Calculate the spatial correlation function of :math:`\Psi_6`.

        The hexagonal order parameter is defined by

        .. math::
            \Psi_6(\vec{r}_j)=\frac{1}{N_j}\sum_{k=1}^{N_j}\exp(i6\theta_{jk})

        where the sum goes over the six next neighbors of particle j.
        The angle :math:`\theta_{jk}` denotes the angle between the vector
        that connects particle j with its k-th nearest neighbor and
        the x-axis. The hexagonal order correlation function is then
        defined by

        .. math::
            g_6(r=|\vec{r}_j-\vec{r}_i|)=<\Psi_6(\vec{r}_j)\Psi_6(\vec{r}_i)> /
            <\Psi_6^2(\vec{r}_j)>.

        See Refs. [1]_ [2]_ [3]_ for further information.

        References
        ----------
        .. [1] Nelson, D. R., Rubinstein, M., & Spaepen, F. (1982).
           Order in two-dimensional binary random arrays.
           Philosophical Magazine A, 46(1), 105–126.
           https://doi.org/10.1080/01418618208236211

        .. [2] Bernard, E. P., & Krauth, W. (2011).
           Two-Step Melting in Two Dimensions:
           First-Order Liquid-Hexatic Transition.
           Physical Review Letters, 107(15), 155704.
           https://doi.org/10.1103/PhysRevLett.107.155704

        .. [3] Digregorio, P., Levis, D., Suma, A., Cugliandolo, L. F.,
           Gonnella, G., & Pagonabarraga, I. (2018).
           Full Phase Diagram of Active Brownian Disks:
           From Melting to Motility-Induced Phase Separation.
           Physical Review Letters,  121(9), 098003.
           https://doi.org/10.1103/PhysRevLett.121.098003


        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with particle-based simulation data.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory. The default
            is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float, optional
            Particle type. The default is None.
        other : float, optional
            Other particle type. If None, ptype is used. The default is None.
        **kwargs
            All other keyword arguments are forwarded to
            `amep.spatialcor.spatialcor`.

        Examples
        --------
        >>> import amep
        >>> import numpy as np
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> hoc = amep.evaluate.HexOrderCor(
        ... traj, skip=0.9, nav=2, njobs=4, rmax=30
        ... )
        >>> hoc.save("./eval/hoc.h5")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(hoc.r, np.real(hoc.avg))
        >>> axs.set_xlabel(r'$r$')
        >>> axs.set_ylabel(r'$g_6(r)$')
        >>> axs.loglog()
        >>> fig.savefig('./figures/evaluate/evaluate-HexOrderCor.png')

        .. image:: /_static/images/evaluate/evaluate-HexOrderCor.png
          :width: 400
          :align: center

        '''
        super(HexOrderCor, self).__init__()
        
        self.name = 'hoc'
        
        self.__traj   = traj
        self.__skip   = skip
        self.__nav    = nav
        self.__ptype  = ptype
        self.__other  = other
        self.__kwargs = kwargs
        
        self.__frames, res, self.__indices = average_func(
            self.__compute,
            self.__traj,
            skip = self.__skip,
            nr = self.__nav,
            indices = True
        )
        
        self.__times = self.__traj.times[self.__indices]
        self.__avg   = res[0]
        self.__r     = res[1]


    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
              One frame of particle-based simulation data.
              
        Returns
        -------
        g6 : np.ndarray
            Correlation function g_6(r).
        r : np.ndarray
            Distances.
        '''
        # spatial correlation function
        if self.__other is None:
            # hexagonal order parameter per particle
            hexorder = psi_k(
                frame.coords(ptype = self.__ptype),
                frame.box,
                k = 6
            )
            
            # spatial correlation function of hexagonal order parameter
            g6, r = spatialcor(
                frame.coords(ptype = self.__ptype),
                frame.box,
                hexorder,
                **self.__kwargs
            )
        else:
            # hexagonal order parameter per particle
            hexorder = psi_k(
                frame.coords(ptype = self.__ptype), 
                frame.box,
                k = 6
            )
            other_hexorder = psi_k(
                frame.coords(ptype = self.__other),
                frame.box,
                k = 6
            )
            
            # spatial correlation function of hexagonal order parameter
            g6, r = spatialcor(
                frame.coords(ptype = self.__ptype),
                frame.box,
                hexorder, 
                other_coords = frame.coords(ptype = self.__other),
                other_values = other_hexorder,
                **self.__kwargs
            )
        return g6, r
    
    @property
    def r(self):
        r'''
        Distances.

        Returns
        -------
        np.ndarray
            Distances.

        '''
        return self.__r
    
    @property
    def frames(self):
        r'''
        HexOrderCor for each frame.

        Returns
        -------
        np.ndarray
            HexOrderCor for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the HexOrderCor is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the HexOrderCor is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged HexOrderCor (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged HexOrderCor.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the HexOrderCor has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


# =============================================================================
# DISTRIBUTIONS
# =============================================================================
class LDdist(BaseEvaluation):
    """Local density distribution.
    """

    def __init__(
            self, traj: ParticleTrajectory | FieldTrajectory,
            skip: float = 0.0, nav: int = 10, nbins: int = 50,
            xmin: float = 0.0, xmax: float = 1.5, ftype: str | None = None,
            ptype: int | None = None, other: int | None = None,
            mode: str = 'number',
            use_voronoi: bool = False,
            **kwargs) -> None:
        r'''
        Calculate the local density distribution and takes an average
        over several frames (= time average). For particle-based data, the
        local density is determined via averages over circles with radius rmax
        if `use_voronoi=False` or via Voronoi tesselation if `use_voronoi=True`.
        For field data, the histogram of the grid values is calculated.
        
        Notes
        -----
        To calculate the local density distribution for particle-based
        simulation data, the radius of the particles is needed. If your
        trajectory does not contain the radii of the particles, you can
        add the radius (here 0.5) to the trajectory using
        
        .. code-block:: python

            for frame in traj:
                frame.add_data("radius", 0.5*np.ones(len(frame.n())))


        Parameters
        ----------
        traj : ParticleTrajectory or FieldTrajectory
            Trajectory object.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        nbins : int
            Number of bins for the histogram. The default is 50.
        xmin : float, optional
            Minimum value of the bins. The default is 0.0.
        xmax : float, optional
            Maximum value of the bins. The default is 1.5.
        ftype : str or None, optional
            Allows to specify for which field in a given FieldTrajectory
            the local density distribution should be calculated. If None, the
            local density is calculated for all fields. The default is None.
        ptype : int or None, optional
            Particle type of query particle for which the local density is
            calculated. The default is None (use all particles).
        other : int or None, optional
            Particle type of environmental particles. The defautl is None (all
            particles).
        mode : str, optional
            Mode `'number'` uses the number density, mode `'mass'` the mass
            density, and mode `'packing'` the packing fraction.
            The default is `'number'`. This keyword is only considered if 
            `type(traj)=ParticleTrajectory`.
        use_voronoi : bool, optional
            If True, Voronoi tesselation is used to determine the local
            density. If False, averages over circles of radius `rmax` are used.
            Note that `other` is ignored if `use_voronoi` is set to True.
            The default is False.
        **kwargs
            Other keyword arguments are forwarded to the local density functions
            used such as `rmax`, `pbc`, `enforce_nd`.
            (see :py:func:`amep.order.voronoi_density`,
            :py:func:`amep.order.local_number_density`,
            :py:func:`amep.order.local_mass_density`,
            :py:func:`amep.order.local_packing_fraction`)


        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> ld1 = amep.evaluate.LDdist(
        ...     traj, nav=2, mode="number", skip=0.9, xmin=0.0, xmax=1.3
        ... )
        >>> ld2 = amep.evaluate.LDdist(
        ...     traj, nav=2, mode="mass", skip=0.9, xmin=0.0, xmax=1.3
        ... )
        >>> ld3 = amep.evaluate.LDdist(
        ...     traj, nav=2, mode="packing", skip=0.9, xmin=0.0, xmax=1.0
        ... )
        >>> ld1.save("./eval/ld_abp.h5", database=True, name="number")
        >>> ld2.save("./eval/ld_abp.h5", database=True, name="mass")
        >>> ld3.save("./eval/ld_abp.h5", database=True, name="packing")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(ld1.ld, ld1.avg, label="number")
        >>> axs.plot(ld2.ld, ld2.avg, label="mass")
        >>> axs.plot(ld3.ld, ld3.avg, label="packing")
        >>> axs.legend()
        >>> axs.set_xlabel(r"$\rho$")
        >>> axs.set_ylabel(r"$p(\rho)$")
        >>> axs.set_title("active Brownian particles")
        >>> fig.savefig("./figures/evaluate/evaluate-LDdist_1.png")

        .. image:: /_static/images/evaluate/evaluate-LDdist_1.png
          :width: 400
          :align: center

        >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
        >>> ld = amep.evaluate.LDdist(
        ...     traj, nav=2, skip=0.9, xmin=0.0, xmax=1.1, ftype="c"
        ... )
        >>> ld.save("./eval/ld.h5")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(ld.ld, ld.avg)
        >>> axs.set_xlabel(r'$c$')
        >>> axs.set_ylabel(r'$p(c)$')
        >>> axs.set_title("Keller-Segel model")
        >>> fig.savefig("./figures/evaluate/evaluate-LDdist_2.png")

        .. image:: /_static/images/evaluate/evaluate-LDdist_2.png
          :width: 400
          :align: center

        '''
        super(LDdist, self).__init__()
        
        self.name = 'lddist'
        
        self.__traj = traj
        self.__ptype = ptype
        self.__other = other
        self.__skip = skip
        self.__nav = nav
        self.__xmin = xmin
        self.__xmax = xmax
        self.__nbins = nbins
        self.__mode = mode
        self.__ftype = ftype
        self.__use_voronoi = use_voronoi
        self.__kwargs = kwargs
        
        # check mode
        if mode not in ['number', 'mass', 'packing']:
            raise ValueError(
                f'amep.evaluate.LDdist: Invalid mode {self.__mode}. Choose one'\
                ' of ["number", "mass", "packing"].'
            )
        
        if type(self.__traj)==ParticleTrajectory:
            # calculation for particles
            self.__frames, res, self.__indices = average_func(
                self.__compute_particles,
                self.__traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        elif type(self.__traj)==FieldTrajectory:
            # calculation for fields
            self.__frames, res, self.__indices = average_func(
                self.__compute_fields,
                self.__traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        else:
            raise TypeError(f'Invalid type of traj: {type(self.__traj)}.')

        self.__times = self.__traj.times[self.__indices]
        self.__avg   = res[0]
        self.__rho   = res[1]
        
    def __compute_particles(self, frame):
        r'''
        Calculation for a single frame of a ParticleTrajectory.

        Parameters
        ----------
        frame : BaseFrame
            Frame.

        Returns
        -------
        hist : np.ndarray
            Histogram.
        bins : np.ndarray
            Bin positions. Same shape as hist.
        '''
        if self.__use_voronoi:
            if self.__mode=='number':
                ld = voronoi_density(
                    frame.coords(ptype=self.__ptype),
                    frame.box,
                    radius=None, 
                    mass=None,
                    mode=self.__mode,
                    **self.__kwargs
                )
            elif self.__mode=='mass':
                ld = voronoi_density(
                    frame.coords(ptype=self.__ptype),
                    frame.box,
                    radius=None, 
                    mass=frame.mass(ptype=self.__ptype),
                    mode=self.__mode,
                    **self.__kwargs
                )
            elif self.__mode=='packing':
                ld = voronoi_density(
                    frame.coords(ptype=self.__ptype),
                    frame.box,
                    radius=frame.radius(ptype=self.__ptype), 
                    mass=None,
                    mode=self.__mode,
                    **self.__kwargs
                )
        else:
            if self.__mode == 'number':
                ld = local_number_density(
                    frame.coords(ptype=self.__ptype),
                    frame.box,
                    frame.radius(ptype=self.__other),
                    other_coords = frame.coords(ptype=self.__other),
                    **self.__kwargs
                )
            elif self.__mode == 'mass':
                ld = local_mass_density(
                    frame.coords(ptype=self.__ptype),
                    frame.box,
                    frame.radius(ptype=self.__other),
                    frame.mass(ptype=self.__other),
                    other_coords = frame.coords(ptype=self.__other),
                    **self.__kwargs
                )
            elif self.__mode == 'packing':
                ld = local_packing_fraction(
                    frame.coords(ptype=self.__ptype),
                    frame.box,
                    frame.radius(ptype=self.__other),
                    other_coords = frame.coords(ptype=self.__other),
                    **self.__kwargs
                )
        # calculate distribution
        hist, bins = distribution(
            ld, xmin=self.__xmin, xmax=self.__xmax, nbins=self.__nbins
        )
        return hist, bins
    
    def __compute_fields(self, frame):
        r'''
        Calculation for a single frame of a field trajectory.

        Parameters
        ----------
        frame : BaseField
            Frame.

        Returns
        -------
        hist : np.ndarray
            Histogram.
        bins : np.ndarray
            Bin positions. Same shape as hist.
        '''
        if self.__ftype is None and len(frame.keys)>1:
            hist = []
            for key in frame.keys:
                h, bins = distribution(
                    frame.data(key),
                    xmin=self.__xmin,
                    xmax=self.__xmax,
                    nbins=self.__nbins
                )
                hist.append(h)
            return np.array(hist), bins
        else:
            if self.__ftype is None:
                hist, bins = distribution(
                    frame.data(frame.keys[0]),
                    xmin=self.__xmin,
                    xmax=self.__xmax,
                    nbins=self.__nbins
                )
            else:
                hist, bins = distribution(
                    frame.data(self.__ftype),
                    xmin=self.__xmin,
                    xmax=self.__xmax,
                    nbins=self.__nbins
                )
            return hist, bins
    
    @property
    def ld(self):
        r'''
        Local density values.

        Returns
        -------
        np.ndarray
            Local density values.

        '''
        return self.__rho
    
    @property
    def frames(self):
        r'''
        LDdist for each frame.

        Returns
        -------
        np.ndarray
            LDdist for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the LDdist is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the LDdist is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged LDdist (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged function value.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the LDdist has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class Psi6dist(BaseEvaluation):
    """Distribution of the hexagonal order parameter.
    """

    def __init__(
            self, traj: ParticleTrajectory, skip: float = 0.0, 
            nav: int = 10, nbins: int = 50, 
            ptype: int | None = None, other: int | None = None, 
            **kwargs) -> None:
        r'''
        Calculate the distribution of the :math:`\Psi_6`.

        :math:`\Psi_6` is the magnitude of the hexagonal order
        parameter.
        Average over several frames (= time average).

        Notes
        -----
        The hexagonal order parameter is defined by

        .. math::
            \Psi_6(\vec{r}_j) = \frac{1}{6} \sum_{n=1}^6\exp(i6\theta_{jn}),

        where the sum goes of the six nearest neighbors of the particle at
        position :math:`\vec{r}_j`. The value of :math:`\theta_{jn}` is equal
        to the angle between the connection line from :math:`\vec{r}_j`
        to :math:`\vec{r}_n` and the
        x axis. See also Refs. [1]_ [2]_ [3]_ for further information.

        References:

        .. [1] Nelson, D. R., Rubinstein, M., & Spaepen, F. (1982).
           Order in two-dimensional binary random arrays.
           Philosophical Magazine A, 46(1), 105–126.
           https://doi.org/10.1080/01418618208236211

        .. [2] Digregorio, P., Levis, D., Suma, A., Cugliandolo, L. F., 
           Gonnella, G., & Pagonabarraga, I. (2018).
           Full Phase Diagram of Active  Brownian Disks:
           From Melting to Motility-Induced Phase Separation. 
           Physical Review Letters, 121(9), 098003.
           https://doi.org/10.1103/PhysRevLett.121.098003
           
        .. [3] Cugliandolo, L. F., & Gonnella, G. (2018). Phases of active matter 
           in two dimensions. ArXiv:1810.11833 [Cond-Mat.Stat-Mech]. 
           http://arxiv.org/abs/1810.11833

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory. 
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        nbins : int, optional
            Number of bins. The default is 50.
        ptype : float, optional
            Particle type. If None, all particles are used.
            The default is None.
        other : float, optional
            Other particles. These are the ones which are counted as neighbors.
            If None, ptypes is used. The default is None.
        **kwargs
            Other keyword arguments are forwarded to 
            :py:func:`amep.order.psi_k`.

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> p6d = amep.evaluate.Psi6dist(
        ...     traj, skip=0.9, nav=2, pbc=True
        ... )
        >>> p6d.save("./eval/p6d.h5")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(p6d.psi6, p6d.avg)
        >>> axs.set_xlabel(r"$\psi_6$")
        >>> axs.set_ylabel(r"$p(\psi_6)$")
        >>> fig.savefig("./figures/evaluate/evaluate-Psi6dist.png")

        .. image:: /_static/images/evaluate/evaluate-Psi6dist.png
          :width: 400
          :align: center

        '''
        super(Psi6dist, self).__init__()
        
        self.name = 'psi6dist'
        
        self.__traj = traj
        self.__skip = skip
        self.__nav = nav
        self.__nbins = nbins
        self.__ptype = ptype
        self.__other = other
        self.__kwargs = kwargs
        
        self.__frames, res, self.__indices = average_func(
            self.__compute, np.arange(self.__traj.nframes), skip=self.__skip,
            nr=self.__nav, indices=True)
        
        self.__times = self.__traj.times[self.__indices]
        self.__avg   = res[0]
        self.__psi6  = res[1]
        
    def __compute(self, ind):
        r'''
        Calculation for a single frame,

        Parameters
        ----------
        ind : int
            Frame index.

        Returns
        -------
        hist : np.ndarray
            Histogram.
        bins : np.ndarray
            Bin positions. Same shape as hist.
        '''
        if self.__other is None:
            hexorder = psi_k(
                self.__traj[ind].coords(ptype=self.__ptype),
                self.__traj[ind].box,
                k = 6,
                **self.__kwargs
            )
        else:
            hexorder = psi_k(
                self.__traj[ind].coords(ptype=self.__ptype), 
                self.__traj[ind].box,
                other_coords=self.__traj[ind].coords(ptype=self.__other),
                k = 6,
                **self.__kwargs
            )
            
        hist, bins = distribution(
            np.abs(hexorder), xmin=0.0, xmax=1.0, nbins=self.__nbins
        )
        return hist, bins
    
    @property
    def psi6(self):
        r'''
        Hexagonal order parameters.

        Returns
        -------
        np.ndarray
            Hexagonal order parameters.

        '''
        return self.__psi6
    
    @property
    def frames(self):
        r'''
        Psi6dist for each frame.

        Returns
        -------
        np.ndarray
            Psi6dist for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the Psi6dist is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the Psi6dist is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged Psi6dist (averaged over the given number
        of frames).

        Returns
        -------
        various
            Time-averaged Psi6dist.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the Psi6dist has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class VelDist(BaseEvaluation):
    """Velocity distribution.
    """
    
    def __init__(
            self, traj, skip: float = 0.0, nav: int = 10,
            nbins: int = 50, ptype: int | None = None,
            vmin: float | None = None, vmax: float | None = None,
            v2min: float | None = None) -> None:
        r'''
        Calculate the distribution of velocities.

        Namely the components :math:`v_x, v_y, v_z`
        as well as the magnitude :math:`v` of the velocity 
        and its square :math:`v^2`. It also
        takes an average over several frames (= time average).

        For the :math:`v^2` distribution, logarithmic
        bins are used. Therefore `v2min`:math:`\ge 0`
        needs to be ensured. For the maximum value,
        `vmax` is used as :math:`3v_{max}^2`.
        Analogously for the maximum of :math:'v',
        where :math:`\sqrt{3v_{max}}` is used. For the minimum
        of :math:'v', the square root of `v2min` is used.

        Parameters
        ----------
        traj : Traj
            Trajectory object.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        nbins : int, optional
            Number of bins. The default is 50.
        ptype : float, optional
            Particle type. The default is None.
        vmin : float | None, optional
            Minimum value for the histogram in each spatial dimension
            :math:`v_x, v_y, v_z`.
            If None, then the minimum value of the last frame will be used
        vmax : float | None, optional
            Maximum value for the histogram in each spatial dimension
            :math:`v_x, v_y, v_z`.
            If None, then the maximum value of the last frame will be used
        v2min : float | None, optional
            Minimum value for the velocity-squared histogram.
            This value has to be :math:`\ge 0` due to the use of
            logarithmic bins for the :math:`v^2` distribution.
            If None, then the minimum value of the last frame will be used.

        Returns
        -------
        None
        
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> vdist = amep.evaluate.VelDist(traj, skip=0.9, nav=2)
        >>> vdist.save("./eval/vdist.h5")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(vdist.v, vdist.vdist)
        >>> axs.set_xlabel(r"$|\vec{v}|$")
        >>> axs.set_ylabel(r"$p(|\vec{v}|)$")
        >>> fig.savefig("./figures/evaluate/evaluate-VelDist.png")

        .. image:: /_static/images/evaluate/evaluate-VelDist.png
          :width: 400
          :align: center

        '''
        super(VelDist, self).__init__()
        
        self.name = 'veldist'
        
        self.__traj  = traj
        self.__skip  = skip
        self.__nav   = nav
        self.__nbins = nbins
        self.__ptype = ptype
        self.__vmin  = vmin
        self.__vmax  = vmax
        self.__v2min = v2min
        
        if self.__vmin is None:
            self.__vmin = np.min(self.__traj[-1].velocities(ptype=self.__ptype))
            
        if self.__vmax is None:
            self.__vmax = np.max(self.__traj[-1].velocities(ptype=self.__ptype))
            
        # due to logbins v2max must not be zero
        # set it explicitely here by using the last frame as reference
        if self.__v2min is None:
            self.__v2min = np.min(
                np.sum(self.__traj[-1].velocities(ptype=self.__ptype)**2,
                       axis=1)) 
        
        self.__frames, res, self.__indices = average_func(
            self.__compute, np.arange(self.__traj.nframes), skip=self.__skip,
            nr=self.__nav, indices=True)
        
        self.__times  = self.__traj.times[self.__indices]
        self.__vxdist = res[0]
        self.__vx     = res[1]
        self.__vydist = res[2]
        self.__vy     = res[3]
        self.__vzdist = res[4]
        self.__vz     = res[5]
        self.__vdist  = res[6]
        self.__v      = res[7]
        self.__v2dist = res[8]
        self.__v2     = res[9]
        
    def __compute(self, ind):
        r'''
        Calculation for a single frame,

        Parameters
        ----------
        ind : int
            Frame index.

        Returns
        -------
        hist : np.ndarray
            Histogram.
        bins : np.ndarray
            Bin positions. Same shape as hist.
        '''        
        vel = self.__traj[ind].velocities(ptype=self.__ptype)
        
        xhist, xbins = distribution(vel[:,0], nbins=self.__nbins,
                                    xmin=self.__vmin, xmax=self.__vmax)
        yhist, ybins = distribution(vel[:,1], nbins=self.__nbins,
                                    xmin=self.__vmin, xmax=self.__vmax)
        zhist, zbins = distribution(vel[:,2], nbins=self.__nbins,
                                    xmin=self.__vmin, xmax=self.__vmax)
        
        v2 = np.sum(vel**2, axis=1)
        
        
        v2hist, v2bins = distribution(
            v2, nbins=self.__nbins, logbins=True, xmin=self.__v2min,
            xmax=3*self.__vmax**2)
        vhist, vbins = distribution(
            np.sqrt(v2), nbins=self.__nbins, xmin=np.sqrt(self.__v2min),
            xmax=np.sqrt(3*self.__vmax**2))


# (KD, 2024.07.16) was:
        # v2hist, v2bins = distribution(
        #     v2, nbins=self.__nbins, logbins=True, xmin=self.__v2min,
        #     xmax=(3*self.__vmin**2+3*self.__vmax**2)/2)
        # vhist, vbins = distribution(
        #     np.sqrt(v2), nbins=self.__nbins, xmin=0,
        #     xmax=np.sqrt((3*self.__vmin**2+3*self.__vmax**2)/2))
        
        return xhist, xbins, yhist, ybins, zhist, zbins, vhist, vbins, v2hist,\
               v2bins 
    
    @property
    def vxdist(self) -> np.ndarray:
        r'''
        Time-averaged distribution of the x-component of the velocity.

        Returns
        -------
        np.ndarray
            Distribution of the x-component of the velocity.
        '''
        return self.__vxdist
    
    @property
    def vx(self):
        r'''
        x-component of the velocities.

        Returns
        -------
        np.ndarray
            x-component of the velocities..

        '''
        return self.__vx
    
    @property
    def vydist(self) -> np.ndarray:
        r'''
        Time-averaged distribution of the y-component of the velocity.

        Returns
        -------
        np.ndarray
            Distribution of the y-component of the velocity.
        '''
        return self.__vydist
    
    @property
    def vy(self):
        r'''
        y-component of the velocities.

        Returns
        -------
        np.ndarray
            y-component of the velocities..

        '''
        return self.__vy
    
    @property
    def vzdist(self)-> np.ndarray:
        r'''
        Time-averaged distribution of the z-component of the velocity.

        Returns
        -------
        np.ndarray
            Distribution of the z-component of the velocity.
        '''
        return self.__vzdist
    
    @property
    def vz(self):
        r'''
        z-component of the velocities.

        Returns
        -------
        np.ndarray
            z-component of the velocities..

        '''
        return self.__vz

    @property
    def vdist(self):
        r'''
        Time-averaged distribution of the magnitude of the velocity.

        Returns
        -------
        np.ndarray
            Distribution of the magnitude of the velocity.
        '''
        return self.__vdist
    
    @property
    def v(self):
        r'''
        Magnitude of the velocities.

        Returns
        -------
        np.ndarray
            Magnitude of the velocities.

        '''
        return self.__v
    
    @property
    def v2dist(self):
        r'''
        Time-averaged distribution of the squared magnitude of the velocity.

        Returns
        -------
        np.ndarray
            Distribution of the squared magnitude of the velocity.
        '''
        return self.__v2dist
    
    @property
    def v2(self):
        r'''
        Squared magnitude of the velocities.

        Returns
        -------
        np.ndarray
            Squared magnitude of the velocities.

        '''
        return self.__v2
    
    @property
    def frames(self):
        r'''
        VelDist for each frame.

        Returns
        -------
        np.ndarray
            VelDist for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the VelDist is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the VelDist is evaluated.

        '''
        return self.__times
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the VelDist has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices



class Dist(BaseEvaluation):
    """General distribution.
    """
    
    def __init__(
            self, traj: ParticleTrajectory | FieldTrajectory,
            keys: str | list[str, ...], func: Callable | None = None, skip: float = 0.0,
            nav: int = 10, nbins: int = 50, ptype: float | None = None,
            ftype: str | None = None, logbins: bool = False,
            xmin: float | None = None, xmax: float | None = None,
            **kwargs):
        r'''
        Calculate the distribution of a user-defined key or keys.

        Namely the components :math:`v_x, v_y, v_z`
        as well as the magnitude :math:`v` of the velocity 
        and its square :math:`v^2`. It also
        takes an average over several frames (= time average).

        Parameters
        ----------
        traj : Traj
            Trajectory object.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        keys : str, list(str)
        name keys, func=None, ...todo...
        xmin : float | None, optional
            Minimum value for the histogram. If None, then the
            minimum value of the last frame will be used
        xmax : float | None, optional
            Maximum value for the histogram. If None, then the
            maximum value of the last frame will be used
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        nbins : int, optional
            Number of bins. The default is None.
        ptype : float, optional
            Particle type. The default is None.

        Returns
        -------
        None
        
        Examples
        --------
        >>> import amep
        >>> path="/home/dormann/Documents/git_amep/examples/data/lammps.h5amep"
        >>> traj= amep.load.traj(path)
        >>> # distribution of the absolute velocity
        >>> dist=amep.evaluate.Dist(traj, "v*", func=np.linalg.norm, axis=1, skip=0.5, logbins=True)
        >>> # save results in hdf5 format
        >>> dist.save("./eval/dist-eval.h5", name="velocitydistribution")

        >>> fig,axs=amep.plot.new()
        >>> axs.plot(dist.x, dist.xdist)
        >>> axs.set_xlabel("Velocity")
        >>> axs.set_ylabel("P(Velocity)")
        >>> axs.semilogx()
        >>> fig.savefig("/home/dormann/Documents/git_amep/doc/source/_static/images/evaluate/evaluate-Dist.png")

        >>> # more examples:
        >>> # distribution of the x-position
        >>> dist=amep.evaluate.Dist(traj, "x", skip=0.5, logbins=True)
        >>> # distribution of the angular velocity
        >>> dist=amep.evaluate.Dist(traj, "omega*", func=np.linalg.norm, axis=1, skip=0.5, logbins=True)

        .. image:: /_static/images/evaluate/evaluate-Dist.png
          :width: 400
          :align: center

        '''
        super(Dist, self).__init__()
        
        if func is None:
            func = lambda x: x
        
        self.name = "dist"
        
        self.__traj  = traj
        self.__keys  = keys
        self.__skip  = skip
        self.__nav   = nav
        self.__nbins = nbins
        self.__ptype = ptype
        self.__xmin  = xmin
        self.__xmax  = xmax
        self.__func  = func
        self.__logbins = logbins
        self.__kwargs = kwargs

        if self.__xmin is None or self.__xmax is None:
            if isinstance(traj, FieldTrajectory):
                minmaxdata=self.__traj[-1].data(self.__keys, ftype=self.__ftype)
            else:
                minmaxdata=self.__traj[-1].data(self.__keys, ptype=self.__ptype)
            minmaxdata = func(minmaxdata, **kwargs)
        if self.__xmin is None:
            self.__xmin = np.min(minmaxdata)
            
        if self.__xmax is None:
            self.__xmax = np.max(minmaxdata)

        self.__frames, res, self.__indices = average_func(
            self.__compute, np.arange(self.__traj.nframes), skip=self.__skip,
            nr=self.__nav, indices=True)
        
        self.__times  = self.__traj.times[self.__indices]
        self.__xdist  = res[0]
        self.__x      = res[1]
        
    def __compute(self, ind):
        r'''
        Calculation for a single frame,

        Parameters
        ----------
        ind : int
            Frame index.

        Returns
        -------
        hist : np.ndarray
            Histogram.
        bins : np.ndarray
            Bin positions. Same shape as hist.
        '''
        data=self.__traj[ind].data(self.__keys, ptype=self.__ptype)
        data=self.__func(data, **self.__kwargs)

        keyhist, keybins = distribution(
            data, nbins=self.__nbins, xmin=self.__xmin, xmax=self.__xmax, logbins=self.__logbins)
        
        return keyhist, keybins
    
    @property
    def xdist(self):
        r'''
        Time-averaged distribution of the magnitude of the velocity.

        Returns
        -------
        np.ndarray
            Distribution of the magnitude of the velocity.
        '''
        return self.__xdist
    
    @property
    def x(self):
        r'''
        Magnitude of the velocities.

        Returns
        -------
        np.ndarray
            Magnitude of the velocities.

        '''
        return self.__x
    
    @property
    def frames(self):
        r'''
        VelDist for each frame.

        Returns
        -------
        np.ndarray
            VelDist for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the VelDist is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the VelDist is evaluated.

        '''
        return self.__times
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the VelDist has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices



# =============================================================================
# CLUSTER ANALYSIS
# =============================================================================
class ClusterSizeDist(BaseEvaluation):
    """Clustersize distribution.
    """

    def __init__(
            self, traj: ParticleTrajectory | FieldTrajectory,
            skip: float = 0.0, nav: int = 10,
            include_single: bool = False, ptype: int | None = None,
            ftype: str | list | None = None,
            use_density: bool = True, logbins: bool = False, 
            xmin: float | None = None, xmax: float | None = None,
            nbins: int | None = None, **kwargs
            ) -> None:
        r'''
        Calculate the weighted cluster size distribution.

        Takes an average over several frames.
        The calculation is based on a histogram of the cluster sizes.

        Notes
        -----
        The weighted cluster size distribution is defined by

        .. math::
            p(m) = m \frac{n_m}{N},

        where m is the cluster size, n_m the number of clusters with size m,
        and N the total number of particles. See Ref. [1]_ for further
        information.

        References
        ----------
        .. [1] Peruani, F., Schimansky-Geier, L., & Bär, M. (2010).
           Cluster dynamics and cluster size distributions in systems of self-
           propelled particles. The European Physical Journal Special Topics,
           191(1), 173–185. https://doi.org/10.1140/epjst/e2010-01349-1

        Parameters
        ----------
        traj : ParticleTrajectory or FieldTrajectory
            Trajectory object.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        include_single : bool, optional
            If True, single-particle clusters are included in the histogram.
            The default is False.
        ptype : float, optional
            Particle type. If None, all particles are considered.
            The default is None.
        ftype : str or None, optional
            Allows to specify for which field in the given FieldTrajectory
            the cluster size distribution should be calculated. If None, the
            cluster size distribution is calculated for the first field in the
            list of ftype keys and a warning is printed. The default is None.
        use_density: bool, optional
            Decides wether to use the integrated value of the field or the area
            as clustersize. Only set to false if your field cannot be 
            interpreted as a density, i.e., if it can be negative. The default 
            is True.
        logbins : bool, optional
            If True, logarithmic bins are used. The default is False.
        xmin : float or None, optional
            Minimum value of the bins. The default is None.
        xmax : float or None, optional
            Maximum value of the bins. The default is None.
        nbins : int or None, optional
            Number of bins. The default is None.
        **kwargs
            Other keyword arguments are forwarded to amep.cluster.identify and 
            amep.continuum.identify_clusters for ParticleTrajectories and 
            FieldTrajectories, respectively.

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> ptraj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> ftraj = amep.load.traj("../examples/data/continuum.h5amep")
        >>> pcsd = amep.evaluate.ClusterSizeDist(
        ...     ptraj, nav=5, pbc=True, skip=0.8
        ... )
        >>> fcsd = amep.evaluate.ClusterSizeDist(
        ...     ftraj, nav=5, pbc=True, skip=0.5, ftype="p",
        ...     use_density=False, threshold=0.2
        ... )
        >>> pcsd.save("./eval/csd-eval.h5", database=True, name="particles")
        >>> fcsd.save("./eval/csd-eval.h5", database=True, name="field")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(
        ...     pcsd.sizes, pcsd.avg, marker="*", ls="",
        ...     label="active Brownian particles"
        ... )
        >>> axs.plot(
        ...     fcsd.sizes, fcsd.avg, marker="+", ls="",
        ...     label="Keller-Segel model", color="darkorange"
        ... )
        >>> axs.loglog()
        >>> axs.set_xlabel(r"$m$")
        >>> axs.set_ylabel(r"$p(m)$")
        >>> axs.legend()
        >>> fig.savefig("./figures/evaluate/evaluate-ClusterSizeDist.png")

        .. image:: /_static/images/evaluate/evaluate-ClusterSizeDist.png
          :width: 400
          :align: center

        '''
        super(ClusterSizeDist, self).__init__()
        
        self.name = 'clustersizedist'
        
        self.__traj = traj
        self.__skip = skip
        self.__nav = nav
        self.__include_single = include_single
        self.__ptype = ptype
        self.__ftype = ftype
        self.__use_density = use_density
        self.__logbins = logbins
        self.__xmin = xmin
        self.__xmax = xmax
        self.__nbins = nbins
        self.__kwargs = kwargs
        
        if isinstance(self.__traj, ParticleTrajectory):
            # get limits for histogram bins
            if self.__xmax is None:
                self.__xmax = self.__traj[-1].n(ptype=self.__ptype)
            if self.__xmin is None:
                self.__xmin = 1
            # get number of bins
            if self.__nbins is None:
                self.__nbins = self.__traj[-1].n(ptype=self.__ptype)
            # calculation for particles
            self.__frames, res, self.__indices = average_func(
                self.__compute_particles,
                self.__traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        elif isinstance(self.__traj, FieldTrajectory):
            if self.__ftype is None:
                warnings.warn(
                    "amep.evaluate.ClusterSizeDist: The field is not "\
                    "specified. Use the ftype keyword to specify the field. "\
                    f"Results are calculated for {self.__traj[0].keys[0]}."
                )
                self.__ftype = self.__traj[0].keys[0]
            # get limits for histogram bins
            if self.__xmax is None:
                if self.__use_density:
                    # integrated density
                    if Version(np.__version) < Version("2.0.0"):
                        val_total_x = np.trapz(
                            self.__traj[-1].data(self.__ftype),
                            x=self.__traj[-1].grid[0],
                            axis=1
                        )
                        val_total = np.trapz(
                            val_total_x,
                            x=self.__traj[-1].grid[1][:, 0]
                        )
                    else:
                        val_total_x = np.trapezoid(
                            self.__traj[-1].data(self.__ftype),
                            x=self.__traj[-1].grid[0],
                            axis=1
                        )
                        val_total = np.trapezoid(
                            val_total_x,
                            x=self.__traj[-1].grid[1][:, 0]
                        )
                    self.__xmax = val_total
                else:
                    # total number of grid points
                    self.__xmax = self.__traj[-1].data(self.__ftype).size
            if self.__xmin is None:
                self.__xmin = self.__xmax/self.__traj[-1].data(self.__ftype).size
            # get number of bins
            if self.__nbins is None:
                self.__nbins = self.__traj[-1].data(self.__ftype).size
            # calculation for fields
            self.__frames, res, self.__indices = average_func(
                self.__compute_fields,
                self.__traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        else:
            raise TypeError(f'Invalid type of traj: {type(self.__traj)}.')

        self.__times = self.__traj.times[self.__indices]
        self.__avg = res[0]
        self.__sizes = res[1]

    def __compute_particles(self, frame):
        r'''
        Calculation for a single frame of particle-based simulation data.

        Parameters
        ----------
        frame : BaseFrame
            Frame of particle-based simulation data.

        Returns
        -------
        hist : np.ndarray
            Histogram.
        bins : np.ndarray
            Bin positions. Same shape as hist.
        '''
        sorted_clusters, _ = identify(
            frame.coords(ptype=self.__ptype),
            frame.box,
            **self.__kwargs
        )
        sizes = cluster_sizes(sorted_clusters)
            
        if self.__include_single is False:
            sizes = sizes[sizes>1]
            
        hist, bins = distribution(
            sizes, xmin=self.__xmin, xmax=self.__xmax, nbins=self.__nbins,
            density=False, logbins=self.__logbins
        )
            
        # normalization
        hist = hist*bins/np.sum(hist*bins)
        
        return hist, bins

    def __compute_fields(self, frame):
        r'''
        Calculation for a single frame of continuum simulation data.

        Parameters
        ----------
        frame : BaseField
            Field of particle-based simulation data.

        Returns
        -------
        hist : np.ndarray
            Histogram.
        bins : np.ndarray
            Bin positions. Same shape as hist.
        '''
        # get grid coordinates
        X, Y = frame.grid
        # calculate clusters
        field = frame.data(self.__ftype)
        ids, labels = identify_clusters(
            field,
            **self.__kwargs
        )
        if "pbc" in self.__kwargs:
            sizes, _, _, _, _, _, _ = cluster_properties(
                field, X, Y, ids, labels, pbc=self.__kwargs["pbc"]
            )
        else:
            sizes, _, _, _, _, _, _ = cluster_properties(
                field, X, Y, ids, labels
            )

        if self.__use_density:
            # remove the first entry (background)
            value = sizes[1:]
        else:
            # get number of grid points for each cluster
            _, occurance = np.unique(labels, return_counts=True)
            # remove the first entry (background)
            value = occurance[1:]

        # calculate the histogram
        hist, bins = distribution(
            value, xmin=self.__xmin, xmax=self.__xmax,
            nbins=self.__nbins, density=False, logbins=self.__logbins
        )
        # normalization (weighted distribution)
        hist = hist*bins/np.sum(hist*bins)

        return hist, bins


    @property
    def sizes(self):
        r'''
        Cluster sizes (as number of particles).

        Returns
        -------
        np.ndarray
            Cluster sizes.

        '''
        return self.__sizes

    @property
    def frames(self):
        r'''
        ClusterSizeDist for each frame.

        Returns
        -------
        np.ndarray
            ClusterSizeDist for each frame.

        '''
        return self.__frames

    @property
    def times(self):
        r'''
        Times at which the ClusterSizeDist is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the ClusterSizeDist is evaluated.

        '''
        return self.__times

    @property
    def avg(self):
        r'''
        Time-averaged ClusterSizeDist (averaged over the given number
        of frames).

        Returns
        -------
        various
            Time-averaged ClusterSizeDist.

        '''
        return self.__avg

    @property
    def indices(self):
        r'''
        Indices of all frames for which the ClusterSizeDist has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class ClusterGrowth(BaseEvaluation):
    """Growth of clusters over time.
    """

    def __init__(
            self, traj: ParticleTrajectory | FieldTrajectory,
            skip: float = 0.0, nav: int | None = 10,
            min_size: int = 0,
            ptype: int | None = None, ftype: str | list | None = None,
            mode: str = "largest", use_density: bool = True,
            **kwargs) -> None:
        r'''
        Calculates the size of the largest cluster, the mean cluster size, or 
        the weighted mean cluster size for each time step as studied in Refs. 
        [1]_ and [2]_ for example.
        
        References
        ----------
        
        .. [1] G. S. Redner, M. F. Hagan, and A. Baskaran, Structure and 
           Dynamics of a Phase-Separating Active Colloidal Fluid, 
           Phys. Rev. Lett. 110, 055701 (2013). 
           https://doi.org/10.1103/PhysRevLett.110.055701

        .. [2] C. B. Caporusso, L. F. Cugliandolo, P. Digregorio, G. Gonnella, 
           D. Levis, and A. Suma, Dynamics of Motility-Induced Clusters: 
           Coarsening beyond Ostwald Ripening, Phys. Rev. Lett. 131, 
           068201 (2023). https://doi.org/10.1103/PhysRevLett.131.068201

        Parameters
        ----------
        traj : ParticleTrajectory | FieldTrajectory
            Trajectory object.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        min_size : int, optional
            Consider only clusters with at least this size. If a 
            ParticleTrajectory is supplied, the minimum size is given in number
            of particles. If a FieldTrajectory is supplied, it is either given 
            in area units (use_density=False) or density units 
            (use_density=True). The default is 2.
        ptype : float, optional
            Particle type. If None, all particles are considered.
            The default is None.
        ftype : str or None, optional
            Allows to specify for which field in the given FieldTrajectory
            the cluster growth should be calculated. If None, the
            cluster growth is calculated for the first field in the
            list of ftype keys and a warning is printed. The default is None.
        mode: str, otptional
            Quantity to be used for calculating the cluster growth. Possible 
            options are "largest" (size of the largest cluster), "mean" (mean 
            cluster size), and "weighted mean" (weighted mean cluster size).
            The default is "largest".
        use_density: bool, optional
            Decides wether to use the integrated value of the field or the area
            as clustersize. Only set to false if your field cannot be 
            interpreted as a density, i.e., if it can be negative. The default 
            is True.
        **kwargs
            Other keyword arguments are forwarded to amep.cluster.cluster and 
            amep.continuum.identify_clusters for ParticleTrajectories and 
            FieldTrajectories, respectively.

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> fig, axs = amep.plot.new()
        >>> clg = amep.evaluate.ClusterGrowth(traj, mode="largest")
        >>> axs.plot(clg.times, clg.frames, label="largest")
        >>> clg = amep.evaluate.ClusterGrowth(traj, mode="mean")
        >>> axs.plot(clg.times, clg.frames, label="mean")
        >>> clg = amep.evaluate.ClusterGrowth(traj, mode="mean", min_size=20)
        >>> axs.plot(clg.times, clg.frames, label="mean min 20")
        >>> clg = amep.evaluate.ClusterGrowth(traj, mode="weighted mean")
        >>> axs.plot(clg.times, clg.frames, label="weighted mean")
        >>> axs.loglog()
        >>> axs.legend()
        >>> axs.set_xlabel(r"$t$")
        >>> axs.set_ylabel(r"$\langle m\rangle$")
        >>> fig.savefig("./figures/evaluate/evaluate-ClusterGrowth_1.png")

        .. image:: /_static/images/evaluate/evaluate-ClusterGrowth_1.png
          :width: 400
          :align: center

        >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
        >>> clg = amep.evaluate.ClusterGrowth(
        ...     traj, ftype="p", threshold=0.2
        ... )
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(clg.times, clg.frames)
        >>> axs.loglog()
        >>> axs.set_xlabel(r"$t$")
        >>> axs.set_ylabel(r"$\langle m\rangle$")
        >>> fig.savefig("./figures/evaluate/evaluate-ClusterGrowth_2.png")

        .. image:: /_static/images/evaluate/evaluate-ClusterGrowth_2.png
          :width: 400
          :align: center

        '''
        super().__init__()

        self.name = 'clustergrowth'

        self.__traj = traj
        self.__ptype = ptype
        self.__nav = nav
        self.__skip = skip
        self.__ptype = ptype
        self.__ftype = ftype
        self.__mode = mode
        self.__min_size = min_size
        self.__use_density = use_density
        self.__kwargs = kwargs

        if nav is None:
            nav = self.__traj.nframes

        # check mode
        if self.__mode not in ["largest", "mean", "weighted mean"]:
            raise ValueError(
                "amep.evaluate.ClusterGrowth: Invalid value for mode. Got "\
                f"{self.__mode}. Please choose one of ['average', 'mean', "\
                "'weighted mean']."
            )

        if isinstance(self.__traj, ParticleTrajectory):
            # calculation for particles
            self.__frames, self.__avg, self.__indices = average_func(
                self.__compute_particles,
                traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        elif isinstance(self.__traj, FieldTrajectory):
            # calculation for fields
            if self.__ftype is None:
                warnings.warn(
                    "amep.evaluate.ClusterSizeDist: The field is not "\
                    "specified. Use the ftype keyword to specify the field. "\
                    f"Results are calculated for {self.__traj[0].keys[0]}."
                )
                self.__ftype = self.__traj[0].keys[0]
            self.__frames, self.__avg, self.__indices = average_func(
                self.__compute_fields,
                traj,
                skip=self.__skip,
                nr=self.__nav,
                indices=True
            )
        else:
            raise TypeError(
                f'Invalid type of traj: {type(self.__traj)}.'
            )

        self.__times = self.__traj.times[self.__indices]


    def __compute_particles(self, frame) -> int | float:
        r'''
        Calculation for a single frame.

        Parameters
        ----------
        frame : BaseFrame
            Frame.

        Returns
        -------
        size : int or float
           Largest, mean, or weighted mean cluster size.
        '''
        # determine clusters for particles
        sorted_clusters, idx = identify(
            frame.coords(ptype=self.__ptype),
            frame.box,
            **self.__kwargs
        )
        sizes = cluster_sizes(sorted_clusters)
        
        # get relevant values
        values = sizes[sizes >= self.__min_size]
        
        # if no clusters are detected
        if len(values) == 0:
            values = [0.0]
            weights = [1.0]
        else:
            weights = values
        
        if self.__mode == "mean":
            # mean cluster size
            return np.mean(values)
        elif self.__mode == "weighted mean":
            # weighted mean cluster size
            return np.average(values, weights=weights)
        else:
            # size of largest cluster
            return values[0]


    def __compute_fields(self, frame):
        r'''
        Calculation for a single frame.

        Parameters
        ----------
        frame : BaseFrame
            Frame.

        Returns
        -------
        size : int
            Size of the largest cluster which is the mass of the cluster.
        '''
        # calculate clusters
        field = frame.data(self.__ftype)
        ids, labels = identify_clusters(
            field,
            **self.__kwargs
        )
        if "pbc" in self.__kwargs:
            sizes, _, _, _, _, _, _ = cluster_properties(
                field, *frame.grid, ids, labels, pbc=self.__kwargs["pbc"]
            )
        else:
            sizes, _, _, _, _, _, _ = cluster_properties(
                field, *frame.grid, ids, labels
            )

        if self.__use_density:
            # remove the first entry (background)
            values = sizes[1:]
        else:
            # get number of grid points for each cluster
            _, occurance = np.unique(labels, return_counts=True)
            # remove the first entry (background)
            values = occurance[1:]
            
        # get relevant values
        values = values[values >= self.__min_size]
            
        # case if there is no cluster
        if len(values) == 0:
            values = [0.0]
            weights = [1.0]
        else:
            weights = values
        
        if self.__mode == "mean":
            # mean cluster size
            return np.mean(values)
        elif self.__mode == "weighted mean":
            # weighted mean cluster size
            return np.average(values, weights=weights)
        else:
            # size of largest cluster
            return np.max(values)


    @property
    def frames(self):
        r'''
        Size of the largest cluster for each frame.

        Returns
        -------
        np.ndarray
            Function value for each frame.

        '''
        return self.__frames

    @property
    def times(self):
        r'''
        Times at which the size of the largest cluster is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the size of the largest cluster is evaluated.

        '''
        return self.__times

    @property
    def avg(self):
        r'''
        Time-averaged size of the largest cluster (averaged over the given 
        number of frames).

        Returns
        -------
        float
            Time-averaged size of the largest cluster.

        '''
        return self.__avg

    @property
    def indices(self):
        r'''
        Indices of all frames for which the size of the largest cluster has 
        been calculated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


# =============================================================================
# TIME CORRELATION FUNCTIONS
# =============================================================================
class MSD(BaseEvaluation):
    """Mean-square displacement.
    """
    
    def __init__(
            self, traj: ParticleTrajectory, ptype: int | None = None,
            skip: float = 0.0, nav: int | None = 10,
            use_nojump: bool = False, pbc: bool = True) -> None:
        r'''
        Calculates the mean-square displacement over time. If periodic boundary
        conditions are applied, the unwrapped coordinates are used if they are
        available. If not, nojump coordinates are used. If periodic boundary
        conditions are not applied, the normal coordinates inside the box are
        used.

        Parameters
        ----------
        traj : amep.trajectory.ParticleTrajectory
            ParticleTrajectory object.
        ptype : int or None, optional
            Particle type. The default is None.
        skip : float, optional
            Skip this fraction of frames at the beginning 
            of the trajectory. The default is 0.0.
        nav : int or None, optional
            Number of time steps at which the mean square
            displacement should be evaluated. The default is 10.
        use_nojump : bool, optional
            Forces the use of nojump coordinates. The default is False.
        pbc : bool, optional
            If True, periodic boundary conditions are applied and the unwrapped
            coordinates are used for the calculation. If those are not 
            available, nojump coordinates will be used instead. The default is
            False.

        Returns
        -------
        None.
        
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> traj.nojump()
        >>> msd_normal = amep.evaluate.MSD(traj, pbc=False)
        >>> msd_unwrapped = amep.evaluate.MSD(traj, pbc=True)
        >>> msd_nojump = amep.evaluate.MSD(traj, use_nojump=True)
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(
        ...     msd_normal.times, msd_normal.frames, ls="--",
        ...     marker="", label="normal coordinates", c="k"
        ... )
        >>> axs.plot(
        ...     msd_unwrapped.times, msd_unwrapped.frames, ls="-",
        ...     marker='', label="unwrapped coordinates", c="gray"
        ... )
        >>> axs.plot(
        ...     msd_nojump.times, msd_nojump.frames, ls=":",
        ...     marker='', label="nojump coordinates", c="darkorange"
        ... )
        >>> axs.legend()
        >>> axs.loglog()
        >>> axs.set_xlabel(r"$t$")
        >>> axs.set_ylabel(r"MSD$(t)$")
        >>> fig.savefig('./figures/evaluate/evaluate-MSD.png')

        .. image:: /_static/images/evaluate/evaluate-MSD.png
          :width: 400
          :align: center

        '''
        # TODO: Use shifted correlation for better averaging
        # TODO: Add possibility to use logarithmic times
        super(MSD, self).__init__()
        
        self.name = 'msd'
        
        self.__traj = traj
        self.__ptype = ptype
        self.__nskip = int(skip*traj.nframes)
        self.__skip = skip
        self.__nav = nav
        self.__pbc = pbc
        self.__use_nojump = use_nojump
        
        if self.__nav is None:
            self.__nav = self.__traj.nframes
            
        # get reference frame at time t0:
        self.__frame0 = self.__traj[self.__nskip]
        
        # check data availability
        if self.__pbc:
            if self.__use_nojump:
                try:
                    a = self.__traj[0].nojump_coords()
                except:
                    raise ValueError(
                        'No nojump coordinates available. Call traj.nojump() '\
                        'to calculate the nojump coordinates.'
                    )
            else:
                try:
                    a = self.__traj[0].unwrapped_coords()
                except:
                    raise ValueError(
                        'There are no unwrapped coordinates available. '\
                        'Please set use_nojump=True to use nojump coordinates '\
                        'instead or do not apply periodic boundary conditions.'
                    )
        # get mode
        if self.__pbc and self.__use_nojump:
            self.__mode = 'nojump'
        elif self.__pbc:
            self.__mode = 'unwrapped'
        else:
            self.__mode = 'normal'

        # calculation
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute, self.__traj, skip=self.__skip,
            nr=self.__nav, indices=True
        )
        self.__times = self.__traj.times[self.__indices]


    def __compute(self, frame):
        r'''
        Calculation for a single frame.

        Parameters
        ----------
        frame : BaseFrame
            Frame.

        Returns
        -------
        float
            Mean-square displacement.

        '''
        if self.__mode == 'unwrapped':
            return msd(
                self.__frame0.unwrapped_coords(ptype=self.__ptype),
                frame.unwrapped_coords(ptype=self.__ptype)
            )
        elif self.__mode == 'nojump':
            return msd(
                self.__frame0.nojump_coords(ptype=self.__ptype),
                frame.nojump_coords(ptype=self.__ptype)
            )
        else:
            return msd(
                self.__frame0.coords(ptype=self.__ptype),
                frame.coords(ptype=self.__ptype)
            )
        
    @property
    def ptype(self):
        r'''
        Particle type(s) for which the MSD has been calculated.

        Returns
        -------
        float or list of floats
            Particle type(s).

        '''
        return self.__ptype
    
    @property
    def frames(self):
        r'''
        MSD for each frame.

        Returns
        -------
        np.ndarray
            MSD for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the MSD is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the MSD is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged MSD (averaged over the given number
        of frames).

        Returns
        -------
        float
            Time-averaged MSD.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the MSD has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class VACF(BaseEvaluation):
    """Velocity autocorrelation function.
    """

    def __init__(
            self, traj: ParticleTrajectory, ptype: int | None = None,
            skip: float = 0.0, nav: int | None = 10,
            direction: str = 'xyz') -> None:
        r'''
        Calculate the velocity autocorrelation function.

        Averages over all particles of the given type.

        Parameters
        ----------
        traj : amep.base.BaseTrajectory
            Trajectory object.
        ptype : int, optional
            Particle type. The default is None.
        skip : float, optional
            Skip this fraction of frames at the beginning
            of the trajectory. The default is 0.0.
        nav : int, optional
            Number of time steps at which the autocorrelation
            function should be evaluated. The default is 10.
        direction : str, optional
            'x', 'y', 'z', or any combination of it.
            The default is 'xyz' (average over all directions).

        Returns
        -------
        None

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> vacf = amep.evaluate.VACF(traj, direction="xy")
        >>> vacf.save("./eval/vacf.h5")
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(vacf.times[1:], vacf.frames[1:]/vacf.frames[0])
        >>> axs.axhline(0.0, color="k", lw=1, ls="--")
        >>> axs.semilogx()
        >>> axs.set_ylim(-0.05, 1.0)
        >>> axs.set_xlabel(r"$t$")
        >>> axs.set_ylabel(r"$C_v(t)/C_v(0)$")
        >>> fig.savefig("./figures/evaluate/evaluate-VACF.png")

        .. image:: /_static/images/evaluate/evaluate-VACF.png
          :width: 400
          :align: center

        '''
        # TODO: Use shifted correlation for better averaging
        # TODO: Add possibility to use logarithmic times
        super(VACF, self).__init__()
        
        self.name = 'vacf'
        
        self.__traj      = traj
        self.__ptype     = ptype
        self.__nskip     = int(skip*traj.nframes)
        self.__direction = direction
        self.__skip      = skip
        self.__nav       = nav
        
        # get directions to be considered
        self.__components = []
        if 'x' in self.__direction:
            self.__components.append(0)
        if 'y' in self.__direction:
            self.__components.append(1)
        if 'z' in self.__direction:
            self.__components.append(2)
            
        if self.__components == []:
            raise ValueError(
                """amep.evaluate.VACF: Wrong direction specified. Allowed
                direction values are 'x', 'y', and 'z' as well as all possible
                combinations up to 'xyz'. Note that the order does not
                matter."""
            )
        
        if self.__nav is None:
            self.__nav = self.__traj.nframes
            
        # get reference frame at t0
        self.__frame0 = self.__traj[self.__nskip]
        
        # main calculation
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute, self.__traj, skip=self.__skip,
            nr=self.__nav, indices=True
        )
        
        # get times
        self.__times = self.__traj.times[self.__indices]        
        
    def __compute(self, frame):
        r'''
        Computation for a single frame.

        Parameters
        ----------
        frame : BaseFrame
            Frame.

        Returns
        -------
        float
            Velocity autocorrelation function.

        '''
        v0 = self.__frame0.velocities(ptype=self.__ptype)
        v = frame.velocities(ptype=self.__ptype)

        return acf(v0[:,self.__components], v[:,self.__components])
            
    @property
    def direction(self):
        r'''
        Considered spatial directions.

        Returns
        -------
        str
            Direction specification.

        '''
        return self.__direction
    
    @property
    def ptype(self):
        r'''
        Particle type(s) for which the VACF has been calculated.

        Returns
        -------
        float or list of floats
            Particle type(s).

        '''
        return self.__ptype
    
    @property
    def frames(self):
        r'''
        VACF for each frame.

        Returns
        -------
        np.ndarray
            VACF for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the VACF is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the VACF is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged VACF (averaged over the given number
        of frames).

        Returns
        -------
        float
            Time-averaged VACF.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the VACF has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class OACF(BaseEvaluation):
    """Orientational autocorrelation function.
    """

    def __init__(self, traj, ptype=None, skip=0.0, nav: int | None = 10, direction='xyz'):
        r'''
        Calculate the orientational autocorrelation function
        averaged over all particles of the given type.

        Parameters
        ----------
        traj : amep.base.BaseTrajectory
            Trajectory object.
        ptype : int, optional
            Particle type. The default is None.
        skip : float, optional
            Skip this fraction of frames at the beginning
            of the trajectory. The default is 0.0.
        nav : int, optional
            Number of time steps at which the autocorrelation
            function should be evaluated. The default is 10.
        direction : str, optional
            'x', 'y', 'z', or any combination of it.
            The default is 'xyz' (average over all directions).

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> oacf = amep.evaluate.OACF(traj)
        >>> oacf.save('./eval/oacf.h5')
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(
        ...     oacf.times[1:], oacf.frames[1:],
        ...     label="data", ls="", c="darkorange"
        ... )
        >>> axs.plot(
        ...     oacf.times[1:], np.exp(-oacf.times[1:]), c="k",
        ...     label=r"$\exp(-t)$", lw=1, ls="--",
        ...     marker=""
        ... )
        >>> axs.set_xlabel(r"$t$")
        >>> axs.set_ylabel(r"$\langle\vec{\mu}(t)\cdot\vec{\mu}(0)\rangle$")
        >>> axs.legend()
        >>> axs.semilogx()
        >>> fig.savefig("./figures/evaluate/evaluate-OACF.png")

        .. image:: /_static/images/evaluate/evaluate-OACF.png
          :width: 400
          :align: center

        '''
        # TODO: Use shifted correlation for better averaging
        # TODO: Add possibility to use logarithmic times
        super(OACF, self).__init__()
        
        self.name = 'oacf'
        
        self.__traj      = traj
        self.__ptype     = ptype
        self.__nskip     = int(skip*traj.nframes)
        self.__direction = direction
        self.__skip      = skip
        self.__nav       = nav
        
        # get directions to be considered
        self.__components = []
        if 'x' in self.__direction:
            self.__components.append(0)
        if 'y' in self.__direction:
            self.__components.append(1)
        if 'z' in self.__direction:
            self.__components.append(2)
            
        if self.__components == []:
            raise ValueError(
                """amep.evaluate.VACF: Wrong direction specified. Allowed
                direction values are 'x', 'y', and 'z' as well as all possible
                combinations up to 'xyz'. Note that the order does not
                matter."""
            )
        
        if self.__nav is None:
            self.__nav = self.__traj.nframes
            
        # get the reference frame at t0
        self.__frame0 = self.__traj[self.__nskip]
        
        # main calculation
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute, self.__traj, skip=self.__skip,
            nr=self.__nav, indices=True
        )

        # get times
        self.__times = self.__traj.times[self.__indices]         
        
    def __compute(self, frame):
        r'''
        Computation for a single frame.

        Parameters
        ----------
        frame : BaseFrame
            Frame.

        Returns
        -------
        float
            Orientation autocorrelation function.

        '''
        v0 = self.__frame0.orientations(ptype=self.__ptype)
        v = frame.orientations(ptype=self.__ptype)

        return acf(v0[:,self.__components], v[:,self.__components])
            
    @property
    def direction(self):
        r'''
        Considered spatial directions.

        Returns
        -------
        str
            Direction specification.

        '''
        return self.__direction
    
    @property
    def ptype(self):
        r'''
        Particle type(s) for which the OACF has been calculated.

        Returns
        -------
        float or list of floats
            Particle type(s).

        '''
        return self.__ptype
    
    @property
    def frames(self):
        r'''
        OACF for each frame.

        Returns
        -------
        np.ndarray
            OACF for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the OACF is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the OACF is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged OACF (averaged over the given number
        of frames).

        Returns
        -------
        float
            Time-averaged OACF.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the OACF has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class TimeCor(BaseEvaluation):
    """Calculate the autocorrelation function in time of a user-defined 
    quantity.
    """

    def __init__(self, traj, *args, ptype=None, skip=0.0, nav: int | None = 10):
        r'''
        Calculate the autocorrelation function.
        Averages over all particles of the given type.
        
        Notes
        -----
        If only one data key is given, the 1D time correlation function
        
        .. math::
            
            c(t)=\langle x(0)x(t)\rangle
            
        is calculated. If more than one data key is given, the data is combined
        in a vector and the calculated time correlation function reads
        
        .. math::

            c(t)=\langle\vec{x}(0)\cdot\vec{x}(t)\rangle.

        Parameters
        ----------
        traj : amep.base.BaseTrajectory
            Trajectory object.
        *args : str
            Data keys.
        ptype : int, optional
            Particle type. The default is None.
        skip : float, optional
            Skip this fraction of frames at the beginning 
            of the trajectory. The default is 0.0.
        nav : int, optional
            Number of time steps at which the autocorrelation
            function should be evaluated. The default is 10.

        Returns
        -------
        None.
        
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> timecor = amep.evaluate.TimeCor(traj, "mux")
        >>> timecor.save('./eval/my-time-cor.h5')
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(timecor.times, timecor.frames)
        >>> axs.set_xlabel(r"$t$")
        >>> axs.set_ylabel(r"$C_{\mu_x}(t)$")
        >>> axs.semilogx()
        >>> fig.savefig("./figures/evaluate/evaluate-TimeCor.png")

        .. image:: /_static/images/evaluate/evaluate-TimeCor.png
          :width: 400
          :align: center

        '''
        # TODO: Use shifted correlation for better averaging
        # TODO: Add possibility to use logarithmic times
        super(TimeCor, self).__init__()
        
        self.name = 'acf'
        
        self.__traj      = traj
        self.__args      = args
        self.__ptype     = ptype
        self.__nskip     = int(skip*traj.nframes)
        self.__skip      = skip
        self.__nav       = nav
        
        if self.__nav is None:
            self.__nav = self.__traj.nframes
            
        # get reference frame at t0
        self.__frame0 = self.__traj[self.__nskip]
        
        # main calculation
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute, self.__traj, skip=self.__skip,
            nr=self.__nav, indices=True)

        # get times
        self.__times = self.__traj.times[self.__indices]        
        
    def __compute(self, frame):
        r'''
        Computation for a single frame.

        Parameters
        ----------
        frame : BaseFrame
            Frame.

        Returns
        -------
        float
            Autocorrelation function.

        '''
        return acf(
            self.__frame0.data(*self.__args, ptype=self.__ptype),
            frame.data(*self.__args, ptype=self.__ptype)
        )
    
    @property
    def args(self):
        r'''
        Data keys.

        Returns
        -------
        tuple of floats
            Data keys.

        '''
        return self.__args
       
    @property
    def ptype(self):
        r'''
        Particle type(s) for which the TimeCor has been calculated.

        Returns
        -------
        float or list of floats
            Particle type(s).

        '''
        return self.__ptype
    
    @property
    def frames(self):
        r'''
        TimeCor for each frame.

        Returns
        -------
        np.ndarray
            TimeCor for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the TimeCor is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the TimeCor is evaluated.

        '''
        return self.__times
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the TimeCor has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices

# =============================================================================
# ENERGY FUNCTIONS
# =============================================================================
class EkinTot(BaseEvaluation):
    """Total kinetic energy.
    """

    def __init__(
            self, traj: ParticleTrajectory, mass: float | np.ndarray,
            inertia: float | np.ndarray, skip: float = 0.0, nav: int = 10, **kwargs):
        r'''Calculate the total kinetic energy of each particle over a
        trajectory.

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with simulation data.
        skip : float, default=0.0
            Skip this fraction at the beginning of
            the trajectory.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        mass : float or np.ndarray
            Mass(es) of the particles.
        inertia : float or np.ndarray
            Moment of inertia of the particles.
        **kwargs : Keyword Arguments
            General python keyword arguments to be
            forwarded to the function f
              
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> ekintotal = amep.evaluate.EkinTot(
        ...     traj, mass=1.0, inertia=0.1, nav=traj.nframes
        ... )
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(ekintotal.times, ekintotal.frames)
        >>> axs.axhline(ekintotal.avg, ls="--", c="k")
        >>> axs.set_xlabel(r'$t$')
        >>> axs.set_ylabel(r'$E_{\rm tot}$')
        >>> axs.semilogy()
        >>> fig.savefig("./figures/evaluate/evaluate-EkinTot.png")

        .. image:: /_static/images/evaluate/evaluate-EkinTot.png
          :width: 400
          :align: center

        '''
        super().__init__()
        
        self.name = 'ekintot'
        
        self.__traj      = traj
        self.__skip      = skip
        self.__mass      = mass
        self.__inertia   = inertia
        self.__nav       = nav
        self.__kwargs    = kwargs
        
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute, np.arange(self.__traj.nframes),
            mass=self.__mass, inertia=self.__inertia, skip=self.__skip,
            nr=self.__nav, indices=True)
        
        self.__times = self.__traj.times[self.indices]

    def __compute(self, ind, **kwargs):
        r'''
        Calculation for a single frame,

        Parameters
        ----------
        ind : int
            Frame index.
        **kwargs : Keyword Arguments
            Keyword arguments to be passed to the
            analysis function.

        Returns
        -------
        res : np.ndarray
            result.
        '''
        return thermo.total_kinetic_energy(self.__traj[ind], **kwargs)
    
    @property
    def frames(self):
        r'''
        Total kinetic energy for each frame.

        Returns
        -------
        np.ndarray
            Total kinetic energy for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the total kinetic energy is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the total kinetic energy is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged total kinetic energy (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged total kinetic energy of each particle.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the total kinetic energy has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class EkinTrans(BaseEvaluation):
    """Translational kinetic energy.
    """

    def __init__(
            self, traj, mass: float | np.ndarray,
            mode: str = "total",
            skip: float = 0.0, nav: int = 10, **kwargs):
        r'''Calculate the translational kinetic energy of each particle.

        Calculates energy over a whole trajectory.

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with simulation data.
        mass : float | np.ndarray
            Function to be used for analysis.
            Its signature should be `func(frame, **kwargs)->float`
        mode : "total" or "particle"
            How to return the energy, c.f. thermo.py
                total : sum over all particles
                particle : individual particle-energies
        skip : float, default=0.0
            Skip this fraction at the beginning of
            the trajectory.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        **kwargs : Keyword Arguments
            General python keyword arguments to be
            forwarded to the function f

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> ekintrans = amep.evaluate.EkinTrans(
        ...     traj, mass=1.0, nav=traj.nframes
        ... )
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(ekintrans.times, ekintrans.frames)
        >>> axs.axhline(ekintrans.avg, ls="--", c="k")
        >>> axs.set_xlabel(r'$t$')
        >>> axs.set_ylabel(r'$E_{\rm trans}$')
        >>> axs.semilogy()
        >>> fig.savefig("./figures/evaluate/evaluate-EkinTrans.png")

        .. image:: /_static/images/evaluate/evaluate-EkinTrans.png
          :width: 400
          :align: center

        '''
        super(EkinTrans, self).__init__()
        
        self.name = 'ekintrans'
        
        self.__traj      = traj
        self.__skip      = skip
        self.__mass      = mass
        self.__nav       = nav
        self.__kwargs    = kwargs
        
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute, np.arange(self.__traj.nframes), mass=self.__mass,
            skip=self.__skip, nr=self.__nav, indices=True)

        self.__times = self.__traj.times[self.indices]

    def __compute(self, ind, **kwargs):
        r'''
        Calculation for a single frame,

        Parameters
        ----------
        ind : int
            Frame index.
        **kwargs : Keyword Arguments
            Keyword arguments to be passed to the
            analysis function.

        Returns
        -------
        res : np.ndarray
            result.
        '''
        return thermo.translational_kinetic_energy(self.__traj[ind], **kwargs)
    
    @property
    def frames(self):
        r'''
        Translational kinetic energy for each particle and in each frame.

        Returns
        -------
        np.ndarray
            Translational kinetic energy for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the translational kinetic energy is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the translational kinetic energy is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged translational kinetic energy (averaged over the given
        number of frames).

        Returns
        -------
        np.ndarray
            Time-averaged translational kinetic energy of each particle.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the translational kinetic energy has
        been calculated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices


class EkinRot(BaseEvaluation):
    """Rotational kinetic energy.
    """

    def __init__(self, traj: ParticleTrajectory,
                 inertia: float | np.ndarray,
                 mode: str = "total",
                 skip: float = 0.0,
                 nav: int = 10,
                 avgstyle=None, **kwargs):
        r'''
        Calculate the rotational kinetic energy of each particle over a
        trajectory.

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with simulation data.
        mode : "total" or "particle"
            How to return the energy, c.f. thermo.py
                total : sum over all particles
                particle : individual particle-energies
        skip : float, default=0.0
            Skip this fraction at the beginning of
            the trajectory.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        **kwargs : Keyword Arguments
            General python keyword arguments to be
            forwarded to the function f

        '''
        super(EkinRot, self).__init__()
        
        self.name = 'ekinrot'
        
        self.__traj      = traj
        self.__skip      = skip
        self.__inertia   = inertia
        self.__nav       = nav
        self.__kwargs    = kwargs
        
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute, np.arange(self.__traj.nframes), indices=True,
            inertia=self.__inertia, skip=self.__skip, nr=self.__nav)
        
        self.__times = self.__traj.times[self.__indices]
        
    def __compute(self, ind, **kwargs):
        r'''
        Calculation for a single frame,

        Parameters
        ----------
        ind : int
            Frame index.
        **kwargs : Keyword Arguments
            Keyword arguments to be passed to the
            analysis function.

        Returns
        -------
        res : np.ndarray
            result.
        '''
        return thermo.rotational_kinetic_energy(self.__traj[ind], **kwargs)
    
    @property
    def frames(self):
        r'''
        Rotational kinetic energy of each particle for each frame.

        Returns
        -------
        np.ndarray
            Rotational kinetic energy for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the rotational kinetic energy is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the rotational kinetic energy is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged rotational kinetic energy (averaged over the given number
        of frames).

        Returns
        -------
        np.ndarray
            Time-averaged rotational kinetic energy of each particle.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the rotational kinetic energy has been
        evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices
    
    
# =============================================================================
# TEMPERATURES
# =============================================================================
class Tkin(BaseEvaluation):
    """Kinetic temperature based on the second moment of the velocity
    distribution.
    """
    def __init__(
            self, traj: ParticleTrajectory, skip: float = 0.0, nav: int = 10,
            ptype: int | None = None, **kwargs
            ) -> None:
        r'''
        Calculates the kinetic temperature based on the second moment of the
        velocity distribution. [1]_
        
        References
        ----------
        
        .. [1] L. Hecht, L. Caprini, H. Löwen, and B. Liebchen, 
           How to Define Temperature in Active Systems?, J. Chem. Phys. 161, 
           224904 (2024). https://doi.org/10.1063/5.0234370

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with simulation data.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float, optional
            Particle type. The default is None.
        **kwargs
            Other keyword arguments are forwarded to
            `amep.thermo.Tkin`.

        '''
        super(Tkin, self).__init__()

        self.name = 'Tkin'
        
        self.__traj   = traj
        self.__skip   = skip
        self.__nav    = nav
        self.__ptype  = ptype
        self.__kwargs = kwargs
        
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute,
            self.__traj,
            skip = self.__skip,
            nr = self.__nav,
            indices = True
        )
        
        self.__times = self.__traj.times[self.__indices]

        
    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
            A single frame of particle-based simulation data.
              
        Returns
        -------
        temp: float
            Mean temperature.
        '''
        temp = thermo.Tkin(
            frame.velocities(ptype=self.__ptype),
            frame.mass(ptype=self.__ptype),
            **self.__kwargs
        )
        return temp

    
    @property
    def frames(self):
        r'''
        Mean kinetic temperature for each frame.

        Returns
        -------
        np.ndarray
            Function value for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the mean kinetic temperature is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the function is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged kinetic temperature 
        (averaged over the given number of frames).

        Returns
        -------
        np.ndarray
            Time-averaged spatial velocity correlation function.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the mean kinetic temperature
        has been evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices

class Tkin4(BaseEvaluation):
    """Kinetic temperature based on the 4th moment of the velocity
    distribution.
    """
    def __init__(
            self, traj: ParticleTrajectory, 
            skip: float = 0.0, nav: int = 10,
            ptype: int | None = None, **kwargs
            ) -> None:
        r'''
        Calculates the kinetic temperature based on the 4th moment of the
        velocity distribution. [1]_
        
        References
        ----------
        
        .. [1] L. Hecht, L. Caprini, H. Löwen, and B. Liebchen, 
           How to Define Temperature in Active Systems?, J. Chem. Phys. 161, 
           224904 (2024). https://doi.org/10.1063/5.0234370

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with simulation data.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float, optional
            Particle type. The default is None.
        **kwargs
            Other keyword arguments are forwarded to
            `amep.thermo.Tkin4`.

        '''
        super(Tkin4, self).__init__()

        self.name = 'Tkin4'
        
        self.__traj   = traj
        self.__skip   = skip
        self.__nav    = nav
        self.__ptype  = ptype
        self.__kwargs = kwargs
        
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute,
            self.__traj,
            skip = self.__skip,
            nr = self.__nav,
            indices = True
        )
        
        self.__times = self.__traj.times[self.__indices]

        
    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
            A single frame of particle-based simulation data.
              
        Returns
        -------
        temp: float
            Mean temperature.
        '''
        temp = thermo.Tkin4(
            frame.velocities(ptype=self.__ptype),
            frame.mass(ptype=self.__ptype),
            **self.__kwargs
        )
        return temp

    
    @property
    def frames(self):
        r'''
        Mean 4th-moment kinetic temperature for each frame.

        Returns
        -------
        np.ndarray
            Function value for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the mean 4th-moment kinetic temperature is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the function is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged 4th-moment kinetic temperature 
        (averaged over the given number of frames).

        Returns
        -------
        np.ndarray
            Time-averaged spatial velocity correlation function.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the mean 4th-moment kinetic temperature
        has been evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices

class Tosc():
    """Oscillator temperature.
    """
    def __init__(
            self, traj: ParticleTrajectory, k: float,
            skip: float = 0.0, nav: int = 10,
            ptype: int | None = None,
            ) -> None:
        r'''
        Calculates the oscillator temperature. [1]_
        
        References
        ----------
        
        .. [1] L. Hecht, L. Caprini, H. Löwen, and B. Liebchen, 
           How to Define Temperature in Active Systems?, J. Chem. Phys. 161, 
           224904 (2024). https://doi.org/10.1063/5.0234370

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with simulation data.
        k : float
            Strength of the harmonic confinement.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float, optional
            Particle type. The default is None.

        '''
        super(Tosc, self).__init__()

        self.name = 'Tosc'
        
        self.__traj = traj
        self.__skip = skip
        self.__nav = nav
        self.__ptype = ptype
        self.__k = k
        
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute,
            self.__traj,
            skip = self.__skip,
            nr = self.__nav,
            indices = True
        )
        
        self.__times = self.__traj.times[self.__indices]

        
    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
            A single frame of particle-based simulation data.
              
        Returns
        -------
        temp: float
            Mean temperature.
        '''
        temp = thermo.Tosc(
            frame.coords(ptype=self.__ptype),
            self.__k
        )
        return temp

    @property
    def frames(self):
        r'''
        Mean oscillator temperature for each frame.

        Returns
        -------
        np.ndarray
            Function value for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the mean oscillator temperature is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the function is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged oscillator temperature
        (averaged over the given number of frames).

        Returns
        -------
        np.ndarray
            Time-averaged spatial velocity correlation function.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the mean oscillator temperature
        has been evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices

class Tconf(BaseEvaluation):
    """Configurational temperature.
    """
    def __init__(
            self, traj: ParticleTrajectory, drU: Callable, dr2U: Callable,
            skip: float = 0.0, nav: int = 10,
            ptype: int | None = None, **kwargs
            ) -> None:
        r'''
        Calculates the configurational temperature.

        For more details, see Refs. [1]_, [2]_ and [3]_.
        
        References
        ----------
        
        .. [1] L. Hecht, L. Caprini, H. Löwen, and B. Liebchen, 
           How to Define Temperature in Active Systems?, J. Chem. Phys. 161, 
           224904 (2024). https://doi.org/10.1063/5.0234370
           
        .. [2] S. Saw, L. Costigliola, and J. C. Dyre, Configurational Temperature 
           in Active Matter. I. Lines of Invariant Physics in the Phase Diagram of 
           the Ornstein-Uhlenbeck Model, Phys. Rev. E 107, 024609 (2023).
           https://doi.org/10.1103/PhysRevE.107.024609
        
        .. [3] S. Saw, L. Costigliola, and J. C. Dyre, Configurational Temperature 
           in Active Matter. II. Quantifying the Deviation from Thermal 
           Equilibrium, Phys. Rev. E 107, 024610 (2023).
           https://doi.org/10.1103/PhysRevE.107.024610

        Parameters
        ----------
        traj : ParticleTrajectory
            Trajectory object with simulation data.
        drU : function
            First derivative of the potential energy function of one particle.
            For example, one can use amep.utils.dr_wca.
        dr2U : function
            Second derivative of the potential energy function of one particle.
            For example, one can use amep.utils.dr2_wca.
        skip : float, optional
            Skip this fraction at the beginning of the trajectory.
            The default is 0.0.
        nav : int, optional
            Number of frames to consider for the time average.
            The default is 10.
        ptype : float, optional
            Particle type. The default is None.
        **kwargs
            Other keyword arguments are forwarded to
            `amep.thermo.Tkin`.

        '''
        super(Tconf, self).__init__()

        self.name = 'Tconf'
        
        self.__traj   = traj
        self.__skip   = skip
        self.__nav    = nav
        self.__ptype  = ptype
        self.__kwargs = kwargs
        self.__drU = drU
        self.__dr2U = dr2U
        
        self.__frames, self.__avg, self.__indices = average_func(
            self.__compute,
            self.__traj,
            skip = self.__skip,
            nr = self.__nav,
            indices = True
        )
        
        self.__times = self.__traj.times[self.__indices]

        
    def __compute(self, frame):
        r'''
        Calculation for a single frame.
        
        Parameters
        ----------
        frame : BaseFrame
            A single frame of particle-based simulation data.
              
        Returns
        -------
        temp: float
            Mean temperature.
        '''
        temp = thermo.Tconf(
            frame.coords(ptype=self.__ptype),
            frame.box,
            self.__drU,
            self.__dr2U,
            **self.__kwargs
        )
        return temp

    
    @property
    def frames(self):
        r'''
        Mean configurational temperature for each frame.

        Returns
        -------
        np.ndarray
            Function value for each frame.

        '''
        return self.__frames
    
    @property
    def times(self):
        r'''
        Times at which the mean configurational temperature is evaluated.

        Returns
        -------
        np.ndarray
            Times at which the function is evaluated.

        '''
        return self.__times
    
    @property
    def avg(self):
        r'''
        Time-averaged configurational temperature 
        (averaged over the given number of frames).

        Returns
        -------
        np.ndarray
            Time-averaged spatial velocity correlation function.

        '''
        return self.__avg
    
    @property
    def indices(self):
        r'''
        Indices of all frames for which the mean configurational temperature
        has been evaluated.

        Returns
        -------
        np.ndarray
            Frame indices.

        '''
        return self.__indices
