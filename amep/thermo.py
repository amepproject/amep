# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) 2023-2024 Lukas Hecht and the AMEP development team.
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
Thermodynamic Observables
=========================

.. module:: amep.thermo

The AMEP module :mod:`amep.thermo` contains thermodynamic functions and 
observables.
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import numpy as np

# =============================================================================
# TEMPERATURE
# =============================================================================
def kintemp(
        v: np.ndarray, mass: float | None | np.ndarray = None) -> np.ndarray:
    r'''
    Calculates the kinetic temperature per particle.

    Notes
    -----
    The kinetic temperature is defined as the kinetic energy. In the
    overdamped regime (i.e. mass=0),
    it is simply :math:`\langle v^2\rangle` (see Ref. [1]_).

    References
    ----------

    .. [1] Caprini, L., & Marini Bettolo Marconi, U. (2020). Active matter at
       high density: Velocity distribution and kinetic temperature.
       The Journal of Chemical Physics, 153(18), 184901.
       https://doi.org/10.1063/5.0029710

    Parameters
    ----------
    v : np.ndarray
        Velocity array of shape (N,3).
    mass : float or np.ndarray or None, optional
        Particle mass(es). The default is None.

    Returns
    -------
    temp : np.ndarray
        Kinetic temperature of each particle. Same length as v.

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> tkin = amep.thermo.kintemp(
    ...     frame.velocities(), mass=frame.mass()
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius(),
    ...     values=tkin, cscale="log"
    ... )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r"$T_{\rm kin}$"
    ... )
    >>> fig.savefig('./figures/thermo/thermo-kintemp.png')
    >>>
    
    .. image:: /_static/images/thermo/thermo-kintemp.png
      :width: 400
      :align: center

    '''
    if mass is None:
        mass = 2.0

    if len(v.shape)==2:
        # multiple particles
        vmean = np.mean(v, axis=0)
        temp = 0.5 * mass * np.sum((v-vmean)**2, axis=1)
    elif len(v.shape)==1:
        # single particle
        vmean = 0
        temp = 0.5 * mass * np.sum((v-vmean)**2)
    else:
        raise RuntimeError('kintemp: v has the wrong shape. Only shape (N,3) or (3,) is allowed.')

    return temp


# =============================================================================
# Kinetic energy
# =============================================================================
def total_kinetic_energy(frame, mass, inertia, mode="total"):
    r'''
    Calculates the kinetic energies per particle.

    Parameters
    ----------
    frame : amep trajectory frame (one time step)
        One time step of the simulation data as amep frame.
    mass : float, optional
        Particle mass.
    inertia : float, optional
        Particle moment of inertia.
    mode : string, optional
        Mode of the calculation.
            total : sum over all particles
            particle : individual particles

    Returns
    -------
    temp : np.ndarray
        Kinetic energy
            mode = total : float, total kinetic energy of the system
            mode = particle : np.ndarray of length N (particle number)

    '''
    return translational_kinetic_energy(frame, mass=mass, mode=mode)\
           + rotational_kinetic_energy(frame, inertia=inertia, mode=mode)


# =============================================================================
# Translational kinetic energy
# =============================================================================
def translational_kinetic_energy(
        frame,
        mass: float | np.ndarray,
        mode = "total"):
    r'''
    Calculates the translational kinetic energies per particle.

    Parameters
    ----------
    frame : amep trajectory frame (one time step)
        One time step of the simulation data as amep frame.
    mass : float | np.ndarray
        Particle mass.
    mode : string, optional
        Mode of the calculation.
            total : sum over all particles
            particle : individual particles

    Returns
    -------
    temp : np.ndarray
        Kinetic energy
            mode = total : float, total kinetic energy of the system
            mode = particle : np.ndarray of length N (particle number)

    '''
    v=frame.velocities()
    if mode=="total":
        return np.sum(np.sum(.5*mass*v**2, axis=1), axis=0)
    elif mode=="particle":
        return np.sum(.5*mass*v**2, axis=1)
    else:
        Exception(
            f"amep.thermo.translational_kinetic_energy: mode {mode} does not "\
            "exist. Choose 'total' or 'particle'."
        )

# =============================================================================
# Rotational kinetic energy
# =============================================================================
def rotational_kinetic_energy(frame, inertia, mode="total"):
    r'''
    Calculates the rotational kinetic energies per particle.

    Parameters
    ----------
    frame : amep trajectory frame (one time step)
        One time step of the simulation data as amep frame.
    inertia : float, optional
        Particle moment of inertia.
    mode : string, optional
        Mode of the calculation.
            total : sum over all particles
            particle : individual particles

    Returns
    -------
    temp : np.ndarray
        Kinetic energy
            mode = total : float, total kinetic energy of the system
            mode = particle : np.ndarray of length N (particle number)

    '''    
    omegas=frame.omegas()
    if mode=="total":
        return np.sum(np.sum(.5*inertia*omegas**2, axis=1), axis=0)
    elif mode=="particle":
        return np.sum(.5*inertia*omegas**2, axis=1)
    else:
        Exception(
            f"amep.thermo.rotational_kinetic_energy: mode {mode} does not "\
            "exist. Choose 'total' or 'particle'."
        )
