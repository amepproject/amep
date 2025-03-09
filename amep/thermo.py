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

from .pbc import distance_matrix, pbc_diff
# =============================================================================
# KINETIC TEMPERATURE
# =============================================================================
def kintemp(
        v: np.ndarray, m: float | None | np.ndarray = None,
        d: int = 2) -> np.ndarray:
    r'''
    Calculates the kinetic temperature per particle based on the 2nd
    moment of the velocity distribution as described in Ref. [1]_.

    Notes
    -----
    The kinetic temperature is defined as the kinetic energy. In the
    overdamped regime (i.e. mass=0),
    it is simply :math:`\langle v^2\rangle` (see Ref. [2]_).

    References
    ----------

    .. [1] L. Hecht, L. Caprini, H. Löwen, and B. Liebchen, 
       How to Define Temperature in Active Systems?, J. Chem. Phys. 161, 
       224904 (2024). https://doi.org/10.1063/5.0234370

    .. [2] Caprini, L., & Marini Bettolo Marconi, U. (2020). Active matter at
       high density: Velocity distribution and kinetic temperature.
       The Journal of Chemical Physics, 153(18), 184901.
       https://doi.org/10.1063/5.0029710

    Parameters
    ----------
    v : np.ndarray
        Velocity array of shape (N,3).
    m : float or np.ndarray or None, optional
        Particle mass(es). The default is None.
    d : int, optional
        Spatial dimension. The default is 2.

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
    ...     frame.velocities(), m=frame.mass()
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
    if m is None:
        # use version for overdamped particles
        m = 2.0
    if len(v.shape)==2:
        # multiple particles
        vmean = np.mean(v, axis=0)
        temp = m * np.sum((v-vmean)**2, axis=1) / d
    elif len(v.shape)==1:
        # single particle
        vmean = 0
        temp = m * np.sum((v-vmean)**2) / d
    else:
        raise RuntimeError('kintemp: v has the wrong shape. Only shape (N,3) or (3,) is allowed.')
    return temp


def Tkin(v: np.ndarray, m: float | np.ndarray, d: int = 2):
    '''
    Kinetic temperature based on the second moment of the velocity distribution
    and averaged over all particles. [1]_
    
    References
    ----------
    
    .. [1] L. Hecht, L. Caprini, H. Löwen, and B. Liebchen, 
       How to Define Temperature in Active Systems?, J. Chem. Phys. 161, 
       224904 (2024). https://doi.org/10.1063/5.0234370
    
    Parameters
    ----------
    v : np.ndarray of shape (N,3)
        Velocities.
    m : float | np.ndarray
        Mass(es).
    d : int, optional
        Spatial dimension. The default is 2.
        
    Returns
    -------
    float
        Mean kinetic temperature.
    '''
    vmean = np.mean(v, axis=0)
    temp = m * np.sum((v-vmean)**2, axis=1) / d
    return temp.mean()

def Tkin4(v: np.ndarray, m: float | np.ndarray, d: int = 2):   
    '''
    Kinetic temperature based on the 4th moment of the velocity distribution
    and averaged over all particles. [1]_
    
    References
    ----------
    
    .. [1] L. Hecht, L. Caprini, H. Löwen, and B. Liebchen, 
       How to Define Temperature in Active Systems?, J. Chem. Phys. 161, 
       224904 (2024). https://doi.org/10.1063/5.0234370
    
    Parameters
    ----------
    v : np.ndarray of shape (N,3)
        Velocities.
    m : float | np.ndarray
        Mass(es).
    d : int, optional
        Spatial dimension. The default is 2.
        
    Returns
    -------
    float
        Mean kinetic temperature.
    '''
    vmean = np.mean(v, axis=0)
    v4 = np.sum((v-vmean[None,:])**2, axis=1)**2
    return np.mean(0.5 * m * np.sqrt(4 * v4 / (d*(d+2))))

# =============================================================================
# CONFIGURATIONAL TEMPERATURE
# =============================================================================
def Tconf(
        coords: np.ndarray, box_boundary: np.ndarray, drU, dr2U, d = 2,
        rcut: float = 1.122, pbc: bool = True):
    """
    Calculates the configurational temperature averaged over all particles.

    Also see Refs. [1]_, [2]_ and [3]_.
    
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
    coords : np.ndarray of shape (N,3)
        Particle coordinates.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    drU : function
        First derivative of the potential energy function of one particle.
    dr2U : function
        Second derivative of the potential energy function of one particle.
    d : int, optional
        Spatial dimension. The current version only works for d=2 and d=3. 
        The default is 2.
    rcut : float, optional
        Cutoff radius of the interaction potential. The default is 1.122.
    pbc : bool, optional
        If True, periodic boundary conditions are applied. The default is True.
        
    Returns
    -------
    temp : float
        Average configurational temperature.
    nom : float
        Nominator.
    denom : float
        Denominator.
    """
    # calculate distances
    dmatrix = distance_matrix(
        coords, box_boundary,
        maxdist = rcut, pbc = pbc
    )
    
    # get only relevant distances within the cutoff
    mask = np.where(dmatrix!=0)
    distances = dmatrix[mask]
    #print(distances.shape)
    
    # pairwise difference vectors for relevant pairs
    diff = pbc_diff(
        coords[mask[0]], coords[mask[1]], box_boundary, pbc = pbc
    )
    
    # calculate force contribution for each particle
    nom = 0.0
    idx_old = 0
    force = 0.0
    for i, idx in enumerate(mask[0]):
        
        if idx != idx_old:
            nom += np.sum(force**2)
            force = 0.0
            
        force += (drU(distances[i])/distances[i])*diff[i]
        idx_old = idx
        
    # calculate denominator
    if d==2:
        denom = np.sum(dr2U(distances) + drU(distances)/distances)
    elif d==3:
        denom = np.sum(dr2U(distances) + 2*drU(distances)/distances)
    else:
        raise ValueError("Invalid value for d. Use 2 or 3.")
    
    return nom/denom, nom, denom

    
# =====================================================================
# OSCILLATOR TEMPERATURE
# =====================================================================
def Tosc(coords: np.ndarray, k: float):
    """
    Oscillator temperature averaged over all particles. [1]_
    
    References
    ----------
    
    .. [1] L. Hecht, L. Caprini, H. Löwen, and B. Liebchen, 
       How to Define Temperature in Active Systems?, J. Chem. Phys. 161, 
       224904 (2024). https://doi.org/10.1063/5.0234370
    
    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Particle coordinates.
    k : float
        Strength of the harmonic potential.
        
    Returns
    -------
    float
        Mean oscillator temperature.
    """
    return 0.5 * k * np.sum(coords**2, axis=1).mean()


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
