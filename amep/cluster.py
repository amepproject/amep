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
Cluster Analysis
================

.. module:: amep.cluster

The AMEP module :mod:`amep.cluster` provides methods for particle-based 
simulation data that (i) identify clusters and (ii) calculate various cluster 
properties. Each method can be used with and without periodic boundary 
conditions.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import numpy as np

from numpy import unique, ndarray
from .pbc import fold, find_pairs, distance_matrix, pbc_diff


# =============================================================================
# CLUSTER PROPERTIES
# =============================================================================
def geometric_center(
        coords: np.ndarray, box_boundary: np.ndarray,
        pbc: bool = True, clusters: list | None = None) -> np.ndarray:
    r"""
    Calculate the geometric center of the cluster containing the given points.

    Uses the minimum image convention if pbc is True.

    Notes
    -----
    To consider periodic boundary conditions, we here use the projection
    method as described in Ref. [1]_ generalized to three spatial dimensions.

    References
    ----------
    .. [1] Bai, L., & Breen, D. (2008). Calculating Center of Mass in an
       Unbounded 2D Environment. Journal of Graphics Tools, 13(4), 53–60.
       https://doi.org/10.1080/2151237X.2008.10129266

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates of all particles inside the cluster.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    clusters : list or None, optional
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. If None, the geometric center of the
        given coords is calculated. If not None, the geometric center of each 
        cluster in clusters is calculated, where coords must be the coordinates
        of the particles in clusters. The default is None.

    Returns
    -------
    np.ndarray of shape (3,) or (N,3)
        Geometric center of coords or of each cluster in clusters.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> coords = [
    ...     np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]),
    ...     np.array([[0,-1.5,0],[0,1,0]]),
    ...     np.array([[-3,0,0],[4,0,0]]),
    ...     np.array([[-3,0,0],[4,0,0]])
    ... ]
    >>> box_boundary = [
    ...     np.array([[-2,2],[-2,2],[-0.5,0.5]]),
    ...     np.array([[-2,2],[-2,2],[-0.5,0.5]]),
    ...     np.array([[-5,5],[-5,5],[-0.5,0.5]]),
    ...     np.array([[-5,5],[-5,5],[-0.5,0.5]])
    ... ]
    >>> gc = [
    ...     amep.cluster.geometric_center(
    ...         coords[0], box_boundary[0], pbc=True
    ...     ), # expected: [0, 0, 0]
    ...     amep.cluster.geometric_center(
    ...         coords[1], box_boundary[1], pbc=True
    ...     ), # expected: [0, 1.75, 0]
    ...     amep.cluster.geometric_center(
    ...         coords[2], box_boundary[2], pbc=False
    ...     ), # expected: [0.5, 0, 0]
    ...     amep.cluster.geometric_center(
    ...         coords[3], box_boundary[3], pbc=True
    ...     ) # expected: [-4.5, 0, 0]
    ... ]
    >>> titles = ['pbc=True', 'pbc=True', 'pbc=False', 'pbc=True']
    >>> print(gc)
    [array([0., 0., 0.]), array([0.  , 1.75, 0.  ]), array([0.5, 0. , 0. ]), array([-4.5,  0. ,  0. ])]


    >>> fig, axs = amep.plot.new(figsize=(4,4), nrows=2, ncols=2)
    >>> axs = axs.flatten()
    >>> for i in range(4):
    ...     axs[i].scatter(coords[i][:,0], coords[i][:,1], s=50, c='k')
    ...     axs[i].scatter(gc[i][0], gc[i][1], s=100, marker='x', color='red')
    ...     amep.plot.box(axs[i], box_boundary[i])
    ...     axs[i].set_title(titles[i])
    ...     axs[i].set_xlabel(r"$x$")
    ...     axs[i].set_ylabel(r"$y$")
    >>> fig.savefig('./figures/cluster/cluster-geometric_center_1.png')

    .. image:: /_static/images/cluster/cluster-geometric_center_1.png
      :width: 400
      :align: center

    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> clusters, idx = amep.cluster.identify(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> gmc = amep.cluster.geometric_center(
    ...     frame.coords(), frame.box, pbc=True, clusters=clusters
    ... )
    >>> imax = 5
    >>> fig, axs = amep.plot.new(figsize=(4,3.5))
    >>> colors = ["red", "orange", "yellow", "green", "blue", "purple"]
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords()[idx<=imax], frame.box, radius=0.5,
    ...     values=idx[idx<=imax], cmap=colors, vmin=-0.5, vmax=5.5
    ... )
    >>> amep.plot.particles(
    ...     axs, frame.coords()[idx>imax], frame.box, radius=0.5, color="gray"
    ... )
    >>> axs.scatter(
    ...     gmc[:imax+1,0], gmc[:imax+1,1], color="k", s=50, marker="x",
    ...     label="geometric center"
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label="cluster id"
    ... )
    >>> amep.plot.format_axis(cax, ticks=False)
    >>> axs.legend(frameon=True)
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/cluster/cluster-geometric_center_2.png")
    
    .. image:: /_static/images/cluster/cluster-geometric_center_2.png
      :width: 400
      :align: center
    
    """
    if clusters is not None:
        # calculate geometric center of each cluster in clusters
        gmcs = np.zeros((len(clusters), 3))
        for i,cluster in enumerate(clusters):
            gmcs[i] = geometric_center(
                coords[cluster], box_boundary, pbc=pbc
            )
        return gmcs
    else:
        # calculate geometric center of given coords
        if pbc:
            # get box length
            box = box_boundary[:,1] - box_boundary[:,0]
            
            # projection angles
            theta = 2*np.pi*coords/box
            
            # projection of coordinates
            projcoords = np.cos(theta)*box/2/np.pi
            
            # coordinates in new dimension
            eta = np.sin(theta)*box/2/np.pi
            
            # projected center
            projcenter = np.mean(projcoords, axis=0)
            etacenter  = np.mean(eta, axis=0)   
            
            # back projection
            thetaprime = np.arctan2(-etacenter, -projcenter) + np.pi
            
            return fold(thetaprime*box/2/np.pi, box_boundary)
        return np.mean(coords, axis=0)


def center_of_mass(
        coords: np.ndarray, box_boundary: np.ndarray, mass: np.ndarray,
        pbc: bool = True, clusters: list | None = None)-> np.ndarray:
    r"""
    Calculates the center of mass of the cluster containing the given points.
    Uses the minimum image convention if pbc is True.
    
    Notes
    -----
    To consider periodic boundary conditions, we here use the projection
    method (weighted by the masses) as described in Ref. [1]_ generalized to
    three spatial dimensions. Note that this method can give slightly
    inaccurate results when applied to only a few particles with distances to
    each other which are comparable to the box size (see also example below).
    
    References
    ----------
    .. [1] Bai, L., & Breen, D. (2008). Calculating Center of Mass in an 
       Unbounded 2D Environment. Journal of Graphics Tools, 13(4), 53–60. 
       https://doi.org/10.1080/2151237X.2008.10129266
    
    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates of all particles inside the cluster.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    mass : np.ndarray of shape (N,)
        Mass of all particles inside the cluster.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    clusters : list or None, optional
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. If None, the geometric center of the
        given coords is calculated. If not None, the geometric center of each 
        cluster in clusters is calculated, where coords must be the coordinates
        of the particles in clusters. The default is None.
        
    Returns
    -------
    np.ndarray of shape (3,)
        Center of mass of the cluster.
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> coords = [
    ...     np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]),
    ...     np.array([[0,-1.5,0],[0,1,0]]),
    ...     np.array([[-3,0,0],[4,0,0]]),
    ...     np.array([[-3,0,0],[4,0,0]])
    ... ]
    >>> box_boundary = [
    ...     np.array([[-2,2],[-2,2],[-0.5,0.5]]),
    ...     np.array([[-2,2],[-2,2],[-0.5,0.5]]),
    ...     np.array([[-5,5],[-5,5],[-0.5,0.5]]),
    ...     np.array([[-5,5],[-5,5],[-0.5,0.5]])
    ... ]
    >>> mass = [
    ...     np.array([2,1,1,2]),
    ...     np.array([1,2]),
    ...     np.array([1,2]),
    ...     np.array([1,2])
    ... ]
    >>> com = [
    ...     amep.cluster.center_of_mass(
    ...         coords[0], box_boundary[0], mass[0], pbc=True
    ...     ),
    ...     amep.cluster.center_of_mass(
    ...         coords[1], box_boundary[1], mass[1], pbc=True
    ...     ),
    ...     amep.cluster.center_of_mass(
    ...         coords[2], box_boundary[2], mass[2], pbc=False
    ...     ),
    ...     amep.cluster.center_of_mass(
    ...         coords[3], box_boundary[3], mass[3], pbc=True
    ...     )
    ... ]
    >>> titles = ['pbc=True', 'pbc=True', 'pbc=False', 'pbc=True']
    >>> print(com)
    [array([ 0.20483276, -0.20483276,  0.        ]), array([0.        , 1.31861167, 0.        ]), array([1.66666667, 0.        , 0.        ]), array([4.81540634, 0.        , 0.        ])]
    
    
    >>> fig, axs = amep.plot.new(figsize=(4,4), nrows=2, ncols=2)
    >>> axs = axs.flatten()
    >>> for i in range(4):
    ...     axs[i].scatter(coords[i][:,0], coords[i][:,1], s=50*mass[i]**2, c='k')
    ...     axs[i].scatter(com[i][0], com[i][1], s=100, marker='x', color='red')
    ...     amep.plot.box(axs[i], box_boundary[i])
    ...     axs[i].set_title(titles[i])
    ...     axs[i].set_xlabel(r"$x$")
    ...     axs[i].set_ylabel(r"$y$")
    >>> fig.savefig('./figures/cluster/cluster-center_of_mass_1.png')

    .. image:: /_static/images/cluster/cluster-center_of_mass_1.png
      :width: 400
      :align: center

    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> clusters, idx = amep.cluster.identify(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> com = amep.cluster.center_of_mass(
    ...     frame.coords(), frame.box, frame.mass(),
    ...     pbc=True, clusters=clusters
    ... )
    >>> imax = 5
    >>> fig, axs = amep.plot.new(figsize=(4,3.5))
    >>> colors = ["red", "orange", "yellow", "green", "blue", "purple"]
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords()[idx<=imax], frame.box, radius=0.5,
    ...     values=idx[idx<=imax], cmap=colors, vmin=-0.5, vmax=5.5
    ... )
    >>> amep.plot.particles(
    ...     axs, frame.coords()[idx>imax], frame.box, radius=0.5, color="gray"
    ... )
    >>> axs.scatter(
    ...     com[:imax+1,0], com[:imax+1,1], color="k", s=50,
    ...     marker="x", label="center of mass"
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label="cluster id"
    ... )
    >>> amep.plot.format_axis(cax, ticks=False)
    >>> axs.legend(frameon=True)
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig('./figures/cluster/cluster-center_of_mass_2.png')

    .. image:: /_static/images/cluster/cluster-center_of_mass_2.png
      :width: 400
      :align: center

    """
    if clusters is not None:
        # calculate center of mass of each cluster in clusters
        coms = np.zeros((len(clusters), 3))
        for i,cluster in enumerate(clusters):
            coms[i] = center_of_mass(
                coords[cluster], box_boundary, mass[cluster], pbc=pbc
            )
        return coms
    else:
        # calculate center of mass of given coords and masses
        if pbc:
            # get box length
            box = box_boundary[:,1] - box_boundary[:,0]
            
            # projection angles
            theta = 2*np.pi*coords/box
            
            # projection of coordinates
            projcoords = np.cos(theta)*box/2/np.pi
            
            # coordinates in new dimension
            eta = np.sin(theta)*box/2/np.pi
            
            # projected center (weighted by the mass of the particles)
            projcenter = np.sum(mass[:,None]*projcoords, axis=0)/np.sum(mass)
            etacenter  = np.sum(mass[:,None]*eta, axis=0)/np.sum(mass)   
            
            # back projection
            thetaprime = np.arctan2(-etacenter, -projcenter) + np.pi
            
            return fold(thetaprime*box/2/np.pi, box_boundary)
        return np.sum(mass[:,None]*coords, axis=0)/np.sum(mass)


def radius_of_gyration(
        coords: np.ndarray, box_boundary: np.ndarray, mass: np.ndarray,
        pbc: bool = True, clusters: list | None = None) -> float:
    r"""
    Calculates the radius of gyration of the cluster containing the given coordinates.
    Uses the minimum image convention if pbc is True.
    
    Notes:
    ------
    For a cluster composed of :math:`n` particles of masses :math:`m_i, i=1,2, \ldots, n`, 
    located at fixed distances :math:`s_i` from the centre of mass, the radius of gyration is 
    the square-root of the mass average of :math:`s_i^2` over all mass elements, i.e.,
    :math:`R_g=\left(\sum_{i=1}^n m_i s_i^2 / \sum_{i=1}^n m_i\right)^{1 / 2}`.
    
    Parameters:
    -----------
    coords : np.ndarray of shape (N,3)
        Coordinates of all particles inside the cluster.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    mass : np.ndarray of shape (N,)
        Mass of all particles inside the cluster.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    clusters : list or None, optional
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. If None, the geometric center of the
        given coords is calculated. If not None, the geometric center of each 
        cluster in clusters is calculated, where coords must be the coordinates
        of the particles in clusters. The default is None.
        
    Returns
    -------
    float or np.ndarray of shape (N,)
        Radius of gyration of the given points or each cluster in clusters.
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[25]
    >>> clusters, idx = amep.cluster.identify(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> coords = frame.coords()[idx==1]
    >>> mass = frame.mass()[idx==1]
    >>> Rg = amep.cluster.radius_of_gyration(
    ...     coords, frame.box, mass, pbc=True
    ... )
    >>> print(Rg)
    8.16761958224252


    >>> rgs = amep.cluster.radius_of_gyration(
    ...     frame.coords(), frame.box, frame.mass(),
    ...     pbc=True, clusters=clusters
    ... )
    >>> centers = amep.cluster.geometric_center(
    ...     frame.coords(), frame.box, clusters=clusters
    ... )
    >>> i = 1
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.particles(
    ...     axs, frame.coords()[idx==i], frame.box, color="orange",
    ...     radius=frame.radius()[idx==i]
    ... )
    >>> amep.plot.particles(
    ...     axs, frame.coords()[idx!=i], frame.box, color="gray",
    ...     radius=frame.radius()[idx!=i]
    ... )
    >>> axs.scatter(
    ...     centers[i,0], centers[i,1], color="k", s=50, marker='x',
    ...     label="geometric center"
    ... )
    >>> angle = np.linspace(0, 2*np.pi, 150)
    >>> x = rgs[i]*np.cos(angle)+centers[i,0]
    >>> y = rgs[i]*np.sin(angle)+centers[i,1]
    >>> axs.plot(
    ...     x, y, c="k", ls="--", lw=2, marker="", label="radius of gyration"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> axs.legend(frameon=True)
    >>> fig.savefig("./figures/cluster/cluster-radius_of_gyration.png")

    .. image:: /_static/images/cluster/cluster-radius_of_gyration.png
      :width: 400
      :align: center

    """
    if clusters is not None:
        # calculate radius of gyration of each cluster in clusters
        rgs = np.zeros(len(clusters))
        for i,cluster in enumerate(clusters):
            rgs[i] = radius_of_gyration(
                coords[cluster], box_boundary, mass[cluster], pbc=pbc
            )
        return rgs
    else:
        # get length of the simulation box
        boxlength = box_boundary[:,1] - box_boundary[:,0]
        
        # get center of mass of the given set of points
        com = center_of_mass(coords, box_boundary, mass, pbc=pbc)
        
        # get distances to the center of mass
        dists = distance_matrix(
            coords,
            box_boundary,
            other=np.array(com).reshape(-1,3),
            pbc=pbc,
            maxdist=np.max(boxlength)
        )
        return np.sqrt(np.sum(mass[:,None]*dists**2)/np.sum(mass[:,None]))


def linear_extension(
        coords: np.ndarray, box_boundary: np.ndarray, mass: np.ndarray,
        pbc: bool = True, clusters: list | None = None) -> float:
    r"""
    Calculate the linear extension (end-to-end distance) of one cluster.

    Takes the coordinates of each particle in the cluster.
    Uses the minimum image convention if pbc is True.

    Notes:
    ------
    The linear extension is defined as the maximal distance between two
    particles in the same cluster [1]_. This is an O(N²) algorithm. An O(N)
    implementation (which is implemented in this function) is to calculate
    twice the in-cluster maximum distance between a cluster particle and the
    cluster center of mass [2]_.

    References:
    -----------
    .. [1] Levis, Demian, and Ludovic Berthier. "Clustering and heterogeneous
        dynamics in a kinetic Monte Carlo model of self-propelled hard disks."
        Physical Review E 89.6 (2014): 062301.
        https://journals.aps.org/pre/abstract/10.1103/PhysRevE.89.062301

    .. [2] Kyriakopoulos, Nikos, Hugues Chaté, and Francesco Ginelli.
        "Clustering and anisotropic correlated percolation in polar flocks."
        Physical Review E 100.2 (2019): 022606.
        https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.022606


    Parameters:
    -----------
    coords : np.ndarray of shape (N,3)
        Coordinates of all particles inside the cluster.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    center : np.ndarray of shape (3,)
        Center of the simulation box.
    mass : np.ndarray of shape (N,)
        Mass of all particles inside the cluster.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    clusters : list or None, optional
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. If None, the geometric center of the
        given coords is calculated. If not None, the geometric center of each 
        cluster in clusters is calculated, where coords must be the coordinates
        of the particles in clusters. The default is None.

    Returns
    -------
    float or np.ndarray of shape (N,)
        Linear extension of the cluster.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[25]
    >>> clusters, idx = amep.cluster.identify(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> coords = frame.coords()[idx==1]
    >>> mass = frame.mass()[idx==1]
    >>> Le = amep.cluster.linear_extension(
    ...     coords, frame.box, mass, pbc=True
    ... )
    >>> print(Le)
    30.3604296244119
    
    
    >>> les = amep.cluster.linear_extension(
    ...     frame.coords(), frame.box, frame.mass(),
    ...     pbc=True, clusters=clusters
    ... )
    >>> centers = amep.cluster.geometric_center(
    ...     frame.coords(), frame.box, clusters=clusters
    ... )
    >>> i = 1
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.particles(
    ...     axs, frame.coords()[idx==i], frame.box, color="orange",
    ...     radius=frame.radius()[idx==i]
    ... )
    >>> amep.plot.particles(
    ...     axs, frame.coords()[idx!=i], frame.box, color="gray",
    ...     radius=frame.radius()[idx!=i]
    ... )
    >>> axs.scatter(
    ...     centers[i,0], centers[i,1], color="k", s=50, marker='x',
    ...     label="geometric center"
    ... )
    >>> angle = np.linspace(0, 2*np.pi, 150)
    >>> x = les[i]/2*np.cos(angle)+centers[i,0]
    >>> y = les[i]/2*np.sin(angle)+centers[i,1]
    >>> axs.plot(
    ...     x, y, c="k", ls="--", lw=2, marker="", label="linear extension"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> axs.legend(frameon=True)
    >>> fig.savefig("./figures/cluster/cluster-linear_extension.png")

    .. image:: /_static/images/cluster/cluster-linear_extension.png
      :width: 400
      :align: center

    """
    if clusters is not None:
        # calculate linear extension of each cluster in clusters
        les = np.zeros(len(clusters))
        for i,cluster in enumerate(clusters):
            les[i] = linear_extension(
                coords[cluster], box_boundary, mass[cluster], pbc=pbc
            )
        return les
    else:
        # get length of the simulation box
        boxlength = box_boundary[:,1] - box_boundary[:,0]
        
        # get center of mass of the given set of points
        com = center_of_mass(coords, box_boundary, mass, pbc=pbc)
        
        # get distances to the center of mass
        dists = distance_matrix(
            coords,
            box_boundary,
            other=np.array(com).reshape(-1,3),
            pbc=pbc,
            maxdist=np.max(boxlength)
        )
        return 2.0*np.max(dists)


def gyration_tensor(
        coords: np.ndarray, box_boundary: np.ndarray, mass: np.ndarray,
        pbc: bool = True, clusters: list | None = None) -> np.ndarray:
    r"""
    Calculate the gyration tensor of one cluster.

    Uses th cluster containing the given coordinates.
    Uses the minimum image convention if pbc is True.

    Notes:
    ------
    The gyration tensor is a tensor that describes the second moments of
    position of a collection of particles normalized by particle number.
    The formula used here is taken from Ref. [1]_ .

    References:
    -----------
    .. [1] Arkin, Handan, and Wolfhard Janke. "Gyration tensor based analysis
        of the shapes of polymer chains in an attractive spherical cage."
        The Journal of chemical physics 138.5 (2013).
        https://doi.org/10.1063/1.4788616

    Parameters:
    -----------
    coords : np.ndarray of shape (N,3)
        Coordinates of all particles inside the cluster.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    mass : np.ndarray of shape (N,)
        Mass of all particles inside the cluster.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    clusters : list or None, optional
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. If None, the geometric center of the
        given coords is calculated. If not None, the geometric center of each 
        cluster in clusters is calculated, where coords must be the coordinates
        of the particles in clusters. The default is None.
        
    Returns
    -------
    np.ndarray of shape (3,3) or (N,3,3)
        Gyration tensor of the given coords or each cluster in clusters.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> clusters, idx = amep.cluster.identify(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> i = 1
    >>> coords = frame.coords()[idx==i]
    >>> mass = frame.mass()[idx==i]
    >>> gt = amep.cluster.gyration_tensor(
    ...     coords, frame.box, mass, pbc=True
    ... )
    >>> print(gt)
    [[2298.3718 1562.3932    0.    ]
     [1562.3932 3079.9683    0.    ]
     [   0.        0.        0.    ]]
    >>> gts = amep.cluster.gyration_tensor(
    ...     frame.coords(), frame.box, frame.mass(),
    ...     pbc=True, clusters=clusters
    ... )
    >>> print(gts.shape)
    (52, 3, 3)
    >>> 
    
    """
    if clusters is not None:
        # calculate gyration tensor of each cluster in clusters
        gyrtensors = np.zeros((len(clusters),3,3))
        for i,cluster in enumerate(clusters):
            gyrtensors[i] = gyration_tensor(
                coords[cluster], box_boundary, mass[cluster], pbc=pbc
            )
        return gyrtensors
    else:
        # get center of mass of the given set of points
        com = center_of_mass(coords, box_boundary, mass, pbc=pbc)
        
        # get the difference vectors to the center of mass
        diff = pbc_diff(coords, com, box_boundary, pbc=pbc)

        return np.dot(diff.T, diff) / diff.shape[0]


def inertia_tensor(
        coords: np.ndarray, box_boundary: np.ndarray, mass: np.ndarray,
        pbc: bool = True, clusters: list | None = None) -> np.ndarray:
    r"""
    Calculates the moment of inertia tensor of the cluster containing the given
    coordinates. Uses the minimum image convention if pbc is True.
    
    Parameters:
    -----------
    coords : np.ndarray of shape (N,3)
        Coordinates of all particles inside the cluster.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    mass : np.ndarray of shape (N,)
        Mass of all particles inside the cluster.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    clusters : list or None, optional
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. If None, the geometric center of the
        given coords is calculated. If not None, the geometric center of each 
        cluster in clusters is calculated, where coords must be the coordinates
        of the particles in clusters. The default is None.
        
    Returns
    -------
    np.ndarray of shape (3,3) or (N,3,3)
        Moment of inertia tensor of the given set of points or each cluster in
        clusters.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> clusters, idx = amep.cluster.identify(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> i = 1
    >>> coords = frame.coords()[idx==i]
    >>> mass = frame.mass()[idx==i]
    >>> it = amep.cluster.inertia_tensor(
    ...     coords, frame.box, mass, pbc=True
    ... )
    >>> print(it)
    [[ 24639.74804688 -12499.14453125      0.        ]
     [-12499.14453125  18386.97460938      0.        ]
     [     0.              0.          43026.7265625 ]]
    >>> its = amep.cluster.inertia_tensor(
    ...     frame.coords(), frame.box, frame.mass(),
    ...     pbc=True, clusters=clusters
    ... )
    >>> print(its.shape)
    (52, 3, 3)
    >>> 
    """
    if clusters is not None:
        # calculate inertia tensor of each cluster in clusters
        intensors = np.zeros((len(clusters),3,3))
        for i,cluster in enumerate(clusters):
            intensors[i] = inertia_tensor(
                coords[cluster], box_boundary, mass[cluster], pbc=pbc
            )
        return intensors
    else:
        # get center of mass of the given set of points
        com = center_of_mass(coords, box_boundary, mass, pbc=pbc)
        
        # get the difference vectors to the center of mass
        diff = pbc_diff(coords, com, box_boundary, pbc=pbc)
        
        # initialize inertia tensor
        I_k = np.zeros((diff.shape[1], diff.shape[1]))

        # calculate components
        I_k[0, 0] = np.sum(mass * (diff[:, 1]**2 + diff[:, 2]**2))
        I_k[1, 1] = np.sum(mass * (diff[:, 0]**2 + diff[:, 2]**2))
        I_k[2, 2] = np.sum(mass * (diff[:, 0]**2 + diff[:, 1]**2))
    
        I_k[0, 1] = I_k[1, 0] = np.sum(-mass * diff[:, 0] * diff[:, 1])
        I_k[0, 2] = I_k[2, 0] = np.sum(-mass * diff[:, 0] * diff[:, 2])
        I_k[1, 2] = I_k[2, 1] = np.sum(-mass * diff[:, 1] * diff[:, 2])
        
        return I_k


def sizes(clusters: list) -> np.ndarray:
    r"""
    Calculates the sizes of the given clusters.

    Parameters:
    -----------
    clusters : list
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. If None, the geometric center of the
        given coords is calculated. If not None, the geometric center of each 
        cluster in clusters is calculated, where coords must be the coordinates
        of the particles in clusters. The default is None.
        
    Returns
    -------
    np.ndarray
        Sizes of the clusters.
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> coords = np.array([[1,0,0], [4,0,0], [0.5,0,0], [4.5,0,0]])
    >>> box = np.array([[-5,5],[-5,5],[-0.5,0.5]])
    >>> clusters, idx = amep.cluster.identify(
    ...     coords, box, pbc=True, rmax=0.5
    ... )
    >>> sizes = amep.cluster.sizes(clusters)
    >>> print(sizes)
    [2 2]
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> clusters, idx = amep.cluster.identify(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> sizes = amep.cluster.sizes(clusters)
    >>> print(sizes[:10])
    [3611    8    7    6    6    5    5    5    4    4]
    >>> 

    """
    size_array = np.array([len(item) for item in clusters])
    
    return size_array

def masses(clusters: list, mass: np.ndarray) -> np.ndarray:
    r"""
    Calculates the masses of the given clusters.

    Parameters:
    -----------
    clusters : list
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. If None, the geometric center of the
        given coords is calculated. If not None, the geometric center of each 
        cluster in clusters is calculated, where coords must be the coordinates
        of the particles in clusters. The default is None.
    mass : np.ndarray of shape (N,)
        Mass of all N particles in the system, ordered by particle index.    
        
    Returns
    -------
    np.ndarray
        Masses of the clusters.
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> coords = np.array([[1,0,0], [4,0,0], [0.5,0,0], [4.5,0,0]])
    >>> box = np.array([[-5,5],[-5,5],[-0.5,0.5]])
    >>> mass = np.array([20, 1, 2, 5])
    >>> clusters, idx = amep.cluster.identify(
    ...     coords, box, pbc=True, rmax=0.5
    ... )
    >>> masses = amep.cluster.masses(clusters, mass)
    >>> print(masses)
    [22  6]
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> clusters, idx = amep.cluster.identify(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> masses = amep.cluster.masses(clusters, frame.mass())
    >>> print(masses[:10])
    [3611.    8.    7.    6.    6.    5.    5.    5.    4.    4.]
    >>> 

    """
    mass_array = np.array([np.sum(mass[item]) for item in clusters])
    
    return mass_array

# =============================================================================
# CLUSTER METHODS
# =============================================================================
def identify(
        coords: ndarray, box_boundary: ndarray,
        sizes: np.ndarray | None = None, pbc: bool = True,
        rmax: float = 1.122) -> tuple[list, np.ndarray]:
    r"""
    Identify clusters from particle coordinates, and respective sizes.

    Return an array of particle pairs and the corresponding distances between
    the particles and identifies which particles belong to the same cluster.

    Parameters
    ----------
    coords : np.ndarray
        Particle coordinates as array of shape (N,3).
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    sizes : numpy.ndarray|None, optional
        Array of size (N,) containing the size of each particle
        (assuming that the particles are spherical).
        If None, rmax is used for finding pairs. If
        given, rmax scales the contact distances. The default is None.
    pbc : bool, optional
        If True, periodic boundary conditions will be considered when
        calculating the pairwise distances. The default is True.
    rmax : float, optional
        Maximum distance. If the distance between two particles is smaller
        than rmax, they are considered as belonging to the same cluster.
        The default is 1.122.

    Returns
    -------
    sorted_clusters : list
        List of lists, where each list contains the indices of the particles
        that belong to the same cluster. The list is sorted by the number of
        particles in each cluster. Single particles are also included as
        clusters in this list.
    idx : numpy.ndarray
        Array of shape (N,) containing the cluster ID for each particle. N is 
        the total number of particles.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> coords = np.array([[1,0,0], [4,0,0], [-2,0,0], [4.5,0,0]])
    >>> box = np.array([[-5,5],[-5,5],[-0.5,0.5]])
    >>> clusters, idx = amep.cluster.identify(
    ...     coords, box, pbc=True, rmax=3
    ... )
    >>> print(clusters)
    [[0, 1, 2, 3]]
    >>> print(idx)
    [0. 0. 0. 0.]
    >>> clusters, idx = amep.cluster.identify(
    ...     coords, box, pbc=True, rmax=2
    ... )
    >>> print(clusters)
    [[1, 3]]
    >>> print(idx)
    [1. 0. 2. 0.]
    >>>

    """
    # extracting relevant information, i.e, particles belong to the same
    # cluster, if their distance is smaller than rmax; here we generate an
    # array of shape (N,2) that contains particle pairs, where N is the number
    # of pairs, the first column contains the index of the first particle i,
    # the second column of the second particle j, where particle i and j are
    # closer than rmax to each other
    pairs = find_pairs(
        coords, box_boundary, pbc=pbc, rmax=rmax, sizes=sizes
    )
    # results list
    clusters = []

    # use a set to store the values of `a` and `b` which have been processed
    # already - this makes it faster to check if a value is already in the set
    done = set()

    # find connections between particles (`a` and `b` are particle indices)
    # use a dictionary to store the connections between `a` and `b` -
    # this makes it faster to find the connections for given `a` or `b`
    connections = {}
    for a, b in pairs:
        if a not in connections:
            connections[a] = set()
        if b not in connections:
            connections[b] = set()
        connections[a].add(b)
        connections[b].add(a)

    # iterate over the unique values of `a`
    for a in unique(pairs[:,0]):

        if a in done:
            # skip `a` if it has already been processed
            continue

        # initialize a new cluster
        clr = []

        # use a stack to keep track of the nodes to visit
        # start with the current node `a`
        stack = {a}
        while stack:

            # pop the last node from the stack
            node = int(stack.pop())

            # add the node to the current cluster
            clr.append(node)

            # add the node to the list of processed nodes
            done.add(node)

            # get the connections for the current node
            con = connections.get(node, set())

            # add the unprocessed connections to the stack
            stack.update(con - done)

        # append the current cluster to the list of clusters
        clusters.append(clr)
        
    sorted_clusters = [item for item in sorted(clusters, reverse=True, key=len)]    
    # generate list of cluster IDs for each parrticles
    idx     = np.zeros(len(coords))

    particle_idx  = set(np.arange(len(coords)))
    clustered_idx = set()

    cluster_id = 0
    for cl in sorted_clusters:
        for i in cl:
            idx[i] = cluster_id
            clustered_idx.add(i)
        cluster_id += 1

    # add leftover single particles
    for k in particle_idx - clustered_idx:
        idx[k] = cluster_id
        sorted_clusters.append([k])
        cluster_id += 1
    idx=np.array(idx, dtype=int)

    return sorted_clusters, idx
