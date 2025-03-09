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
Periodic Boundary Conditions
============================

.. module:: amep.pbc

The AMEP module :mod:`amep.pbc` contains methods that apply periodic boundary
condictions and calculate distances and difference vectors considering periodic
boundaries.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist
from warnings import warn
from .utils import dimension, compute_parallel
from .base import get_module_logger

# logger setup
_log = get_module_logger(__name__)

# =============================================================================
# INTERNAL UTILITIES
# =============================================================================
def __convert_pair_list(pair_list: list) -> np.ndarray:
    r'''
    Converts a pair list into an array of pairs.

    Parameters
    ----------
    pair_list : list
        List of list (list 1 contains particle indices of neighbors of particle
        1, list 2 of particle 2, ...).

    Returns
    -------
    np.ndarray
        Array of pairs (particle indices) of shape (N_pairs,2).

    '''
    pairs = []
    for i,p in enumerate(pair_list):
        for j in p:
            pairs.append([i,j])
    return np.asarray(pairs)

def __create_combinations(l: list) -> np.ndarray:
    r'''
    Returns an array of all combinations of two elements in l.

    Parameters
    ----------
    l : list
        List of values.

    Returns
    -------
    np.ndarray
        All possible pairs as array of shape (N_pairs,2).

    '''
    c = []
    for i in l:
        for j in l:
            if j>=i:
                c.append([i,j])
    return np.asarray(c)

# =============================================================================
# APPLY PERIODIC BOUNDARY CONDITIONS TO COORDINATES
# =============================================================================
def fold(coords, box_boundary):
    r'''
    Applies periodic boundary conditions to the given coordinates, i.e.
    folds the coordinates back into the box if they are not inside the box.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame of shape (N,3).
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.

    Returns
    -------
    base : np.ndarray
        Folded coordinate frame.
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> box_boundary = np.array([[-5,5],[-5,5],[-1,1]])
    >>> unwrapped = np.array([[9,0,0], [-8,-3,0],[1,6,0],[0,-13,0]])
    >>> center = np.zeros(3)
    >>> fold = amep.pbc.fold(unwrapped, box_boundary)
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.box(axs, box_boundary, color='k', ls='--')
    >>> colors = ['blue', 'green', 'gray', 'orange']
    >>> for i,c in enumerate(colors):
    ...     axs.scatter(
    ...         unwrapped[i,0], unwrapped[i,1],
    ...         s=50, facecolors='none', edgecolors=c
    ...     )
    ...     axs.scatter(
    ...         fold[i,0], fold[i,1], s=50, facecolor=c
    ...     )
    >>> axs.scatter(
    ...     -100, -100, s=50, facecolors='none',
    ...     edgecolors='k', label='unwrapped'
    ... )
    >>> axs.scatter(
    ...     -100, -100, s=50, facecolors='k',
    ...     edgecolors='k', label='fold back'
    ... )
    >>> axs.set_xlim(-10,10)
    >>> axs.set_ylim(-15,10)
    >>> axs.legend()
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/pbc/pbc-fold.png')
    >>> 
    
    .. image:: /_static/images/pbc/pbc-fold.png
      :width: 400
      :align: center

    '''
    # get the length of the simulation box
    box = box_boundary[:,1] - box_boundary[:,0]
    
    # get center of the simulation box
    center = np.mean(box_boundary, axis=1)
    
    # fold coordinates back into the box
    s = (coords-center) / box
    base = box * (s - s.round()) + center
    
    return base


# =============================================================================
# CALCULATION OF PERIODIC IMAGES
# =============================================================================
def pbc_points(
        coords: np.ndarray, box_boundary: np.ndarray,
        enforce_nd: int | None = None, thickness: float | None = None,
        width: float | None = None, index: bool = False,
        inclusive: bool = True, verbose: bool = False,
        fold_coords: bool = False) -> np.ndarray:
    r'''
    Returns the points and their first periodic images.
    
    Notes
    -----
    Copied from mdevaluate's pbc.py (data/robin/mdevaluate/mdevaluate)
    and slightly modified for 2D use.
    
    
    Parameters
    ----------
    coords: np.ndarray of shape (N,3)
        Particle coordinates.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    enforce_nd: int, None, optional
        enforces the number of dimensions. 2 for 2d, 3 for 3d.
        If None is supplied, a best guess is used by checking
        if all particles have the same dimension in the last
        coordinate. See `utils.dimension()`
    width: float or None, optional
        Width of the periodic images relative to the box dimensions.
        None means the original and all periodic images are returned;
        positive means points are cutoff at box*(1+width);
        negative values mean that less than the box is returned.
        At most one periodic image in each direction is returned.
        This keyword is preferred before thickness. The default is None.
    thickness: float or None, optional
        Absolute width of the periodic images in each direction.
        None means the original and all periodic images are returned;
        positive means points are cutoff at box+thickness;
        negative values mean that less than the box is returned.
        If width is supplied, this keyword is ignored. The default is None.
    index: 
        If True, also the indices with indices of images being their original
        values are returned. The default is False.
    inclusive: bool, optional
        If False, only the images are returned. The default is True.
    fold_coords: bool, optional
        If True, points in coordinates are fold back into the box. The default
        is False.
    verbose: bool, optional
        If True, some information is printed. The default is False.
        
    Returns
    -------
    np.ndarray
        Array of all coordinates (including periodic images; array of floats)
    
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> images = amep.pbc.pbc_points(
    ...     frame.coords(), frame.box, inclusive=False
    ... )
    >>> perm_vectors = np.array([
    ...     [-1, -1,  0],
    ...     [-1,  0,  0],
    ...     [-1,  1,  0],
    ...     [ 0, -1,  0],
    ...     [ 0,  1,  0],
    ...     [ 1, -1,  0],
    ...     [ 1,  0,  0],
    ...     [ 1,  1,  0]
    ... ]).astype(float)
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.particles(
    ...     axs, images, frame.box, 0.5, color="tab:blue",
    ...     set_ax_limits=False
    ... )
    >>> amep.plot.particles(
    ...     axs, frame.coords(), frame.box, 0.5, color="tab:orange",
    ...     set_ax_limits=False
    ... )
    >>> amep.plot.box(axs, frame.box, color='k', linewidth=2)
    >>> for p in perm_vectors:
    ...     shift = p*(frame.box[:,1]-frame.box[:,0])
    ...     amep.plot.box(
    ...         axs, frame.box+shift[:,None], color='k',
    ...         linewidth=1, linestyle='--'
    ...     )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/pbc/pbc-pbc_points.png')
    >>> 

    .. image:: /_static/images/pbc/pbc-pbc_points.png
      :width: 400
      :align: center

    '''
    # get box length
    box = box_boundary[:,1] - box_boundary[:,0]
    
    # get center of the simulation box
    center = np.mean(box_boundary, axis=1)

    # check nd
    if enforce_nd is None:
        dimensions = dimension(coords)
        if dimensions == 2:
            if verbose:
                _log.info("Using 2d mode.")
        elif dimensions == 3:
            if verbose:
                _log.info("Using 3d mode.")
    else:
        dimensions = enforce_nd
        if verbose:
            _log.info(
                f"{dimensions}d mode enforced with enforce_nd={enforce_nd}."
            )
    if dimensions not in [2,3]:
        raise ValueError(
            "amep.pbc.pbc_points: Only 2d or 3d systems are supported. "\
            "Please check coords and enforce_nd."
        )
    if fold_coords:
        base = fold(coords, box_boundary)
    else:
        base = coords
    
    box = box.astype(np.float32)
    base = base.astype(np.float32)
    coords = coords.astype(np.float32)
    
    # This array is defined explicitly since this is probably the fastest way
    if dimensions == 2:
        indices = np.tile(np.arange(len(coords)),(9))
        perm_vectors = np.array([[ 0,  0,  0], [-1, -1,  0], [-1,  0,  0],\
                                 [-1,  1,  0], [ 0, -1,  0], [ 0,  1,  0],\
                                 [ 1, -1,  0], [ 1,  0,  0], [ 1,  1,  0]]).astype(np.float32)
        
    elif dimensions == 3:
        indices = np.tile(np.arange(len(coords)),(27))
        perm_vectors = np.array([[ 0,  0,  0], [-1, -1, -1], [-1, -1,  0],\
                                 [-1, -1,  1], [-1,  0, -1], [-1,  0,  0],\
                                 [-1,  0,  1], [-1,  1, -1], [-1,  1,  0],\
                                 [-1,  1,  1], [ 0, -1, -1], [ 0, -1,  0],\
                                 [ 0, -1,  1], [ 0,  0, -1], [ 0,  0,  1],\
                                 [ 0,  1, -1], [ 0,  1,  0], [ 0,  1,  1],\
                                 [ 1, -1, -1], [ 1, -1,  0], [ 1, -1,  1],\
                                 [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1],\
                                 [ 1,  1, -1], [ 1,  1,  0], [ 1,  1,  1]]).astype(np.float32)
    
    # create images
    # Append new coordinates to the original ones.
    # The first coordinates are still the initial, non-shifted ones.
    allcoords = np.concatenate(list(base+v*box for v in perm_vectors), axis=0)
    
    if not inclusive:
        allcoords = allcoords[len(coords):]
        indices = indices[len(coords):]
    
    if width is not None:
        mask = np.all(allcoords < center+box*(.5+width), axis=1)
        allcoords = allcoords[mask]
        indices = indices[mask]
        mask = np.all(allcoords > center-box*(.5+width), axis=1)
        allcoords = allcoords[mask]
        indices = indices[mask]
    elif thickness is not None: #!= 0:
        mask = np.all(allcoords < center+box/2+thickness, axis=1)
        allcoords = allcoords[mask]
        indices = indices[mask]
        mask = np.all(allcoords > center-box/2-thickness, axis=1)
        allcoords = allcoords[mask]
        indices = indices[mask]
        
    if index:
        return (allcoords, indices)
    
    return allcoords


# =============================================================================
# CALCULATION OF MIRROR IMAGES
# =============================================================================
def mirror_points(
        coords: np.ndarray, box_boundary: np.ndarray,
        width: float | None = None, thickness: float | None = None,
        index: bool = False, inclusive: bool = True, verbose: bool = False,
        fold: bool = False, enforce_nd: int | None = None) -> np.ndarray:
    r'''
    Returns the points and their first periodic mirrors.
    This function is used for Voronoi tessellations without
    periodic boundary conditions.
    The dimension (2d or 3d) is guessed from the shape and values 
    of the supplied coordinates.
    
    Parameters
    ----------
        coordinates: np.ndarray
            coordinate frame (Nx3 array of floats; required)
	box_boundary : np.ndarray of shape (3,2)
            Boundary of the simulation box in the form of
            `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
        width: float or None
            Width of the periodic images relative to the box dimensions.
            None means the original and all periodic images are returned;
            positive means points are cutoff at box*(1+width);
            negative values mean that less than the box is returned.
            At most one periodic image in each direction is returned.
            This keyword is preferred before thickness.
            (default=None)
        thickness: float or None
            Absolute width of the periodic images in each direction.
            None means the original and all periodic images are returned;
            positive means points are cutoff at box+thickness;
            negative values mean that less than the box is returned.
            If width is supplied, this keyword is ignored.
            (default=None)
        index: boolean
            if true, also the indices with indices of images being 
            their original values are returned (default=False)
        inclusive: boolean
            if false only the images are returned (defautl=True)
        fold: boolean
            if true points in coordinates are fold back into the box 
            (default=False)
        verbose: bool, optional
            If True, runtime information is printed. The default is False.
        enforce_nd: int, None, optional
            enforces the number of dimensions. 2 for 2d, 3 for 3d.
            If None is supplied, a best guess is used by checking
            if all particles have the same dimension in the last
            coordinate. See `utils.dimension()`
        
    Returns
    -------
        array of all coordinates (including periodic images; 
        array of floats)

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> images = amep.pbc.mirror_points(
    ...     frame.coords(), frame.box, inclusive=False
    ... )
    >>> perm_vectors = np.array([
    ...     [-1, -1,  0],
    ...     [-1,  0,  0],
    ...     [-1,  1,  0],
    ...     [ 0, -1,  0],
    ...     [ 0,  1,  0],
    ...     [ 1, -1,  0],
    ...     [ 1,  0,  0],
    ...     [ 1,  1,  0]
    ... ]).astype(float)
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.particles(
    ...     axs, images, frame.box, 0.5, color="tab:blue",
    ...     set_ax_limits=False
    ... )
    >>> amep.plot.particles(
    ...     axs, frame.coords(), frame.box, 0.5, color="tab:orange",
    ...     set_ax_limits=False
    ... )
    >>> amep.plot.box(axs, frame.box, color='k', linewidth=2)
    >>> for p in perm_vectors:
    ...     shift = p*(frame.box[:,1]-frame.box[:,0])
    ...     amep.plot.box(
    ...         axs, frame.box+shift[:,None], color='k',
    ...         linewidth=1, linestyle='--'
    ...     )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/pbc/pbc-mirror_points.png')
    >>> 

    .. image:: /_static/images/pbc/pbc-mirror_points.png
      :width: 400
      :align: center

    '''
    # get box length
    box = box_boundary[:,1]-box_boundary[:,0]
    
    # get center of the simulation box
    center = np.mean(box_boundary, axis=1)

    # check nd
    if enforce_nd is None:
        dimensions = dimension(coords)
        if dimensions == 2:
            if verbose:
                _log.info("Using 2d mode.")
        elif dimensions == 3:
            if verbose:
                _log.info("Using 3d mode.")
    else:
        dimensions = enforce_nd
        if verbose:
            _log.info(
                f"{dimensions}d mode enforced with enforce_nd={enforce_nd}."
            )
    if dimensions not in [2,3]:
        raise ValueError(
            "Only 2d or 3d systems are supported. Please check coords "\
            "and enforce_nd."
        )
    if fold:
        base = fold(coords, box_boundary)
    else:
        base = coords
    
    box = box.astype(np.float32)
    base = base.astype(np.float32)
    coords = coords.astype(np.float32)
    
    if dimensions == 2:
        zmirrors=[0]
        indices = np.tile(np.arange(len(coords)),(9))
    elif dimensions == 3:
        zmirrors=[-1,0,1]
        indices = np.tile(np.arange(len(coords)),(27))

    # create mirror images
    mirrored_coords = np.empty((0,3), int)
    for xmirr in [-1,0,1]:
        for ymirr in [-1,0,1]:
            for zmirr in zmirrors:
                if xmirr==ymirr==zmirr==0:
                    continue
                else:
                    xyzmirror=np.asarray([xmirr, ymirr, zmirr])
                    mirror=np.diag(box_boundary[:,(xyzmirror+2)//2])
                    mirrored=coords + 2*np.add(mirror, -coords)*abs(xyzmirror)
                    mirrored_coords=np.append(mirrored_coords,mirrored, axis=0)
    # Append new coordinates to the original ones.
    # The first coordinates are still the initial, non-shifted ones.
    allcoords=np.append(coords, mirrored_coords, axis=0)

    if not inclusive:
        allcoords = allcoords[len(coords):]
        indices = indices[len(coords):]
        
    if width is not None: #!= 0:
        mask = np.all(allcoords < center+box*(.5+width), axis=1)
        allcoords = allcoords[mask]
        indices = indices[mask]
        mask = np.all(allcoords > center-box*(.5+width), axis=1)
        allcoords = allcoords[mask]
        indices = indices[mask]
    elif thickness is not None: #!= 0:
        mask = np.all(allcoords < center+box/2+thickness, axis=1)
        allcoords = allcoords[mask]
        indices = indices[mask]
        mask = np.all(allcoords > center-box/2-thickness, axis=1)
        allcoords = allcoords[mask]
        indices = indices[mask]
        
    if index:
        return (allcoords, indices)
    
    return allcoords


# =============================================================================
# CALCULATION OF DIFFERENCE VECTORS
# =============================================================================
def pbc_diff(
        v1: np.ndarray, v2: np.ndarray, box_boundary: np.ndarray,
        pbc: bool = True) -> np.ndarray:
    r"""
    Calculates the difference vector(s) between v1 and v2 considering 
    periodic boundary conditions.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector. 
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.

    Returns
    -------
    out : np.ndarray
        Difference vector(s).

    """
    if pbc:
        out = pbc_diff_rect(v1, v2, box_boundary)
    else:
        out = v1 - v2
    return out


def pbc_diff_rect(v1, v2, box_boundary):
    r"""
    Calculate the difference of two vectors, considering periodic boundary 
    conditions within a rectangular box.
    
    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector. 
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
        
    Returns
    -------
    v: np.ndarray
        Difference vector.
    """
    if v2 is None:
        v = v1
    else:
        v = v1 - v2
        
    # fold distance vectors back into the box
    # must fold with respect to center of box
    v = fold(v, box_boundary-np.mean(box_boundary, axis=1)[:,None])
    return v


# =============================================================================
# KDTREE
# =============================================================================
def kdtree(
        coords: np.ndarray, box_boundary: np.ndarray,
        pbc: bool = True) -> KDTree:
    r'''
    Creates a scipy.spatial._kdtree.KDTree object with and without considering
    periodic boundary conditions.

    Parameters
    ----------
    coords : np.ndarray
        Particle coordinates as array of shape (N,3).
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.

    Returns
    -------
    scipy.spatial._kdtree.KDTree
        KDTree object.

    '''
    box = None
    if pbc:
        # box length
        box = box_boundary[:,1] - box_boundary[:,0]
        
        # get center of the simulation box
        center = np.mean(box_boundary, axis=1)
        
        # shift all coordinates to be within [0,L_i), i=x,y,z
        # (this is required by the KDTree algorithm)
        coords = coords + box/2. - center
        
        # fold coords back into box (to avoid errors)
        coords = fold(coords, box_boundary-box_boundary[:,0][:,None])
        
        # shift particles at the right border to the left border
        # to avoid errors occuring if a particle is placed at L_i
        coords[coords[:,0]==box[0],0]=0
        coords[coords[:,1]==box[1],1]=0
        coords[coords[:,2]==box[2],2]=0
        # should not be done for fields! all right boundaries will
        # be shifted to the left and would be double-occupied!

    return KDTree(coords, boxsize=box)


# =============================================================================
# CALCULATION OF PAIRWISE DISTANCES
# =============================================================================
def find_pairs(
        coords: np.ndarray, box_boundary: np.ndarray,
        ids: np.ndarray | None = None, sizes: np.ndarray | None = None,
        other_coords: np.ndarray | None = None,
        other_ids: np.ndarray | None = None,
        other_sizes: np.ndarray | None = None, pbc: bool = True,
        rmax: float = 1.122) -> np.ndarray:
    r'''
    Identifies pairs of particles based on the pairwise distance. Particles
    are considered as pairs, if their distance to each other is smaller than
    rmax (or smaller than rmax times the contact distance of the particles
    assuming that the particles are spherical).
    
    Parameters
    ----------
    coords : numpy.ndarray
        Particle coordinates as array of shape (N,3).
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    ids : numpy.ndarrays|None, optional
        Particle IDs of particles in coords as array of shape (N,).
        The default is None.
    sizes : numpy.ndarray|None, optional
        Array of size (N,) containing the size, i.e., diameter of each particle
        (assuming that the particles are spherical). If None, rmax is used for
        finding pairs. If given, rmax scales the contact distances.
        The default is None.
    other_coords : numpy.ndarray|None, optional
        Other coordinates as array of shape (N_other, 3). If None, coords is
        used. The default is None.
    other_ids : numpy.ndarrays|None, optional
        Particle IDs of particles in other_coords as array of shape (N_other,).
        The default is None.
    other_sizes : numpy.ndarray|None, optional
        Array of size (N_other,) containing the size of each particle in other
        (assuming that the particles are spherical). If None, rmax is used for 
        finding pairs. If given, rmax scales the contact distances. 
        The default is None.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    rmax : float, optional
        Maximum distance between two particles for which they are considered
        to be pairs. The default is 1.122.

    Returns
    -------
    pairs : numpy.ndarray
        Particle pairs as 2d array of particle indices.
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> coords = np.array(
    ...     [[2,0,0], [1,0,0], [4,0,0], [7,0,0], [9,0,0]],
    ...     dtype=float
    ... )
    >>> ids = np.array([0,1,2,3,4])
    >>> sizes = np.array([1,1,3,3,1], dtype=float)
    >>> box_boundary = np.array([[0,10],[-5,5],[0,1.0]])
    >>> pairs = amep.pbc.find_pairs(
    ...     coords, box_boundary, rmax=2, sizes=sizes
    ...     )
    >>> print(pairs)
    [[0 1]
     [1 4]
     [0 2]
     [1 2]
     [1 3]
     [4 3]
     [2 3]]
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.particles(axs, coords, box_boundary, sizes/2.)
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/pbc/pbc-find_pairs.png")
    >>> 

    .. image:: /_static/images/pbc/pbc-find_pairs.png
      :width: 400
      :align: center

    '''
    if ids is None:
        ids = np.arange(len(coords))
        
    if other_ids is None and other_coords is not None:
        other_ids = np.arange(len(other_coords))

    if sizes is None:
        if other_coords is None:
            # create KDTree object
            tree = kdtree(coords, box_boundary, pbc=pbc)

            # get list of pairs (particle indices)
            pairs = np.asarray(tree.query_pairs(rmax, output_type='ndarray'))
            
            if pairs.shape == (0,):
                # raise an error if no pairs found
                raise RuntimeError(
                    'No pairs found. Please check rmax and sizes.'
                )
            
            # get correct indices
            pairs[:,0] = ids[pairs[:,0]]
            pairs[:,1] = ids[pairs[:,1]]

        else:
            
            if other_sizes is not None:
                warn('sizes is None but not other_sizes. Ignore other_sizes.')

            tree       = kdtree(coords, box_boundary, pbc=pbc)
            other_tree = kdtree(other_coords, box_boundary, pbc=pbc)

            pair_list = tree.query_ball_tree(other_tree, rmax)

            # convert 
            pairs = __convert_pair_list(pair_list)
            
            if pairs.shape == (0,):
                # raise an error if no pairs found
                raise RuntimeError(
                    'No pairs found. Please check rmax and sizes.'
                )

            # set to correct indices
            pairs[:,0] = ids[pairs[:,0]]
            pairs[:,1] = other_ids[pairs[:,1]]
        
    else:
        if other_coords is None:
            # split data according to sizes and process the splitted data individually
            # first, sort the data by the particle sizes
            sort_index    = np.argsort(sizes)
            sorted_sizes  = sizes[sort_index]
            sorted_coords = coords[sort_index]
            sorted_ids    = ids[sort_index]                

            # get all sizes
            unique_sizes = np.unique(sizes)

            # create all pairwise combinations of given sizes
            size_pairs = __create_combinations(unique_sizes)

            # list containing results for each size pair
            container = []

            # loop through all combinations
            for s1, s2 in size_pairs:
                if s1==s2:
                    # threshold
                    max_dist = s1*rmax
                    
                    # get indices
                    indices = np.argwhere(sorted_sizes==s1).T[0]

                    # get pairs
                    p = find_pairs(
                        sorted_coords[indices],
                        box_boundary,
                        pbc=pbc,
                        rmax=max_dist,
                        ids=sorted_ids[indices]
                    )

                else:
                    # threshold
                    max_dist = (s1+s2)*rmax/2

                    # indices
                    ids1 = np.argwhere(sorted_sizes==s1).T[0]
                    ids2 = np.argwhere(sorted_sizes==s2).T[0]

                    # get pairs
                    p = find_pairs(
                        sorted_coords[ids1],
                        box_boundary,
                        other_coords=sorted_coords[ids2],
                        rmax=max_dist,
                        pbc=pbc,
                        ids=sorted_ids[ids1],
                        other_ids=sorted_ids[ids2]
                    )

                container.append(p)

            pairs = np.concatenate(container)
            
        else:
            if other_sizes is None:
                warn('other_sizes is None but not sizes. Ignore sizes.')
                pairs = find_pairs(
                    coords, box_boundary, other_coords=other_coords,
                    rmax=rmax, pbc=pbc, ids=ids, other_ids=other_ids
                )
            else:
                # split data according to sizes and process the splitted data individually
                # first, sort the data by the particle sizes
                sort_index    = np.argsort(sizes)
                sorted_sizes  = sizes[sort_index]
                sorted_coords = coords[sort_index]
                sorted_ids    = ids[sort_index]
                
                sort_other_index    = np.argsort(other_sizes)
                sorted_other_sizes  = other_sizes[sort_other_index]
                sorted_other_coords = other_coords[sort_other_index]
                sorted_other_ids    = other_ids[sort_other_index] 

                # get all sizes
                unique_sizes       = np.unique(sizes)
                unique_other_sizes = np.unique(other_sizes)

                # list containing results for each size pair
                container = []

                # loop through all size pairs
                for s in unique_sizes:
                    for o in unique_other_sizes:
                        # threshold
                        max_dist = (s+o)*rmax/2

                        # indices
                        ids1 = np.argwhere(sorted_sizes==s).T[0]
                        ids2 = np.argwhere(sorted_other_sizes==o).T[0]

                        # get pairs
                        p = find_pairs(
                            sorted_coords[ids1],
                            box_boundary,
                            other_coords=sorted_other_coords[ids2],
                            rmax=max_dist,
                            pbc=pbc,
                            ids=sorted_ids[ids1],
                            other_ids=sorted_other_ids[ids2]
                        )

                        container.append(p)

                pairs = np.concatenate(container)

    return pairs


def __dis_chunk(
        coords: np.ndarray, otree: KDTree, box_boundary: np.ndarray,
        pbc: bool, maxdist: float) -> np.ndarray:
    r'''
    Calculates a chunk (i.e., one row) of the distance matrix. This is used
    in `amep.pbc.distance_matrix`.

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates to which the distances should be calculated.
    otree : KDTree
        KDTree of coordinates from which the distances are calculated.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    pbc : bool
        If True, periodic boundary conditions are considered.
    maxdist : float
        Maxmimum distance to consider. Always use the smallest suitable value
        saves a lot computation time.

    Returns
    -------
    np.ndarray
        One row of the distance matrix.

    '''
    # Create tree object
    ctree = kdtree(coords, box_boundary, pbc=pbc)

    # Calculate the sparse distance matrix for the current row and update it directly
    d = ctree.sparse_distance_matrix(otree, maxdist).toarray()
    
    return d[0]


def distance_matrix_parallel(
        coords: np.ndarray, box_boundary: np.ndarray,
        other: np.ndarray | None = None, pbc: bool = True,
        maxdist: float = 1.122, njobs: int = 1) -> np.ndarray:
    r'''
    Calculates the distance matrix. Distances larger than `maxdist` are
    ignored and set to zero in the distance matrix. This is the parallel
    version using less RAM but taking more time to be calculated. 

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates of the particles as array of shape (N,3).
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other : numpy.ndarray, optional
        Other coordinates to which the pairwise distances are calculated as 
        array of shape (N_other,3). If None, coords is used.
        The default is None.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    maxdist : float, optional
        Maxmimum distance to consider. Always use the smallest suitable value
        saves a lot computation time. The default is 1.122.
    njobs : int, optional
        Number of workers used for the parallelization. If this number exceeds
        the number of CPU cores, it is set to the number of available CPU
        cores. The default is 2.

    Returns
    -------
    d : np.ndarray
        Distance matrix of shape (N,N_other).
        
    Examples
    --------
    Create a simple test setup:
        
    >>> import amep
    >>> import numpy as np
    >>> coords = np.array([[1,0,0], [4,0,0], [-2,0,0], [4.5,0,0]])
    >>> box_boundary = np.array([[-5,5],[-5,5],[-0.5,0.5]])
    >>> box = box_boundary[:,1] - box_boundary[:,0]
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> axs.grid(visible=True)
    >>> amep.plot.particles(axs, coords, box_boundary, 0.25)
    >>> amep.plot.box(axs, box_boundary)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/pbc/pbc-distance_matrix_parallel.png')
    >>> 
    
    .. image:: /_static/images/pbc/pbc-distance_matrix_parallel.png
      :width: 400
      :align: center
      
     
    Calculate the distance matrix without periodic boundary conditions:
    
    >>> D = amep.pbc.distance_matrix_parallel(
    ...     coords, box_boundary, pbc=False, maxdist=5
    ... )
    >>> print(D)
    [[0.  3.  3.  3.5]
     [3.  0.  0.  0.5]
     [3.  0.  0.  0. ]
     [3.5 0.5 0.  0. ]]
    >>> 
    
    
    Calculate the distance matrix with periodic boundary conditions:
    
    >>> D = amep.pbc.distance_matrix_parallel(
    ...     coords, box_boundary, pbc=True, maxdist=5
    ... )
    >>> print(D)
    [[0.  3.  3.  3.5]
     [3.  0.  4.  0.5]
     [3.  4.  0.  3.5]
     [3.5 0.5 3.5 0. ]]
    >>> 
    
    
    Calculate the distance matrix for `other_coords`:
        
    >>> D = amep.pbc.distance_matrix_parallel(
    ...     coords, box_boundary, other=coords[:2],
    ...     pbc=True, maxdist=5
    ... )
    >>> print(D)
    [[0.  3. ]
     [3.  0. ]
     [3.  4. ]
     [3.5 0.5]]
    >>> 

    '''
	# Check other coordinates
    if other is None:
        other = coords

    # create kdtree for other coordinates
    otree = kdtree(other, box_boundary, pbc=pbc)

    # add additional axis (needed for dis_chunk)
    coords = coords[:,None,:]

	# calculate each row of the distance matrix within a separate worker
    result = compute_parallel(
        __dis_chunk,
        coords,
        otree,
        box_boundary,
        pbc,
        maxdist,
        njobs = njobs,
        verbose = True
    )
    return np.asarray(result)


def distance_matrix(
        coords: np.ndarray, box_boundary: np.ndarray,
        other: np.ndarray | None = None, pbc: bool = True,
        maxdist: float = 1.122) -> np.ndarray:
    r'''
    Calculates the distance matrix. Distances larger than `maxdist` are
    ignored and set to zero in the distance matrix. Note that this method
    requires a large amount of RAM. If there is not enough RAM available, use
    the `amep.pbc.distance_matrix_parallel` method.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates of the particles as array of shape (N,3).
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other : numpy.ndarray, optional
        Other coordinates to which the pairwise distances are calculated as 
        array of shape (N_other,3). If None, coords is used.
        The default is None.
    pbc : bool, optional
        If True, periodic boundary conditions are considered.
        The default is True.
    maxdist : float, optional
        Maxmimum distance to consider. Always use the smallest suitable value
        saves a lot computation time. The default is 1.122.

    Returns
    -------
    d : np.ndarray
        Distance matrix of shape (N,N_other).
        
    Examples
    --------
    Create a simple test setup:
        
    >>> import amep
    >>> import numpy as np
    >>> coords = np.array([[1,0,0], [4,0,0], [-2,0,0], [4.5,0,0]])
    >>> box_boundary = np.array([[-5,5],[-5,5],[-0.5,0.5]])
    >>> box = box_boundary[:,1] - box_boundary[:,0]
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> axs.grid(visible=True)
    >>> amep.plot.particles(axs, coords, box_boundary, 0.25)
    >>> amep.plot.box(axs, box_boundary)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/pbc/pbc-distance_matrix.png')
    >>> 
    
    .. image:: /_static/images/pbc/pbc-distance_matrix.png
      :width: 400
      :align: center
      
     
    Calculate the distance matrix without periodic boundary conditions:
    
    >>> D = amep.pbc.distance_matrix(
    ...     coords, box_boundary, pbc=False, maxdist=5
    ... )
    >>> print(D)
    [[0.  3.  3.  3.5]
     [3.  0.  0.  0.5]
     [3.  0.  0.  0. ]
     [3.5 0.5 0.  0. ]]
    >>> 
    
    
    Calculate the distance matrix with periodic boundary conditions:
    
    >>> D = amep.pbc.distance_matrix(
    ...     coords, box_boundary, pbc=True, maxdist=5
    ... )
    >>> print(D)
    [[0.  3.  3.  3.5]
     [3.  0.  4.  0.5]
     [3.  4.  0.  3.5]
     [3.5 0.5 3.5 0. ]]
    >>> 
    
    
    Calculate the distance matrix for `other_coords`:
        
    >>> D = amep.pbc.distance_matrix(
    ...     coords, box_boundary, other=coords[:2],
    ...     pbc=True, maxdist=5
    ... )
    >>> print(D)
    [[0.  3. ]
     [3.  0. ]
     [3.  4. ]
     [3.5 0.5]]
    >>> 

    '''
    # check other coordinates
    if other is None:
        other = coords
        
    # create tree objects   
    ctree = kdtree(coords, box_boundary, pbc = pbc)
    otree = kdtree(other, box_boundary, pbc = pbc)
    
    # calculate distance matrix
    d = ctree.sparse_distance_matrix(otree, maxdist).toarray()
    
    return d


# TODO: PBC handling
def distances(coords, other_coords=None):
    r'''
    Returns a 1D array of all pairwise distances.
    
    Notes
    -----
    If other=None, only the upper half of the distance matrix with the
    diagonal excluded is returned as a flattened 1D array.
    
    This method does not consider periodic boundary conditions.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array of shape (N,3).
    other_coords : np.ndarray, optional
        Coordinate array of shape (N,3). The default is None.

    Returns
    -------
    d : np.ndarray
        1D array of pairwise distances.

    '''
    if other_coords is None:
        d = pdist(coords)
    else:
        d = cdist(coords, other_coords).flatten()
    return d
