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
Spatial Correlation Functions
=============================

.. module:: amep.spatialcor

The AMEP module :mod:`amep.spatialcor` provides methods that calculate spatial
correlation functions for a single frame of particle-based data.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
# import warnings
import os

import numpy as np

from scipy import special
from numba import jit

from .utils import runningmean, sq_from_sf2d, rotate_coords, unit_vector_2D
from .utils import unit_vector_3D, dimension, compute_parallel, optimal_chunksize
from .pbc import pbc_points, pbc_diff, distances, kdtree, fold
from .continuum import coords_to_density
from .continuum import sf2d as csf2d
from .base import MAXMEM, get_module_logger

# logger setup
_log = get_module_logger(__name__)


# =============================================================================
# GENERAL SPATIAL CORRELATION FUNCTION
# =============================================================================
def __scor_chunk(
        chunk, chunksize, coords, other_coords, box, values, other_values,
        dbin_edges, tree):
    r'''
    Calculates the spatial correlation function of the given
    values for one chunk of size chunksize of the system with
    coordinates given by coords.

    Parameters
    ----------
    chunk : int
        Current chunk.
    chunksize : int
        Size of each chunk.
    coords : np.ndarray of shape (N,3)
        Coordinate frame.
    other_coords : np.ndarray of shape (M,3)
        Other coordinates.
    box : np.ndarray of shape (3,) 
        Box length as array of the form `np.array([Lx,Ly,Lz])`.
    values : np.ndarray of shape (N,) or (N,d)
        Contains per atom values for which the spatial correlation function
        should be calculated.
    other_values : np.ndarray of shape (M,) or (M,d)
        Values corresponding to `other_coords`.
    dbin_edges : np.ndarray
        Distances to evaluate the correlation function at.
    tree : scipy.spatial.cKDtree
        Kdtree of the given system.

    Returns
    -------
    list

    '''
    # total number of particles in tree
    N = len(coords)
    
    # check values dtypes
    if values.dtype in ['complex64', 'complex128', np.complex64]:
        cor = np.zeros(len(dbin_edges)-1, dtype=np.complex64)
    else:
        cor = np.zeros(len(dbin_edges)-1, dtype=float)
    
    # normalization
    norm = np.zeros(len(dbin_edges)-1, dtype=float)
    
    # calculate correlation function for the given chunk using slice object
    sl = slice(chunk, chunk + chunksize)
        
    # next neighbor search
    dists, indices = tree.query(
        other_coords[sl],
        N,
        distance_upper_bound = max(box)/2 # max. half box dimension
    )
        
    # correlation function
    for i,d in enumerate(dbin_edges[:-1]):
        # get all entries in distance bin
        mask = (dists >= dbin_edges[i]) & (dists < dbin_edges[i+1])
        # returns all indices of atoms in chunk for which there are nonzero
        # values, i.e. distances inside the specified bin
        row = np.nonzero(mask)[0]

        if len(values.shape) == 1:
            # scalar values
            cor[i] += np.dot(
                other_values[chunk + row%chunksize].conjugate(),
                values[indices[mask]%N]
            )
        else:
            # vector quantities
            for n in range(values.shape[1]):
                cor[i] += np.dot(
                    other_values[chunk + row%chunksize, n].conjugate(),
                    values[indices[mask]%N, n]
                )
                
        norm[i] += len(row)
        
    return [cor, norm]


def spatialcor(
        coords: np.ndarray, box_boundary: np.ndarray,
        values: np.ndarray, other_coords: np.ndarray | None = None,
        other_values: np.ndarray | None = None,
        dbin_edges: np.ndarray | None = None,
        ndists: int | None = None, chunksize: int | None = None,
        njobs: int = 1, rmax: float | None = None, verbose: bool = False,
        pbc: bool = True) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Calculates the spatial correlation function of the values for atoms 
    at positions given by coords. Distances are calculated by the cKDTree
    algorithm.

    Notes
    -----
    Works only for 2D systems in this version.
    
    The spatial correlation function is defined by the following equation
    in which f can be a complex-valued scalar or vector:
    
    .. math::
                C(r) = <f(r)*f(0)> / <f(0)^2>.
    
    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinate frame of all particles.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    values : np.ndarray of shape (N,) or (N,d)
        Contains per atom values for which the spatial correlation function
        should be calculated. For scalar values, provide a np.ndarray of shape
        (N,), where N is the number of particles. For vector quantities,
        provide a np.ndarray of shape (N,d), where d is the dimension of the
        vector. For example, to calculate the spatial velocity correlation
        function, an array of shape (N,3) would be required for a 3d system
        containing the speeds in each spatial dimension.
    other_coords : np.ndarray of shape (M,3) or None, optional
        Coordinates to which the correlation is calculated. If None, `coords` 
        is used. The default is None.
    other_values : None or np.ndarray of shape (M,) or (M,d), optional
        Values corresponding to `other_coords`. If None, `values` is used. The
        default is None.
    dbin_edges : np.ndarray, optional
        Edges of the distance bins. The default is None.
    ndists : int, optional
        Number of distances. The default is None.
    chunksize : int, optional
        Divide the system into chunks of this size
        to save memory during the processing (the
        value gives the number of atoms for each chunk;
        it is suggested to use values between 1000-5000).
        The default is None.
    njobs : int, optional
        Number of jobs used for parallel computation. The default is 1.
    rmax : float, optional
        Maximal distance to consider. The default is None.
    verbose : bool, optional
        If True, additional information is printed and a progress bar is shown.
        The default is False.
    pbc : bool, optional
        If True, periodic boundary conditions are applied. The default is True.

    Returns
    -------
    np.ndarray of shape (ndists,), dtype float
        Averaged correlation function.
    np.ndarray of shape (ndists,), dytpe float
        Distances.

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> cor0, d0 = amep.spatialcor.spatialcor(
    ...     frame.coords(), frame.box, frame.data('vx'),
    ...     chunksize=None, njobs=4, rmax=25, verbose=True
    ... )
    >>> cor1, d1 = amep.spatialcor.spatialcor(
    ...     frame.coords(), frame.box, frame.data('vx'),
    ...     chunksize=100, njobs=4, rmax=25, verbose=True
    ... )
    >>> cor2, d2 = amep.spatialcor.spatialcor(
    ...     frame.coords(), frame.box, frame.data('vx'),
    ...     chunksize=500, njobs=4, rmax=25, verbose=True
    ... )
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(d0, cor0, label='chunksize None', ls='-')
    >>> axs.plot(d1, cor1, label='chunksize 100', ls='--')
    >>> axs.plot(d2, cor2, label='chunksize 500', ls=':')
    >>> axs.set_xlabel(r'$r$')
    >>> axs.set_ylabel(r'$C(r)$')
    >>> axs.legend()
    >>> fig.savefig('./figures/spatialcor/spatialcor-spatialcor.png')
    >>> 
    
    .. image:: /_static/images/spatialcor/spatialcor-spatialcor.png
      :width: 400
      :align: center

    '''
    # check number of CPUs
    if njobs > os.cpu_count():
        njobs = os.cpu_count()
    
    # get box length
    box = box_boundary[:,1]-box_boundary[:,0]
    
    # check values
    if len(values.shape) != 1 and len(values.shape) != 2:
        raise ValueError(
            f'''amep.spatialcor.spatialcor: values have an invalid shape
            {values.shape}. The shape must be (N,) for scalar values and (N,d)
            for vector quantities.'''
        )
    
    # check other coords and values
    if other_coords is None:
        other_coords = coords
    if other_values is None:
        other_values = values
    if other_coords is None and other_values is not None:
        raise ValueError(
            '''amep.spatialcor.spatialcor: other_values are supplied but no
            other_coords. Please provide other_coords as well.'''
        )
    if other_values is None and other_coords is not None:
        raise ValueError(
            '''amep.spatialcor.spatialcor: other_coords are supplied but no
            other_values. Please provide other_values as well.'''
        )

    # total number of particles
    N = len(other_coords)
        
    # get optimal chunk size to reduce RAM usage
    if chunksize is None:
        chunksize = optimal_chunksize(N, 4*N)
        
    # limit chunksize
    if chunksize > N:
        chunksize = int(N/njobs)
        
    # check distance limit
    if rmax is None:
        rmax = max(box)/2.
    
    # check/create distance bins
    if dbin_edges is None and ndists is not None:
        dbin_edges = np.linspace(0.0, rmax, ndists)
    elif dbin_edges is None and ndists is None:
        # step size via trial and error: 1.05 seems to be a good value; for
        # smaller steps, there are too much fluctuations; for larger steps:
        # r=0 correlation smaller than 1
        dbin_edges = np.arange(0.0, rmax, 1.05)
        
    # generate the kdtree for neighbor searching
    tree = kdtree(coords, box_boundary, pbc = pbc)
    
    # need to shift coordinates to be compatible with periodic kdtree
    if pbc:
        # get center of the simulation box
        center = np.mean(box_boundary, axis=1)
        # shift to center (required for pbc kdtree)
        other_coords = fold(other_coords + box/2. - center, box_boundary)
    
    # main calculation (parallelized in chunks)
    cor = 0
    norm = 0
    
    results = compute_parallel(
        __scor_chunk,
        range(0, N, chunksize),
        chunksize,
        coords,
        other_coords,
        box,
        values,
        other_values,
        dbin_edges,
        tree,
        njobs = njobs,
        verbose = verbose
    )
    
    for res in results:
        cor += res[0]
        norm += res[1]
    
    # if norm==0 and cor==0: fix norm
    # this avoids nan-values in empty bins
    norm[(norm==0) & (cor==0)] = 1

    cor = cor/norm

    # use only the real part for complex values
    if cor.dtype in [np.complex64, np.complex128, complex]:
        cor = np.real(cor)

    return cor/cor[0], runningmean(dbin_edges, 2)


# =============================================================================
# PAIR CORRELATION FUNCTIONS
# =============================================================================
def __rdf_diff(chunk, chunksize, coords, other_coords, box_boundary, bins, pbc):
    r'''
    Calculates the rdf for particle n with pbc_diff.

    Parameters
    ----------
    chunk : int
        Current chunk.
    chunksize : int
        Size of the chunk.
    coords : np.ndarray
        Coordinate frame.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    bins : np.ndarray
        distance bins.
    pbc : boolean
        If True, periodic boundary conditions are applied.

    Returns
    -------
    hist : np.ndarray
        Histogram of distances.

    '''
    N = len(coords)
    sl = slice(chunk, chunk+chunksize)
    dist = np.zeros(int(len(other_coords[sl])*N))
    for i,c in enumerate(other_coords[sl]):
        # need the exact distances here (no need to fold the distance
        # vectors back into the box!)
        #diff = pbc_diff(c, coords, box_boundary, pbc=pbc)
        diff = c - coords
        dist[i*N:(i+1)*N] = (diff**2).sum(axis=1)**0.5
    
    # remove zero distances
    dist = dist[dist > 1e-3]
    
    hist = np.histogram(dist, bins, density=False)[0]
    return hist
    
def __rdf_kdtree(chunk, chunksize, coords, other_coords, bins, tree):
    r'''
    Calculates the rdf for one chunk
    with cKDtree.

    Parameters
    ----------
    chunk : int
        Current chunk.
    chunksize : int
        Size of the chunk.
    coords : np.ndarray
        Coordinate frame.
    bins : np.ndarray
        Distance bins.
    tree : scipy.spatial.cKDtree
        kdtree of the system.

    Returns
    -------
    hist : np.ndarray
        Histogram of distances.

    '''
    sl = slice(chunk, chunk+chunksize)
    dist = tree.query(other_coords[sl], k=len(coords))[0].flatten()
    dist = dist[(dist < np.inf) & (dist > 1e-3)]
    hist = np.histogram(dist, bins, density=False)[0]
    return hist


def rdf(
        coords: np.ndarray, box_boundary: np.ndarray,
        other_coords: np.ndarray | None = None, nbins: int | None = None,
        rmax: float | None = None, mode: str = 'diff',
        chunksize: int | None = None, njobs: int = 1, pbc: bool = True,
        verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Calculate the radial pair-distribution function for a single frame.
    Provides two modes: `mode='diff'` loops through all particles and
    calculates the distance to all others as simple difference of the position
    vectors and `use='kdtree'` uses the `scipy.spatial.cKDtree` to determine
    the distances between the particles. It depends on your hardware which
    option is the fastest. Using the `'kdtree'` usually needs more memory but
    could be slightly faster than `'diff'`.


    Notes
    -----
    The radial pair-distribution function between two particle species
    i and j is defined by (see Refs. [1]_ [2]_)

    .. math::
                g_{ij}(r) = \frac{1}{\rho_j N_i}\sum\limits_{k\in S_i}
                            \sum\limits_{\substack{l\in S_j \\ l\neq k}}
                            \left\langle\frac{\delta\left(r-\left|\vec{r}_k(t)-\vec{r}_l(t)\right|\right)}{4\pi r^2}\right\rangle_t

    For i=j we have

    .. math::
                g(r) = \frac{1}{\rho N}\sum\limits_{k}\sum\limits_{l\neq k}
                       \left\langle\frac{\delta\left(r-\left|\vec{r}_k(t)-\vec{r}_l(t)\right|\right)}{2\pi r}\right\rangle_t


    References
    ----------

    .. [1] Abraham, M. J., Hess, B., Spoel, D. van der, Lindahl, E., Apostolov,
       R., Berendsen, H. J. C., Buuren, A. van, Bjelkmar, P., Drunen, R. van,
       Feenstra, A., Fritsch, S., Groenhof, G., Junghans, C., Hub, J., Kasson,
       P., Kutzner, C., Lambeth, B., Larsson, P., Lemkul, J. A., … Maarten, W.
       (2018). GROMACS User Manual version 2018.3. 258. www.gromacs.org

    .. [2] Hecht, L., Horstmann, R., Liebchen, B., & Vogel, M. (2021).
       MD simulations of charged binary mixtures reveal a generic relation
       between high- and low-temperature behavior.
       The Journal of Chemical Physics, 154(2), 024501.
       https://doi.org/10.1063/5.0031417


    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray or None, optional
        Coordinate frame of the other species to which the pair correlation
        is calculated. The default is None (uses coords).
    nbins : int or None, optional
        Number of distance bins. The default is None.
    rmax : float or None, optional
        Maximum distance to consider. The default is None.
    mode : str, optional
        Allows to choose the calculation method. Available modes are `'diff'`
        and `'kdtree'`. The default is 'diff'.
    chunksize : int or None, optional
        Divide calculation into chunks of this size. The default is None.
    njobs : int, optional
        Number of jobs for multiprocessing. The default is 1.
    pbc : boolean, optional
        If True, periodic boundary conditions are applied. The default is True.

    Returns
    -------
    gr : np.ndarray
        g(r) (1D array of floats)
    r : np.ndarray
        distances (1D array of floats)
        
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> gr1, r1 = amep.spatialcor.rdf(
    ...     frame.coords(), frame.box, mode='diff', rmax=10,
    ...     verbose=True, njobs=1, pbc=True
    ... )
    >>> gr2, r2 = amep.spatialcor.rdf(
    ...     frame.coords(), frame.box, mode='kdtree', rmax=10,
    ...     verbose=True, njobs=1, pbc=True
    ... )
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(r1, gr1, label='diff', markersize=4)
    >>> axs.plot(r2, gr2, label='kdtree', ls='--', c='b')
    >>> axs.legend()
    >>> axs.set_xlabel(r'$r$')
    >>> axs.set_ylabel(r'$g(r)$')
    >>> axs.set_xlim(0,5)
    >>> fig.savefig('./figures/spatialcor/spatialcor-rdf.png')
    >>> 
    
    .. image:: /_static/images/spatialcor/spatialcor-rdf.png
      :width: 400
      :align: center

    '''
    # check mode
    if mode not in ['diff', 'kdtree']:
        raise ValueError(
            f"amep.spatialcor.rdf: Invalid mode '{mode}'. Available modes "\
            "are 'diff' and 'kdtree'."
        )
    # get box length
    box = box_boundary[:,1]-box_boundary[:,0]
    
    # check other coords
    if other_coords is None:
        other_coords = coords

    # total number of particles
    Nother = len(other_coords)
    N = len(coords)

    # check number of distance bins
    if nbins is None:
        nbins = 500

    # check max. distance
    if rmax is None:
        rmax = max(box_boundary[:,1]-box_boundary[:,0])/2
    
    # check number of jobs
    if njobs > os.cpu_count():
        njobs = os.cpu_count()
        
    # get spatial dimension
    dim = dimension(coords)
    
    # bin edges
    bins = np.linspace(0.0, rmax, nbins+1) 

    # calculation for diff mode
    if mode == 'diff':
        
        # get optimal chunk size to reduce RAM usage
        if chunksize is None:
            chunksize = optimal_chunksize(Nother, 8*Nother)
            
        # limit chunksize
        if chunksize > Nother:
            chunksize = int(Nother/njobs)

        # pbc
        if pbc:
            coords = pbc_points(
                coords, box_boundary, width=0.5, fold_coords=False
            )
            # not needed anymore due to correction in __rdf_diff
            #N = len(coords)

        # compute the histogram
        hist = 0
        
        results = compute_parallel(
            __rdf_diff,
            range(0, Nother, chunksize),
            chunksize,
            coords,
            other_coords,
            box_boundary,
            bins,
            pbc,
            njobs = njobs,
            verbose = verbose
        )
        for res in results:
            hist += res

    # calculation for kdtree mode
    elif mode == 'kdtree':
        
        # get optimal chunk size to reduce RAM usage
        if chunksize is None:
            chunksize = optimal_chunksize(N, 4*N)
            
        # limit chunksize
        if chunksize > N:
            chunksize = int(N/njobs)

        # create kdtree
        to_tree = kdtree(coords, box_boundary, pbc = pbc)
        
        if pbc:
            # get center of the simulation box
            center = np.mean(box_boundary, axis=1)
            # shift to center (required for pbc kdtree)
            other_coords = fold(other_coords + box/2. - center, box_boundary)
        
        # calculate the histogram of distances
        hist = 0
        
        results = compute_parallel(
            __rdf_kdtree,
            range(0, Nother, chunksize),
            chunksize,
            coords,
            other_coords,
            bins,
            to_tree,
            njobs = njobs,
            verbose = verbose
        )
        for res in results:
            hist += res
    
    # normalization
    if dim == 2:
        area = np.pi * (bins[1:]**2 - bins[:-1]**2)
        density = N / (box[0]*box[1])
        res = hist / area / density / Nother
    else: # means dim=3
        volume = 4 / 3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
        density = N / np.prod(box)
        res = hist / volume / density / Nother

    return res, runningmean(bins, 2)
        

def __dhist2d(
        chunk, chunksize, coords, box_boundary, other_coords, xbins, ybins,
        angle, same, pbc):
    r'''
    Calculates the 2D histogram of distances from chunk particles to all other
    particles in the system (directonally resolved in x and y direction). 
    
    Notes
    -----
    This method is needed for the pcf2d method.

    Parameters
    ----------
    chunk : int
        Current chunk.
    chunksize : int
        Size of the chunk.
    coords : np.ndarray
        Coordinate frame.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray
        Coordinate frame of the other species to which the pair correlation
        is calculated.
    xbins : np.ndarray
        Bin edges for x direction.
    ybins : np.ndarray
        Bin edges for y direction.
    angle : np.ndarray
        Angle to rotate the system in order to orient its mean orientation
        along the x-axis.
    same : bool
        True if coords and other_coords are the same.
    pbc : bool
        If True, periodic boundary conditions are applied.

    Returns
    -------
    hist : np.ndarray
        2D histogram of distances.

    '''
    sl = slice(chunk, chunk+chunksize)
    
    hist = np.zeros((len(xbins)-1,len(ybins)-1), dtype=float)

    for n in np.arange(len(other_coords))[sl]:
        # calculate distance vectors
        if same:
            # diff = pbc_diff(
            #     other_coords[n],
            #     coords[np.arange(len(coords)) != n], # exclude the particle itself
            #     box_boundary,
            #     pbc=pbc
            # )
            # exclude the particle itself
            diff = other_coords[n] - coords[np.arange(len(coords)) != n]
        else:
            # diff = pbc_diff(other_coords[n], coords, box_boundary, pbc=pbc)
            diff = other_coords[n] - coords
        # orient x-axis along mean sample orientation
        # The rotation is applied to the difference vectors instead of the
        # particle positions, since the distances can only be calculated 
        # correctly if the particle coordinates and the simulation box
        # fit to each other (which is no longer the case when coordinates
        # are rotated)!
        if angle != 0.0:
            # get center of the simulation box
            center = np.mean(box_boundary, axis=1)
            
            # rotate all coords
            diff = rotate_coords(diff, -angle, center)


        # calculate 2D histogram
        hist += np.histogram2d(diff[:,0], diff[:,1], [xbins,ybins])[0]
    
    return hist
        

def pcf2d(
        coords: np.ndarray, box_boundary: np.ndarray,
        other_coords: np.ndarray | None = None, nxbins: int | None = None,
        nybins: int | None = None, rmax: float | None = None,
        psi: np.ndarray | None = None, njobs: int = 1, pbc: bool = True,
        verbose: bool = False, chunksize: int | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r'''
    Calculates the 2D pair correlation function by calculating histograms.
    To allow for averaging the result (either with respect to time or to make
    an ensemble average), the coordinates are rotated such that the mean
    orientation (given by the vector psi) points along the x axis
    (see Ref. [1]_ for details).


    Notes
    -----
    The 2D pair correlation function between the particles themselfes, i.e.,
    for i=j, is defined by

    .. math::
               g(x,y) = \frac{1}{\rho N}\sum\limits_{i=1}^{N}
               \sum\limits_{j\neq i}^{N}\delta(x-x_{ij})\delta(y-y_{ij}).

    The cross correlation between different particle species is given by

    .. math::
                g_{ij}(x,y)=\frac{1}{\rho_i N_j}\sum\limits_{n=1}^{N_i}
               \sum\limits_{m=1}^{N_j}\delta(x-x_{nm})\delta(y-y_{nm}).

    This version only works for (quadratic) 2D systems. For 3D systems,
    the z coordinate is discarded.

    References:

    .. [1] Bernard, E. P., & Krauth, W. (2011). Two-Step Melting in Two
       Dimensions: First-Order Liquid-Hexatic Transition.
       Physical Review Letters, 107(15), 155704.
       https://doi.org/10.1103/PhysRevLett.107.155704


    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray of shape (M,3), optional
        Coordinate frame of the other species to which the pair correlation
        is calculated. The default is None (uses coords).
    nxbins : int or None, optional
        Number of x bins. The default is None.
    nybins : int or None, optional
        Number of y bins. The default is None.
    rmax : float or None, optional
        Maximum distance to consider (should be smaller than the largest box
        length divided by 2*sqrt(2)). The default is None.
    psi : np.ndarray of shape (2,) or None, optional
        Mean orientation of the whole system. If not None, the particles
        positions are rotated such that the x-axis of the system points into
        the direction of `psi`. According to Ref. [1], use 
        `psi=(Re{Psi_6},Im{Psi_6})`, where Psi_6 is the mean hexagonal order
        parameter of the whole system.
    njobs : int, optional
        Number of jobs for parallel computation. The default is 1.
    pbc : bool, optional
        If True, periodic boundary conditions are applied. The default is True.
    verbose : bool, optional
        If True, a progress bar is shown. The default is False.
    chunksize : int or None, optional
        Divide calculation into chunks of this size. The default is None.
        

    Returns
    -------
    np.ndarray
        g(x,y) (2D array of floats)
    np.ndarray
        x coordinate values (1D array of floats)
    np.ndarray
        y coordinate (1D array of floats)
        
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> gxy, x, y = amep.spatialcor.pcf2d(
    ...     frame.coords(), frame.box, njobs=4,
    ...     nxbins=1000, nybins=1000
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.7,3))
    >>> mp = amep.plot.field(axs, gxy, x, y)
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r'$g(\Delta x, \Delta y)$'
    ... )
    >>> axs.set_xlim(-10,10)
    >>> axs.set_ylim(-10,10)
    >>> axs.set_xlabel(r'$\Delta x$')
    >>> axs.set_ylabel(r'$\Delta y$')
    >>> fig.savefig('./figures/spatialcor/spatialcor-pcf2d.png')
    >>> 
    
    .. image:: /_static/images/spatialcor/spatialcor-pcf2d.png
      :width: 400
      :align: center
    
    '''
    # get box length
    box = box_boundary[:,1]-box_boundary[:,0]
    
    same = False
    if other_coords is None:
        other_coords = coords
        same = True
    
    # enforce 2D
    coords[:,-1] = 0.0
    other_coords[:,-1] = 0.0
    
    # total number of particles
    N = len(coords)
    Nother = len(other_coords) 

    if nxbins is None:
        nxbins = 500
    if nybins is None:
        nybins = 500

    # divide by 2*np.sqrt(2) to allow averaging with rotation
    # (difference vectors with length larger than max(box)/2/np.sqrt(2)
    # might not stay insight the box after rotation)
    if rmax is None:
        rmax = max(box)//(2*np.sqrt(2))#2
    
    # check number of jobs
    if njobs > os.cpu_count():
        njobs = os.cpu_count()
        
    # angle to rotate (to orient x-axis along mean sample orientation)
    angle = 0.0
    if psi is not None:
        ex = np.array([1,0])
        angle = np.arccos(np.dot(ex, psi)/np.sqrt(psi[0]**2+psi[1]**2))
        if psi[1] < 0:
            angle = 2*np.pi - angle
        
    # get optimal chunk size to reduce RAM usage
    if chunksize is None:
        chunksize = optimal_chunksize(N, 8*N)
        
    # limit chunksize
    if chunksize > N:
        chunksize = int(N/njobs)

    # create bin edges for the histogram
    xbins = np.linspace(-rmax, rmax, nxbins+1)
    ybins = np.linspace(-rmax, rmax, nybins+1)

    # compute the 2D histogram
    hist = np.zeros((nxbins,nybins), dtype=float)
    
    results = compute_parallel(
        __dhist2d,
        range(0, Nother, chunksize),
        chunksize,
        coords,
        box_boundary,
        other_coords,
        xbins,
        ybins,
        angle,
        same,
        pbc,
        njobs = njobs,
        verbose = verbose
    )
     
    for res in results:
        hist += res

    # normalization
    area = (xbins[1:] - xbins[:-1])[:,None] * (ybins[1:]-ybins[:-1])
    density = N / (box[0]*box[1])
    res = hist / area / density / Nother 
    
    x = runningmean(xbins, 2)
    y = runningmean(ybins, 2)
    
    X,Y = np.meshgrid(x,y)

    return res, X, Y 


def __dhist_angle(
        chunk, chunksize, coords, box_boundary, other_coords, dbins, abins, e,
        angle, same, pbc):
    r'''
    Calculates the 2D histogram of distances and angles for all particles
    in a chunk with respect to all other particles in the system.
    
    Notes
    -----
    This method is needed for the pcf_angle method.

    Parameters
    ----------
    chunk : int
        Current chunk.
    chunksize : int
        Size of the chunk.
    coords : np.ndarray
        Coordinate frame.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray
        Coordinate frame of the other species to which the pair correlation
        is calculated.
    dbins : np.ndarray
        Bin edges for distances.
    abins : np.ndarray
        Bin edges for angles.
    e : np.ndarray
        Direction/axis to which the angle is measured.
    angle : np.ndarray
        Angle to rotate the system in order to orient its mean orientation
        along the x-axis.
    pbc : bool
        If True, periodic boundary conditions are applied.

    Returns
    -------
    hist : np.ndarray
        2D histogram.

    '''
    sl = slice(chunk, chunk+chunksize)
    
    # compute the 2D histogram (r,theta)
    hist = np.zeros((len(dbins)-1,len(abins)-1), dtype=float)

    for n in np.arange(len(other_coords))[sl]:
        # calculate distance vectors
        if same:
            # diff = pbc_diff(
            #     other_coords[n],
            #     coords[np.arange(len(coords)) != n], # exclude the particle itself
            #     box_boundary,
            #     pbc=pbc
            # )
            # exclude the particle itself
            diff = other_coords[n] - coords[np.arange(len(coords)) != n]
        else:
            # diff = pbc_diff(other_coords[n], coords, box_boundary, pbc=pbc)
            diff = other_coords[n] - coords
        
        # get distances
        dist = (diff**2).sum(axis=1)**0.5

        # calculate angles
        theta = np.arccos(np.dot(e, diff.T)/dist)
        theta[diff[:,1] < 0.0] = 2*np.pi - theta[diff[:,1] < 0.0] # use angles between 0 and 2\pi
        
        # orient x-axis along mean sample orientation
        # The rotation is applied to the difference vectors instead of the
        # particle positions, since the distances can only calculated 
        # correctly if the particle coordinates and the simulation box
        # fit to each other (which is no longer the case when coordinates
        # are rotated)!
        if angle != 0.0:
            theta = theta - angle
            theta[theta<0.0] = theta[theta<0.0] + 2*np.pi
            theta[theta>2*np.pi] = theta[theta>2*np.pi] - 2*np.pi

        # calculate 2D histogram
        hist += np.histogram2d(dist, theta, [dbins,abins])[0]
        
    return hist


def pcf_angle(
        coords: np.ndarray, box_boundary: np.ndarray, 
        other_coords: np.ndarray | None = None, ndbins: int | None = None,
        nabins: int | None = None, rmax: int | None = None,
        psi: np.ndarray | None = None, njobs: int = 1, pbc: bool = True,
        verbose: bool = False, chunksize: int | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r'''Calculate the angle dependent radial pair correlation function.

    Do that by calculating histograms.
    To allow for averaging the result (either with respect to time or to make
    an ensemble average), the coordinates are rotated such that the mean
    orientation (given by the vector psi) points along the x axis
    (see Ref. [1]_ for details).

    Notes
    -----
    The angle-dependent pair correlation function is defined by (see Ref. [2]_)

    .. math::
       g(r,\theta) = \frac{1}{\langle \rho\rangle_{local,\theta} N}\sum\limits_{i=1}^{N}
       \sum\limits_{j\neq i}^{N}\frac{\delta(r_{ij} -r)\delta(\theta_{ij}-\theta)}{2\pi r^2 \sin(\theta)}


    The angle :math:`\theta` is defined with respect to a certain axis :math:`\vec{e}` and
    is given by

    .. math::
       \cos(\theta)=\frac{\vec{r}_{ij}\cdot\vec{e}}{r_{ij}e}

    Here, we choose :math:`\vec{e}=\hat{e}_x`.

    This version only works for 2D systems. For systems in 3D,
    the z coordinate is discarded.

    References
    ----------
    .. [1] Bernard, E. P., & Krauth, W. (2011). Two-Step Melting in Two
       Dimensions: First-Order Liquid-Hexatic Transition.
       Physical Review Letters, 107(15), 155704.
       https://doi.org/10.1103/PhysRevLett.107.155704

    .. [2] Abraham, M. J., Hess, B., Spoel, D. van der, Lindahl, E., Apostolov,
       R., Berendsen, H. J. C., Buuren, A. van, Bjelkmar, P., Drunen, R. van,
       Feenstra, A., Fritsch, S., Groenhof, G., Junghans, C., Hub, J., Kasson,
       P., Kutzner, C., Lambeth, B., Larsson, P., Lemkul, J. A., … Maarten, W.
       (2018). GROMACS User Manual version 2018.3. www.gromacs.org

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Particle coordinates.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray of shape (M,3), optional
        Coordinate frame of the other species to which the pair correlation
        is calculated. The default is None (uses coords).
    ndbins : int or None, optional
        Number of distance bins. The default is None.
    nabins : int or None, optional
        Number of angle bins. The default is None.
    rmax : float or None, optional
        Maximum distance to consider. The default is None.
    psi : np.ndarray of shape (2,) or None, optional
        Mean orientation of the whole system. If not None, the particles
        positions are rotated such that the x-axis of the system points into
        the direction of psi. According to Ref. [1]_, use
        :math:`\psi=(Re(\Psi_6),Im(\Psi_6))`, where :math:`\Psi_6` is the mean
        hexagonal order parameter of the whole system. The default is None.
    njobs : int, optional
        Number of jobs used for parallel computing.The default is 1.
        Rule of thumb: Check how many your system has available with

        >>> import os
        >>> os.cpu_count()

        and set njobs to between 1x this number and 2x this number.
    pbc : bool, optional
        If True, periodic boundary conditions are applied. The default is True.
    verbose : bool, optional
        If True, a progress bar is shown. The default is False.
    chunksize : int or None, optional
        Divide calculation into chunks of this size. The default is None.

    Returns
    -------
    gr : np.ndarray
        :math:`g(r,\theta)` (2D array of floats)
    R : np.ndarray
        distances (meshgrid)
    T : np.ndarry
        Angles (meshgrid)
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> grt, rt, theta = amep.spatialcor.pcf_angle(
    ...     frame.coords(), frame.box, njobs=4,
    ...     ndbins=1000, nabins=1000
    ... )
    >>> X = rt*np.cos(theta)
    >>> Y = rt*np.sin(theta)
    >>> fig, axs = amep.plot.new(figsize=(3.8,3))
    >>> mp = amep.plot.field(axs, grt.T, X, Y)
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r'$g(\Delta x, \Delta y)$'
    ... )
    >>> axs.set_xlim(-10,10)
    >>> axs.set_ylim(-10,10)
    >>> axs.set_xlabel(r'$\Delta x$')
    >>> axs.set_ylabel(r'$\Delta y$')
    >>> fig.savefig('./figures/spatialcor/spatialcor-pcf_angle.png')
    >>> 
    
    .. image:: /_static/images/spatialcor/spatialcor-pcf_angle.png
      :width: 400
      :align: center
    
    '''
    # get box length
    box = box_boundary[:,1]-box_boundary[:,0]
    
    same = False
    if other_coords is None:
        other_coords = coords
        same = True
        
    # enforce 2D
    coords[:,-1] = 0.0
    other_coords[:,-1] = 0.0
    
    # total number of particles
    N = len(coords) 
    Nother = len(other_coords)
    
    # calculate angle with respect to this unit vector
    e = np.array([1.,0.,0.]) 

    if ndbins is None:
        ndbins = 500
    if nabins is None:
        nabins = 100

    if rmax is None:
        rmax = max(box_boundary[:,1]-box_boundary[:,0])//2
        
    # angle to rotate (to orient x-axis along mean sample orientation)
    angle = 0.0
    if psi is not None:
        ex = np.array([1,0])
        angle = np.arccos(np.dot(ex, psi)/np.sqrt(psi[0]**2+psi[1]**2))
        if psi[1] < 0:
            angle = 2*np.pi - angle
            
    # check number of jobs
    if njobs > os.cpu_count():
        njobs = os.cpu_count()
        
    # get optimal chunk size to reduce RAM usage
    if chunksize is None:
        chunksize = optimal_chunksize(N, 8*N)
        
    # limit chunksize
    if chunksize > N:
        chunksize = int(N/njobs)

    # generate distance and angle bin edges
    dbins = np.linspace(0.0, rmax, ndbins+1)
    abins = np.linspace(0.0, 2*np.pi, nabins+1)

    # compute the 2D histogram (r,theta)
    hist = np.zeros((ndbins,nabins), dtype=float)
    
    results = compute_parallel(
        __dhist_angle, 
        range(0, Nother, chunksize),
        chunksize,
        coords,
        box_boundary,
        other_coords,
        dbins,
        abins,
        e,
        angle,
        same,
        pbc,
        njobs = njobs,
        verbose = verbose
    )
    for res in results:
        hist += res

    # normalization
    area = 0.5 * (dbins[1:]**2 - dbins[:-1]**2)[:,None] * (abins[1:]-abins[:-1])
    density = N / (box[0]*box[1])
    res = hist / area / density / Nother
    
    r = runningmean(dbins, 2)
    t = runningmean(abins, 2)
    
    R,T = np.meshgrid(r,t)

    return res.T, R, T


# =============================================================================
# STRUCTURE FACTORS
# =============================================================================
def __sf_iso_chunk_std(chunk, chunksize, coords, other_coords, q, twod):
    r'''
    Calculates the isotropic static structure factor for one chunk of the 
    system by using the Debye scattering formula (see Ref. [1] for details).
    
    
    Notes
    -----
    This method is used by the the sf_iso method when mode='std' is selected.
    
    References:
        
    .. [1] Wieder, T. (2012). The Debye scattering formula in n dimensions. 
       Journal of Mathematical and Computational Science, 2, 1086. 
       https://scik.org/index.php/jmcs/article/view/263

    Parameters
    ----------
    chunk : int
        Current chunk.
    chunksize : int
        Size of each chunk.
    coords : np.ndarray
        Coordinate frame.
    other_coords : np.ndarray
        Coordinate frame of the other species to which the correlation
        is calculated.
    q : np.ndarray
        Array of wave numbers.
    twod : bool
        If True, the formula for the 2D case is used.

    Returns
    -------
    np.ndarray
        Structure factor for the given chunk.

    '''
    # max. memory in bytes
    maxmem = MAXMEM*1e9
    
    # convert to float32 to save memory
    q = q.astype(np.float32)
    coords = coords.astype(np.float32)
    other_coords = other_coords.astype(np.float32)
    
    # slicing
    sl = slice(chunk, chunk+chunksize)
    
    # get pairwise distances
    dists = distances(coords, other_coords=other_coords[sl])
    dists = dists.astype(np.float32)
    
    # add dists=0 contribution to results array
    res = np.zeros(len(q), dtype=np.float32)
    res += np.sum(dists==0)
    
    # exclude zero distances to avoid error due to divide by zero
    dists = dists[dists!=0]
    
    if twod:
        if (dists.nbytes*len(q)/maxmem) > 1: 
            for i,Q in enumerate(q):
                res[i] += np.sum(special.j0(dists*Q))
        else:
            x = dists[:,None]*(q[None,:])
            res += np.sum(special.j0(x), axis=0)
    else:
        if (dists.nbytes*len(q)/maxmem) > 1:
            for i,Q in enumerate(q):
                res[i] += np.sum(np.sinc(dists*Q/np.pi))
        else:
            x = dists[:,None]*(q[None,:]/np.pi)
            res += np.sum(np.sinc(x), axis=0)
    
    return res


def __sf_iso_chunk_fast(u, coords, other_coords, q):
    r'''
    Calculates the isotropic static structure factor for one chunk of the 
    system by using the rewritten form for homogeneous and isotropic system
    provided in Ref. [1]. This reduces the number of computations from N^2
    to N, where N is the number of particles.
    
    
    Notes
    -----
    This method is used by the the sf_iso method when mode='fast' is selected.
    
    References:
    -----------
        
    .. [1] Aichele, M., & Baschnagel, J. (2001). Glassy dynamics of simulated 
       polymer melts: Coherent scattering and van Hove correlation functions.
       The European Physical Journal E, 5(2), 229–243. 
       https://doi.org/10.1007/s101890170078

    Parameters
    ----------
    u : np.ndarray
        Unit vector of shape (3,).
    coords : np.ndarray
        Coordinate frame.
    other_coords : np.ndarray
        Coordinate frame of the other species to which the correlation
        is calculated.
    q : np.ndarray
        Array of wave numbers.

    Returns
    -------
    np.ndarray
        Structure factor for the given chunk.

    '''
    cos = np.zeros(len(q))
    sin = np.zeros(len(q))
    for i,k in enumerate(q):
        cos[i] = np.sum(np.cos(k*np.sum(u*coords,axis=1)))*np.sum(np.cos(k*np.sum(u*other_coords,axis=1)))
        sin[i] = np.sum(np.sin(k*np.sum(u*coords,axis=1)))*np.sum(np.sin(k*np.sum(u*other_coords,axis=1)))
    
    S = (cos+sin)/np.sqrt(len(coords)*len(other_coords))
    
    return S


def sfiso(
        coords: np.ndarray, box_boundary: np.ndarray, N: int,
        qmax: float = 20.0, twod: bool = True, njobs: int = 1,
        chunksize: int | None = None, mode: str = 'std',
        other_coords: np.ndarray | None = None, accuracy: float = 0.5,
        num: int = 64, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Calculates the isotropic static structure factor for a given set of
    coordinates.

    Notes
    -----
    The isotropic static structure factor is defined by

    .. math::
        S_{3D}(q) = \frac{1}{N}\left\langle\sum_{m,l=1}^N\frac{\sin(qr_{ml})}{qr_{ml}}\right\rangle

    .. math::
        S_{2D}(q) = \frac{1}{N}\left\langle\sum_{m,l=1}^N J_0(qr_{ml})\right\rangle

    with :math:`r_{ml}=|\vec{r}_m-\vec{r}_l|` and the Bessel function of the first
    kind :math:`J_0(x)`.

    For `mode='fast'`, a rewritten form for homogeneous and isotropic systems
    is used that reduces the number of computations
    from :math:`\mathcal{O}(N^2)` to :math:`\mathcal{O}(N)`,
    where :math:`N` is the number of particles [1]_.

    Mode 'fft' only works in 2D!!!
    
    The minimum wave vector is fixed to :math:`2\pi/L`, where :math:`L` is the
    box length.
    
    
    References
    ----------

    .. [1] Aichele, M., & Baschnagel, J. (2001). Glassy dynamics of simulated
       polymer melts: Coherent scattering and van Hove correlation functions.
       The European Physical Journal E, 5(2), 229–243.
       https://doi.org/10.1007/s101890170078

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    N : int
        Total number of particles.
    qmax : float, optional
        Maximum wave number to consider. The default is 20.
    twod : bool, optional
        If True, the 2D form is used. The default is True.
    njobs : int, optional
        Number of jobs for multiprocessing. The default is 1.
    chunksize : int, optional
        In 'std' mode, the calculation is divided into chunks of this size.
        The default is 1000.
    mode : str, optional
        One of ['std', 'fast', 'fft']. The 'fft' mode only works
        if twod is True. The default is 'std'.
    other_coords : np.ndarray, optional
        Coordinate frame of the other species to which the correlation
        is calculated. The default is None (uses coords).
    accuracy : float, optional
        Accuracy for fft mode. 0.0 means least accuracy, 1.0 best accuracy.
        The default is 0.5. Note that a higher accuracy needs more memory
        for the computation. accuracy must be in (0,1].
    num : int, optional
        Number of q vectors to average over in 'fast' mode. If twod is False,
        the number of q vectors is equal to num^2. The default is 64.
    verbose : bool, optional
        If True, additional information is printed and a progress bar is shown.
        The default is False.


    Returns
    -------
    S : np.ndarray
        Isotropic static structure factor.
    q : np.ndarray
        Wave numbers.

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> S0, q0 = amep.spatialcor.sfiso(
    ...     frame.coords(), frame.box, frame.n(),
    ...     njobs=6, twod=True, chunksize=1000, mode='std'
    ... )
    >>> S1, q1 = amep.spatialcor.sfiso(
    ...     frame.coords(), frame.box, frame.n(),
    ...     njobs=6, twod=True, chunksize=1000, mode='fast'
    ... )
    >>> S2, q2 = amep.spatialcor.sfiso(
    ...     frame.coords(), frame.box, frame.n(),
    ...     njobs=6, twod=True, chunksize=1000, mode='fft', accuracy=0.5
    ... )
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(q0, S0, ls='-', label='std')
    >>> axs.plot(q1, S1, ls=':', label='fast')
    >>> axs.plot(q2, S2, ls='--', label='fft')
    >>> axs.set_ylim(-0.5, 8)
    >>> axs.set_xlim(0, 20)
    >>> axs.legend(title='mode')
    >>> axs.set_xlabel(r'$q$')
    >>> axs.set_ylabel(r'$S(q)$')
    >>> axs.axhline(0, c='k', ls='--', lw=1)
    >>> fig.savefig('./figures/spatialcor/spatialcor-sfiso.png')
    >>>
    
    .. image:: /_static/images/spatialcor/spatialcor-sfiso.png
      :width: 400
      :align: center

    '''
    # possible modes
    modes = ['std', 'fast', 'fft']
    
    # check number of jobs for parallelization
    if njobs > os.cpu_count():
        njobs = os.cpu_count()
    
    # check other_coords
    if other_coords is None:
        other_coords = coords
    N = len(other_coords)
    
    # get optimal chunk size to reduce RAM usage
    if chunksize is None:
        chunksize = optimal_chunksize(N, 2*N)
        
    # limit chunksize
    if chunksize > N:
        chunksize = int(N/njobs)
        
    # get maximum distance
    rmax = np.max(box_boundary[:,1]-box_boundary[:,0])
    
    # smallest possible q value
    dq = 2*np.pi/rmax
    
    # generate q values
    q = np.arange(dq, qmax, dq)
    
    # check mode
    if mode not in modes:
        _log.warning(
            f"Mode '{mode}' not available. Using mode 'fast'. "\
            f"Available modes are {modes}."
        )
        mode = 'fast'
    if not twod and mode == 'fft':
        _log.warning(
            "Mode 'fft' does only work for 2d systems. Since twod is False, "\
            "mode 'std' is used."
        )
        mode = 'std'
    
    # calculation with standard mode in chunks
    if mode == 'std':
        
        # compute in parallel
        results = compute_parallel(
            __sf_iso_chunk_std,
            range(0, N, chunksize),
            chunksize,
            coords,
            other_coords,
            q,
            twod,
            njobs = njobs,
            verbose = verbose
        )

        # put data from the chunks together
        S = 0
        for res in results:
            S += res
        S = S/np.sqrt(len(coords)*len(other_coords))
        
    # calculation with fast mode
    elif mode == 'fast':
        
        # get unit vectors to average over
        if twod:
            theta        = np.linspace(0, 360, num=num, endpoint=False)
            unit_vectors = unit_vector_2D(theta)
        else:
            theta = np.linspace(0, 180, num=int(num/2), endpoint=False)
            phi   = np.linspace(0, 360, num=num, endpoint=False)
            t,p   = np.meshgrid(theta, phi)
            unit_vectors = unit_vector_3D(t.flatten(), p.flatten())
        
        # compute in parallel
        results = compute_parallel(
            __sf_iso_chunk_fast,
            unit_vectors,
            coords,
            other_coords,
            q,
            njobs = njobs,
            verbose = verbose
        )

        # put data from the chunks together
        S = 0
        for res in results:
            S += res            
        S = S/len(unit_vectors)
    
    # calculation with fft mode              
    elif mode == 'fft':
        if verbose:
            _log.info(
                "Using mode 'fft' is experimental. It is recommended "\
                "to use mode 'std' or 'fast'."
            )
        S2d, Kx, Ky = sf2d(
            coords,
            box_boundary, 
            Ntot=N, 
            other_coords=other_coords, 
            mode='fft', 
            accuracy=accuracy,
            njobs=njobs,
            qmax=qmax
        )
        S,q = sq_from_sf2d(S2d, Kx, Ky)
        S = S[(q<qmax) & (q>=dq)]
        q = q[(q<qmax) & (q>=dq)]

    return S, q


@jit(nopython=True)
def __sf2d_chunk_std(chunk, chunksize, coords, Qx, Qy):
    '''
    2d static structure factor for a chunk of particles.

    Parameters
    ----------
    chunk : int
        Start index of the given chunk.
    chunksize : int
        Size of each chunk.
    coords : np.ndarray
        Coordinate frame.
    Q : np.ndarray
        Wave vectors.

    Returns
    -------
    S : np.ndarray
        Static structure factor.

    '''
    # slicing
    sl = slice(chunk, chunk+chunksize)
    # Fourier transform of microscopic density
    rhoq = np.zeros(Qx.shape, dtype=np.complex64)
    # loop over all particles in the given chunk
    for c in coords[sl]:
        rhoq += np.exp(-1j*(Qx*c[0]+Qy*c[1]))
    return rhoq


def sf2d(
        coords: np.ndarray, box_boundary: np.ndarray,
        other_coords: np.ndarray | None = None, qmax: float = 20.0,
        njobs: int = 1, mode: str = 'std', Ntot: int | None = None,
        accuracy: float = 0.1, chunksize: int | None = None,
        verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r'''
    Calculates the 2d static structure factor.

    Notes
    -----
    The static structure factor is defined by

    .. math::

        S(\vec{q}) = \frac{1}{N} \left\langle\sum_{j=1}^{N}\sum_{k=1}^{N}\exp\left\lbrace-i\vec{q}\cdot(\vec{r}_j-\vec{r}_k)\right\rbrace\right\rangle\\
                   = \frac{1}{N} \left\langle\sum_{j=1}^{N}\left\lvert\exp\left\lbrace-i\vec{q}\cdot\vec{r}_j\right\rbrace\right\rvert^2\right\rangle,

    where :math:`\rho(\vec{q})` is the Fourier transform of the particle number
    density (see Ref. [1]_ for further information).

    S(0,0) is set to 0

    References
    ----------

    .. [1] Hansen, J.-P., & McDonald, I. R. (2006). Theory of Simple Liquids
       (3rd ed.). Elsevier. https://doi.org/10.1016/B978-0-12-370535-8.X5000-9

    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame (3D).
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray, optional
        Coordinate frame of the other species to which the pair correlation
        is calculated. The default is None (uses coords).
    qmax : float, optional
        Maximum wave number to consider. The default is 20.
    njobs : int, optional
        Number of jobs used for parallel computation. The default is 4.
    mode : str, optional
        Calculation method. Mode 'fft' converts the particle coordinates to
        a density field and uses the continuum method (FFT) to calculate the
        structure factor. Mode 'std' uses the particle coordinates
        directly, which is much slower. The default is 'fft'.
    Ntot : int, optional
        Total number of particles in the system. This value is needed for the
        correct normalization of the structure factor. If None, Ntot is
        calculated from coords. The default is None.
    accuracy : float, optional
        Accuracy for fft mode. 0.0 means least accuracy, 1.0 best accuracy.
        The default is 0.1. Note that a higher accuracy needs more memory
        for the computation. accuracy must be in (0,1].
    verbose : bool, optional
        If True, a progress bar is shown. The default is False.
    chunksize : int or None, optional
        Divide calculation into chunks of this size. The default is None.

    Returns
    -------
    np.ndarray
        Two dimensional static structure factor.
    np.ndarray
        Wave vector's x component.
    np.ndarray
        Wave vector's y component.

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> sxy, qx, qy = amep.spatialcor.sf2d(
    ...     frame.coords(), frame.box, verbose=True, mode='std', njobs=4
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.7,3))
    >>> mp = amep.plot.field(axs, sxy, qx, qy)
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r'$S(q_x, q_y$'
    ... )
    >>> axs.set_xlabel(r'$q_x$')
    >>> axs.set_ylabel(r'$q_y$')
    >>> fig.savefig('./figures/spatialcor/spatialcor-sf2d.png')
    >>>
    
    .. image:: /_static/images/spatialcor/spatialcor-sf2d.png
      :width: 400
      :align: center

    '''
    # box dimensions
    box = box_boundary[:,1]-box_boundary[:,0]
    
    # total number of particles
    if Ntot is None:
        Ntot = len(coords)
    
    # check other_coords
    same = False
    if other_coords is None:
        other_coords = coords
        same = True
        
    # check accuracy value (only needed for fft mode)
    if accuracy < 0.0 or accuracy > 1.0:
        raise ValueError(
            f"amep.spatialcor.sf2d: Accuracy must be between 0.0 and 1.0. "\
            f"Got {accuracy}."
        )  
    # check mode
    if mode not in ['std', 'fft']:
        raise ValueError(
            f"Mode '{mode}' not available. Available modes are "\
            f"{['std', 'fft']}."
        )    
    # partial particle numbers
    N = len(coords)
    Nother = len(other_coords)
        
    # check njobs
    if njobs > os.cpu_count():
        njobs = os.cpu_count()

    # get maximum distance
    rmax = np.abs(np.max(box))
    
    # smallest q value
    dq = 2*np.pi/rmax
    
    # generate q values
    q  = np.arange(dq, qmax, dq)
    qx = np.append(np.flip(-q), q)
    qy = np.append(np.flip(-q), q)
    
    if mode == 'std':
        # get optimal chunk size to reduce RAM usage
        if chunksize is None:
            chunksize = min(
                optimal_chunksize(Nother, N),
                optimal_chunksize(N, Nother)
            )
            
        # limit chunksize
        if chunksize > Nother:
            chunksize = int(Nother/njobs)
        elif chunksize > N:
            chunksize = int(N/njobs)

        # wave vectors
        Qx, Qy = np.meshgrid(qx, qy)
        
        # parallel computation of Fourier-transformed microscopic density
        rhoq = np.zeros(Qx.shape, dtype=np.complex64)
        rhoq_other = np.zeros(Qx.shape, dtype=np.complex64)
        
        results = compute_parallel(
            __sf2d_chunk_std,
            range(0, N, chunksize),
            chunksize,
            coords,
            Qx,
            Qy,
            njobs = njobs,
            verbose = verbose
        )
        for res in results:
            rhoq += res
            
        if same:
            rhoq_other = rhoq
        else:
            results = compute_parallel(
                __sf2d_chunk_std,
                range(0, Nother, chunksize),
                chunksize,
                other_coords,
                Qx,
                Qy,
                njobs = njobs,
                verbose = verbose
            )
            for res in results:
                rhoq_other += res
        S = np.real(rhoq*rhoq_other.conj())/Ntot
        return S, Qx, Qy
    
    elif mode == 'fft':
        if verbose:
            _log.info(
                "Using fft mode is experimental. It is recommended "\
                "to use std mode."
            )
        dmin = 2*np.pi/2/qmax/(accuracy*10) # to match qmax and the correct q spacing (accuracy factor for better accuracy)
        d, X, Y = coords_to_density(coords, box_boundary, dmin=dmin)
        dother, _, _ = coords_to_density(other_coords, box_boundary, dmin=dmin)
        S, Qx, Qy = csf2d(d, X, Y, other_dfield=dother, Ntot=Ntot, njobs=njobs)

        if np.max(Qx) > qmax or np.max(Qy) > qmax:
            # only return the results for Q values in [-qmax,qmax]
            dqx   = Qx[0,1]-Qx[0,0]
            dqy   = Qy[1,0]-Qy[0,0]
            qx    = np.arange(dqx, qmax, dqx)
            qy    = np.arange(dqy, qmax, dqy)
            
            mask = (np.abs(Qx)<=qmax) & (np.abs(Qy)<=qmax)
            sx   = int(2*len(qx)+1)
            sy   = int(2*len(qy)+1)

            S = S[mask].reshape(sx,sy)
            Qx = Qx[mask].reshape(sx,sy)
            Qy = Qy[mask].reshape(sx,sy)

        # Modify S to exluced value for q=0 (replace it with 0)
        position_of_q0 = S.shape[0]//2
        S[position_of_q0, position_of_q0] = 0
        return S, Qx, Qy
