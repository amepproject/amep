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
Utility Methods
===============

.. module:: amep.utils

The AMEP module :mod:`amep.utils` is a collection of various utility methods.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import time
from packaging.version import Version

import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from scipy import special, signal

from .functions import gaussian2d, gaussian
from .base import MAXMEM, get_module_logger
from tqdm.autonotebook import tqdm
from concurrent.futures import ProcessPoolExecutor

# logger setup
_log = get_module_logger(__name__)

# =============================================================================
# AVERAGING
# =============================================================================
def traj_slice(N: int, skip: float, nr_averages: int) -> slice:
    r'''
    Slicing object to slice a trajectory.

    Parameters
    ----------
    N : int
        total number of time steps of the trajectory
    skip : float
           fraction of trajectory to skip at its beginning
           (float between 0.0 and 1.0)
    nr_averages : int
                  total number of frames to use (int)

    Returns
    -------
    slice
    '''
    step = int(N * (1 - skip) // nr_averages) or 1
    return slice((int(skip * N) + step//2), N, step)

def average_func(
        func: callable, data: list | np.ndarray, skip: float = 0.0,
        nr: int = 10, indices: bool = False, **kwargs):
    r'''
    Compute the average of a certain function.
    
    Parameters
    ----------
    func : func
        function that has to be averaged
    data : list/np.ndarray
        list/array of data for each time step (input for func)
    skip: float, default=0.0
        fraction of the whole trajectory that is skipped at the
        beginning of the trajectory for the calculation of the
        time average (float between 0.0 and 1.0)
    nr : int, default=10
        maximum number of frames to average over
    indices : bool, default=False
        If True, the list indices that the averaging function used are returned
        as last output!
    **kwargs: Keyword Arguments
        keyword arguments that are put to func

    Returns
    -------
    list
        list of func results for each time step
    various
        time-averaged value of func for given trajectory
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> def function(x):
    ...     return x**2
    >>> x = np.linspace(0,100,1000)
    >>> y = np.sin(x)
    >>> function_yvalues, average = amep.utils.average_func(
    ...     function, y, nr=20
    ... )
    >>> function_xvalues, _ = amep.utils.average_func(
    ...     lambda x:x, x, nr=20
    ... )
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(
    ...     function_xvalues, function_yvalues,
    ...     label='data'
    ... )
    >>> axs.axhline(average, label='average', ls='--')
    >>> axs.legend()
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-average_func.png')
    >>> 

    .. image:: /_static/images/utils/utils-average_func.png
      :width: 400
      :align: center
    
    '''
    N = len(data)   # number of time steps

    if(nr == None or nr > N - skip * N):
        nr = max(1,int(N-skip*N))

    evaluated_indices = np.array(np.ceil(np.linspace(skip*N, N-1, nr)), dtype=int)
    func_result = [func(x, **kwargs) for x in tqdm(data[evaluated_indices])]
    evaluated = np.array(func_result)
    if indices:
        return evaluated, np.mean(evaluated, axis=0), evaluated_indices
    return evaluated, np.mean(evaluated, axis=0)


def runningmean(data: np.ndarray, nav: int, mode: str = 'valid') -> np.ndarray:
    r'''
    Compute the running mean of a 1-dimensional array.

    Parameters
    ----------
    data : np.ndarray
           input data of shape (N, )
    nav : int
          number of points over which the data will be averaged
    mode : str
           'same', 'valid', or 'full'

    Returns
    -------
    np.ndarray of shape (N-(nav-1), )
        averaged data

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(0,2*np.pi,500)
    >>> y = np.sin(x)*(1+np.random.rand(len(x)))
    >>> yav = amep.utils.runningmean(y, nav=20)
    >>> xav = amep.utils.runningmean(x, nav=20)
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(x, y, label='data', ls='')
    >>> axs.plot(
    ...     xav, yav, label='running mean',
    ...     marker='', lw=2, c='orange'
    ... )
    >>> axs.legend()
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-runningmean.png')
    >>>
    
    .. image:: /_static/images/utils/utils-runningmean.png
      :width: 400
      :align: center

    '''
    return np.convolve(data, np.ones((nav,)) / nav, mode=mode)


def runningmean2d(
        data: np.ndarray, nav: int, mode: str = 'valid') -> np.ndarray:
    r'''
    Computes the running mean of two-dimensional data, e.g., of a 2d density
    field.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (N1,N2,).
    nav : int
        Number of points in each direction to average over (the total number
        over which the data is averaged is given by nav^2).
    mode : str, optional
        'same', 'full', or 'valid'. The default is 'valid'.

    Returns
    -------
    np.ndarray
        Averaged data.
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> a = np.zeros((7,7))
    >>> a[3,3] = 1
    >>> print(a)
    [[0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]]
    >>> print(amep.utils.runningmean2d(a, 3, mode='valid'))
    [[0.         0.         0.         0.         0.        ]
     [0.         0.11111111 0.11111111 0.11111111 0.        ]
     [0.         0.11111111 0.11111111 0.11111111 0.        ]
     [0.         0.11111111 0.11111111 0.11111111 0.        ]
     [0.         0.         0.         0.         0.        ]]
    >>> 

    '''
    return convolve2d(data, np.ones((nav,nav,))/nav**2, mode=mode)


def segmented_mean(
        data: np.ndarray, n: int, verbose: bool = False
        ) -> np.ndarray:
    r'''Compute the mean of n adjacent data points.

    Pools n adjacent data points into one bin.
    Creates array of len(data)/n bins.
    The used data may be reduced in case the length is not divisable by n.

    Parameters
    ----------
    data : np.ndarray
        Input data to be averaged.
    n : int
        Number of adjacent data points to be averaged.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Returns
    -------
    np.ndarray
        Averaged data
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(0,2*np.pi,500)
    >>> y = np.sin(x)*(1+np.random.rand(len(x)))
    >>> yav = amep.utils.segmented_mean(y, n=20)
    >>> xav = amep.utils.segmented_mean(x, n=20)
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(x, y, label='data', ls='')
    >>> axs.plot(
    ...     xav, yav, label='segmented mean',
    ...     marker='', lw=2, c='orange'
    ... )
    >>> axs.legend()
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-segmented_mean.png')
    >>> 

    .. image:: /_static/images/utils/utils-segmented_mean.png
      :width: 400
      :align: center

    '''
    if len(data) % n != 0 and verbose:
        _log.info(
            f"Data length not divisible by {n}. "\
            f"Proceeding by omitting the last {len(data)%n} steps"
        )
    return data[:len(data)-(len(data) % n)].reshape(-1, n).mean(axis=1)
    # [:len(a)-(len(a)%m)] reduces array length by proper amount
    # but also works if no elements are to be removed


def profile(
        coords: np.ndarray, box_boundary: np.ndarray, values: np.ndarray,
        binwidth: float = 10.0, axis: str = 'x', return_indices: bool = False
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r'''
    Calculate the profile of given values.

    Work per particle along the given axis of the simulation box.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    values : np.ndarray
        Per-particle values (1D array).
    binwidth : float, optional
        Width of the distance bins along the given axis. The default is 10.0.
    axis : str, optional
        Axis along which the profile should be calculated. The default is 'x'.
    return_indices : bool, optional
        If True, for each bin the indices of included particles are returned.
        The default is False.

    Returns
    -------
    np.ndarray
        Profile of the given values (=average of values of the particles in
        each bin).
    np.ndarray
        Bin positions.
    np.ndarray
        Object array of lists of indices for each bin giving the particle
        indices of all particles inside the specific bin.

    Examples
    --------
    Here, we calculate the x-velocity profile along the x direction:

    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> y, x = amep.utils.profile(
    ...     frame.coords(), frame.box, frame.data("vx"),
    ...     binwidth=1, axis='x'
    ... )
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(x, y)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$\langle v_x\rangle_y$')
    >>> fig.savefig('./figures/utils/utils-profile_1.png')
    >>>
    
    .. image:: /_static/images/utils/utils-profile_1.png
      :width: 400
      :align: center

    '''
    # create bin edges
    if axis == 'y':
        ind = 1
    elif axis == 'z':
        ind = 2
    else:
        ind = 0
    bins = np.arange(
        box_boundary[ind,0],
        box_boundary[ind,1]+binwidth/2,
        binwidth
    )
    # average values over all other directions (except given axis)
    res = np.zeros(len(bins)-1)
    indices = np.empty(len(bins)-1, dtype=object)
    for i in range(len(res)):
        mask = np.where((coords[:,ind] >= bins[i]) & (coords[:,ind] < bins[i+1]))[0]
        res[i] = np.mean(values[mask])
        indices[i] = mask
        
    if return_indices:
        return res, runningmean(bins, 2), indices
    else:
        return res, runningmean(bins, 2)


# =============================================================================
# MATRIX AND VECTOR OPERATIONS
# =============================================================================
def unit_vector_2D(theta: float | np.ndarray) -> np.ndarray:
    r'''
    Generates 2D unit vectors in the x-y plane with angle theta to the x axis.

    Parameters
    ----------
    theta : float or np.ndarray
        Angle in degrees. A single value can be given as a float or multiple
        values as an array of shape (N,).

    Returns
    -------
    np.ndarray
        Unit vector(s).
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> print(amep.utils.unit_vector_2D(60))
    [0.5       0.8660254 0.       ]
    >>> theta = np.array([0, 60, 120, 180, 240, 300, 360])
    >>> vectors = amep.utils.unit_vector_2D(theta)
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> axs.scatter(vectors[:,0], vectors[:,1], s=50)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> axs.axhline(0, c='k', lw=1, ls='--')
    >>> axs.axvline(0, c='k', lw=1, ls='--')
    >>> fig.savefig('./figures/utils/utils-unit_vector_2D.png')
    >>> 

    .. image:: /_static/images/utils/utils-unit_vector_2D.png
      :width: 400
      :align: center

    '''
    # transform into radiants
    theta = theta*2*np.pi/360
    
    if type(theta) == np.ndarray:
        return np.array([np.cos(theta), np.sin(theta), np.zeros(len(theta))]).T
    elif type(theta) == float:
        return np.array([np.cos(theta), np.sin(theta), 0])
    

def unit_vector_3D(
        theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
    r'''
    Generates 3D unit vectors with components
    
    .. math::
        x = \sin(\theta)\cos(\phi)
        
        y = \sin(\theta)\sin(\phi)
        
        z = \cos(\theta).
    

    Parameters
    ----------
    theta : float or np.ndarray
        Angle in degrees. A single value can be given as a float or multiple
        values as an array of shape (N,).
    phi : float or np.ndarray
        Angle in degrees. A single value can be given as a float or multiple
        values as an array of the same shape as theta.

    Returns
    -------
    np.ndarray
        Unit vector(s).
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> print(amep.utils.unit_vector_3D(60, 60))
    >>> theta = np.array([0, 30, 60, 90, 120, 150, 180])
    >>> phi = np.array([0, 60, 120, 180, 240, 300, 360])
    >>> t, p = np.meshgrid(theta, phi)
    >>> vectors = amep.utils.unit_vector_3D(
    ...     t.flatten(), p.flatten()
    ... )
    >>> fig = plt.figure()
    >>> axs = fig.add_subplot(111, projection='3d')
    >>> axs.quiver(
    ...     0, 0, 0, vectors[:,0], vectors[:,1],
    ...     vectors[:,2], length=0.05
    ... )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> axs.set_zlabel(r'$z$')
    >>> fig.savefig('./figures/utils/utils-unit_vector_3D.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-unit_vector_3D.png
      :width: 400
      :align: center

    '''
    
    # transform into radiants
    theta = theta*2*np.pi/360
    phi   = phi*2*np.pi/360
    
    if type(theta)==np.ndarray and type(phi)==np.ndarray and len(theta)==len(phi):
        return np.array([np.sin(theta)*np.cos(phi),
                         np.sin(theta)*np.sin(phi),
                         np.cos(theta)]).T
    elif type(theta)==float and type(phi)==float:
        return np.array([np.sin(theta)*np.cos(phi),
                         np.sin(theta)*np.sin(phi),
                         np.cos(theta)])
    

def rotation(theta: float, axis: str = 'z') -> np.ndarray:
    r'''
    Rotation matrix in 3D for rotation around a certain axis.
    
    Notes
    -----
    Only axis='z' is implemented in this version.
    
    Parameters
    ----------
    theta : float
            rotation angle
    axis : str
           'x', 'y', or 'z'; defines the rotation axis.
           

    Returns
    -------
    np.ndarray
        rotation matrix of shape (3,3)
    '''
    M = np.zeros((3,3))
    
    if axis == 'x':
        pass
    elif axis == 'y':
        pass
    elif axis == 'z':
        M = np.array([[np.cos(theta),-np.sin(theta),0],
                      [np.sin(theta),np.cos(theta),0],
                      [0,0,1]])
    return M


def rotate_coords(
        coords: np.ndarray, theta: float, center: np.ndarray) -> np.ndarray:
    r'''
    Rotates all coordinate vectors given in coords by the angle theta
    around the origin of the system given by center.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array.
    theta : float
        Angle.
    center : np.ndarray
        Center of the system (this is the origin around the coordinates
        are rotated).

    Returns
    -------
    coords : np.ndarray
        Rotated coordinates array (same shape as input coords array).
    
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> theta = 0.25*np.pi
    >>> rcoords = amep.utils.rotate_coords(
    ...     frame.coords(), theta, np.mean(frame.box, axis=1)
    ... )
    >>> fig, axs = amep.plot.new(ncols=2, figsize=(6,3))
    >>> amep.plot.particles(
    ...     axs[0], frame.coords(), frame.box, frame.radius(),
    ...     set_ax_limits=False
    ... )
    >>> amep.plot.box(axs[0], frame.box)
    >>> axs[0].set_xlim(-10, 90)
    >>> axs[0].set_ylim(-10, 90)
    >>> axs[0].set_xlabel(r'$x$')
    >>> axs[0].set_ylabel(r'$y$')
    >>> axs[0].set_title('original')
    >>> amep.plot.particles(
    ...     axs[1], rcoords, frame.box, frame.radius(),
    ...     set_ax_limits=False
    ... )
    >>> amep.plot.box(axs[1], frame.box)
    >>> axs[1].set_xlim(-10, 90)
    >>> axs[1].set_ylim(-10, 90)
    >>> axs[1].set_xlabel(r'$x$')
    >>> axs[1].set_ylabel(r'$y$')
    >>> axs[1].set_title('rotated')
    >>> fig.savefig('./figures/utils/utils-rotate_coords.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-rotate_coords.png
      :width: 600
      :align: center

    '''
    # get rotation matrix
    R = rotation(theta)
    
    # shift to put [0,0,0] into the center
    coords = coords - center
    
    # rotate coordinates
    coords = np.array([np.dot(R,coords[i]) for i in range(len(coords))])
    
    # redo the shift operation
    coords = coords + center
    
    return coords


def in_box(
        coords: np.ndarray, box_boundary: np.ndarray,
        values: np.ndarray | None = None, reverse: bool = False,
        indices: bool = False) -> np.ndarray:
    r'''
    Returns all coordinates in coords that are in the given simulation
    box boundaries (including the boundary).

    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    values : np.ndarray, optional
        Array of shape (N,p), where N is the number of particles and p the
        number of values per particle. The default is None.
    reverse : bool, optional
        If True, coordinates outside the specified region are returned.
        The default is False.
    indices : bool, optional
        If True, also particle indices are returned. The default is False.

    Returns
    -------
    values : np.ndarray
        Values inside the box.
    ind : np.ndarray
        Particle indices inside the box (only if indices is True).
    
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> box = np.array([[40, 60], [40, 60], [-0.5, 0.5]])
    >>> inside, idxin = amep.utils.in_box(
    ...     frame.coords(), box, indices=True
    ... )
    >>> outside, idxout = amep.utils.in_box(
    ...     frame.coords(), box, reverse=True, indices=True
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.particles(
    ...     axs, inside, frame.box, frame.radius()[idxin],
    ...     set_ax_limits = False, color='orange'
    ... )
    >>> amep.plot.particles(
    ...     axs, outside, frame.box, frame.radius()[idxout],
    ...     set_ax_limits = False
    ... )
    >>> amep.plot.box(axs, frame.box)
    >>> amep.plot.box(axs, box)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-in_box.png')
    >>> 

    .. image:: /_static/images/utils/utils-in_box.png
      :width: 400
      :align: center

    '''
    if values is None:
        values = coords
    
    if reverse:
        mask = ((coords[:,0]<box_boundary[0,0]) | (coords[:,0]>box_boundary[0,1])) | \
               ((coords[:,1]<box_boundary[1,0]) | (coords[:,1]>box_boundary[1,1])) | \
               ((coords[:,2]<box_boundary[2,0]) | (coords[:,2]>box_boundary[2,1]))
        
    else:
        mask = (coords[:,0]>=box_boundary[0,0]) & \
               (coords[:,0]<=box_boundary[0,1]) & \
               (coords[:,1]>=box_boundary[1,0]) & \
               (coords[:,1]<=box_boundary[1,1]) & \
               (coords[:,2]>=box_boundary[2,0]) & \
               (coords[:,2]<=box_boundary[2,1])  
        
    ind = np.arange(len(values))[mask]
    values = values[mask]
        
    if indices:
        return values, ind
    else:
        return values

    
def in_circle(
        coords: np.ndarray, radius: float,
        center: np.ndarray = np.array([0,0,0]),
        values: np.ndarray | None = None, reverse: bool = False,
        indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    r'''
    Returns the coordinates/values for only particles inside or outside
    a circular/spherical region of a given radius and centered around a
    given center position.
    
    Parameters
    ----------
    coords : np.ndarray
        Particle coordinates.
    radius : float
        Radius of the spherical/circular region.
    center : np.ndarray, optional
        Center of the region. The default is np.array([0,0,0]).
    values : np.ndarray, optional
        Per-particle values/vectors to return. The default is None.
    reverse : bool
        If True, particle values for particles outside the region
        are returned. The default is False.
    indices : bool
        If True, particle indices are also returned. The default is False.
        
    Return
    ------
    values : np.ndarray
        Values for particles in or outside the region.
    ind : np.ndarray
        Particle indices.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> center = frame.box.mean(axis=1)
    >>> R = 25
    >>> inside, idxin = amep.utils.in_circle(
    ...     frame.coords(), R, indices=True, center=center
    ... )
    >>> outside, idxout = amep.utils.in_circle(
    ...     frame.coords(), R, reverse=True,
    ...     indices=True, center=center
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.particles(
    ...     axs, inside, frame.box, frame.radius()[idxin],
    ...     set_ax_limits = False, color='orange'
    ... )
    >>> amep.plot.particles(
    ...     axs, outside, frame.box, frame.radius()[idxout],
    ...     set_ax_limits = False
    ... )
    >>> amep.plot.box(axs, frame.box)
    >>> angle = np.linspace(0, 2*np.pi, 150)
    >>> x = R*np.cos(angle) + center[0]
    >>> y = R*np.sin(angle) + center[1]
    >>> axs.plot(x, y, c='k', lw=2, marker='')
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-in_circle.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-in_circle.png
      :width: 400
      :align: center
    
    '''
    if values is None:
        values = coords
    
    # shift coords
    shifted = coords-center
    
    # apply mask
    if reverse:
        ind = np.where(shifted[:,0]**2+shifted[:,1]**2>=radius**2)[0]
        values = values[ind]
    else:
        ind = np.where(shifted[:,0]**2+shifted[:,1]**2<radius**2)[0]
        values = values[ind] 
    
    # returns
    if indices:
        return values, ind
    else:
        return values


def upper_triangle(
        matrix: np.ndarray, diagonal: bool = False,
        indices: bool = False) -> np.ndarray:
    r'''
    Returns the upper diagonal of a quadratic matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix of shape (N,N).
    diagonal : bool, optional
        If True, the diagonal elements are returned as well.
        The default is False.
    indices : bool, optional
        If True, the indices of the matrix elements are returned
        as well. The default is False.

    Returns
    -------
    np.ndarray
        Array of shape (M,) or (3,M) if indices is True.
        
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> m = np.array([[0,1,2],[3,4,5],[6,7,8]])
    >>> print(m)
    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    >>> m1 = amep.utils.upper_triangle(m, diagonal=False, indices=False)
    >>> print(m1)
    [1 2 5]
    >>> m2 = amep.utils.upper_triangle(m, diagonal=True, indices=False)
    >>> print(m2)
    [0 1 2 4 5 8]
    >>> m3 = amep.utils.upper_triangle(m, diagonal=False, indices=True)
    >>> print(m3)
    [[0 0 1]
     [1 2 2]
     [1 2 5]]
    >>> 

    '''
    if diagonal:
        res = matrix[np.triu_indices_from(matrix, k=0)]
        idx = np.vstack(np.triu_indices_from(matrix, k=0))
    else:
        res = matrix[np.triu_indices_from(matrix, k=1)]
        idx = np.vstack(np.triu_indices_from(matrix, k=1))
        
    if indices:
        return np.vstack((idx, res))
    else:
        return res


# =============================================================================
# FIND PEAKS
# =============================================================================
def kpeaks(
        Sxy: np.ndarray, kx: np.ndarray, ky: np.ndarray, a: float = 1.0,
        dk: float = 4.0, mode: str = 'hexagonal') -> np.ndarray:
    r'''
    Calculates the wave vectors that correspond to the first peaks
    of the 2D structure factor. The mode parameter specifies what kind of 
    lattice is assumed (e.g. for a hexagonal lattice one expects six peaks). 
    This information is also used to make a first estimate of the peak
    positions. The peaks are then determined by fitting a 2D Gaussian
    to the structure factor around the estimates.
    
    Notes
    -----
    This code only works for 2D systems.    
    
    Parameters
    ----------
    Sxy : np.ndarray
        2D static structure factor.
    kx : np.ndarray
        x-components of wave vectors (same shape as Sxy; meshgrid).
    ky : np.ndarray
        y-components of wave vectors (same shape as Sxy; meshgrid).
    a : float, optional
        Lattice spacing - this is needed the make a first 
        estimate for the peak positions. The default is 1.0.
    dk : float, optional
        Size of the area in k-space around each estimate used 
        for the Gaussian fit. The default is 4.0.
    mode : str, optional
        The current version only uses mode='hexagonal'. There
        are no other modes available yet. The default is 'hexagonal'.
            
    Returns
    -------
    np.ndarray
        Nkx3 array of floats (containing the Nk vectors
        corresponding to the Nk first peaks - z-component is set to zero)
    '''
    if mode == 'hexagonal':
        # estimates for k0
        G = 4*np.pi/a/np.sqrt(3) * np.array([1.0, 0.0, 0.0]) # reciprocal lattice vector
    
        estimates = [np.dot(rotation(n*np.pi/3), G) for n in range(6)]
        estimates = np.array(estimates)
    else:
        # estimates for k0
        G = 4*np.pi/a/np.sqrt(3) * np.array([1.0, 0.0, 0.0]) # reciprocal lattice vector
    
        estimates = [np.dot(rotation(n*np.pi/3), G) for n in range(6)]
        estimates = np.array(estimates)
    
    # get area of size dk^2 around estimates
    areas = []
    Kxvals = []
    Kyvals = []

    for vec in estimates:
    
        # get mask for specified area dk**2 around the estimate
        mask = (kx<vec[0]+dk/2) & (kx>vec[0]-dk/2) & (ky<vec[1]+dk/2) & (ky>vec[1]-dk/2)

        Kxvals.append(kx[np.where(mask)])
        Kyvals.append(ky[np.where(mask)])
        areas.append(np.abs(Sxy)[np.where(mask)])
        

    # fit 2d Gaussian to the area
    sigx = 1.0; sigy = 1.0; theta = 0.0; offset = 0.0 # initial values

    k0vecs = []

    for i in range(6):
        try:
            popt, pcov = curve_fit(gaussian2d, (Kxvals[i], Kyvals[i]), areas[i].ravel(),\
                                   p0=[np.max(areas[i]), estimates[i][0], estimates[i][1], sigx, sigy, theta, offset],\
                                   maxfev=20000, bounds=([0.0, -np.max(G)-dk, -np.max(G)-dk, 0.0, 0.0, 0.0, -np.inf],\
                                                         [np.inf, np.max(G)+dk, np.max(G)+dk, np.inf, np.inf, 2*np.pi, np.inf]))
                
            err = np.sqrt(np.diag(pcov))

            # check if norm of fitted vector is inside a small area around
            # the estimate (if not use estimated value)
            if np.linalg.norm(G)-0.1*dk < np.sqrt(popt[1]**2 + popt[2]**2) < np.linalg.norm(G)+0.1*dk and np.all(err!=0):
                k0vecs.append([popt[1], popt[2], 0.0])
            else:
                k0vecs.append([estimates[i][0], estimates[i][1], 0.0])
        except:
            # if fit fails use the estimated value (corresponds to an optimal hexagonal lattice)
            k0vecs.append([estimates[i][0], estimates[i][1], 0.0])

    k0vecs = np.array(k0vecs)
    
    return k0vecs



def detect2peaks(
        xdata: np.ndarray, ydata: np.ndarray, nav: int = 11, distance: int = 8,
        height: float = 0.1, width: int = 4
        ) -> tuple[list, list, np.ndarray, np.ndarray]:
    r'''
    Detects the two highest peaks in the given data after smoothing it
    with a running average. Includes also global maxima at the boundaries.

    Notes
    -----
    This method only produces correct results if there are at most two
        peaks present! Please check the behavior in your individual case.

    Parameters
    ----------
    xdata : np.ndarray
        x values as 1D array of floats.
    ydata : np.ndarray
        y values as 1D array of floats (same shape as xdata).
    nav : int, optional
        Number of points to average over via running mean. The default is 11.
    distance : int, optional
        Minimum distance in samples between two peaks. The default is 8.
    height : float, optional
        Minium height of a peak. The default is 0.1.
    width : int, optional
        Minimum peak width in samples. The default is 4.

    Returns
    -------
    low : list
        Coordinates [x,y] of the peak at smaller x value.
    high : list
        Coordinates [x,y] of the peak at larger x value.
    avydata : np.ndarray
        Averaged y values.
    avxdata : np.ndarray
        Averaged x values (same shape as avydata).
    
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(1,100,1000)
    >>> y = np.exp(-(x-20)**2/5)
    >>> y += np.exp(-(x-70)**2/10)
    >>> y += np.random.rand(len(x))*0.1
    >>> low, high, avy, avx = amep.utils.detect2peaks(
    ...     x, y, nav=10, width=2
    ... )
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(x, y, label='original', c='k', ls='')
    >>> axs.plot(avx, avy, label='averaged', c='r', marker='')
    >>> axs.plot(
    ...     low[0], low[1], linestyle='', marker='.',
    ...     c='b', markersize=10, label='peaks'
    ... )
    >>> axs.plot(
    ...     high[0], high[1], linestyle='',
    ...     marker='.', c='b', markersize=10
    ... )
    >>> axs.legend()
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$f(x)$')
    >>> fig.savefig('./figures/utils/utils-detect2peaks.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-detect2peaks.png
      :width: 400
      :align: center
    
    '''
    # smooth data with running mean
    avxdata = runningmean(xdata, nav, mode='valid')
    avydata = runningmean(ydata, nav, mode='valid')

    # detecting maxima at the borders
    # adding "virtual" points at the beginning and end of data
    dx=avxdata[1]-avxdata[0]
    avxdata=np.concatenate(([avxdata[0]-dx], avxdata, [avxdata[-1]+dx]))
    avydata=np.concatenate(([np.min(avydata)], avydata, [np.min(avydata)]))
    
    # determine peaks in averaged data
    ind, other = signal.find_peaks(avydata, height=height, distance=distance, width=width)

    # determine high- and low-x peaks (accounts also for global maxima at the boundary)
    if len(ind) == 0:
        if avydata[-1] >= np.mean(avydata[-(width+1):-1]):#max(avydata[-(width+1):-1]):
            high = [avxdata[-1], avydata[-1]]
            low = high
        else:
            high = None
            low = None
    elif len(ind) == 1:
        if avydata[-1] > np.mean(avydata[-(width+1):-1]):#max(avydata[-(width+1):-1]):
            high = [avxdata[-1], avydata[-1]]
            low = [avxdata[ind][0], avxdata[ind][0]]
        else:
            low = [avxdata[ind][0], avydata[ind][0]]
            high = low
    else:
        largest = np.argpartition(avydata[ind],-2)[-2:] # two largest peaks
    
        if avydata[-1] > np.mean(avydata[-(width+1):-1]):#max(avydata[-(width+1):-1]):
            high = [avxdata[-1], avydata[-1]]
            low = [np.max(avxdata[ind[largest]]),\
                   avydata[ind[largest]][np.argmax(avxdata[ind[largest]])]]  # low x peak
        else:
            low = [np.min(avxdata[ind[largest]]),\
                   avydata[ind[largest]][np.argmin(avxdata[ind[largest]])]]  # low x peak
            high = [np.max(avxdata[ind[largest]]),\
                    avydata[ind[largest]][np.argmax(avxdata[ind[largest]])]] # high x peak

    # remove virtual points in avxdata and avydata
    return low, high, avydata[1:-1], avxdata[1:-1]



# =============================================================================
# DATA CONVERSION
# =============================================================================
def mesh_to_coords(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    r'''
    Converts positions in two-dimensional space given as a meshgrid into an
    array of two-dimensional position vectors of shape (N,2), where N is the
    total number of positions in the given meshgrid.

    Parameters
    ----------
    X : np.ndarray
        Two-dimensional array of x positions on a grid in form of a meshgrid.
    Y : np.ndarray
        Two-dimensional array of y positions on a grid in form of a meshgrid.

    Returns
    -------
    np.ndarray
        Array of shape (N,2) containing the position vector of each grid point.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(-1,1,3)
    >>> y = np.linspace(-2,2,5)
    >>> X,Y = np.meshgrid(x,y)
    >>> print(X)
    [[-1.  0.  1.]
     [-1.  0.  1.]
     [-1.  0.  1.]
     [-1.  0.  1.]
     [-1.  0.  1.]]
    >>> print(Y)
    [[-2. -2. -2.]
     [-1. -1. -1.]
     [ 0.  0.  0.]
     [ 1.  1.  1.]
     [ 2.  2.  2.]]
    >>> print(amep.utils.mesh_to_coords(X,Y))
    [[-1. -2.]
     [ 0. -2.]
     [ 1. -2.]
     [-1. -1.]
     [ 0. -1.]
     [ 1. -1.]
     [-1.  0.]
     [ 0.  0.]
     [ 1.  0.]
     [-1.  1.]
     [ 0.  1.]
     [ 1.  1.]
     [-1.  2.]
     [ 0.  2.]
     [ 1.  2.]]
    >>> 
    
    '''
    return np.vstack([X.ravel(), Y.ravel()]).T


def sq_from_gr(
        r: np.ndarray, gr: np.ndarray, q: np.ndarray, rho: float,
        twod: bool = True) -> np.ndarray:
    r'''
    Compute the static structure factor as fourier transform of the pair 
    correlation function.

    Notes
    -----
    The relation between g(r) and S(q) is given by (here in 3D; see Ref. [1]_
    for further information)

    .. math::
        S(q) = 1 + \frac{4\pi\rho}{q}\int\limits_0^\infty dr\,r\sin(qr)(g(r)-1)

    References
    ----------

    .. [1] Yarnell, J. L., Katz, M. J., Wenzel, R. G., & Koenig, S. H. (1973).
       Structure Factor and Radial Distribution Function for Liquid Argon at
       85 °K. Physical Review A, 7(6), 2130–2144.
       https://doi.org/10.1103/PhysRevA.7.2130

    Parameters
    ----------
    r : np.ndarray
        Distances.
    gr : np.ndarray
        Radial pair-distribution function.
    q : np.ndarray
        Wave numbers.
    rho : float
        Number density.
    twod : bool, default=True
        If True, the slightly different formula for the two-dimensional case
        is used (which includes a Bessel function and can be derived by
        using zylindrical coordinates and calculating the angular integral by
        setting w.l.o.g. q parallel to e_x.)

    Returns
    -------
    np.ndarray
        Isotropic static structure factor.

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> rdf = amep.evaluate.RDF(traj, nav=10, skip=0.8, njobs=4)
    >>> boxlength = np.max(traj[0].box[:,1] - traj[0].box[:,0])
    >>> q = np.arange(2*np.pi/boxlength, 20.0, 2*np.pi/boxlength)
    >>> S = amep.utils.sq_from_gr(
    ...     rdf.r, rdf.avg, q, traj[0].density(), twod=True
    ... )
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(q, S)
    >>> axs.axhline(0.0, ls='--', c='k')
    >>> axs.set_xlabel(r'$q$')
    >>> axs.set_ylabel(r'$S(q)$')
    >>> axs.set_ylim(-1, 10)
    >>> fig.savefig('./figures/utils/utils-sq_from_gr.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-sq_from_gr.png
      :width: 400
      :align: center
    
    '''
    if twod:
        ydata = ((gr - 1) * r).reshape(-1, 1) * special.jv(0, r.reshape(-1, 1) * q.reshape(1, -1))
        if Version(np.__version__) < Version("2.0.0"):
            return np.trapz(x=r, y=ydata, axis=0) * (2 * np.pi * rho) + 1
        return np.trapezoid(x=r, y=ydata, axis=0) * (2 * np.pi * rho) + 1
    else:
        ydata = ((gr - 1) * r).reshape(-1, 1) * np.sin(r.reshape(-1, 1) * q.reshape(1, -1))
        if Version(np.__version__) < Version("2.0.0"):
            return np.trapz(x=r, y=ydata, axis=0) * (4 * np.pi * rho / q) + 1
        return np.trapezoid(x=r, y=ydata, axis=0) * (4 * np.pi * rho / q) + 1
    

def sq_from_sf2d(
        S: np.ndarray, Qx: np.ndarray, Qy: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Calculates the isotropic static structure factor from the two-dimensional
    static structure factor by making a spherical average/radial mean.

    Parameters
    ----------
    S : np.ndarray
        2D static structure factor.
    Qx : np.ndarray
        x-components of the k vectors (same shape as S; meshgrid).
    Qy : np.ndarray
        y-components of the k vectors (same shape as S; meshgrid).

    Returns
    -------
    sq : np.ndarray
        Isotropic static structure factor.
    q : np.ndarray
        Wave numbers.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> S, Kx, Ky = amep.spatialcor.sf2d(frame.coords(), frame.box)
    >>> Sq, q = amep.utils.sq_from_sf2d(S, Kx, Ky)
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(q, Sq)
    >>> axs.axhline(0.0, ls='--', c='k')
    >>> axs.set_xlabel(r'$q$')
    >>> axs.set_ylabel(r'$S(q)$')
    >>> axs.set_ylim(-1, 10)
    >>> fig.savefig('./figures/utils/utils-sq_from_sf2d.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-sq_from_sf2d.png
      :width: 400
      :align: center    

    '''
    # radii
    R = np.sqrt(Qx**2+Qy**2)
    
    # bin width
    dq = np.max([np.abs(Qx[0,1]-Qx[0,0]), np.abs(Qy[1,0]-Qy[0,0])])
    
    # distances in k space
    dists = np.arange(0, np.min([np.max(Qx), np.max(Qy)]), dq)

    sq, q = np.histogram(R, weights=S, bins=dists, density=False)
    q = runningmean(q, 2)
    sq = sq/np.histogram(R, bins=dists)[0]
    return sq, q


def msd_from_vacf(vacf, times):
    r'''
    Calculates the mean square displacement from the integral of
    the velocity autocorrelation function.

    Parameters
    ----------
    vacf : np.ndarray
        Velocity autocorrelation function as 1D array.
    times : np.ndarray
        Corresponding time values as 1D array.

    Returns
    -------
    msd : np.ndarray
        Mean square displacement as 1D array.
    times : np.ndarray
        Time values as 1D array.

    '''
    if Version(np.__version__) < Version("2.0.0"):
        msd = 2*np.array([np.trapz((times[i]-times[:i+1])*vacf[:i+1], x=times[:i+1]) for i in range(len(times))])
    else:
        msd = 2*np.array([np.trapezoid((times[i]-times[:i+1])*vacf[:i+1], x=times[:i+1]) for i in range(len(times))])
    return msd, times


def domain_length(
        s_fac: np.ndarray, q: np.ndarray, qmin: float | None = None,
        qmax: float | None = None) -> float:
    r'''
    Calculate the domain length from the structure factor. Takes the structure
    factor as a function of the wave numbers as well as the corresponding wave
    numbers.

    Notes
    -----
    The domain length is defined as

    .. math::

        L(t) = 2\pi\frac{\int_{q_{\rm min}}^{q_{\rm max}}{\rm d}q\,S(q,t)}{\int_{q_{\rm min}}^{q_{\rm max}}{\rm d}q\,qS(q,t)}

    where :math:`q=n \frac{2\pi}{L}` with :math:`n\in \mathbb{N}` and box legth :math:`L`.
    It has been used in Refs. [1]_ [2]_ [3]_ [4]_ for example.

    References
    ----------

    .. [1] R. Wittkowski, A. Tiribocchi, J. Stenhammar, R. J. Allen,
           D. Marenduzzo, and M. E. Cates, Scalar Φ4 Field Theory for
           Active-Particle Phase Separation, Nat. Commun. 5, 4351 (2014).
           https://doi.org/10.1038/ncomms5351

    .. [2] A. K. Omar, K. Klymko, T. GrandPre, P. L. Geissler, and J. F. Brady,
           Tuning Nonequilibrium Phase Transitions with Inertia,
           J. Chem. Phys. 158, 42 (2023).
           https://doi.org/10.1063/5.0138256

    .. [3] J. Stenhammar, D. Marenduzzo, R. J. Allen, and M. E. Cates,
           Phase Behaviour of Active Brownian Particles: The Role of
           Dimensionality, Soft Matter 10, 1489 (2014).
           https://doi.org/10.1039/C3SM52813H

    .. [4] V. M. KENDON, M. E. CATES, I. PAGONABARRAGA, J.-C. DESPLAT,
           and P. BLADON, Inertial Effects in Three-Dimensional Spinodal
           Decomposition of a Symmetric Binary Fluid Mixture: A Lattice
           Boltzmann Study, J. Fluid Mech. 440, 147 (2001).
           https://doi.org/10.1017/S0022112001004682


    Parameters
    ----------
    s_fac: np.ndarray
        structure factor corresponding to the given q-values
    q: np.ndarray
        The wave number corresponding to the structure factor
    qmin : float or None, optional
        Lower integration limit. The default is None.
    qmax : float or None, optional
        Upper integration limit. The default is None.

    Returns
    -------
    l: float
        Domain length as inverse expectation value of q.

    
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
    >>> frame = traj[3]
    >>> C = frame.data('c')
    >>> X, Y = frame.grid
    >>> sq2d, qx, qy = amep.continuum.sf2d(C, X, Y)
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(axs, C, X, Y)
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r'$c(x,y)$'
    ... )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-domain_length_1.png')
    >>>
    
    .. image:: /_static/images/utils/utils-domain_length_1.png
      :width: 400
      :align: center   
      
    >>> sq, q = amep.utils.sq_from_sf2d(sq2d, qx, qy)
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(q[1:], sq[1:])
    >>> axs.set_xlabel(r'$q$')
    >>> axs.set_ylabel(r'$S(q)$')
    >>> fig.savefig('./figures/utils/utils-domain_length_2.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-domain_length_2.png
      :width: 400
      :align: center  
      
      
    >>> L = amep.utils.domain_length(sq, q)
    >>> print(L)
    10.400560044952304
    >>> 
    
    '''
    # apply integration limits if given
    if qmin is not None:
        s_fac = s_fac[q>=qmin]
        q = q[q>=qmin]
    if qmax is not None:
        s_fac = s_fac[q<=qmax]
        q = q[q<=qmax]
    
    if Version(np.__version__) < Version("2.0.0"):
        return 2*np.pi*(np.trapz(s_fac, x=q)/np.trapz(s_fac*q, x=q))
    return 2*np.pi*(np.trapezoid(s_fac, x=q)/np.trapezoid(s_fac*q, x=q))


# =============================================================================
# FOURIER TRANSFORM
# =============================================================================
def rfft(
        data: np.ndarray, delta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Real fast Fourier transform. Requires the corresponding time values
    to be equally spaced with distance delta.

    Parameters
    ----------
    data : np.ndarray
        Real data to transform as array of shape (N,).
    delta : float, optional
        Spacing of the corresponding x (time) values to calculated correct 
        frequencies. The default is 1.0.

    Returns
    -------
    ft : np.ndarray
        Fourier-transformed data.
    fq : np.ndarray
        Corresponding frequencies.

    '''
    #TODO: Use scipy.fft.rfft (can be parallelized!)
    ft = np.fft.rfft(data)
    fq = np.fft.fftfreq(len(ft), d=delta)
    return ft, fq


def spectrum(xdata, ydata):
    r'''
    Calculates the spectrum of one-dimensional real data by using fft.
    Assumes equally spaced xdata.

    Parameters
    ----------
    xdata : np.ndarray
        Equally spaced x-values as array of shape (N,).
    ydata : np.ndarray
        Corresponding y-values as array of shape (N,).

    Returns
    -------
    np.ndarray
        Spectrum.
    np.ndarray
        Frequencies.
    
    
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(0,10,500)
    >>> y = np.sin(2*np.pi*2*x) + 0.5*np.cos(2*np.pi*x)
    >>> s, f = amep.utils.spectrum(x, y)
    >>> fig, axs = amep.plot.new(nrows=2, figsize=(3,3))
    >>> axs[0].plot(x,y)
    >>> axs[0].set_xlabel('Time')
    >>> axs[0].set_ylabel('Data')
    >>> axs[1].plot(f, s)
    >>> axs[1].set_xlabel('Frequency')
    >>> axs[1].set_ylabel('Spectrum')
    >>> fig.savefig('./figures/utils/utils-spectrum.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-spectrum.png
      :width: 400
      :align: center 

    '''
    delta = xdata[1]-xdata[0]
    ft, fq = rfft(ydata, delta=delta)
    return np.abs(ft[:len(ft)//2])**2, fq[:len(ft)//2]


# =============================================================================
# MISC
# =============================================================================
def log_indices(first: int, last: int, num: int = 100) -> np.ndarray:
    r'''
    Creates logarithmically spaced indices.

    Parameters
    ----------
    first : int
        First index.
    last : int
        Last index.
    num : int, optional
        Number of indices. The default is 100.

    Returns
    -------
    np.ndarray
        Indices.

    '''
    ls = np.logspace(0, np.log10(last - first + 1), num=num)
    return np.unique(np.int_(ls) - 1 + first)


def envelope(
        f: np.ndarray, x: np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Calculates the envelope of the data f by determining all local peaks
    of the data.

    Parameters
    ----------
    f : np.ndarray
        Data values.
    x : np.ndarray, optional
        x values (same shape as f). If not specified, x is just a list of
        of indices from 0 to len(f)-1. The default is None.

    Returns
    -------
    np.ndarray
        Envelope of f.
    x : np.ndarray
        Corresponding x values (same shape as envelope.
                                
    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(0,10*np.pi,1000)
    >>> y = np.sin(10*x) + np.sin(0.25*x)
    >>> yenv, xenv = amep.utils.envelope(y, x=x)
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(x, y, label='data')
    >>> axs.plot(
    ...     xenv, yenv, label='envelope',
    ...     marker='', c='orange', lw=2
    ... )
    >>> axs.legend()
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-envelope.png')
    >>> 
    
    .. image:: /_static/images/utils/utils-envelope.png
      :width: 400
      :align: center

    '''
    
    peakind = signal.find_peaks(f, distance=1)[0]
    
    if x is None:
        x = np.arange(len(peakind))
    else:
        x = x[peakind]
    
    return f[peakind], x


def dimension(coords: np.ndarray, verbose: bool = False) -> int:
    r'''
    Returns the spatial dimension of simulation data. Requires a coordinate
    frame of three spatial dimension with coordinates of the unused dimensions
    set to a constant value.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Returns
    -------
    dim : int
        Spatial dimension.

    '''
    # check coords shape
    correct_shape = len(coords.shape)==2
    
    # check spatial dimension
    dim = None
    if correct_shape:
        if np.shape(coords)[1]==2:
            dim = 2
        elif np.shape(coords)[1]==3:
            dim = 0
            for i in range(3):
                if not np.all(coords[:,i] == coords[0,i]):
                    dim += 1

            if dim==0 and np.shape(coords)==(1,3):
                dim = 3 # revert to 3d if only one particle

            # check for correct shape:
            # x or y component must always be set. only z is optional
            if dim==0 or dim==1 or\
            (dim==2 and not np.all(coords[:,2] == coords[0,2])):
                if verbose:
                    _log.info(
                        "Coordinate shape may not be supported. Only (x,y), "\
                        "(x,y,z) or (x,y,a) with constant a are supported."
                    )
                # this warning may also be thrown if all coordinates are the same.
        else:
            correct_shape = False
    if not correct_shape:
        raise ValueError(f"coordinates do not have a known shape. {np.shape(coords)}")
    return dim


def weighted_runningmean(
        data, nav: int, kernel: str | np.ndarray = "homogenous",
        mode: str = 'valid', width: float = 1.) -> np.ndarray:
    r'''
    Compute the weighted running mean of a 1-dimensional array.

    The weights can be specified as a kernel matrix,
    or one of the preset variants.

    Parameters
    ----------
    data : np.ndarray
        input data of shape (N, )
    nav : int
        length of the kernel if standard is chosen
    kernel : str|np.ndarray
        Kind of kernel to be used for weights.
        Possible are homogenous, triangle and gauss.
        Provided a fitting array will also use
        normed version of this to weight.
        Weight array should have positive entries.
        Otherwise use np.convolve directly.
    mode : str
        'same', 'valid', or 'full'
    width : float
        width of the gaussian. Only effective when using gaussian kernel.

    Returns
    -------
    np.ndarray of shape (N-(nav-1), )
        averaged data

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(0,2*np.pi,500)
    >>> y = np.sin(x)*(1+np.random.rand(len(x)))
    >>> yav = amep.utils.weighted_runningmean(
    ...     y, kernel='triangle', nav=20
    ... )
    >>> xav = amep.utils.weighted_runningmean(
    ...     x, kernel='homogenous', nav=20
    ... )
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(x, y, label='data')
    >>> axs.plot(
    ...     xav, yav, label='weighted running mean',
    ...     marker='', c='orange'
    ... )
    >>> axs.legend()
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-weighted_runningmean.png')
    >>>
    
    .. image:: /_static/images/utils/utils-weighted_runningmean.png
      :width: 400
      :align: center
    
    '''
    match kernel:
        case "homogenous":
            kernel = np.ones((nav,))
        case "triangle":
            kernel = nav/2-abs(np.arange(-nav/2, nav/2))
        case "gauss":
            kernel = gaussian(np.arange(nav+1), nav/2, width*nav/2,)
    if isinstance(kernel, np.ndarray):
        normed_kernel = kernel / np.sum(kernel)
    else:
        raise ValueError(
            "The chosen kernel is not available. Choose one of "\
            "['homogenous', 'triangle', 'gauss'] or supply a custom kernel "\
            "as np.ndarray."
        )
    return np.convolve(data, normed_kernel, mode=mode)


def weighted_runningmean2d(
        data: np.ndarray, nav: int, kernel: str | np.ndarray = "homogenous",
        mode: str = 'valid', width: float = 1.) -> np.ndarray:
    r'''Compute the running mean of two-dimensional data.

    This is of a 2d density field.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (N1,N2,).
    nav : int
        length of the kernel if standard is chosen
    kernel : str|np.ndarray
        Kind of kernel to be used for weights.
        Possible are homogenous, triangle and gauss.
        Provided a fitting array will also use
        normed version of this to weight.
        Weight array should have positive entries.
        Otherwise use np.convolve directly.
    mode : str
        'same', 'valid', or 'full'
    width : float
        width of the gaussian. Only effective when using gaussian kernel.

    Returns
    -------
    np.ndarray
        Averaged data.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> a = np.zeros((7,7))
    >>> a[3,3] = 1
    >>> print(a)
    [[0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]]
    >>> print(amep.utils.weighted_runningmean2d(a, 3, mode='valid'))
    [[0.         0.         0.         0.         0.        ]
     [0.         0.11111111 0.11111111 0.11111111 0.        ]
     [0.         0.11111111 0.11111111 0.11111111 0.        ]
     [0.         0.11111111 0.11111111 0.11111111 0.        ]
     [0.         0.         0.         0.         0.        ]]
    >>>

    '''
    match kernel:
        case "homogenous":
            kernel = np.ones((nav, nav,))
        case "circle":
            x_axe = np.arange(-nav/2+.5, nav/2)
            y_axe = np.arange(-nav/2+.5, nav/2)
            x_v, y_v = np.meshgrid(x_axe, y_axe)
            kernel = np.where(x_v**2+y_v**2 < (nav/2)**2,
                              np.ones_like(x_v),
                              np.zeros_like(y_v))
        case "gauss":
            x_axe = np.arange(-nav/2, nav/2+.5)
            y_axe = np.arange(-nav/2, nav/2+.5)
            x_v, y_v = np.meshgrid(x_axe, y_axe)
            kernel = gaussian((x_v**2+y_v**2)**.5, 0, width)
        case "cone":
            x_axe = np.arange(-nav/2, nav/2+.5)
            y_axe = np.arange(-nav/2, nav/2+.5)
            x_v, y_v = np.meshgrid(x_axe, y_axe)
            kernel = np.where(x_v**2+y_v**2 < (nav/2)**2,
                              nav/2-(x_v**2+y_v**2)**.5,
                              np.zeros_like(y_v))

    if isinstance(kernel, np.ndarray):
        normed_kernel = kernel/np.sum(kernel)
    else:
        raise ValueError(
            "The chosen kernel is not avaliable. Use one of ['homogeneous', "\
            "'circle', 'gauss', 'cone'] or supply a custom kernel as "\
            "np.ndarray."
        )
        #normed_kernel = np.ones((nav, nav,))/nav**2
    return convolve2d(data, normed_kernel, mode=mode)


def segmented_mean2d(
        data: np.ndarray, n: int, verbose: bool = False
        ) -> np.ndarray:
    r'''Compute the mean of n*n adjacent data points.

    Pools n*n square of data points into one bin.
    Creates array of data.shape/n bins.
    The used data may be reduced in case the length is not divisable by n.

    Parameters
    ----------
    data : np.ndarray
        Input data to be averaged.
    n : int
        Edge length of the square summed over
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Returns
    -------
    np.ndarray
        Averaged data
    
    Example
    ------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
    >>> frame = traj[-1]
    >>> X, Y = frame.grid
    >>> C = frame.data('c')
    >>> X_COARSE = amep.utils.segmented_mean2d(X, 6)
    >>> Y_COARSE = amep.utils.segmented_mean2d(Y, 6)
    >>> C_COARSE = amep.utils.segmented_mean2d(C, 6)
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(axs, C, X, Y)
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r'$c(x,y)$'
    ... )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/utils/utils-segmented_mean2d.png')
    >>>
    
    .. image:: /_static/images/utils/utils-segmented_mean2d.png
      :width: 400
      :align: center    

    '''
    if len(data) % n != 0 and verbose:
        _log.info(
            f"Data length not divisible by {n}. "\
            f"Proceeding by omitting the last {len(data)%n} steps"
        )
    x_length = data.shape[1]
    y_length = data.shape[0]
    cut_data = data[:x_length-(x_length % n), :y_length - (y_length % n)]
    return cut_data.reshape(int(x_length/n), n,
                            -1, n).mean(axis=3).mean(axis=1)
    # [:len(a)-(len(a)%m)] reduces array length by proper amount
    # but also works if no elements are to be removed


def group_jobs(joblist: list, nworkers: int) -> list:
    r'''
    Groups the given jobs into groups of length `nworkers`. This method is used
    for the `amep.utils.compute_parallel` method.

    Parameters
    ----------
    joblist : list
        List of jobs.
    nworkers : int
        Distribute the jobs to this number of workers.

    Returns
    -------
    jobgroups : list of list
        Grouped jobs.

    '''
    jobgroups = []
    for i in range((len(joblist)-len(joblist)%nworkers)//nworkers + 1):
        imin = i*nworkers
        imax = (i+1)*nworkers
        if imax < len(joblist):
            jobgroups.append(joblist[imin:imax])
        elif imin < len(joblist):
            jobgroups.append(joblist[imin:])
    return jobgroups


def compute_parallel(
        func: callable, chunks: list | np.ndarray | tuple, *args,
        njobs: int = 2, verbose: bool = False, **kwargs) -> list:
    r'''
    Calls the given function for all given chunks in parallel with a given 
    number of parallel jobs.

    Parameters
    ----------
    func : callable
        Function to call for each chunk. The first argument of `func` must be
        the chunk.
    chunks : list or np.ndarray or tuple
        Iterable of chunks.
    *args :
        Additional arguments forwarded to `func`.
    njobs : int, optional
        Number of workers used for the parallelization. If this number exceeds
        the number of CPU cores, it is set to the number of available CPU
        cores. The default is 2.
    verbose : bool, optional
        If True, a progress bar is shown. The default is False.
    **kwargs :
        Other keyword arguments are forwarded to `func`.

    Returns
    -------
    results : list
        List of results from each of the workers.

    '''    
    # check number of jobs for parallelization
    if njobs > os.cpu_count():
        njobs = os.cpu_count()

    # setup multiprossing environment
    execution = ProcessPoolExecutor(max_workers = njobs)
    
    # create list of jobs (one job for each chunk)
    joblist = [chunk for chunk in chunks]

    # create groups of jobs
    jobgroups = group_jobs(joblist, njobs)

    # main loop
    results = []
    for group in tqdm(jobgroups, disable = not verbose):
        
        # create list of jobs that should run in parallel
        jobs = []
        for chunk in group:
                        
            task = execution.submit(
                func,
                chunk,
                *args,
                **kwargs
            )
            
            # append to list of jobs
            jobs.append(task)
            
        try:
            # loop checking status of jobs every 5 ms - it counts the number
            # of jobs that are done already and stops if all jobs are done
            # we cannot use future.result or concurrent.futures.wait because
            # they act like a time.sleep with the time all workers need to be
            # finished - hence, we use a simple loop to check the status 
            # frequently which finally allows us to cancel all child processes
            # with a keyboard interrupt
            running = True
            while running:
                n = 0
                for future in jobs:
                    if future.done():
                        n += 1    
                if n == len(jobs):
                    # all jobs done
                    running = False
                time.sleep(0.005)
        except KeyboardInterrupt:
            # kill all child processes immediately
            for pid, process in execution._processes.items():
                process.kill()
            raise
        else:
            # collect results if no exception (called here and not within the
            # try block because task.result waits until the task has been
            # finished and during that time, a keyboard interrupt is ignored
            # prevenint the user from cancelling the jobs)
            for task in jobs:
                results.append(task.result())

    return results


def optimal_chunksize(
        data_length: int, number_of_elements: int,
        bit_per_element: int = 32, buffer: float = 0.25,
        maxmem: float = MAXMEM) -> int:
    r'''
    Estimates an optimal chunksize for limiting the RAM usage.

    Notes
    -----
    The memory estimate in GB for the computation with one chunk can be written
    `chunksize*number_of_elements*bit_per_element/8/1e9+buffer`. Setting it
    equal to `maxmem` leads to the estimate of `chunksize`.

    Parameters
    ----------
    data_length : int
        Length of the data to be chunked.
    number_of_elements : int
        Number of (array) elements to be stored for each chunk.
    bit_per_element : int, optional
        Number of bits per element, e.g., 32 for float32 numbers or 64 for
        float64 numbers. The default is 32.
    buffer : float, optional
        Additional buffer to avoid filling up the RAM. The default is 0.25.
    maxmem : float, optional
        Maxmimum RAM usage per CPU core. The default is MAXMEM.

    Returns
    -------
    int
        Chunksize.

    '''
    # limit to maxmem per CPU core
    # (memory estimate in GB for float32: 2*chunksize*N*32/8/1e9+buffer,
    # where chunksize*N is the size of the distance matrix and the factor
    # 2 is needed due to copying numpy arrays)
    # additionally we add some buffer
    chunksize = int((maxmem-buffer)*8e9/bit_per_element/number_of_elements)
    return chunksize


def quaternion_rotate(quat: np.ndarray, vec: np.ndarray):
    r'''
    Calculates the 3d vector rotated by the quaternion.

    Notes
    -----
    Method from http://people.csail.mit.edu/bkph/articles/Quaternions.pdf

    Parameters
    ----------
    quat : np.ndarray
        4d quaternion the vector `vec` will be rotated with.
        Shapes (4,) or (N,4,) allowed.
    vec : np.ndarray
        3d vector that will be rotated by the quaternion `quat`.
        Shapes (3,) or (N,3,) allowed.

    Returns
    -------
    np.ndarray
        Rotated 3d vector.

    '''
    if len(np.shape(quat))==2:
        return ((quat[:,0] * quat[:,0] - np.sum(quat[:,1:]*quat[:,1:], axis=1))[:,None] * vec 
            + 2 * quat[:,0,None] * np.cross(quat[:,1:], vec) 
            + 2 * np.sum(quat[:,1:]*vec, axis=1)[:,None] * quat[:,1:])
    return ((quat[0] * quat[0] - np.dot(quat[1:], quat[1:])) * vec 
        + 2 * quat[0] * np.cross(quat[1:], vec) 
        + 2 * np.dot(quat[1:], vec) * quat[1:])


def quaternion_conjugate(quat : np.ndarray):
    r'''
    Calculates the conjugated quaternion.

    Parameters
    ----------
    quat : np.ndarray
        4d quaternion that will be conjugated.
        Shapes (4,) or (N,4,) allowed.

    Returns
    -------
    np.ndarray
        Conjugated quaternion.

    '''
    return quat*np.array([1,-1,-1,-1])


def quaternion_multiply(a : np.ndarray, b : np.ndarray):
    r'''
    Calculates the multiplication of two quaternions.
    Quaternion multiplication is not commutative.

    Parameters
    ----------
    a : np.ndarray
        4d quaternion.
        Shapes (4,) or (N,4,) allowed.
    b : np.ndarray
        4d quaternion.
        Shapes (4,) or (N,4,) allowed.

    Returns
    -------
    np.ndarray
        Multiplied quaternion.
        Shapes (4,) or (N,4,) respectively to input.

    '''
    ab=np.empty(np.shape(a))
    if len(np.shape(a))==2:
        ab[:,0]=((a[:,0] * b[:,0]) - np.sum(a[:,1:]*b[:,1:], axis=1))
        ab[:,1:]=(a[:,0,None] * b[:,1:]
                + a[:,1:] * b[:,0,None]
                + np.cross(a[:,1:], b[:,1:]))
        return ab
    ab[0]=(a[0] * b[0] - np.dot(a[1:], b[1:]))
    ab[1:]=(a[0] * b[1:]
            + a[1:] * b[0]
            + np.cross(a[1:], b[1:]))
    return ab



# =============================================================================
# INTERACTION POTENTIALS
# =============================================================================
def wca(r: float | np.ndarray, eps: float = 10.0, sig: float = 1.0):
    """
    Weeks-Chandler-Anderson (WCA) potential. [1]_
    
    References
    ----------
    
    .. [1] J. D. Weeks, D. Chandler, and H. C. Andersen, Role of Repulsive 
       Forces in Determining the Equilibrium Structure of Simple Liquids, 
       J. Chem. Phys. 54, 5237 (1971). https://doi.org/10.1063/1.1674820

    Parameters
    ----------
    r : float | np.ndarray
        Distances.
    eps : float, optional
        Strength of the potential. The default is 10.0.
    sig : float, optional
        Effective particle diameter. The default is 1.0.

    Returns
    -------
    epot : float | np.ndarray
        Potential energy.

    """
    rcut = 2**(1/6)*sig
    epot = np.where(r<=rcut, 4*eps*((sig/r)**12-(sig/r)**6)+eps, 0)
    return epot

def dr_wca(r: float | np.ndarray, eps: float = 10.0, sig: float = 1.0):
    """
    First derivative of the Weeks-Chandler-Anderson (WCA) potential. [1]_
    
    References
    ----------
    
    .. [1] J. D. Weeks, D. Chandler, and H. C. Andersen, Role of Repulsive 
       Forces in Determining the Equilibrium Structure of Simple Liquids, 
       J. Chem. Phys. 54, 5237 (1971). https://doi.org/10.1063/1.1674820

    Parameters
    ----------
    r : float | np.ndarray
        Distances.
    eps : float, optional
        Strength of the potential. The default is 10.0.
    sig : float, optional
        Effective particle diameter. The default is 1.0.

    Returns
    -------
    dr : float | np.ndarray
        First derivative.

    """
    rcut = 2**(1/6)*sig
    dr = np.where(r<=rcut, 4*eps*(6*sig**6/r**7 - 12*sig**12/r**13), 0)
    return dr

def dr2_wca(r: float | np.ndarray, eps: float = 10.0, sig: float = 1.0):
    """
    Second derivative of the Weeks-Chandler-Anderson (WCA) potential. [1]_
    
    References
    ----------
    
    .. [1] J. D. Weeks, D. Chandler, and H. C. Andersen, Role of Repulsive 
       Forces in Determining the Equilibrium Structure of Simple Liquids, 
       J. Chem. Phys. 54, 5237 (1971). https://doi.org/10.1063/1.1674820

    Parameters
    ----------
    r : float | np.ndarray
        Distances.
    eps : float, optional
        Strength of the potential. The default is 10.0.
    sig : float, optional
        Effective particle diameter. The default is 1.0.

    Returns
    -------
    dr2 : float | np.ndarray
        Second derivative.

    """
    rcut = 2**(1/6)*sig
    dr2 = np.where(r<=rcut, 4*eps*(156*sig**12/r**14 - 42*sig**6/r**8), 0)
    return dr2