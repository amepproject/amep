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
Positional and Orientational Order Analysis
===========================================

.. module:: amep.order

The AMEP module :mod:`amep.order` contains methods for calculating positional and
orientational order parameters, analyzing the nearest and next neighbors, and
voronoi tesselation of particle-based simulation data.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import numpy as np

from scipy import spatial
from .pbc import pbc_points, mirror_points
from .utils import dimension
from .base import get_module_logger

# logger setup
_log = get_module_logger(__name__)

# =============================================================================
# VORONOI TESSELATION
# =============================================================================
def voronoi(
        coords: np.ndarray, box_boundary: np.ndarray, pbc: bool = True, 
        width: float | None = .5, closed: bool = True,
        enforce_nd: int | None = None, verbose: bool = False
        ) -> spatial.Voronoi:
    r'''    
    Calculates the Voronoi tesselation for given coordinates by using the scipy
    package (see Ref. [1]_). For periodic boundary conditions, a part of the
    box with partial width is periodically repeated around the original box.
    Periodic mirrors are created in case of non-periodic boundary conditions to
    create the correct closed voronoi diagram.
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html 
    
    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates of the particles for the voronoi tesselation.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    pbc : bool or None, optional
        Specifier whether periodic boundary conditions are used (True) or not
        (False). The default is True.
    width : float, optional
        Relative width (to the box) of the periodic images or mirrors used.
        For very low densities, this value might have to be set to a larger value
        than the default. For higher densities, this value can easily be set
        to 0.3 or 0.1 or even smaller. Please check the Voronoi plot to ensure
        the correct calculation.
        Range: [0,1]. Default is 0.5.
    closed : bool, optional
        If True, the closed Voronoi diagram is calculated using the mirrored
        boxes. If False, the open Voronoi diagram is returned which can result
        in unphysical measures of number of neighbors and local density.
        This keyword is ignored if pbc is True. The default is True.
    enforce_nd: int, None, optional
        enforces the number of dimensions. 2 for 2d, 3 for 3d.
        If None is supplied, a best guess is used by checking
        if all particles have the same dimension in the last
        coordinate. See `utils.dimension()`
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Returns
    -------
    scipy.spatial.voronoi
        scipy Voronoi object
    np.ndarray or None
        Indices of the points within the original box (None for opened Voronoi
        diagram or width=0.0).
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> vor, idx = amep.order.voronoi(
    ...     frame.coords(), frame.box, pbc=True
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.voronoi(
    ...     axs, vor, show_vertices=False,
    ...     line_colors='orange', line_width=1,
    ...     line_alpha=0.6, point_size=1, c="b"
    ... )
    >>> amep.plot.box(axs, frame.box)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> axs.set_xlim(frame.box[0])
    >>> axs.set_ylim(frame.box[1])
    >>> fig.savefig('./figures/order/order-voronoi.png')

    .. image:: /_static/images/order/order-voronoi.png
      :width: 400
      :align: center

    '''
    if not pbc and not closed:
        # for standard open Voronoi diagram
        points = coords
        if verbose:
            _log.info(
                "The open Voronoi diagram is calculated. Note that it can "\
                "lead to unphysical results for the number of next neighbors "\
                "and the local density."
            )
    elif not pbc and closed:
        # for closed Voronoi diagram (using mirror images)
        points_extented = mirror_points(
            coords, box_boundary=box_boundary, width=width
        )
        points = points_extented
    elif pbc:
        # for considering periodic boundary conditions
        if not closed and verbose:
            _log.info(
                "The closed keyword is ignored for pbc=True."
            )
        points_extented = pbc_points(
            coords, box_boundary, width=width
        )
        points = points_extented
    else:
        raise ValueError(
            "amep.order.voronoi: Invalid value for pbc or closed. Only True "\
            "or False are allowed."
        )
    # Differentiate the Voronoi computation for 3d and 2d.
    if enforce_nd==2 or dimension(points)==2:
        # 2d calculation
        vor = spatial.Voronoi(points[:,:2])
    elif enforce_nd==3 or dimension(points)==3:
        # 3d calculation
        vor = spatial.Voronoi(points[:,:])
    else:
        raise ValueError(
            "Coordinates may have incorrect shape. Try to use 'enforce_nd'."
        )
    return vor, np.arange(len(coords))


# =============================================================================
# LOCAL DENSITY
# =============================================================================
def local_number_density(
        coords : np.ndarray, box_boundary : np.ndarray,
        radius : float | np.ndarray,
        other_coords : np.ndarray | None = None,
        rmax : float = 5.0, pbc : bool = True,
        enforce_nd : int | None = None, verbose : bool = False) -> np.ndarray:
    r'''
    Calculates the local number density based on averages over circles with
    radius rmax. Fractional neighbor counting is used instead of calculating
    the exact volume or area overlaps between the particles and the circle. A
    linear mapping is used (as in Ref. [1]_).

    References
    ----------

    .. [1] https://freud.readthedocs.io/en/latest/modules/density.html#freud.density.LocalDensity

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates at which the local density is calculated.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray of shape (M,3) or None, optional
        Coordinates of particles whose local density is calculated. If None,
        coords is used. The default is None.
    radius : float or np.ndarray
        Radius of the particles. If given as an array (e.g., for differently
        sized particles), the array must have the same shape as coords (or 
        other_coords if given).
    rmax : float, optional
        Radius of the sphere/circle over which the density is calculated.
        The default is 1.0.
    pbc : bool, optional
        If True, periodic boundary conditions are used. The default is True.
    enforce_nd : int or None, optional
        Enforce to perform the calculation in this number of dimensions. If 
        None, a suitable number of dimensions is guessed from the given coords.
        The default is None.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Raises
    ------
    ValueError
        Invalid input for radius, rmax, or enforce_nd.

    Returns
    -------
    np.ndarray of shape (N,)
        Local number density for each particle in coords.

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> ld = amep.order.local_number_density(
    ...     frame.coords(), frame.box, frame.radius(),
    ...     rmax=5.0, pbc=True
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius(),
    ...     values=ld
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label="local number density"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/order/order-local_number_density.png")

    .. image:: /_static/images/order/order-local_number_density.png
      :width: 400
      :align: center

    '''
    # check other coords
    if other_coords is None:
        other_coords = coords    
    
    # check radius
    if isinstance(radius, float):
        radius = np.repeat(radius, len(other_coords))
    elif not isinstance(radius, np.ndarray):
        raise ValueError(
            'amep.order.local_density: radius must either be a float or an '\
            f'array of the same length as coords. Got type {type(radius)}.'
        )
    elif len(radius) != len(other_coords) or len(radius.shape) != 1:
        raise ValueError(
            'amep.order.local_density: radius has invalid shape. Got shape '\
            f'{radius.shape} instead of ({other_coords.shape[0]},).'
        )
    # get box length
    box = box_boundary[:,1]-box_boundary[:,0]
    
    # check rmax
    if 2*rmax > np.max(box):
        raise ValueError(
            'amep.order.local_number_density: rmax must be smaller than half '\
            'of the largest box length.'
        )
    # check number of dimensions
    if enforce_nd is None:
        enforce_nd = dimension(coords)
        if verbose:
            _log.info(
                f"enforce_nd is not given. Guessing {enforce_nd} spatial "\
                "dimension(s) from the given coords. Set enforce_nd if you "\
                "would like to do the calculation for another number of "\
                "spatial dimensions."
            )
    elif not isinstance(enforce_nd, int):
        raise ValueError(
            'amep.order.local_nunmber_density: enforce_nd must be of type int'\
            f' or None. Got {type(enforce_nd)}.'
        )
    elif enforce_nd <= 0 or enforce_nd > 3:
        raise ValueError(
            'amep.order.local_nunmber_density: enforce_nd must be larger than'\
            f'0 and smaller than 3. Got {enforce_nd}.'
        )
    # get neighbors
    number, distances, neighbors = nearest_neighbors(
        coords,
        box_boundary,
        other_coords = other_coords,
        rmax = rmax + np.max(radius), # need to count neighbors within this
        pbc = pbc,                    # radius for fractional neighbor counting
        exclude = False
    )
    # fractional neighbor counting (linear mapping)
    number_density = np.zeros(len(coords))
    for i,num in enumerate(number):
        for j,d in enumerate(distances[i]):
            # get neighbor id
            nid = neighbors[i][j]
            if d <= rmax-radius[nid]:
                # full counting
                number_density[i] += 1.0
            elif rmax-radius[nid] < d < rmax+radius[nid]:
                # fractional counting
                frac = (rmax+radius[nid])/2/radius[nid] - d/2/radius[nid]
                number_density[i] += frac
                    
    # normalization
    if enforce_nd == 1:
        return number_density/rmax/2
    elif enforce_nd == 2:
        return number_density/np.pi/rmax**2
    elif enforce_nd == 3:
        return number_density/(4*np.pi*rmax**3/3)

    
def local_mass_density(
        coords : np.ndarray, box_boundary : np.ndarray,
        radius : float | np.ndarray, mass : float | np.ndarray,
        other_coords : np.ndarray | None = None,
        rmax : float = 5.0, pbc : bool = True,
        enforce_nd : int | None = None, verbose : bool = False) -> np.ndarray:
    r"""
    Calculates the local mass density based on averages over circles with
    radius rmax. Fractional neighbor counting is used instead of calculating
    the exact volume or area overlaps between the particles and the circle. A
    linear mapping is used (as in Ref. [1]_).
    
    References
    ----------
    
    .. [1] https://freud.readthedocs.io/en/latest/modules/density.html#freud.density.LocalDensity

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates at which the local density is calculated.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    radius : float or np.ndarray
        Radius of the particles. If given as an array (e.g., for differently
        sized particles), the array must have the same shape as coords (or 
        other_coords if given).
    mass : float or np.ndarray
        Mass of the particles. If given as an array (e.g., for differently
        sized particles), the array must have the same shape as coords (or 
        other_coords if given).
    other_coords : np.ndarray of shape (M,3) or None, optional
        Coordinates of particles whose local density is calculated. If None,
        coords is used. The default is None.
    rmax : float, optional
        Radius of the sphere/circle over which the density is calculated.
        The default is 1.0.
    pbc : bool, optional
        If True, periodic boundary conditions are used. The default is True.
    enforce_nd : int or None, optional
        Enforce to perform the calculation in this number of dimensions. If 
        None, a suitable number of dimensions is guessed from the given coords.
        The default is None.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Raises
    ------
    ValueError
        Invalid radius, mass, rmax, or enforce_nd.

    Returns
    -------
    np.ndarray of shape (N,)
        Local mass density for each particle in coords.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> ld = amep.order.local_mass_density(
    ...     frame.coords(), frame.box, frame.radius(),
    ...     frame.mass(), rmax=5.0, pbc=True
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius(),
    ...     values=ld
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label="local mass density"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/order/order-local_mass_density.png")

    .. image:: /_static/images/order/order-local_mass_density.png
      :width: 400
      :align: center

    """
    # check other coords
    if other_coords is None:
        other_coords = coords    
    
    # check radius
    if isinstance(radius, float):
        radius = np.repeat(radius, len(other_coords))
    elif not isinstance(radius, np.ndarray):
        raise ValueError(
            'amep.order.local_density: radius must either be a float or an '\
            f'array of the same length as coords. Got type {type(radius)}.'
        )
    elif len(radius) != len(other_coords) or len(radius.shape) != 1:
        raise ValueError(
            'amep.order.local_density: radius has invalid shape. Got shape '\
            f'{radius.shape} instead of ({other_coords.shape[0]},).'
        )
    # check radius
    if isinstance(mass, float):
        mass = np.repeat(mass, len(other_coords))
    elif not isinstance(mass, np.ndarray):
        raise ValueError(
            'amep.order.local_density: mass must either be a float or an '\
            f'array of the same length as coords. Got type {type(mass)}.'
        )
    elif len(mass) != len(other_coords) or len(mass.shape) != 1:
        raise ValueError(
            'amep.order.local_density: mass has invalid shape. Got shape '\
            f'{mass.shape} instead of ({other_coords.shape[0]},).'
        )
    # get box length
    box = box_boundary[:,1]-box_boundary[:,0]
    
    # check rmax
    if 2*rmax > np.max(box):
        raise ValueError(
            'amep.order.local_number_density: rmax must be smaller than half '\
            'of the largest box length.'
        )
    # check number of dimensions
    if enforce_nd is None:
        enforce_nd = dimension(coords)
        if verbose:
            _log.info(
                f"enforce_nd is not given. Guessing {enforce_nd} spatial "\
                "dimension(s) from the given coords. Set enforce_nd if you "\
                "would like to do the calculation for another number of "\
                "spatial dimensions." 
            )
    elif not isinstance(enforce_nd, int):
        raise ValueError(
            'amep.order.local_nunmber_density: enforce_nd must be of type int'\
            f' or None. Got {type(enforce_nd)}.'
        )
    elif enforce_nd <= 0 or enforce_nd > 3:
        raise ValueError(
            'amep.order.local_nunmber_density: enforce_nd must be larger than'\
            f'0 and smaller than 3. Got {enforce_nd}.'
        )
    # get neighbors
    number, distances, neighbors = nearest_neighbors(
        coords,
        box_boundary,
        other_coords = other_coords,
        rmax = rmax + np.max(radius), # need to count neighbors within this radius
        pbc = pbc,                    # for fractional neighbor counting!
        exclude = False
    )
    # fractional neighbor counting (linear mapping)
    mass_density    = np.zeros(len(coords))
    for i,num in enumerate(number):
        for j,d in enumerate(distances[i]):
            # get neighbor id
            nid = neighbors[i][j]
            if d <= rmax-radius[nid]:
                # full counting
                mass_density[i] += mass[nid]
            elif rmax-radius[nid] < d < rmax+radius[nid]:
                # fractional counting
                frac = (rmax+radius[nid])/2/radius[nid] - d/2/radius[nid]
                mass_density[i] += frac*mass[nid]
                    
    # normalization
    if enforce_nd == 1:
        return mass_density/rmax/2
    elif enforce_nd == 2:
        return mass_density/np.pi/rmax**2
    elif enforce_nd == 3:
        return mass_density/(4*np.pi*rmax**3/3)
    

def local_packing_fraction(
        coords : np.ndarray, box_boundary : np.ndarray,
        radius : float | np.ndarray,
        other_coords : np.ndarray | None = None,
        rmax : float = 5.0, pbc : bool = True,
        enforce_nd : int | None = None, verbose : bool = False) -> np.ndarray:
    r"""
    Calculates the local packing fraction based on averages over circles with
    radius rmax. Fractional neighbor counting is used instead of calculating
    the exact volume or area overlaps between the particles and the circle. A
    linear mapping is used (as in Ref. [1]_).
    
    References
    ----------
    
    .. [1] https://freud.readthedocs.io/en/latest/modules/density.html#freud.density.LocalDensity

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Coordinates at which the local density is calculated.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    radius : float or np.ndarray
        Radius of the particles. If given as an array (e.g., for differently
        sized particles), the array must have the same shape as coords (or 
        other_coords if given).
    other_coords : np.ndarray of shape (M,3) or None, optional
        Coordinates of particles whose local density is calculated. If None,
        coords is used. The default is None.
    rmax : float, optional
        Radius of the sphere/circle over which the density is calculated.
        The default is 1.0.
    pbc : bool, optional
        If True, periodic boundary conditions are used. The default is True.
    enforce_nd : int or None, optional
        Enforce to perform the calculation in this number of dimensions. If 
        None, a suitable number of dimensions is guessed from the given coords.
        The default is None.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Raises
    ------
    ValueError
        Invalid radius, rmax, or enforce_nd.

    Returns
    -------
    np.ndarray of shape (N,)
        Local packing fraction for each particle.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> ld = amep.order.local_packing_fraction(
    ...     frame.coords(), frame.box, frame.radius(),
    ...     rmax=5.0, pbc=True
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius(),
    ...     values=ld
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label="local packing fraction"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/order/order-local_packing_fraction.png")

    .. image:: /_static/images/order/order-local_packing_fraction.png
      :width: 400
      :align: center

    """
    # check other coords
    if other_coords is None:
        other_coords = coords    
    
    # check radius
    if isinstance(radius, float):
        radius = np.repeat(radius, len(other_coords))
    elif not isinstance(radius, np.ndarray):
        raise ValueError(
            'amep.order.local_density: radius must either be a float or an '\
            f'array of the same length as coords. Got type {type(radius)}.'
        )
    elif len(radius) != len(other_coords) or len(radius.shape) != 1:
        raise ValueError(
            'amep.order.local_density: radius has invalid shape. Got shape '\
            f'{radius.shape} instead of ({other_coords.shape[0]},).'
        )
    # get box length
    box = box_boundary[:,1]-box_boundary[:,0]
    
    # check rmax
    if 2*rmax > np.max(box):
        raise ValueError(
            'amep.order.local_number_density: rmax must be smaller than half '\
            'of the largest box length.'
        )
    # check number of dimensions
    if enforce_nd is None:
        enforce_nd = dimension(coords)
        if verbose:
            _log.info(
                f"enforce_nd is not given. Guessing {enforce_nd} spatial "\
                "dimension(s) from the given coords. Set enforce_nd if you "\
                "would like to do the calculation for another number of "\
                "spatial dimensions."
            )
    elif not isinstance(enforce_nd, int):
        raise ValueError(
            'amep.order.local_nunmber_density: enforce_nd must be of type int'\
            f' or None. Got {type(enforce_nd)}.'
        )
    elif enforce_nd <= 0 or enforce_nd > 3:
        raise ValueError(
            'amep.order.local_nunmber_density: enforce_nd must be larger than'\
            f'0 and smaller than 3. Got {enforce_nd}.'
        )
    # get neighbors
    number, distances, neighbors = nearest_neighbors(
        coords,
        box_boundary,
        other_coords = other_coords,
        rmax = rmax + np.max(radius), # need to count neighbors within this radius
        pbc = pbc,                    # for fractional neighbor counting!
        exclude = False
    )
    # fractional neighbor counting (linear mapping)
    volume_fraction = np.zeros(len(coords)) # 3d
    area_fraction   = np.zeros(len(coords)) # 2d
    line_fraction   = np.zeros(len(coords)) # 1d
    for i,num in enumerate(number):
        for j,d in enumerate(distances[i]):
            # get neighbor id
            nid = neighbors[i][j]
            if d <= rmax-radius[nid]:
                # full counting
                volume_fraction[i] += 4*np.pi*radius[nid]**3/3
                area_fraction[i] += np.pi*radius[nid]**2
                line_fraction[i] += 2*radius[nid]
            elif rmax-radius[nid] < d < rmax+radius[nid]:
                # fractional counting
                frac = (rmax+radius[nid])/2/radius[nid] - d/2/radius[nid]
                volume_fraction[i] += frac*4*np.pi*radius[nid]**3/3
                area_fraction[i] += frac*np.pi*radius[nid]**2
                line_fraction[i] += frac*2*radius[nid]
                    
    # normalization
    if enforce_nd == 1:
        return line_fraction/rmax/2
    elif enforce_nd == 2:
        return area_fraction/np.pi/rmax**2
    elif enforce_nd == 3:
        return volume_fraction/(4*np.pi*rmax**3/3)


def voronoi_density(
        coords: np.ndarray | None, box_boundary: np.ndarray | None,
        radius: float | np.ndarray | None = None,
        mass: float | np.ndarray | None = None, width: float | None = None,
        pbc: bool = True, vor: spatial.Voronoi | None = None,
        ids: np.ndarray | None = None, enforce_nd: int | None = None,
        mode: str = "number",
        verbose: bool = False) -> np.ndarray:
    r'''
    Calculates the local mass or number density or the local packing fraction
    by using Voronoi tessellation.

    If the radius of the particles is specified, the local packing fraction of
    each Voronoi cell is returned (assuming the particles to be spherical). If
    the mass is specified, the local density is returned. If neither radius nor
    mass is specified, the local number density is returned.

    Examples of the Voronoi tessellation used, can be found in [1]_ and [2]_

    References
    ----------
    .. [1] B. Steffen, A. Seyfried, "Methods for measuring pedestrian density,
        flow, speed and direction with minimal scatter",
        Physica A: Statistical Mechanics and its Applications,
        Volume 389, Pages 1902-1910, 2010.

    .. [2] Chaoming Song, Ping Wang, Yuliang Jin, Hernán A. Makse,
        "Jamming I: A volume function for jammed matter",
        Physica A: Statistical Mechanics and its Applications,
        Volume 389, Issue 21, Pages 4497-4509, 2010.


    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the particles for the voronoi tessellation.
        If None, the user has to specify the Voronoi tessellation with the
        keyword `vor` (and ids for pbc=True).
    box_boundary : np.ndarray of shape (3,2) | None, optional
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
        If None, the user has to specify the Voronoi tessellation with the
        keyword `vor` (and ids for pbc=True).
    radius : float or np.ndarray or None
        Radius of the particles for calculating the local packing fraction.
        Default is None.
    mass : float or np.ndarray or None
        Mass of the particles for calculating the local mass density.
        Default is None.
    width : float, optional
        Relative width (to the box) of the periodic images and mirrors used.
        For very low densities, this value might have to be set to a larger value
        than the default. For higher densities, this value can easily be set
        to 0.3 or 0.1 or even smaller. Please check the Voronoi plot to ensure
        the correct calculation.
        Range: [0,1]. Default is 0.5.
    vor : scipy.spatialcor.Voronoi or None
        If a Voronoi object vor (and indices ids for periodic boundary 
        conditions pbc=True) are supplied, the Voronoi tessellation
        is not re-calculated.
        Default is None.
    ids : np.ndarray or None
        IDs of the points inside the box when using periodic boundaries.
        If the Voronoi tessellation and the ids are supplied, the Voronoi
        tessellation is not re-calculated.
        Default is None.
    enforce_nd: int, None, optional
        enforces the number of dimensions. 2 for 2d, 3 for 3d.
        If None is supplied, a best guess is used by checking
        if all particles have the same dimension in the last
        coordinate. See `utils.dimension()`
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Returns
    -------
    np.ndarray
        Local densities for each Voronoi cell.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> ld = amep.order.voronoi_density(
    ...     frame.coords(), frame.box, radius=frame.radius(),
    ...     pbc=True, mode="packing"
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius(),
    ...     values=ld
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label="local packing fraction"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/order/order-voronoi_density.png")

    .. image:: /_static/images/order/order-voronoi_density.png
      :width: 400
      :align: center
 
    '''
    if mode not in ["number", "mass", "packing"]:
        raise ValueError("Please choose mode from ['number', 'mass', 'packing'].")

    # calculate Voronoi diagram if not given
    if vor is not None and ids is not None and verbose:
        _log.info(
            "Using supplied vor and ids. Please ensure that the Voronoi "\
            "object matches your data."
        )
    elif vor is None and ids is not None:
        # do not overwrite ids
        vor, idsbox = voronoi(
            coords, box_boundary, pbc=pbc, width=width, enforce_nd=enforce_nd
        )
    elif vor is None and ids is None:
        vor, ids = voronoi(
            coords, box_boundary, pbc=pbc, width=width, enforce_nd=enforce_nd
        )

    # calculate local areas
    local_areas_of_ids = np.zeros(len(ids)) # only local areas of particles with ids will be calculated
    # get local areas only for particles (= point_region-ids) with the given ids
    for i, region_index in enumerate(vor.point_region[ids]):
        region_vertices = vor.vertices[vor.regions[region_index]]
        hull = spatial.ConvexHull(region_vertices)
        local_areas_of_ids[i] = hull.volume

    # calculate local packing fraction/mass density/number density from local
    # areas
    if mode == "packing" and radius is not None:
        if verbose:
            _log.info("Returning local packing fraction.")
        if dimension(coords)==2 or enforce_nd==2:
            return np.pi*radius**2/local_areas_of_ids
        else:
            return 4*np.pi*radius**3/3./local_areas_of_ids
    elif mode == "mass" and mass is not None:
        if verbose:
            _log.info("Returning local mass density.")
        return mass/local_areas_of_ids
    elif mode == "number":
        if verbose:
            _log.info("Returning local number density.")
        return 1./local_areas_of_ids
    else:
        raise Exception(
            "amep.order.voronoi_density: Please check the selected mode "\
            f"'{mode}' and supplied data (mass, radius). Mode 'packing' "\
            "requires the radius. Mode 'mass' requires the mass."
        )


# =============================================================================
# NEIGHBORS
# =============================================================================
def next_neighbors(
        coords: np.ndarray | None, box_boundary: np.ndarray | None,  
        vor: spatial.Voronoi | None = None, ids: np.ndarray | None = None,
        width: float | None = 0.5, pbc: bool = True) -> np.ndarray:
    r'''
    Determines the next neighbors based on a given or calculated Voronoi
    diagram. Returns the number of next neighbors for each particle in coords
    and the next-neighbor distances as well as the indices of the next
    neighbors.
    
    Parameters
    ----------
    coords : np.ndarray of shape (N,3) or None
        Array of coordinates/points to search for neighbors.
    box_boundary : np.ndarray of shape (3,2) or None
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
        If `None`, one has to specify the Voronoi diagram with the keyword
        `vor` (and `ids` for `pbc=True`).
    width : float, optional
        Relative width (to the box) of the periodic images and mirrors used.
        For very low densities, this value might have to be set to a larger
        value than the default. For higher densities, this value can easily be 
        set to smaller values (e.g., 0.1-0.3´). Please check the Voronoi plot
        to ensure the correct calculation. Range: [0,1]. The default is 0.5.
    vor : scipy.spatialcor.Voronoi or None, optional
        If a Voronoi object vor (and indices ids for periodic boundary 
        conditions pbc=True) is given, the Voronoi tessellation
        is not re-calculated. The default is None.
    ids : np.ndarray or None, optional
        IDs of the points for which the number of next neighbors should be
        returned. If a Voronoi diagram is given via the `vor` keyword and
        `pbc=True`, `ids` is required to return only the coordinates inside the
        box. If the Voronoi tessellation and the ids are supplied, the Voronoi
        tessellation is not re-calculated. The default is None.
    pbc: boolean, optional
        If True, periodic boundary conditions are used. The default is True.
        
        
    Returns
    -------
    nnn : np.ndarray of shape (N,)
        Number of next neighbors of other_coords in coords
        (same length as other_coords).
    distances : np.ndarray of shape (N,)
        Array of lists containing the next-neighbor distances for each particle
        in coords (ids) to its next neighbors.
    neighbors : np.ndarray of shape (N,)
        Array of lists containing the next-neighbor indices for each particle
        in coords (ids) to its next neighbors.
    vor : scipy.spatial.voronoi
        The Scipy Voronoi object used for the calculations.
    ids : np.ndarray of shape (N,)
        Particle indices for which the results are returned/which are returned
        from the Voronoi tesselation (see amep.order.voronoi).
        
        
    Examples
    --------
    Without periodic boundary conditions:
        
    >>> import amep
    >>> import numpy as np
    >>> coords = np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-2,0]])
    >>> box_boundary = np.array([[-4,4],[-4,4],[-0.5,0.5]])
    >>> nnn, distances, neighbors, vor, ids = amep.order.next_neighbors(
    ...     coords, box_boundary, pbc=False
    ... )
    >>> print(nnn) # number of next neighbors
    [4 3 3 3 3]
    >>> print(distances) # next neighbor distances
    [list([1.0, 1.0, 1.0, 2.0])
     list([1.4142135623730951, 1.0, 2.23606797749979])
     list([1.4142135623730951, 1.4142135623730951, 1.0])
     list([1.4142135623730951, 1.0, 2.23606797749979])
     list([2.23606797749979, 2.23606797749979, 2.0])]
    >>> print(neighbors) # neighbor ids
    [list([3, 1, 2, 4]) list([2, 0, 4]) list([3, 1, 0]) list([2, 0, 4])
     list([3, 1, 0])]
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.voronoi(axs, vor, show_vertices=False)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> axs.set_title('pbc = False')
    >>> axs.set_xlim(box_boundary[0])
    >>> axs.set_ylim(box_boundary[1])
    >>> fig.savefig('./figures/order/order-next_neighbors-pbc-False.png')
    >>> 
    
    .. image:: /_static/images/order/order-next_neighbors-pbc-False.png
      :width: 400
      :align: center

    
    With periodic boundary conditions:
        
    >>> nnn, distances, neighbors, vor, ids = amep.order.next_neighbors(
    ...     coords, box_boundary, pbc=True
    ... )
    >>> print(nnn) # number of next neighbors
    [4 5 4 5 6]
    >>> print(distances) # next neighbor distances
    [list([1.0, 1.0, 2.0, 1.0])
     list([2.23606797749979, 6.0, 1.0, 1.4142135623730951, 6.082762530298219])
     list([1.4142135623730951, 1.4142135623730951, 1.0, 5.0])
     list([2.23606797749979, 6.0, 1.0, 1.4142135623730951, 6.082762530298219])
     list([5.0, 2.23606797749979, 7.280109889280518, 2.23606797749979, 7.280109889280518, 2.0])]
    >>> print(neighbors) # neighbor ids
    [list([3, 1, 4, 2]) list([4, 3, 0, 2, 2]) list([3, 1, 0, 2])
     list([4, 0, 0, 2, 2]) list([1, 3, 0, 1, 3, 0])]
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.voronoi(axs, vor, show_vertices=False)
    >>> amep.plot.box(axs, box_boundary=box_boundary, ls='--', c='r')
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> axs.set_title('pbc = True')
    >>> fig.savefig('./figures/order/order-next_neighbors-pbc-True.png')
    
    .. image:: /_static/images/order/order-next_neighbors-pbc-True.png
      :width: 400
      :align: center

    '''
    # voronoi tessalation
    if vor is None and ids is not None:
        # the user can set ids to be returned. those will not be overwritten when calculating the Voronoi tessellation
        vor, idsinbox = voronoi(
            coords,
            pbc=pbc,
            box_boundary=box_boundary,
            width=width
        )
        N = len(idsinbox) # number of points (needed to get correct indices)
    elif vor is None and ids is None:
        vor, ids = voronoi(
            coords,
            pbc=pbc,
            box_boundary=box_boundary,
            width=width
        )
        N = len(coords) # number of points (needed to get correct indices)
    elif vor is not None and ids is None:
        N = len(vor.points) # number of points (needed to get correct indices)
    else:
        N = len(ids) # number of points (needed to get correct indices)

    # connected points
    connections = vor.ridge_points
            
    # generate neighbor list and determine distances and indices
    points = vor.points
    distances = [ [] for _ in range(len(points))]
    neighbors = [ [] for _ in range(len(points))]
    for pair in connections:
        if pbc:
            neighbors[pair[0]].append(pair[1]%N) # get indices in box
            neighbors[pair[1]].append(pair[0]%N) # get indices in box
            d = np.linalg.norm(points[pair[0]] - points[pair[1]])
            distances[pair[0]].append(d)
            distances[pair[1]].append(d)
        else:
            if pair[0] in ids and pair[1] in ids:
                neighbors[pair[0]].append(pair[1])
                neighbors[pair[1]].append(pair[0])
                d = np.linalg.norm(points[pair[0]] - points[pair[1]])
                distances[pair[0]].append(d)
                distances[pair[1]].append(d)
    neighbors = np.asarray(neighbors, dtype=object)[ids]
    distances = np.asarray(distances,dtype=object)[ids]

    # count number of next neighbors
    nnn = []
    for i in range(len(coords)):
        nnn.append(len(neighbors[i]))
    nnn = np.asarray(nnn)[ids]

    return nnn, distances, neighbors, vor, ids
        
        
def nearest_neighbors(
        coords: np.ndarray, box_boundary: np.ndarray,
        other_coords: np.ndarray | None = None,
        rmax: float = 1.0, pbc : bool = True, exclude : bool = True,
        enforce_nd: int | None = None) -> np.ndarray:
    r'''
    Calculates the nearest neighbors within a distance cutoff and returns the 
    number of nearest neighbors, the nearest-neighbor distances, and the
    nearest-neighbor indices.

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Array of coordinates to search for neighbors of.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray of shape (M,3) or None, optional
        Array of coordinates of the particles (neighbors).
        The default is None (use coords).
    rmax : float, optional
        Distance cutoff. Only neighbors with distance smaller than the cutoff
        are returned. The default is 1.0.
    pbc: boolean, optional
        If True, periodic boundary conditions are used. The default is True.
    exclude : boolean, optional
        If True, the particle itself is excluded in the neighbor counting.
        The default is True.
    enforce_nd: int, None, optional
        enforces the number of dimensions. 2 for 2d, 3 for 3d.
        If None is supplied, a best guess is used by checking
        if all particles have the same dimension in the last
        coordinate. See `utils.dimension()`
    

    Returns
    -------
    nnn : np.ndarray of shape (N,)
        Number of nearest neighbors.
    distances : np.ndarray of shape (N,)
        Array of lists containing the nearest-neighbor distances.
    neighbors : np.ndarray of shape (N,)
        Array of lists containing the nearest-neighbor indices.
        
    Examples
    --------
    Create a test setup of a box and some particles at certain positions:
        
    >>> import amep
    >>> import numpy as np
    >>> coords = np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-2,0]])
    >>> box_boundary = np.array([[-4,4],[-4,4],[-0.5,0.5]])
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.box(axs, box_boundary, ls='--')
    >>> axs.plot(coords[:,0], coords[:,1], "x", c="k")
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/order/order-nearest_neighbors.png')
    >>> 
    
    .. image:: /_static/images/order/order-nearest_neighbors.png
      :width: 400
      :align: center
      
    
    Calculate nearest neighbors without periodic boundary conditions:
        
    >>> nnn, distances, neighbors = amep.order.nearest_neighbors(
    ...     coords, box_boundary, rmax=5, pbc=False
    ... )
    >>> print(nnn) # number of nearest neighbors
    [4 4 4 4 4]
    >>> print(distances) # nearest-neighbor distances
    [[1.0 1.0 1.0 2.0]
     [1.0 1.4142135623730951 2.0 2.23606797749979]
     [1.0 1.4142135623730951 1.4142135623730951 3.0]
     [1.0 2.0 1.4142135623730951 2.23606797749979]
     [2.0 2.23606797749979 3.0 2.23606797749979]]
    >>> print(neighbors) # nearest neighbor ids
    [[1 2 3 4]
     [0 2 3 4]
     [0 1 3 4]
     [0 1 2 4]
     [0 1 2 3]]
    >>> 


    Calculate nearest neighbors with periodic boundary conditions:
        
    >>> nnn, distances, neighbors = amep.order.nearest_neighbors(
    ...     coords, box_boundary, rmax=5, pbc=True
    ... )
    >>> print(nnn) # number of nearest neighbors
    [4 4 5 4 5]
    >>> print(distances) # nearest-neighbor distances
    [list([1.0, 1.0, 1.0, 2.0])
     list([1.0, 1.4142135623730951, 2.0, 2.23606797749979])
     list([1.0, 1.4142135623730951, 1.4142135623730951, 3.0, 5.0])
     list([1.0, 2.0, 1.4142135623730951, 2.23606797749979])
     list([2.0, 2.23606797749979, 3.0, 2.23606797749979, 5.0])]
    >>> print(neighbors) # nearest neighbor ids
    [list([1, 2, 3, 4]) list([0, 2, 3, 4]) list([0, 1, 3, 4, 4])
     list([0, 1, 2, 4]) list([0, 1, 2, 3, 2])]
    >>> 
    
    
    Calculate nearest neighbors of all `coords` in `other_coords`:
        
    >>> nnn, distances, neighbors = amep.order.nearest_neighbors(
    ...     coords[1:], box_boundary, rmax=5,
    ...     pbc=False, other_coords=coords[0:1]
    ... )
    >>> print(nnn) # number of nearest neighbors
    [1 1 1 1]
    >>> print(distances) # nearest-neighbor distances
    [[1.0]
     [1.0]
     [1.0]
     [2.0]]
    >>> print(neighbors) # nearest neighbor ids
    [[0]
     [0]
     [0]
     [0]]
    >>> 
    
    '''
    other = True
    if other_coords is None:
        other_coords = coords
        other = False
        
    N = len(other_coords)
    
    # generate a kdtree for neighbor searching
    tree = spatial.KDTree(coords)
        
    # apply pbc
    if pbc:
        other_coords = pbc_points(
            other_coords,
            box_boundary,
            fold_coords=False,
            enforce_nd=enforce_nd
        )
    other_tree = spatial.KDTree(other_coords)
    
    # search for nearest neighbors
    nndata = tree.query_ball_tree(other_tree, rmax)
    
    # get number, distances, and ids
    neighbors = [ [] for _ in range(len(coords))]
    distances = [ [] for _ in range(len(coords))]
    nnn = []
    for i,c in enumerate(coords):
        if other or not exclude:
            nnn.append(len(nndata[i]))
            for j in nndata[i]:
                d = np.linalg.norm(coords[i] - other_coords[j])
                distances[i].append(d)
                neighbors[i].append(j%N) # only indices in box
        else:
            nnn.append(len(nndata[i])-1) # substract the particle itself
            for j in nndata[i]:
                if j != i: # exclude the particle itself
                    d = np.linalg.norm(coords[i] - other_coords[j])
                    distances[i].append(d)
                    neighbors[i].append(j%N) # only indices in box

    nnn = np.asarray(nnn)
    distances = np.asarray(distances, dtype=object)
    neighbors = np.asarray(neighbors, dtype=object)

    return nnn, distances, neighbors


def k_nearest_neighbors(
        coords: np.ndarray, box_boundary: np.ndarray,
        other_coords: np.ndarray | None = None,
        k: int = 1, pbc : bool = True,
        rmax : float = np.inf, enforce_nd: int | None = None) -> np.ndarray:
    r'''
    Calculates the k nearest neighbors and returns the number of nearest
    neighbors (can be different from k if a distance cutoff is given),
    the k nearest-neighbor distances, and the k nearest-neighbor indices.

    Parameters
    ----------
    coords : np.ndarray of shape (N,3)
        Array of coordinates/points to search for neighbors.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray of shape (M,3) or None, optional
        Array of points to search for neighbors of. The default is None (use 
        coords).
    rmax : float, optional
        Distance cutoff. Only neighbors with distance smaller than the cutoff
        are returned. The default is np.inf.
    pbc: boolean, optional
        If True, periodic boundary conditions are used. The default is True.
    k : int, optional
        Number of nearest neighbors to determine. The default is 1.
    enforce_nd: int, None, optional
        enforces the number of dimensions. 2 for 2d, 3 for 3d.
        If None is supplied, a best guess is used by checking
        if all particles have the same dimension in the last
        coordinate. See `utils.dimension()`

    Returns
    -------
    nnn : np.ndarray of shape (N,)
        Number of nearest neighbors.
    distances : np.ndarray of shape (N,)
        Array of lists containing the nearest-neighbor distances.
    neighbors : np.ndarray of shape (N,)
        Array of lists containing the nearest-neighbor indices.
        
    Examples
    --------
    Create a test setup of a box and some particles at certain positions:

    >>> import amep
    >>> import numpy as np
    >>> coords = np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-2,0]])
    >>> box_boundary = np.array([[-4,4],[-4,4],[-0.5,0.5]])
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.box(axs, box_boundary, ls='--')
    >>> axs.plot(coords[:,0], coords[:,1], "x", c="k")
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/order/order-k_nearest_neighbors.png')
    >>> 

    .. image:: /_static/images/order/order-k_nearest_neighbors.png
      :width: 400
      :align: center
      
      
    Calculate k nearest neighbors without periodic boundary conditions:
        
    >>> nnn, distances, neighbors = amep.order.k_nearest_neighbors(
    ...     coords, box_boundary, k=5, pbc=False
    ... )
    >>> print(nnn) # number of nearest neighbors
    [4 4 4 4 4]
    >>> print(distances) # nearest-neighbor distances
    [[1.0 1.0 1.0 2.0]
     [1.0 1.4142135623730951 2.0 2.23606797749979]
     [1.0 1.4142135623730951 1.4142135623730951 3.0]
     [1.0 1.4142135623730951 2.0 2.23606797749979]
     [2.0 2.23606797749979 2.23606797749979 3.0]]
    >>> print(neighbors) # nearest neighbor ids
    [[2 1 3 4]
     [0 2 3 4]
     [0 1 3 4]
     [0 2 1 4]
     [0 1 3 2]]
    >>> 


    Calculate k nearest neighbors with periodic boundary conditions:
        
    >>> nnn, distances, neighbors = amep.order.k_nearest_neighbors(
    ...     coords, box_boundary, k=5, pbc=True
    ... )
    >>> print(nnn) # number of nearest neighbors
    [5 5 5 5 5]
    >>> print(distances) # nearest-neighbor distances
    [[1.0 1.0 1.0 2.0 6.0]
     [1.0 1.4142135623730951 2.0 2.23606797749979 6.0]
     [1.0 1.4142135623730951 1.4142135623730951 3.0 5.0]
     [1.0 1.4142135623730951 2.0 2.23606797749979 6.0]
     [2.0 2.23606797749979 2.23606797749979 3.0 5.0]]
    >>> print(neighbors) # nearest neighbor ids
    [[3 1 2 4 4]
     [0 2 3 4 3]
     [0 3 1 4 4]
     [0 2 1 4 1]
     [0 3 1 2 2]]
    >>> 


    Calculate k nearest neighbors of all `coords` in `other_coords`:
        
    >>> nnn, distances, neighbors = amep.order.k_nearest_neighbors(
    ...     coords[1:], box_boundary, k=5, pbc=False, other_coords=coords[0:1]
    ... )
    >>> print(nnn) # number of nearest neighbors
    [1 1 1 1]
    >>> print(distances) # nearest-neighbor distances
    [[1.0]
     [1.0]
     [1.0]
     [2.0]]
    >>> print(neighbors) # nearest neighbor ids
    [[0]
     [0]
     [0]
     [0]]
    >>> 

    '''
    other = True
    if other_coords is None:
        other_coords = coords
        k = list(range(2,k+2)) # exclude the particle itself
        other = False
        
    N = len(other_coords)
        
    # apply pbc
    if pbc:
        other_coords = pbc_points(
            other_coords,
            box_boundary,
            fold_coords=False,
            enforce_nd=enforce_nd
        )
    # generate a kdtree for neighbor searching
    tree = spatial.KDTree(other_coords)
    
    # search for nearest neighbors
    dists, neighs = tree.query(coords, k=k, distance_upper_bound=rmax)
    
    # get number, distances, and ids
    neighbors = [ [] for _ in range(len(coords))]
    distances = [ [] for _ in range(len(coords))]
    nnn = []
    npoints = len(other_coords)
    for i,d in enumerate(dists):
        n = neighs[i,neighs[i]!=npoints]
        if other:
            nnn.append(len(n))
            distances[i] = list(d[d!=np.inf])
            neighbors[i] = list(n%N)
        else:
            d = d[d!=np.inf][n%N!=i]
            n = n[n%N!=i]
            nnn.append(len(n))
            distances[i] = list(d)
            neighbors[i] = list(n%N)

    nnn = np.asarray(nnn)
    distances = np.asarray(distances, dtype=object)
    neighbors = np.asarray(neighbors, dtype=object)

    return nnn, distances, neighbors



# =============================================================================
# ORDER PARAMETERS
# =============================================================================
def psi_k(
        coords: np.ndarray, box_boundary: np.ndarray,
        other_coords: np.ndarray | None = None,
        rmax: float = 1.122, k: int = 6, pbc: bool = True) -> np.ndarray:
    r'''
    Calculates the k-atic bond order parameter for an entire 2D system.

    In a first step, the indexes of the first k next neighbors of each
    atom is calculated with the KDTree algorithm and with periodic
    boundary conditions appplied with pbc_points.


    Notes
    -----
    The k-atic order parameter is defined by

    .. math::
        \Psi_k(\vec{r}_j) = \frac{1}{k} \sum_{n=1}^k\exp(ik\theta_{jn}),

    where the sum goes over the k nearest neighbors of the particle at
    position :math:`\vec{r}_j`. The value of :math:`\theta_{jn}` is equal to the angle
    between the connection line from :math:`\vec{r}_j` to :math:`\vec{r}_n` and the
    x axis. See also Refs. [1]_ [2]_ [3]_ for further information.

    References:

    .. [1] Nelson, D. R., Rubinstein, M., & Spaepen, F. (1982).
       Order in two-dimensional binary random arrays. Philosophical Magazine A,
       46(1), 105–126. https://doi.org/10.1080/01418618208236211

    .. [2] Digregorio, P., Levis, D., Suma, A., Cugliandolo, L. F.,
       Gonnella, G., & Pagonabarraga, I. (2018). Full Phase Diagram of Active
       Brownian Disks: From Melting to Motility-Induced Phase Separation.
       Physical Review Letters, 121(9), 098003.
       https://doi.org/10.1103/PhysRevLett.121.098003

    .. [3] Cugliandolo, L. F., & Gonnella, G. (2018). Phases of active matter
       in two dimensions. ArXiv:1810.11833 [Cond-Mat.Stat-Mech].
       http://arxiv.org/abs/1810.11833


    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame of all particles.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    other_coords : np.ndarray, optional
        Coords of the particles which are considered as possible neighbors.
        If None, coords is used. The default is None.
    rmax : float, optional
        Maximum distance between particles to counted as neighbors.
        The default is 1.122 (which is the cutoff radius of the WCA potential).
        This value is ignored in the current version.
    k : int, optional
        Symmetry of the k-atic bond order parameter. The default is 6.
    pbc: bool, optional
        If True, periodic boundary conditions are considered. The default is
        True.

    Returns
    -------
    np.ndarray
        k-atic order parameter of each particle (1D array of
        complex numbers).

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> psi_4 = amep.order.psi_k(
    ...     frame.coords(), frame.box, k=4
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius(),
    ...     values=np.abs(psi_4)
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r'$|\psi_4|$'
    ... )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/order/order-psi4.png')
    >>> 
    
    .. image:: /_static/images/order/order-psi4.png
      :width: 400
      :align: center
      

    >>> psi_6 = amep.order.psi_k(
    ...     frame.coords(), frame.box, k=6
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius(),
    ...     values=np.abs(psi_6)
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r'$|\psi_6|$'
    ... )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/order/order-psi6.png')
    >>> 
    
    .. image:: /_static/images/order/order-psi6.png
      :width: 400
      :align: center

    '''
    # get indices of the k nearest neighbors
    _, _, indices = k_nearest_neighbors(
        coords,
        box_boundary,
        other_coords = other_coords,
        k = k,
        pbc = pbc,
        enforce_nd = 2
    )
    indices = indices.astype(int)

    # coordinates of the first k next neighbors  
    if other_coords is None:
        other_coords = coords
    nncoords = other_coords[indices]

    # calculate angle between the particles
    theta = np.arctan2(
        (nncoords[:,:,1]-coords[:,None,1]),
        (nncoords[:,:,0]-coords[:,None,0])
    )

    # calculate (complex) bond order parameter
    psi = 1./k * np.sum(np.exp(1j*k*theta), axis=1)
    
    return psi
