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
Continuum Field Analysis
========================

.. module:: amep.continuum

The AMEP module :mod:`amep.continuum` provides various methods for analyzing 
field data from continuum simulations as well as conversion methods that 
convert particle-based simulation data into field data by using density 
estimators.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
from packaging.version import Version
import numpy as np
import scipy.fft as fft

from skimage.segmentation import watershed
from skimage import measure

from .utils import runningmean, mesh_to_coords
from .pbc import pbc_points, distance_matrix
from .cluster import geometric_center, center_of_mass
from .cluster import gyration_tensor, inertia_tensor


# =============================================================================
# PARTICLE COORDS TO FIELD CONVERSION (NO COARSE GRAINING!)
# =============================================================================
def coords_to_density(
        coords: np.ndarray, box_boundary: np.ndarray,
        dmin: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Convert the particle coordinates into a 2D density field.

    With size gridsize^2 by calculating a 2D histogram of the
    x and y coordinates of the particles.

    Notes
    -----
    This version only works in 2D.

    To get similar results (e.g., for the structure factor), the size of a
    grid cell must be smaller or equal to the size of a single particle such
    that one grid cell cannot be occupied by more than one particle.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    dmin : float, optional
        Minimum grid spacing. Must be smaller than the diameter of the
        smallest particle in the system. The default is 1.0.

    Returns
    -------
    hist : np.ndarray
        Density field.
    X : np.ndarray
        x coordinates (same shape as hist; meshgrid).
    Y : np.ndarray
        y coordinates (same shape as hist; meshgrid).

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> d, X, Y = amep.continuum.coords_to_density(
    ...     frame.coords(), frame.box, dmin=0.3
    ... )
    >>> print('Integrated density: ', int(np.sum(d)*(X[0,1]-X[0,0])*(Y[1,0]-Y[0,0])))
    Integrated density:  4000
    >>> print('Particle number: ', frame.n())
    Particle number:  4000
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(axs, d, X, Y)
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r"$\rho(x,y)$"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/continuum/continuum-coords_to_density_1.png")

    .. image:: /_static/images/continuum/continuum-coords_to_density_1.png
      :width: 400
      :align: center

    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(axs, d, X, Y)
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r"$\rho(x,y)$"
    ... )
    >>> axs.scatter(
    ...     frame.coords()[:,0], frame.coords()[:,1], s=10,
    ...     marker="x", color="gray", label="particle coordinates"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> axs.set_xlim(20, 40)
    >>> axs.set_ylim(40, 60)
    >>> axs.legend(frameon=True)
    >>> fig.savefig('./figures/continuum/continuum-coords_to_density_2.png')

    .. image:: /_static/images/continuum/continuum-coords_to_density_2.png
      :width: 400
      :align: center

    """
    hist, X, Y = hde(coords, box_boundary, delta=dmin, shift=0.5)

    return hist, X, Y


# =============================================================================
# SPATIAL CORRELATION FUNCTIONS
# =============================================================================
def sf2d(
        dfield: np.ndarray, X: np.ndarray, Y: np.ndarray,
        other_dfield: np.ndarray | None = None, Ntot: int | None = None,
        njobs: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r'''
    Calculate the 2D static structure factor from a density field.

    Is done via Fourier transform.

    Notes
    -----
    The static structure factor is defined by

    .. math::
        S(\vec{q}) = <\rho(\vec{q})\rho(\vec{-q})>= <|\rho(\vec{q})|^2>,

    where :math:`\rho(\vec{q})` is the Fourier transform of the particle number
    density (see Ref. [1]_ [2]_ [3]_ for further information).

    .. [1] Hansen, J.-P., & McDonald, I. R. (2006). Theory of Simple Liquids
       (3rd ed.). Elsevier. https://doi.org/10.1016/B978-0-12-370535-8.X5000-9
       (p. 78 and 83)

    .. [2] Bray, A. J. (1994). Theory of phase-ordering kinetics.
       Advances in Physics, 43(3), 357–459.
       https://doi.org/10.1080/00018739400101505

    .. [3] Wittkowski, Raphael, et al. "Scalar φ 4 field theory for
       active-particle phase separation."
       Nature communications 5.1 (2014): 4351.
       https://doi.org/10.1038/ncomms5351


    S(0,0) is set to 0

    Parameters
    ----------
    dfield : np.ndarray
        Two dimensional density field.
    X : np.ndarray
        x coordinates of grid cells in form of a meshgrid with same shape as
        dfield.
    Y : np.ndarray
        y coordinates of grid cells in form of a meshgrid with same shape as
        dfield.
    other_dfield : np.ndarray, optional
        Two dimensional density field to which the correlation should be
        calculated. The default is None (use dfield).
    Ntot : int, optional
        Total number of particles in the system. This value is needed for the
        correct normalization of the structure factor. If None, Ntot is
        calculated from the integral over dfield. The default is None.
    njobs : int, optional
        Number of processors used for the parallel computation of the fft.
        The default is 1.

    Returns
    -------
    np.ndarray
        2d static structure factor.
    np.ndarray
        x components of the wave vectors.
    np.ndarray
        y components of the wave vectors.

    Examples
    --------
    >>> import amep
    >>> import matplotlib as mpl
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> d, X, Y = amep.continuum.coords_to_density(
    ...     frame.coords(), frame.box, dmin=0.3
    ... )
    >>> S, Kx, Ky = amep.continuum.sf2d(d, X, Y)
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(
    ...     axs, S, Kx, Ky, cscale="log", vmin=1e0
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r"$S(q_x,q_y)$"
    ... )
    >>> axs.set_xlabel(r'$q_x$')
    >>> axs.set_ylabel(r'$q_y$')
    >>> axs.set_title(r"active Brownian particles")
    >>> fig.savefig('./figures/continuum/continuum-sf2d_1.png')

    .. image:: /_static/images/continuum/continuum-sf2d_1.png
      :width: 400
      :align: center

    >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
    >>> frame = traj[3]
    >>> S, Kx, Ky = amep.continuum.sf2d(
    ...     frame.data("p"), *frame.grid
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(
    ...     axs, S, Kx, Ky, cscale="log", vmin=1e0
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r"$S(q_x,q_y)$"
    ... )
    >>> axs.set_xlabel(r'$q_x$')
    >>> axs.set_ylabel(r'$q_y$')
    >>> axs.set_title(r"Keller-Segel model")
    >>> fig.savefig('./figures/continuum/continuum-sf2d_2.png')

    .. image:: /_static/images/continuum/continuum-sf2d_2.png
      :width: 400
      :align: center

    '''
    # normalization of fft
    norm = 'backward'

    # grid spacings
    dx = X[0, 1]-X[0, 0]
    dy = Y[1, 0]-Y[0, 0]

    if other_dfield is None:
        other_dfield = dfield

    # particle number (integral over density)
    Ni = int(np.sum(dfield)*dx*dy)
    Nj = int(np.sum(other_dfield)*dx*dy)

    #if Ntot is None:
    #    Ntot = Ni

    # get static structure factor from 2D Fourier transform of the density
    S = np.real(fft.fftshift(fft.fft2(dfield, norm=norm, workers=njobs)) *
                fft.fftshift(fft.fft2(other_dfield, norm=norm,
                                      workers=njobs)).conj())

    # normalization (S(0,0)=N_i*N_j/N_{\rm tot}, see Ref. [1], p. 83)
    # - FT at k=0 is equal to sum of all values
    if Ntot is not None:
        # to get the same normalization as for the particle-based version
        S = S/np.sum(dfield)/np.sum(other_dfield)*Ni*Nj/Ntot

    # get q vectors
    qx = fft.fftshift(fft.fftfreq(S.shape[0], d=dx))*2*np.pi
    qy = fft.fftshift(fft.fftfreq(S.shape[1], d=dy))*2*np.pi

    # make meshgrid
    Qx, Qy = np.meshgrid(qx, qy)

    # Filter out S(0,0) by replacing it with 0
    qx_index, qy_index = np.where((Qx == 0.0) & (Qy == 0.0))
    S[qx_index, qy_index] = 0

    return S, Qx, Qy


# =============================================================================
# DENSITY ESTIMATORS
# =============================================================================
def hde(
        coords: np.ndarray, box_boundary: np.ndarray,
        weights: np.ndarray | None = None,
        delta: float = 1.0, shift: float = 0.5,
        pbc: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Histogram Density Estimation (HDE).

    Calculates the density estimate for a given set
    of particle coordinates in a simulation box based
    on a 2D histogram of the particle positions.

    Notes
    -----
    This method is only applicable to 2D systems. For a 3D system, only the
    x-y plane is considered.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame of shape (N,3), which gives
        the positions of the particles. The third
        component is disregarded.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    weights : np.ndarray|None, optional
        Weighting factors for each particle as array of shape (N,).
        The default is None.
    delta : float, optional
        Bin width for the histogram bins (same for each
        spatial direction). The default is 1.0.
    shift : float, optional
        Fraction of delta by which the most left bin edges
        are shifted. The default is 0.5.
    pbc : bool, optional
        If True, periodic images are considered. The default is False.

    Returns
    -------
    hist : np.ndarray
        Coarse-grained density field as a two-dimensional array.
    X : np.ndarray
        X-components of the grid points as two-dimensional
        meshgrid.
    Y : np.ndarray
        Y-components of the grid points as two-dimensional
        meshgrid.

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> ld = amep.order.local_number_density(
    ...     frame.coords(), frame.box, frame.radius()
    ... )
    >>> dfield, X, Y = amep.continuum.hde(
    ...     frame.coords(), frame.box,
    ...     weights=ld, delta=2.0, pbc=True
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(
    ...     axs, dfield, X, Y
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r"$\rho_{\rm loc}$"
    ... )
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/continuum/continuum-hde.png')

    .. image:: /_static/images/continuum/continuum-hde.png
      :width: 400
      :align: center

    """
    # account for periodic images
    if pbc:
        coords, indices = pbc_points(
            coords, box_boundary, enforce_nd=2, index=True
        )

        # adapt weights to pbc coords
        if weights is not None:
            weights = weights[indices]

    # create bin edges
    xbin_edges = np.arange(
        box_boundary[0, 0]-shift*delta,
        box_boundary[0, 1]+delta,
        delta
    )
    ybin_edges = np.arange(
        box_boundary[1, 0]-shift*delta,
        box_boundary[1, 1]+delta,
        delta
    )

    # count the number of particles in each grid cell via 2D histogram
    hist = np.histogram2d(
        coords[:, 0], coords[:, 1], [xbin_edges, ybin_edges], weights=weights
    )[0].T/delta**2

    # get grid points
    X, Y = np.meshgrid(runningmean(xbin_edges, 2), runningmean(ybin_edges, 2))

    return hist, X, Y


def gkde(
        coords: np.ndarray, box_boundary: np.ndarray,
        weights: np.ndarray | None = None, bandwidth: float = 1.0,
        gridpoints: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r'''
    Gaussian Kernel Density Estimation (GKDE).

    Uses the kernel density estimation method with a Gaussian
    kernel to estimate the density field from particle coordinates.

    Notes
    -----
    The Kernel Density Estimation (KDE) method uses a continuous kernel
    function, which needs to be normalized to one. In KDE, a kernel function
    k of width h is placed at each position of the particles. Then, all kernel
    functions are summed to get the final density estimate. There two
    "parameters": the form of the kernel function k and its width.

    This method is nicely described on page 13 ff. in Ref. [1]_.

    References
    ----------
    .. [1] Silverman, B. W. (1998). Density Estimation for Statistics and
       Data Analysis. Chapman and Hall/CRC.
       https://doi.org/10.1201/9781315140919.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate frame of shape (N,3), which gives
        the positions of the real particles. The third
        component is disregarded.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    weights : np.ndarray|None, optional
        Weighting factors for each particle as array of shape (N,).
        The default is None.
    bandwidth : float, optional
        Width of the kernel function. The default is 1.0.
    gridpoints : int, optional
        Number of gridpoints in each direction. The
        default is 100.
        
    Returns
    -------
    d : np.ndarray
        Coarse-grained density field as a two-dimensional array.
    X : np.ndarray
        X-components of the grid points as two-dimensional
        meshgrid.
    Y : np.ndarray
        Y-components of the grid points as two-dimensional
        meshgrid.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> d, X, Y = amep.continuum.gkde(
    ...     frame.coords(), frame.box, bandwidth=5.0
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(
    ...     axs, d, X, Y
    ... )
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs, mp, label=r"$\rho(x,y)$"
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig('./figures/continuum/continuum-gkde.png')

    .. image:: /_static/images/continuum/continuum-gkde.png
      :width: 400
      :align: center
    
    '''    
    # create the grid
    X, Y = np.meshgrid(np.linspace(box_boundary[0,0], box_boundary[0,1], gridpoints),
                       np.linspace(box_boundary[1,0], box_boundary[1,1], gridpoints))

    # convert the grid into a coordinate frame
    gridcoords = mesh_to_coords(X,Y)
    
    # add periodic images to account for periodic boundary conditions
    # (this is needed to ensure that the coarse-grained density also
    #  fulfills periodic boundary conditions)
    coords, indices = pbc_points(
        coords, box_boundary, enforce_nd=2,
        thickness=3*bandwidth, index=True
    )
    
    # adapt weights to pbc coords
    if weights is None:
        weights = np.ones(len(coords))
    else:
        weights = weights[indices]
    
    # kernel function
    def gaussian_kernel(x, h):
        return np.sum(weights*np.exp(-0.5*np.sum((x-coords[:,:2])**2, axis=1)/h**2))/2/np.pi/h**2
    
    # density estimation
    d = np.apply_along_axis(gaussian_kernel, 1, gridcoords, h=bandwidth)
    
    return d.reshape(X.shape), X, Y 


# ==============================================================================
# CLUSTER ANALYSIS
#==============================================================================
def identify_clusters(
        dfield: np.ndarray, scale: float = 1.0, pbc: bool = True,
        cutoff : float = 1.0, threshold: float = 0.5, method: str = "threshold"
        ) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Identify clusters in a continuum field.
    Uses either the watershed algorithm or a threshold method.
    The threshold method creates a black and white image by separating pixels
    with values below a threshold and those with values above.
    Then it indexes each continous pixel group of the same value with a number.
    The watershed method is taken from scikit image.

    Parameters
    ----------
    dfield : np.ndarray of shape (N,M)
        Two dimensional density field.
    scale : float, optional
        The scale of the field. This is needed to improve the segmentation
        done in the watershed algorithm. This keyword is ignored when using
        `method='threshold'`. The default is 1.0.
        Can also be set to negative value to make bubble detection
        accesible.
    pbc : bool, optional
        Whether to use periodic boundary conditions. Default is True.
    cutoff: float, optional
        Caps the highest field values to the ratio of the data extent. Default
        value 1.0 means no cut, 0.9 means the highest and lowest 10%
        of the field get cut off.
        Needed for clusters with multiple local minima. Combats
        oversegmentation. This keyword is ignored when using
        `method='threshold'`. The default is 1.0.
    threshold : float, optional
        Relative threshold for `method='threshold'`. Searches for connected
        regions in which the relative values are larger than the threshold,
        i.e., larger than threshold*(np.max(dfield)-np.min(dfield)). This
        keyword is ignored if `method='watershed'`. The default is 0.5.
    method: str, optional
        Chooses between 'watershed' and 'threshold' method. The default is
        'threshold'.


    Returns
    -------
    ids : np.ndarray
        An array of the cluster IDs, starting from 0.
    labels : np.ndarray of shape (N,M)
        Array of the same shape as the input field, where each pixel
        or grid point is labeled with the cluster ID it belongs to.
        


    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> X, Y = np.meshgrid(
    ...     np.linspace(0, np.pi, 101), np.linspace(0, np.pi, 101)
    ... )
    >>> dfield = np.cos(2*np.pi*X)*np.cos(2*np.pi*Y)
    >>> ids, labels = amep.continuum.identify_clusters(
    ...     dfield, scale=1.0, cutoff=0.8, threshold=0.2,
    ...     method="threshold", pbc=True
    ... )
    >>> print("Cluster ids: ", ids)
    Cluster ids:  [ 0  1  2  3  5  6  7  8  9 10 12 13 14 15 16 17 19 20 21]
    
    
    
    >>> fig, axs = amep.plot.new(ncols=2, figsize=(7, 3))
    >>> mp1 = amep.plot.field(axs[0], dfield, X, Y, cmap="viridis")
    >>> cax1 = amep.plot.add_colorbar(fig, axs[0], mp1, label=r"$\rho$")
    >>> axs[0].set_title("density field")
    >>> axs[0].set_xlabel(r"$x$")
    >>> axs[0].set_ylabel(r"$y$")
    >>> mp2 = amep.plot.field(axs[1], labels, X, Y, cmap="nipy_spectral")
    >>> cax2 = amep.plot.add_colorbar(fig, axs[1], mp2, label="cluster id")
    >>> axs[1].set_title("clusters")
    >>> axs[1].set_xlabel(r"$x$")
    >>> axs[1].set_ylabel(r"$y$")
    >>> fig.savefig("./figures/continuum/continuum-identify_clusters.png")

    .. image:: /_static/images/continuum/continuum-identify_clusters.png
      :width: 600
      :align: center

    '''
    match method:
        case "watershed":
            # rescaling and transforming to int for use with watershed
            minimum = np.min(dfield)
            maximum = np.max(dfield)
            upper_cut = maximum-(1-cutoff)*(maximum-minimum)
            lower_cut = minimum+(1-cutoff)*(maximum-minimum)
            dfield_scaled = (np.clip(dfield, lower_cut, upper_cut) * scale).astype(int)

            # apply watershed
            labels = watershed(-dfield_scaled, markers=None, mask=dfield_scaled)
        case _:
            dfield_scaled = dfield > (np.min(dfield) +
                                      (np.max(dfield) - np.min(dfield))
                                      )*threshold
            labels = measure.label(dfield_scaled)

    
    if pbc:
        # check lower and upper edge
        a = labels[0, :] != 0
        b = labels[-1, :] != 0
        c = labels[0, :] != labels[-1, :]
        d = a*b*c
    
        index1 = labels[0, d]
        index2 = labels[-1, d]
    
        i_set_1 = {(index2[i],val) for i,val in enumerate(index1)}
    
        # check left and right edge
        a = labels[:, 0] != 0
        b = labels[:, -1] != 0
        c = labels[:, 0] != labels[:, -1]
        d = a*b*c
    
        index3 = labels[d, 0]
        index4 = labels[d, -1]
    
        i_set_2 = {(index4[i],val) for i,val in enumerate(index3)}
    
        i_set = i_set_1.union(i_set_2)
        i_list = list(i_set)
        c_labels = labels.copy()
        for i in range(len(i_list)):
            c_labels[c_labels==max(*i_list[i])] = min(*i_list[i])
            for j in range (i+1, len(i_list)):
                if max(*i_list[i]) == i_list[j][0]:
                    i_list[j] = (min(*i_list[i]), i_list[j][1])
                elif max(*i_list[i]) == i_list[j][1]:
                    i_list[j] = (i_list[j][0], min(*i_list[i]))
                    
        labels = c_labels

    ids = np.unique(labels)

    return ids, labels



def cluster_properties(
        dfield: np.ndarray, X: np.ndarray, Y: np.ndarray,
        ids: np.ndarray, labels: np.ndarray, pbc: bool = True,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,\
                   np.ndarray, np.ndarray]:
    r'''
    Calculates the size, geometric center, center of mass, radius of gyration
    and linear extension of each cluster in a continuum field.

    Parameters
    ----------
    dfield : np.ndarray of shape (N,M)
        Two dimensional density field.
    X : np.ndarray of shape (N,M)
        x coordinates of grid cells in form of a meshgrid with same shape as
        dfield.
    Y : np.ndarray of shape (N,M)
        y coordinates of grid cells in form of a meshgrid with same shape as
        dfield.
    ids : np.ndarray
        An array of the cluster IDs, starting from 0.
    labels : np.ndarray of shape (N,M)
        Array of the same shape as the input field, where each pixel
        or grid point is labeled with the cluster ID it belongs to.
    pbc : bool, optional
        Whether to use periodic boundary conditions. Default is True.


    Returns
    -------
    sizes : np.ndarray
        The respective size of each cluster. Size in this context means field
        value integrated over the cluster.
    geometric_centers : np.ndarray
        The geometric centers of the clusters.
    mass_centers : np.ndarray
        The centers of masses of the clusters.
    radii_of_gyration : np.ndarray
        The radii of gyration of the clusters.
    linear_extensions : np.ndarray
        The linear extensions of the clusters.
    gyration_tensors : np.ndarray
        The gyration tensor of each cluster.
    inertia_tensors : np.ndarray
        The inertia tensor of each cluster.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> X, Y = np.meshgrid(
    ...     np.linspace(0, np.pi, 101), np.linspace(0, np.pi, 101)
    ... )
    >>> dfield = np.cos(2*np.pi*X)*np.cos(2*np.pi*Y)
    >>> ids, labels = amep.continuum.identify_clusters(
    ...     dfield, scale=1.0, cutoff=0.8, threshold=0.2,
    ...     method="threshold", pbc=True
    ... )
    >>> sizes, gmcs, coms, rgs, les, gts, its = amep.continuum.cluster_properties(
    ...     dfield, X, Y, ids, labels, pbc=True
    ... )
    >>> fig, axs = amep.plot.new(figsize=(3.75, 3))
    >>> mp = amep.plot.field(axs, dfield, X, Y, cmap="gray")
    >>> cax = amep.plot.add_colorbar(fig, axs, mp, label=r"$\rho$")
    >>> axs.scatter(
    ...     gmcs[1:,0], gmcs[1:,1], marker="o", c="blue",
    ...     label="geometric center", s=20
    ... )
    >>> axs.scatter(
    ...     coms[1:,0], coms[1:,1], marker="+", c="red",
    ...     label="center of mass", s=40
    ... )
    >>> angle = np.linspace(0, 2*np.pi, 50)
    >>> x = rgs[4]*np.cos(angle)+coms[4,0]
    >>> y = rgs[4]*np.sin(angle)+coms[4,1]
    >>> axs.plot(x, y, c="orange", ls="-", lw=1, label="radius of gyration", marker="")
    >>> x = 0.5*les[4]*np.cos(angle)+coms[4,0]
    >>> y = 0.5*les[4]*np.sin(angle)+coms[4,1]
    >>> axs.plot(x, y, c="orange", ls='--', lw=1, label="linear extension", marker="")
    >>> axs.legend(frameon=True, loc="upper left")
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/continuum/continuum-cluster_properties.png")

    .. image:: /_static/images/continuum/continuum-cluster_properties.png
      :width: 400
      :align: center

    '''
    # get box length and center
    box_lengths = np.asarray([np.max(X)-np.min(X), np.max(Y)-np.min(Y), 1.0])
    box_boundary = np.asarray([
        [np.min(X), np.max(X)],
        [np.min(Y), np.max(Y)],
        [-0.5, 0.5]
    ])
    # calculate geometric centers
    geometric_centers = [geometric_center(np.stack((
        X[labels==i],
        Y[labels==i],
        np.zeros_like(Y[labels==i])
    )).transpose(), box_boundary, pbc=pbc) for i in ids]
    
    # calculate centers of mass
    mass_centers = [center_of_mass(np.stack((
        X[labels==i],
        Y[labels==i],
        np.zeros_like(Y[labels==i])
    )).transpose(), box_boundary, dfield[labels==i], pbc=pbc) for i in ids]
        
    # calculate sizes=particle number or total mass of each cluster
    sizes = []
    for i in ids:
        y = dfield.copy()
        # set all grid points outside the cluster to zero
        y[labels != i] = 0
        # integral
        if Version(np.__version__)<Version("2.0.0"):
            N = np.trapz(y, x=X, axis=1)
            N = np.trapz(N, x=Y[:, 0])
        else:
            N = np.trapezoid(y, x=X, axis=1)
            N = np.trapezoid(N, x=Y[:, 0])
        sizes.append(N)

    # calculate the radius of gyration and linear extension
    # (end-to-end distance) of each cluster
    radii_of_gyration = []
    linear_extensions = []

    for n,i in enumerate(ids):
        y = dfield.copy()
        # set all grid points outside the cluster to zero
        y[labels != i] = 0
        # calculate distances to the center of mass
        dist = distance_matrix(
            np.vstack([
                X[labels==i].ravel(),
                Y[labels==i].ravel(),
                np.zeros_like(Y[labels==i].ravel())
            ]).transpose(),
            box_boundary = np.array([
                [np.min(X), np.max(X)],
                [np.min(Y), np.max(Y)],
                [-0.001,0.001]
            ]),
            other = np.array(mass_centers)[n].reshape(-1,3),
            pbc = pbc,
            maxdist = np.max(box_lengths)
        )
        radii_of_gyration.append(
            np.sqrt(np.sum(y[labels==i, None]*dist**2)/np.sum(y[labels==i, None]))
        )
        linear_extensions.append(2.0*np.max(dist))
        
    # calculate gyration tensors
    gyration_tensors = [
        gyration_tensor(np.stack(
            (X[labels==i], Y[labels==i], np.zeros_like(Y[labels==i]))
        ).transpose(), box_boundary, dfield[labels==i], pbc=pbc) for i in ids
    ]
    
    # calculate inertia tensors
    inertia_tensors = [
        inertia_tensor(np.stack(
            (X[labels==i], Y[labels==i], np.zeros_like(Y[labels==i]))
        ).transpose(), box_boundary, dfield[labels==i], pbc=pbc) for i in ids
    ]

    return np.asarray(sizes), np.asarray(geometric_centers),\
           np.asarray(mass_centers), np.asarray(radii_of_gyration),\
           np.asarray(linear_extensions), np.asarray(gyration_tensors),\
           np.asarray(inertia_tensors)
