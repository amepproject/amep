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
Statistics and Distributions
============================

.. module:: amep.statistics

The AMEP module :mod:`amep.statistics` provides various statistical quantities 
such as histograms or the Binder cumulant.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import numpy as np

from scipy import stats
from .utils import runningmean


# =============================================================================
# CUMULANTS
# =============================================================================
def binder_cumulant(data: np.ndarray) -> float:
    r'''
    Calculates the Binder cumulant for the data given in the array.

    Notes
    -----
    The Binder cumulant of a quantity x is defined by

    .. math::
        
        U_4 = 1 - \frac{\langle x^4\rangle}{3\langle x^2\rangle^2}.

    See Refs. [1]_ [2]_ [3]_ for further information.

    References
    ----------

    .. [1] Binder, K. (1981). Finite size scaling analysis of ising model
       block distribution functions. Zeitschrift Für Physik B Condensed Matter,
       43(2), 119–140. https://doi.org/10.1007/BF01293604

    .. [2] Binder, K. (1981). Critical Properties from Monte Carlo Coarse
       Graining and Renormalization. Physical Review Letters, 47(9), 693–696.
       https://doi.org/10.1103/PhysRevLett.47.693

    .. [3] Digregorio, P., Levis, D., Suma, A., Cugliandolo, L. F.,
       Gonnella, G., & Pagonabarraga, I. (2018). Full Phase Diagram of Active
       Brownian Disks: From Melting to Motility-Induced Phase Separation.
       Physical Review Letters, 121(9), 098003.
       https://doi.org/10.1103/PhysRevLett.121.098003

    Parameters
    ----------
    data : np.ndarray
        Data as 1D array of floats.

    Returns
    -------
    float
        Binder cumulant.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> ndata = np.random.normal(size=100000)
    >>> rdata = np.random.rand(100000)
    >>> a = amep.statistics.binder_cumulant(ndata)
    >>> b = amep.statistics.binder_cumulant(rdata)
    >>> print(a,b)
    0.0009875868561266543 0.40026594013163164
    >>>
    
    '''
    mu2 = stats.moment(data, moment=2, axis=None)
    mu4 = stats.moment(data, moment=4, axis=None)

    return 1 - mu4/(3*mu2**2)


# =============================================================================
# DISTRIBUTIONS
# =============================================================================
def distribution(
        data: np.ndarray, weights: np.ndarray | None = None,
        xmin: float | None = None, xmax: float | None = None,
        nbins: int | None = 10, density: bool = True,
        logbins: bool = False) -> tuple[np.ndarray, np.ndarray]:
    r'''
    Calculates the distribution function of the given data
    from a histogram.

    Notes
    -----
    An optimal number of bins can be estimated using the Freedman–Diaconis 
    rule (see https://en.wikipedia.org/wiki/Freedman–Diaconis_rule and 
    Ref. [1]_ for further information). If nbins is set to None, this rule will
    be applied. Note that an error could occur for very large data arrays.
    Therefore, for large data arrays, it is recommended to fix the number of
    bins manually.

    References
    ----------

    .. [1] Freedman, D., & Diaconis, P. (1981). On the histogram as a density
       estimator:L 2 theory. Zeitschrift Für Wahrscheinlichkeitstheorie Und
       Verwandte Gebiete, 57(4), 453–476. https://doi.org/10.1007/BF01025868

    Parameters
    ----------
    data : np.ndarray of shape (M,)
        Data of which the distribution should be calculated.
    weights : np.ndarray or None, optional
        Weight each data point with these weights. Must have the same shape as
        `data`. If `density` is True, the weights are normalized. The default 
        is None.
    xmin : float, optional
        Minimum value of the bins. The default is None.
    xmax : float, optional
        Maximum value of the bins. The default is None.
    nbins : int or None, optional
        Number of bins. If None, the Freedman-Diaconis rule [1]_ is used to 
        estimate an optimal number of bins. Using this rule is only recommended
        for small data arrays. The default is 10.
    density : bool, optional
        If True, the distribution is normalized. If False, a simple
        histogram is returned. The default is True.
    logbins : bool, optional
        If True, the bins are logarithmically spaced. Only possible
        when nbins is given. The default is False.

    Returns
    -------
    np.ndarray
        Histogram/Distribution function.
    np.ndarray
        Bins.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> ndata = np.random.normal(
    ...     loc=0.0, scale=1.0, size=100000
    ... )
    >>> a, abins = amep.statistics.distribution(
    ...     ndata, nbins=None
    ... )
    >>> gfit = amep.functions.NormalizedGaussian()
    >>> gfit.fit(abins, a)
    >>> print(gfit.results)
    {'mu': (0.003195771437101405, 0.002420397928883222),
     'sig': (0.9982728437059646, 0.001951630585125867)}
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(abins, a, label='histogram', ls='')
    >>> axs.plot(
    ...     abins, gfit.generate(abins),
    ...     label='Gaussian fit', marker=''
    ... )
    >>> axs.legend()
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$p(x)$')
    >>> fig.savefig('./figures/statistics/statistics-distribution.png')
    >>>
    
    .. image:: /_static/images/statistics/statistics-distribution.png
      :width: 400
      :align: center
    
    '''
    if xmin is None:
        xmin = np.min(data)
    
    if xmax is None:
        xmax = np.max(data)
    
    if nbins==0:
        raise ValueError(
            "The value of nbins is set to 0. Please use nbins > 0."
        )
    if nbins is None:
        # error for logbins
        if logbins:
            raise ValueError(
                "amep.statistics.distribution: logbins=True only possible if "\
                "nbins is given. Please specify nbins."
            )
        # get optimal bin width from Freedman–Diaconis rule
        bin_width = 2*stats.iqr(data)/(len(data))**(1./3.)
        bin_edges = np.arange(xmin, xmax+bin_width, bin_width)
    else:
        if logbins:
            if xmin <= 0 or xmax <= 0:
                raise ValueError(
                    "Logarithmic bins are not compatible with xmin or xmax "\
                    "smaller than or equal to zero. "\
                    f"Got xmin={xmin} and xmax={xmax}."
                )
            bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
        else:
            bin_edges = np.linspace(xmin, xmax, nbins+1)
        
    # histogram
    hist, bin_edges = np.histogram(
        data,
        bins = bin_edges,
        density = density,
        weights = weights
    )
    return hist, runningmean(bin_edges, 2)


# =============================================================================
# HISTOGRAM2D
# =============================================================================
def histogram2d(
        xdata: np.ndarray, ydata: np.ndarray, xmin: float | None = None,
        xmax: float | None = None, ymin: float | None = None,
        ymax: float | None = None, nxbins: int | None = None,
        nybins: int | None = None, density: bool = True,
        xlogbins: bool = False, ylogbins: bool = False
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r'''
    Calculates the two-dimensional distribution function of the given data
    from a 2d-histogram.

    Notes
    -----
    If the number of bins nbins is not given, the optimal number
    is estimated using the Freedman–Diaconis rule (see
    https://en.wikipedia.org/wiki/Freedman–Diaconis_rule or Ref. [1]_ for
    further information).

    References
    ----------

    .. [1] Freedman, D., & Diaconis, P. (1981). On the histogram as a density
       estimator:L 2 theory. Zeitschrift Für Wahrscheinlichkeitstheorie Und
       Verwandte Gebiete, 57(4), 453–476. https://doi.org/10.1007/BF01025868

    Parameters
    ----------
    xdata : np.ndarray
        Input data of first dimension.
    ydata : np.ndarray
        Input data of second dimension.
    xmin : float or None, optional
        Minimum value of the bins in first dimension. The default is None.
    xmax : float or None, optional
        Maximum value of the bins in first dimension. The default is None.
    ymin : float or None, optional
        Minimum value of the bins in second dimension. The default is None.
    ymax : float or None, optional
        Maximum value of the bins in second dimension. The default is None.
    nxbins : int or None, optional
        Number of bins in first dimension. The default is None.
    nybins : int or None, optional
        Number of bins in second dimension. The default is None.
    density : bool, optional
        If True, the distribution is normalized. If False, a simple
        histogram is returned. The default is True.
    xlogbins : bool, optional
        If True, the bins are logarithmically spaced in the first dimension.
        Only possible when nxbins is given. The default is False.
    ylogbins : bool, optional
        If True, the bins are logarithmically spaced in the second dimension.
        Only possible when nxbins is given. The default is False.


    Returns
    -------
    np.ndarray
        Histogram/Distribution function.
    np.ndarray
        Bins.


    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> n = 10000
    >>> x = np.random.standard_normal(n)
    >>> y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
    >>> hist, xedges, yedges = amep.statistics.histogram2d(x, y)
    >>> X,Y = np.meshgrid(xedges, yedges, indexing="ij")
    >>> fig, axs = amep.plot.new()
    >>> mp = amep.plot.field(axs, hist, X, Y)
    >>> cax = amep.plot.add_colorbar(fig, axs, mp, label=r"$p(x,y)$")
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig('./figures/statistics/statistics-histogram2d.png')
    >>> 

    .. image:: /_static/images/statistics/statistics-histogram2d.png
      :width: 400
      :align: center

    '''
    if xmin is None:
        xmin = np.min(xdata)
    
    if xmax is None:
        xmax = np.max(xdata)
        
    if ymin is None:
        ymin = np.min(ydata)
    
    if ymax is None:
        ymax = np.max(ydata)
        
    if nxbins is None:
        # get optimal bin width from Freedman–Diaconis rule
        xbin_width = 2*stats.iqr(xdata)/(len(xdata))**(1./3.)
        xbin_edges = np.arange(xmin, xmax+xbin_width, xbin_width)
    else:
        if xlogbins:
            xbin_edges = np.logspace(np.log10(xmin), np.log10(xmax), nxbins+1)
        else:
            xbin_edges = np.linspace(xmin, xmax, nxbins+1)
            
    if nybins is None:
        # get optimal bin width from Freedman–Diaconis rule
        ybin_width = 2*stats.iqr(ydata)/(len(ydata))**(1./3.)
        ybin_edges = np.arange(ymin, ymax+ybin_width, ybin_width)
    else:
        if ylogbins:
            ybin_edges = np.logspace(np.log10(ymin), np.log10(ymax), nybins+1)
        else:
            ybin_edges = np.linspace(ymin, ymax, nybins+1)
        
    # histogram
    hist, xbin_edges, ybin_edges = np.histogram2d(xdata, ydata, bins=[xbin_edges,ybin_edges], density=density)

    return hist, runningmean(xbin_edges,2), runningmean(ybin_edges,2)
