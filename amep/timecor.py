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
Time Correlation Functions
==========================

.. module:: amep.timecor

The AMEP module :mod:`amep.timecor` provides methods that calculate time
correlation functions between two simulation frames of particle-based data.

"""
# ============================================================================
# IMPORT MODULES
# ============================================================================
import numpy as np


# ============================================================================
# TIME CORRELATION FUNCTIONS
# ============================================================================
def msd(start, frame):
    '''
    Calculates the mean-square displacement.

    Parameters
    ----------
    start : np.ndarray
        Unwrapped or nojump coordinates at the start time.
    frame : np.ndarray
        Unwrapped or nojump coordinates at a later time.

    Returns
    -------
    float
        Mean-square displacement averaged over all given coordinates.

    '''
    vec = start - frame
    return (vec ** 2).sum(axis=1).mean()


def acf(start, frame):
    '''
    Calculates the autocorrelation function of the given data arrays as
    average over all particles.

    Parameters
    ----------
    start : np.ndarray
        Data array at time 0.
    frame : np.ndarray
        Data array at time t>0.

    Returns
    -------
    float
        Autocorrelation function.

    '''
    cor = 0

    if np.shape(np.shape(start))[0] > 1:
        for i in range(np.shape(start)[1]):
            cor += (start[:,i]*frame[:,i]).mean()
    else:
        cor = (start*frame).mean()

    return cor      


def isf(start, frame, k):
    r'''
    Incoherent intermediate scattering function (ISF).

    Notes
    -----
    The incoherent intermediate scattering function is defined as

    .. math::

        F_{\text{s}}\left(\vec{k},t\right)=\frac{1}{N}\left\langle\sum_{j=1}^{N}\exp\left[i\vec{k}\cdot\left(\vec{r}_j(t)-\vec{r}_j(0)\right)\right]\right\rangle

    Here, we use the isotropic ISF

    .. math::

        F_{\text{s}}(k,t)=\frac{1}{N}\left\langle\sum_{j=1}^{N}\frac{\sin\left(k\cdot\left|\vec{r}_j\left(t_0+t\right)-\vec{r}_j\left(t_0\right)\right|\right)}{k\cdot\left|\vec{r}_j\left(t_0+t\right)-\vec{r}_j\left(t_0\right)\right|}\right\rangle_{t_0}.

    that can be obtained via taking the mean over all directions of :math:`\vec{k}`.
    A detailed derivation can be found in Ref. [1]_.

    References
    ----------

    .. [1] C. L. Farrow and S. J. L. Billinge, Relationship between the Atomic
        Pair Distribution Function and Small-Angle Scattering: Implications for
        Modeling of Nanoparticles,
        Acta Crystallogr. Sect. A Found. Crystallogr. 65, 232 (2009).
        https://doi.org/10.1107/S0108767309009714

    Parameters
    ----------
    start : np.ndarray
        Unwrapped or nojump coordinates at the start time.
    frame : np.ndarray
        Unwrapped or nojump coordinates at a later time.
    k : float
        Wave number.

    Returns
    -------
    float
        Incoherent intermediate scattering function.

    '''
    vec = start - frame
    distance = (vec**2).sum(axis=1)**.5
    return np.sinc(distance * k / np.pi).mean()
