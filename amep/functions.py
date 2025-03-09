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
# GNU General Public License for more details.  #
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Contact: Lukas Hecht (lukas.hecht@pkm.tu-darmstadt.de)
# =============================================================================
"""
Functions and Fitting
=====================

.. module:: amep.functions

The AMEP module :mod:`amep.functions` contains regularly used functions to fit
to data as well as fitting methods.
The naming covention is that the fit classes are named in CamelCase while the
mathematical functions that can be fitted are name all lowercase.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import inspect
import numpy as np

from scipy.special import erf, erfc, gamma
from .base import BaseFunction

# =============================================================================
# GAUSSIANS
# =============================================================================
def gaussian(
        x: np.ndarray, mu: float = 0.0, sig: float = 1.0, offset: float = 0.0,
        A: float = 1.0, normalized: bool = False) -> np.ndarray:
    r'''
    Gaussian Bell curve.

    Equivalent to the probability density function of a normal distribution.
    By the central limit theorem this is a good guess for most peak shapes
    that arise from many random processes.
    The function is given by

    .. math::
        g(x)=A\exp\left(-\frac{\left(x-\mu\right)^2}{\sigma^2}\right)+b.

    Parameters
    ----------
    x : np.ndarray
        :math:`x` values.
    mu : float
        Mean value :math:`\mu`.
    sig : float
        Standard deviation :math:`\sigma`.
    offset : float, optional
        Shifts the Gaussian by :math:`b` in y direction. The default is 0.0.
    A : float, optional
        Amplitude :math:`A`. The default is 1.0.
    normalized : bool, optional
        If True, the Gaussian is normalized to unit area. The default is False.

    Returns
    -------
    np.ndarray
        g(x)

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(-10, 10, 200)
    >>> y = amep.functions.gaussian(x, mu=0.0, sig=1.0)
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(x,y)
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$f(x)$")
    >>> fig.savefig("./figures/functions/functions-gaussian-function.png")

    .. image:: /_static/images/functions/functions-gaussian-function.png
      :width: 400
      :align: center

    '''
    if normalized:
        A = 1.0/(np.sqrt(2*np.pi) * sig)

    return A*np.exp(-((x-mu)**2)/2/sig**2) + offset


def gaussian2d(
        data_tuple: tuple[np.ndarray, np.ndarray], A: float, mux: float,
        muy: float, sigx: float, sigy: float, theta: float, offset: float
        ) -> np.ndarray:
    r'''
    2D Gaussian Bell curve.


    Equivalent to the probability density function of a normal distribution
    in two dimensions.
    By the central limit theorem this is a good guess for most peak shapes
    that arise from many random processes.
    This parametrization can include correlations
    via the angle variable :math:`\theta`.
    The function is given by

    .. math::
        g(x)= A\exp\left(-\left(\vec{x}-\vec{\mu}\right)^T
                R(\Theta)^{-1}\sigma^{-2}R(\Theta)
                \left(\vec{x}-\vec{\mu}\right)
        \right)+b

    where :math:`\vec{x}` is the vector composed the x and y coordinates,
    :math:`\vec{\mu}` is the mean vector composed
    of :math:`\mu_x` and :math:`\mu_y` and
    :math:`\sigma^{-2}` is the diagonal matrix with the inverse
    variances :math:`\sigma_x^{-2}` and :math:`\sigma_y^{-2}` as entries.

    Parameters
    ----------
    data : tuple
        tuple (x,y) of x and y values where x and y
        are 1D np.ndarrays.
    A : float
        amplitude.
    mux : float
        mean :math:`\mu_x` in x direction.
    muy : float
        mean :math:`\mu_y` in y direction.
    sigx : float
        standard deviation :math:`\sigma_x` in x direction.
    sigy : float
        standard deviation :math:`\sigma_y`  in y direction.
    theta : float
        Orientation angle :math:`\Theta` of the polar axis.
    offset : float
        offset :math:`b`. Shifts the output value linearly.

    Returns
    -------
    g(x): np.ndarray
        1D array of floats (flattended 2D array!)


    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(-10, 10, 500)
    >>> y = np.linspace(-10, 10, 500)
    >>> X,Y = np.meshgrid(x,y)
    >>> z = amep.functions.gaussian2d(
    ...     (X,Y), 1.0, 0.0, 3.0, 2.0, 5.0, np.pi/3, 0.0
    ... ).reshape((len(x),len(y)))
    >>> fig, axs = amep.plot.new(figsize=(3,3))
    >>> amep.plot.field(axs, z, X, Y)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/functions/functions-gaussian2d-function.png')

    .. image:: /_static/images/functions/functions-gaussian2d-function.png
      :width: 400
      :align: center

    '''
    (x, y) = data_tuple
    mux = float(mux)
    muy = float(muy)
    a = (np.cos(theta)**2)/(2*sigx**2) + (np.sin(theta)**2)/(2*sigy**2)
    b = -(np.sin(2*theta))/(4*sigx**2) + (np.sin(2*theta))/(4*sigy**2)
    c = (np.sin(theta)**2)/(2*sigx**2) + (np.cos(theta)**2)/(2*sigy**2)
    g = offset + A*np.exp(a*((x-mux)**2)
                          + 2*b*(x-mux)*(y-muy)
                          + c*((y-muy)**2))
    return g.ravel()


class Gaussian(BaseFunction):
    """One-dimensional Gaussian.
    """
    def __init__(self) -> None:
        r'''
        Initialize a Function object for a one-dimensional Gaussian function.

        Notes
        -----
        This Gaussian function has three parameters :math:`\mu` (mean value),
        :math:`\sigma` (standard deviation), and :math:`a` (amplitude).
        It can be written as

        .. math::

            g(x) = a\exp\left\lbrace\frac{(x-\mu)^2}{2\sigma^2}\right\rbrace

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> import numpy as np
        >>> g = amep.functions.Gaussian()
        >>> x = np.linspace(-10, 10, 500)
        >>> y = g.generate(
        ...     x, p=[1.5, 2.0, 3.0]
        ... ) + 0.25*np.random.normal(size=x.shape)
        >>> g.fit(x, y)
        >>> print(g.results)
        {'mu': (1.4700012577518546, 0.028756071777006377),
         'sig': (1.9703218241082439, 0.026488071823107234),
         'a': (3.0643031127741156, 0.0341113309891326)}
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(x, y, label='raw data', ls="")
        >>> axs.plot(
        ...     x, g.generate(x), label='Gaussian fit',
        ...     marker="", lw=2
        ... )
        >>> axs.set_xlabel(r'$x$')
        >>> axs.set_ylabel(r'$g(x)$')
        >>> axs.legend()
        >>> fig.savefig('./figures/functions/functions-Gaussian.png')

        .. image:: /_static/images/functions/functions-Gaussian.png
          :width: 400
          :align: center

        '''
        super().__init__(3)

        self.name = 'Gaussian'
        self.keys = ['mu', 'sig', 'a']

    def f(
            self, p: list | np.ndarray,
            x: float | np.ndarray) -> float | np.ndarray:
        r'''
        Non-normalized Gaussian function in 1d of the form

        .. math::

            g(x) = a\exp\left\lbrace\frac{(x-\mu)^2}{2\sigma^2}\right\rbrace.

        Parameters
        ----------
        p : list or np.ndarray
            Parameters :math:`\mu`, :math:`\sigma`,
            and :math:`a` of the Gaussian function.
        x : float or np.ndarray
            Value(s) at which the function is evaluated.

        Returns
        -------
        float or np.ndarray
            Function evaluated at the given x value(s).

        '''
        return p[2] * np.exp(-((x-p[0])**2) / 2 / p[1]**2)


class NormalizedGaussian(BaseFunction):
    r"""Normalized one-dimensional Gaussian.

    Has the mathematical form

        .. math::

            g(x) = \frac{1}{\sqrt{2\pi}\sigma}
                \exp\left\lbrace\frac{(x-\mu)^2}{2\sigma^2}\right\rbrace.
    """
    def __init__(self) -> None:
        r'''
        Initialize the Function object.

        This function is a normalized one-dimensional Gaussian
        function.

        Notes
        -----
        This Gaussian function has two parameters `mu` (mean value) and `sig`
        (standard deviation). It can be written as
        
        .. math..:
            
            g(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left\lbrace\right\frac{(x-\mu)^2}{2\sigma^2}\rbrace.

        Returns
        -------
        None.
        
        Examples
        --------
        >>> import amep
        >>> import numpy as np
        >>> g = amep.functions.NormalizedGaussian()
        >>> x = np.linspace(-10, 10, 500)
        >>> y = g.generate(
        ...     x, p=[1.5, 2.0]
        ... ) + 0.025*np.random.normal(size=x.shape)
        >>> g.fit(x, y)
        >>> print(g.results)
        {'mu': (1.501942256986355, 0.03582565085467049),
         'sig': (2.050890988264332, 0.029231471248178938)}
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(x, y, label='raw data', ls="")
        >>> axs.plot(
        ...     x, g.generate(x),  marker="", lw=2,
        ...     label='normalized Gaussian fit'
        ... )
        >>> axs.set_xlabel(r'$x$')
        >>> axs.set_ylabel(r'$g(x)$')
        >>> axs.legend()
        >>> fig.savefig('./figures/functions/functions-NormalizedGaussian.png')

        .. image:: /_static/images/functions/functions-NormalizedGaussian.png
          :width: 400
          :align: center

        '''     
        super().__init__(2)

        self.name = 'NormalizedGaussian'
        self.keys = ['mu', 'sig']

    def f(
          self, p: list | np.ndarray,
          x: float | np.ndarray) -> float | np.ndarray:
        r'''
        Normalized one-dimensional Gaussian function of the form

        .. math::

            g(x) = \frac{1}{\sqrt{2\pi}\sigma}
                \exp\left\lbrace\frac{(x-\mu)^2}{2\sigma^2}\right\rbrace.

        Parameters
        ----------
        p : list or np.ndarray
            Parameters :math:`\mu` and :math:`\sigma`
            of the normalized Gaussian function.
        x : float | np.ndarray
            Value(s) at which the function is evaluated.

        Returns
        -------
        float or np.ndarray
            Function evaluated at the given x value(s).

        '''
        A = 1.0 / (np.sqrt(2*np.pi) * p[1])
        return A * np.exp(-((x-p[0])**2) / 2 / p[1]**2)


class ExGaussian(BaseFunction):
    """Exponentially modified Gaussian.
    """
    def __init__(self) -> None:
        r'''
        Initialize a function object for an exponentially modified Gaussian.

        Notes
        -----
        The exponentially modified Gaussian function has three parameters
        :math:`\lambda, \mu`, and :math:`\sigma`, and it is defined as

        .. math::

            g_{\rm ex}(x) = \frac{\lambda}{2} e^{\frac{\lambda}{2}
             (2 \mu + \lambda \sigma^2 - 2 x)}{\rm erfc}
             \left(\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma}\right),

        where :math:`{\rm erfc}` is the complementary error function [1]_. The
        parameters are ordered as follows:

            p[0] : :math:`\lambda`

            p[1] : :math:`\mu`

            p[2] : :math:`\sigma`

        The mean value of the distribution is given by :math:`\mu+1/\lambda`
        and the standard deviation by :math:`\sqrt{\sigma^2 + 1/\lambda^2}`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> import numpy as np
        >>> g = amep.functions.ExGaussian()
        >>> x = np.linspace(-10, 10, 500)
        >>> y = g.generate(
        ...     x, p=[1.0, -5.0, 2.0]
        .. ) + 0.002*np.random.normal(size=x.shape)
        >>> g.fit(x, y)
        >>> print(g.results)
        {'lambda': (0.9531534827685344, 0.039017907566076315),
         'mu': (-5.12948790385678, 0.09219256237461067),
         'sigma': (2.0659374733316973, 0.05531580274767071)}
        >>> print(g.mean)
        -4.080338928294738
        >>> print(g.std)
        2.3170695321114207
        

        >>> fig, axs = amep.plot.new()
        >>> axs.plot(x, y, label='raw data', ls="")
        >>> axs.plot(
        ...     x, g.generate(x), marker="", lw=2,
        ...     label='exponentially modified Gaussian fit'
        ... )
        >>> axs.set_xlabel(r'$x$')
        >>> axs.set_ylabel(r'$g(x)$')
        >>> axs.legend(loc="upper left")
        >>> axs.set_ylim(-0.005, 0.035)
        >>> fig.savefig('./figures/functions/functions-ExGaussian.png')

        .. image:: /_static/images/functions/functions-ExGaussian.png
          :width: 400
          :align: center

        '''
        super().__init__(3)

        self.name = 'Exponentially modified Gaussian'
        self.keys = ['lambda', 'mu', 'sigma']

    def f(
            self, p: list | np.ndarray,
            x: float | np.ndarray) -> float | np.ndarray:
        r'''
        Exponentially modified Gaussian function.
        Has the functional form

        .. math..:

            g_{\rm ex}(x) = \frac{\lambda}{2} e^{\frac{\lambda}{2}
                                (2 \mu + \lambda \sigma^2 - 2 x)}
                                {\rm erfc} \left(
                                \frac{\mu + \lambda \sigma^2 - x}{
                                \sqrt{2} \sigma}\right)

        Parameters
        ----------
        p : list or np.ndarray
            Parameters :math:`\lambda, \mu`, and :math:`\sigma`
            of the exponentially modified Gaussian function.
        x : float | np.ndarray
            Value(s) at which the function is evaluated.

        Returns
        -------
        float or np.ndarray
            Function evaluated at the given x value(s).
        '''
        return (p[0]/2) * np.exp(p[0]*(2*p[1] + p[0]*p[2]**2 - 2*x)/2) *\
            erfc(p[1] + p[0]*p[2]**2 - x)/(np.sqrt(2)*p[2])

    @property
    def mean(self) -> float:
        return self.params[1] + 1.0/self.params[0]

    @property
    def std(self) -> float:
        return np.sqrt(self.params[2]**2 + 1.0/self.params[0]**2)


class SkewGaussian(BaseFunction):
    """Skewed Gaussian distribution.
    """
    def __init__(self) -> None:
        r'''
        Initializes a function object of a skewed normal distribution.

        Notes
        -----
        The skewed normal distribution has three parameters :math:`\mu` (location),
        :math:`\sigma` (width), and :math:`\alpha` (skewness). It is given by [1]_

        .. math::

            g_{\rm skeq}(x) = \frac{1}{\sqrt{2\pi}\sigma}
            \exp\left\lbrace-\frac{(x-\mu)^2}{2\sigma^2}\right\rbrace
            \left[1 + {\rm erf}\left(\frac{\alpha (x-\mu)}{\sqrt{2}\sigma}\right)\right].

        The parameters are ordered as follows:

            p[0] : :math:`\mu`
            
            p[1] : :math:`\sigma`
            
            p[2] : :math:`\alpha`

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Skew_normal_distribution

        Returns
        -------
        None

        Examples
        --------
        >>> import amep
        >>> import numpy as np
        >>> g = amep.functions.SkewGaussian()
        >>> x = np.linspace(-10, 10, 500)
        >>> y = g.generate(
        ...     x, p=[-1.0, 2.0, 4.0]
        ... ) + 0.01*np.random.normal(size=x.shape)
        >>> g.fit(x, y)
        >>> print(g.results)
        {'mu': (-0.997718763927072, 0.009639684113005365),
         'sigma': (1.9920368667592616, 0.016343709582216908),
         'alpha': (3.8547382025213865, 0.12910248961985227)}
        >>> print(g.mean)
        0.5407700079557437
        >>> print(g.std)
        1.2654102802326845


        >>> fig, axs = amep.plot.new()
        >>> axs.plot(x, y, label='raw data', ls="")
        >>> axs.plot(
        ...     x, g.generate(x), marker="", lw=2, 
        ...     label='skewed Gaussian fit'
        ... )
        >>> axs.set_xlabel(r'$x$')
        >>> axs.set_ylabel(r'$g(x)$')
        >>> axs.legend()
        >>> fig.savefig('./figures/functions/functions-SkewGaussian.png')

        .. image:: /_static/images/functions/functions-SkewGaussian.png
          :width: 400
          :align: center

        '''
        super().__init__(3)

        self.name = 'Skew Normal Distribution'
        self.keys = ['mu', 'sigma', 'alpha']

    def f(
            self, p: list | np.ndarray,
            x: float | np.ndarray) -> float | np.ndarray:
        r'''
        Skewed Gaussian function of the form

        .. math::

            g_{\rm skeq}(x) = \frac{1}{\sqrt{2\pi}\sigma}
            \exp\left\lbrace-\frac{(x-\mu)^2}{2\sigma^2}\right\rbrace
            \left[1 + {\rm erf}\left(\frac{\alpha (x-\mu)}{\sqrt{2}\sigma}\right)\right].

        Parameters
        ----------
        p : list | np.ndarray
            Parameters :math:`\mu, \sigma`, and :math:`\alpha` of the skewed
            Gaussian function.
        x : float | np.ndarray
            Value(s) at which the function is evaluated.

        Returns
        -------
        float or np.ndarray
            Function evaluated at the given x value(s).

        '''
        return 1.0/np.sqrt(2*np.pi)/p[1] * np.exp(-(x-p[0])**2/2/p[1]**2) *\
            (1 + erf(p[2]*(x-p[0])/np.sqrt(2)/p[1]))

    @property
    def mean(self) -> float:
        delta = self.params[2]/np.sqrt(1+self.params[2]**2)
        return self.params[0] + self.params[1]*delta*np.sqrt(2/np.pi)

    @property
    def std(self) -> float:
        delta = self.params[2]/np.sqrt(1+self.params[2]**2)
        return np.sqrt(self.params[1]**2*(1-2*delta**2/np.pi))


class MaxwellBoltzmann(BaseFunction):
    """Maxwell-Boltzmann distribution.
    """

    def __init__(self,
                 d: int = 1,
                 m: float = 1.):
        r'''Maxwell-Boltzmann velocity distribution.

        Initializes a function object of a Maxwell-Boltzmann distribution.
        This distribution describes velocities of free particles in
        thermal equilibrium.
        It is usually derived in three dimensions where it is

        .. math::

            f(v) = {\left[\frac{m}{2\pi{}k_bT}\right]}^\frac{3}{2}4\pi{}
            v^2\exp\left(-\frac{mv^2}{k_bT}\right)

        in arbitrary spatial dimensions :math:`d` it is given by

        .. math::

            f(v) =  \frac{2}{\Gamma\left(\frac{d}{2}\right)}
                    {\left[\frac{m}{k_bT}\right]}^\frac{d}{2}
                    v^{d-1}\exp\left(-\frac{mv^2}{k_bT}\right)

        Parameters
        ----------
        d : int, optional
            Spatial dimension. The default is 1.
        m : float, optional
            Mass (if None, set to unity). The default is None.

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> vdist = amep.evaluate.VelDist(traj, skip=0.0, nav=100)
        >>> vfit = amep.functions.MaxwellBoltzmann(d=2, m=1.0)
        >>> vfit.fit(vdist.v, vdist.vdist, p0=[1000.0])
        >>> fig, axs = amep.plot.new()
        >>> axs.plot(
        ...     vdist.v, vdist.vdist, label='data', ls=""
        ... )
        >>> axs.plot(
        ...     vdist.v, vfit.generate(vdist.v),
        ...     marker="", label='fit'
        ... )
        >>> axs.legend()
        >>> axs.set_xlabel(r'$|\vec{v}|$')
        >>> axs.set_ylabel(r'$p(|\vec{v}|)$')
        >>> fig.savefig('./figures/functions/functions-MaxwellBoltzmann.png')

        .. image:: /_static/images/functions/functions-MaxwellBoltzmann.png
          :width: 400
          :align: center

        '''
        super().__init__(1)

        self.__d = d
        self.__m = m

        self.name = 'Maxwell-Boltzmann Distribution'
        self.keys = ['kbT']

    def f(self,
          p: list[float] | np.ndarray,
          x: float | np.ndarray
          ) -> float | np.ndarray:
        r"""Calculate the value of the Maxwell-Boltzmann distribution.

        This distribution describes velocities of free particles in
        thermal equilibrium.
        It is usually derived in three dimensions where it is

        .. math::

            f(v) = {\left[\frac{m}{2\pi{}k_bT}\right]}^\frac{3}{2}4\pi{}
            v^2\exp\left(-\frac{mv^2}{k_bT}\right)

        in arbitrary spatial dimensions :math:`d` it is given by

        .. math::

            f(v) =  \frac{2}{\Gamma\left(\frac{d}{2}\right)}
                    {\left[\frac{m}{k_bT}\right]}^\frac{d}{2}
                    v^{d-1}\exp\left(-\frac{mv^2}{k_bT}\right)

        Parameters
        ----------
        p: list | np.ndarray
            Parameters :math:`k_{\rm B}T`.
        x: float | np.ndarray
            Velocities(s) :math:`v` at which the function is evaluated.

        """
        return ((self.__m/(2*p[0]))**(self.__d/2)/gamma(self.__d/2)*2 *
                np.exp(-self.__m*x**2/(2*p[0]))*x**(self.__d-1))

    @property
    def mean(self):
        pass

    @property
    def std(self):
        pass


class Gaussian2d(BaseFunction):
    """Two-dimensional Gaussian.
    """

    def __init__(self):
        r'''
        Initializes a function object of Two-dimensional Gaussian function.

        Equivalent to the probability density function of a normal distribution
        in two dimensions.
        By the central limit theorem this is a good guess for most peak shapes
        that arise from many random processes.
        This parametrization can include correlations
        via the angle variable :math:`\theta`.
        The function is given by

        .. math::
            g(x)= A\exp\left(-\left(\vec{x}-\vec{\mu}\right)^T
                    R(\Theta)^{-1}\sigma^{-2}R(\Theta)
                    \left(\vec{x}-\vec{\mu}\right)
            \right)+b

        where :math:`\vec{x}` is the vector composed the x and y coordinates,
        :math:`\vec{\mu}` is the mean vector composed
        of :math:`\mu_x` and :math:`\mu_y` and
        :math:`\sigma^{-2}` is the diagonal matrix with the inverse
        variances :math:`\sigma_x^{-2}` and :math:`\sigma_y^{-2}` as entries.


        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> import numpy as np
        >>> x = np.linspace(-10, 10, 500)
        >>> y = np.linspace(-10, 10, 500)
        >>> X,Y = np.meshgrid(x,y)
        >>> g2d = amep.functions.Gaussian2d()
        >>> Z = g2d.generate(
        ...     amep.utils.mesh_to_coords(X,Y),
        ...     p=[1,0,0,2,5,np.pi/3]
        ... ).reshape(X.shape)
        >>> fig, axs = amep.plot.new(figsize=(3,3))
        >>> amep.plot.field(axs, Z, X, Y)
        >>> axs.set_xlabel(r'$x$')
        >>> axs.set_ylabel(r'$y$')
        >>> fig.savefig('./figures/functions/functions-Gaussian2d.png')

        .. image:: /_static/images/functions/functions-Gaussian2d.png
          :width: 400
          :align: center

        '''
        super().__init__(6)

        self.name = 'Two-dimensional Gaussian'
        self.keys = ['a', 'mux', 'muy', 'sigx', 'sigy', 'theta']

    def f(self, p, x):
        r'''
        2D Gaussian.
        The function is given by

        .. math::
            g(x)= A\exp\left(-\left(\vec{x}-\vec{\mu}\right)^T
                    R(\Theta)^{-1}\sigma^{-2}R(\Theta)
                    \left(\vec{x}-\vec{\mu}\right)
            \right)+b

        where :math:`\vec{x}` is the vector composed the x and y coordinates,
        :math:`\vec{\mu}` is the mean vector composed
        of :math:`\mu_x` and :math:`\mu_y` and
        :math:`\sigma^{-2}` is the diagonal matrix with the inverse
        variances :math:`\sigma_x^{-2}` and :math:`\sigma_y^{-2}` as entries.

        Parameters
        ----------
        p: list
            Parameters :math:`(A,\mu_x,\mu_y,\sigma_x,\sigma_y,\Theta)`.
        x: np.ndarray
            x-values as 2d array of shape (N,2).

        Returns
        -------
        np.ndarray
            1D array of shape (N,).


        Examples
        --------
        >>> 
        '''
        X = x[:, 0]
        Y = x[:, 1]
        a = (np.cos(p[5])**2)/(2*p[3]**2) + (np.sin(p[5])**2)/(2*p[4]**2)
        b = -(np.sin(2*p[5]))/(4*p[3]**2) + (np.sin(2*p[5]))/(4*p[4]**2)
        c = (np.sin(p[5])**2)/(2*p[3]**2) + (np.cos(p[5])**2)/(2*p[4]**2)
        g = p[0]*np.exp(- (a*((X-p[1])**2) + 2*b*(X-p[1])*(Y-p[2])
                           + c*((Y-p[2])**2)))
        return g

    @property
    def mean(self):
        pass

    @property
    def std(self):
        pass


class Fit(BaseFunction):
    """Fitting a 1d function.
    """

    def __init__(self, g, **kwargs):
        r'''
        General fit class that allows to fit a 1d user-defined function to 
        data.

        Parameters
        ----------
        g: function
            The function that will be fitted to the data
        **kwargs: 
            Other keyword arguments are forwarded to the function g. These 
            parameters will be kept constant in the fitting process.

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> import numpy as np
        >>> x = np.linspace(-np.pi, np.pi, 100)
        >>> y = (1+np.random.random((100,)))*np.sin(2*x)
        >>> def f(x, amp=1, omega=1, phi=0):
        ...     return amp*np.sin(omega*x+phi)
        >>> fit = amep.functions.Fit(f)
        >>> fit.fit(x, y, p0=[1,2,3])
        >>> print(fit.results)
        {'amp': (-1.6707345639977673, 0.03611365715752075),
         'omega': (1.980453842406803, 0.018014280164672856),
         'phi': (3.0950628128156383, 0.03345686981008855)}


        >>> fit = amep.functions.Fit(f, omega=2)
        >>> fit.fit(x, y, p0=[1,2])
        >>> print(fit.results)
        {'amp': (1.6786964141227352, 0.03291999452138418),
         'phi': (0.07093556110237159, 0.03115340006227788)}


        >>> fig, axs = amep.plot.new()
        >>> axs.plot(x, y, label="data", ls="")
        >>> axs.plot(x, fit.generate(x), marker="", label="fit")
        >>> axs.legend()
        >>> axs.set_xlabel(r"$x$")
        >>> axs.set_ylabel(r"$\sin(2x)$")
        >>> fig.savefig("./figures/functions/functions-Fit.png")

        .. image:: /_static/images/functions/functions-Fit.png
          :width: 400
          :align: center

        '''
        # inspect(getargspec(func))
        # 1: extract all kwargs of f using the inspect Python library
        inspected_object = inspect.getfullargspec(g)
        # self.keys = inspected_object[0]

        len_diff = len(inspected_object[0]) - len(inspected_object[3])
        fct_call_params_defaults = inspected_object[3]
        fct_def_params = inspected_object[0][len_diff:]
        # 2: Get user defined function to get all parameters and
        # their default values, then get the parameters from
        # the actual function call and categorize parameters
        # - their function call values if they are mentioned
        # - their default values if they are mentioned as
        # "=None" in the function call
        # - fit to them if neither of the above is true

        # Extract all params from def and call of fct
        self.fct_def_params = fct_def_params
        fct_call_params = list(kwargs.keys())
        fct_call_values = list(kwargs.values())
        # Get the length difference between arguments and kwargs,
        # this allows us to match name to value

        # Determine which params have what values
        # in fit and which are to be fitted
        fit_params_input = []
        fit_params_to_determine = []
        for k, param in enumerate(fct_def_params):
            if param in fct_call_params:
                if fct_call_values[fct_call_params.index(param)]:
                    fit_params_input.append(fct_call_values[fct_call_params.index(param)])
                else:
                    fit_params_input.append(fct_call_params_defaults[fct_call_params.index(param)])
            else:
                fit_params_to_determine.append(param)
        super().__init__(len(fit_params_to_determine))

        self.name = 'Custom Fit'

        self.defaults = inspected_object[3]
        self.g = g

        self.fit_params_input = fit_params_input
        self.keys = fit_params_to_determine

        # Fit stuff

    def f(self, p, x):
        kwargs_dict = {key: p[i] for i, key in enumerate(self.keys)}
        i = 0
        for key in self.fct_def_params:
            if key not in kwargs_dict.keys():
                kwargs_dict[key] = self.fit_params_input[i]
                i += 1
        return self.g(x, **kwargs_dict)
