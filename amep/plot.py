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
Visualization
=============

.. module:: amep.plot

The AMEP module :mod:`amep.plot` contains methods to visualize and animate
simulations, observables, and analysis results. It is based on the Matplotlib
Python library. For a detailed customization of plots see also
https://matplotlib.org/stable/tutorials/introductory/customizing.html.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
from typing import Callable, Iterable
from os.path import abspath, dirname, join
from pathlib import Path
import warnings
warnings.simplefilter('always', PendingDeprecationWarning)
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Rectangle, FancyBboxPatch, ConnectionPatch, Circle, FancyArrow
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.animation import FuncAnimation
from tqdm.autonotebook import tqdm

from .trajectory import FieldTrajectory, ParticleTrajectory
from .base import get_module_logger

# logger setup
_log = get_module_logger(__name__)

# =============================================================================
# SET PLOT DEFAULTS
# =============================================================================
def matplotlib_plot_defaults():
    r'''Shortcut function to revert to the matplotlib rcparams defaults.
    
    Returns
    -------
    None.
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)


def style(style_name: str = "", mpl_default: bool = False) -> None:
    r'''
    Set the plot style.

    .. note:: Deprecates in 2.0.0
          `mpl_default: bool` will be removed in an upcoming major release. Please use "matplotlib".
          "amep_latex" and "amep_standard" will be removed. Please use "latex" and "standard" instead.

    Parameters
    ----------
    style_name : str, optional
        Specifies the name of the style to apply. Apart from the Matplotlib
        styles, one can choose the AMEP styles `'amep_latex'` and 
        `'amep_standard'`. The AMEP styles are used per default when AMEP is
        imported.
        Please switch to the modes "latex", "standard" for AMEP-styles and
        "matplotlib" for the default matplotlib style. Any other 'style_name'
        is forwarded to 'matplotlib.pyplot.style.use(style_name)'.
    mpl_default : bool, optional
        Determines whether to apply a style or revert
        to the default pyplot style. The default is False.

    Returns
    -------
    None.
    '''
    if not mpl_default:
        style=""
        if style_name in ('amep_latex', 'amep_standard'):
            warnings.warn("The options <mpl_default: bool>, 'amep_latex' and 'amep_standard' will be removed in an upcoming major release. Please use the modes 'matplotlib', 'latex' and 'standard' instead.", PendingDeprecationWarning)
            style = join(abspath(dirname(__file__)),
                        './styles/',
                        style_name + '.mplstyle')
        elif style_name=="latex":
            style = join(abspath(dirname(__file__)),
                        './styles/',
                        'amep_latex.mplstyle')
        elif style_name=="standard":
            style = join(abspath(dirname(__file__)),
                        './styles/',
                        'amep_standard.mplstyle')
        elif style_name=="matplotlib":
            style = "default"
        else:
            style = style_name
        plt.style.use(style)
    else:
        warnings.warn("The options <mpl_default: bool>, 'amep_latex' and 'amep_standard' will be removed in an upcoming major release. Please use the modes 'matplotlib', 'latex' and 'standard' instead.", PendingDeprecationWarning)
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use('default')


def amep_plot_defaults():
    r'''Shortcut function to set amep plot defaults.
    
    Returns
    -------
    None.
    '''
    if shutil.which('latex'):
        style('amep_latex')
    else:
        _log.info(
            "Could not find a LaTeX distribution - using amep standard style."
        )
        style('amep_standard')

# set plot style
amep_plot_defaults()

# =============================================================================
# PLOT UTILITIES
# =============================================================================
def to_latex(string: str) -> str:
    r"""Replace the mathematical operators with corresponding LaTeX commands.
    
    Parameters
    ----------
    string: str
        String to convert.
        
    Returns
    -------
    out: str
        Converted string.
    """
    symbols = {'<': r'\textless ',
               '>': r'\textgreater ',
               '^': r'\^{}',
               '&': r'\&',
               '_': r'\_'}
    out = string
    for into, outof in symbols.items():
        out = out.replace(into, outof)
    return out

def set_locators(
        axis: mpl.axes.Axes | np.ndarray,
        which: str = 'both', major: float = 10,
        minor: float = 1) -> None:
    r'''
    Set the locations of major and minor ticks.

    Parameters
    ----------
    axis : AxesSubplot or array of AxesSubplot objects
        Axis (matplotlib.pyplot axis object) for which the location of
        ticks should be modified.
    which : str, optional
        Apply the locators to this axis ('x', 'y', or 'both').
        The default is 'both'.
    major : float, optional
        Distance between major ticks. The default is 10.
    minor : float, optional
        Distance between minor ticks. The default is 1.

    Returns
    -------
    None.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> fig, axs = amep.plot.new(ncols=3, figsize=(9,3))
    >>> x = np.arange(20)
    >>> y = 2*x
    >>> axs[0].plot(x, y)
    >>> axs[1].plot(x, y**2)
    >>> axs[2].plot(x, y/2)
    >>> amep.plot.set_locators(axs[:1], which='x', major=5, minor=1)
    >>> amep.plot.set_locators(axs[0], which='y', major=10, minor=2)
    >>> amep.plot.set_locators(axs[1], which='y', major=200, minor=50)
    >>> amep.plot.set_locators(axs[2], which='both', major=5, minor=1)
    >>> fig.savefig('./figures/plot/plot-set_locators.png')
    >>>

    .. image:: /_static/images/plot/plot-set_locators.png
      :width: 600
      :align: center

    '''
    if isinstance(axis, np.ndarray):
        for axe in axis:
            set_locators(axe, which=which, major=major, minor=minor)
    else:
        if which in ('x', 'both'):
            axis.xaxis.set_major_locator(MultipleLocator(major))
            axis.xaxis.set_minor_locator(MultipleLocator(minor))
            axis.xaxis.set_minor_formatter(NullFormatter())
        if which in ('y', 'both'):
            axis.yaxis.set_major_locator(MultipleLocator(major))
            axis.yaxis.set_minor_locator(MultipleLocator(minor))
            axis.yaxis.set_minor_formatter(NullFormatter())


def format_axis(
        axis : mpl.axes.Axes | np.ndarray,
        which : str = 'both',
        axiscolor : str | None = None,
        axiswidth : float | None = None,
        ticks : bool = True,
        ticksbottom : bool | None = None,
        tickstop : bool | None = None,
        ticksleft : bool | None = None,
        ticksright : bool | None = None,
        direction : str | None = None,
        majorlength : int | None = None,
        minorlength : int | None = None,
        tickwidth : float | None = None,
        tickcolor : str | None = None,
        ticklabels : bool = True,
        ticklabelsize : int | None = None,
        ticklabelcolor : str | None = None,
        ticklabelpad : int | None = None,
        ticklabelbottom : bool | None = None,
        ticklabeltop : bool | None = None,
        ticklabelleft : bool | None = None,
        ticklabelright : bool | None = None,
        labels : bool = True,
        labelsize : int | None = None,
        labelcolor : str | None = None,
        xlabelloc : str | None = None,
        ylabelloc : str | None = None,
        titlesize : int | None = None,
        titlecolor : str | None = None,
        backgroundcolor : str | None = None,
        colors : str | None = None
        ) -> None:
    r"""Format the given axis(axes) of the plot.
    
    This method is a wrapper for various Matplotlib methods that allow to
    format one or multiple axis including the formatting of ticks, labels, and
    titles. The default of most keywords is None. In this case, the defaults
    from the plot style are used.

    Parameters
    ----------
    axis : AxesSubplot or array of AxesSubplot objects
        Axis object to which the changes should be applied.
    which : str, optional
        The axis ('x', 'y', or 'both') to which the changes should be applied.
        The default is 'both'.
    axiswidth : float or None, optional
        Axis width in points. The default is None.
    axiscolor : str or None, optional
        Color of the axis lines. Is set to `colors` if `colors` is not None. 
        The default is None.
    tick : bool, optional
        Turns the ticks on and off. The default is True.
    ticksbottom : bool or None, optional.
        Show ticks at the bottom. The default is None.
    tickstop : bool or None, optional
        Show ticks at the top. The default is None.
    ticksleft : bool or None, optional
        Show ticks at the left side. The default is None.
    ticksright : bool or None, optional
        Show ticks at the right side. The default is None.
    direction : str or None, optional
        Direction in which the ticks should point ('in' or 'out').
        The default is None.
    majorlength : int or None, optional
        Length of the major ticks. The default is None.
    minorlength : int or None, optional
        Length of the minor ticks. The default is None.   
    tickwidth : float or None, optional
        Tick width in points. The default is None.
    tickcolor : str or None, optional
        Color of the ticks and tick labels. Is set to `colors` if `colors` is
        not None.The default is None.
    ticklabels : bool, optional
        Turns the tick labels on and off. The default is True.
    ticklabelsize : int or None, optional
        Size of the tick labels. The default is None.
    ticklabelcolor : str or None, optional
        Color of the tick labels. Is set to `colors` if `colors` is not None. 
        The default is None.
    ticklabelpad : float or None, optional
        Distance in points between tick and label. The default is None.
    ticklabelbottom : bool or None, optional
        Show tick labels at the bottom. The default is None.
    ticklabeltop : bool or None, optional
        Show tick labels at the top. The default is None.
    ticklabelleft : bool or None, optional
        Show tick labels at the left side. The default is None.
    ticklabelright : bool, optional
        Show tick labels at the right side. The default is None.
    labels : bool, optional
        Turns the axis labels on and off. The default is True.
    labelsize : int or None, optional
        Size of the axis labels. The default is None.
    labelcolor : str or None, optional
        Color of the axis labels. Is set to `colors` if `colors` is not None.
        The default is None.
    xlabelloc : str or None, optional
        Location of the x label ('top' or 'bottom'). The default is None.
    ylabelloc : str or None, optional
        Location of the y label ('left' or 'right'). The default is None.
    titlesize : int or None, optional
        Size of the title. The default is None.
    titlecolor : int or None, optional
        Color of the title. Is set to `colors` if `colors` is not None. The
        default is None.
    background : str or None, optional
        Background color of the plot. The default is None.
    colors : str or None, optional
        If not None, this color is used for `titlecolor`, `axiscolor`,
        `tickcolor`, `labelcolor`, and `ticklabelcolor`. The default is None.

    Returns
    -------
    None.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> fig, axs = amep.plot.new(nrows=2, ncols=2, figsize=(6,6))
    >>> axs = axs.flatten()
    >>> x = np.arange(20)
    >>> y = 2*x
    >>> for i,ax in enumerate(axs):
    ...     ax.plot(x, y**(i+1))
    >>> axs[0].set_title('title')
    >>> axs[1].set_xlabel(r'$x$'); axs[1].set_ylabel(r'$x^2$')
    >>> axs[3].set_xlabel(r'$x$'); axs[3].set_ylabel(r'$f(x)$')
    >>> axs[3].loglog()
    >>> amep.plot.format_axis(
    ...     axs[0], titlesize=15, titlecolor='red',
    ...     ticklabelpad=15
    ... )
    >>> amep.plot.format_axis(
    ...     axs[1], which='x', majorlength=16, minorlength=10,
    ...     axiswidth=2, colors='orange'
    ... )
    >>> amep.plot.format_axis(
    ...     axs[2], ticks=False, ticklabels=False,
    ...     axiswidth=4, axiscolor='blue'
    ... )
    >>> amep.plot.format_axis(
    ...     axs[3], which='y', ylabelloc='right', ticklabelleft=False,
    ...     ticklabelright=True, ticksleft=False, tickstop=False
    ... )
    >>> amep.plot.format_axis(
    ...     axs, backgroundcolor='lightgray'
    ... )
    >>> fig.savefig('./figures/plot/plot-format_axis.png')
    >>> 

    .. image:: /_static/images/plot/plot-format_axis.png
      :width: 400
      :align: center
    
    """
    if isinstance(axis, np.ndarray):
        # extract all keyword arguments
        kwargs = locals()
        del kwargs['axis']
        for axe in axis:
            # call the method for each axis in the given array
            format_axis(axe, **kwargs)
    else:
        if isinstance(colors, str):
            tickcolor = colors
            labelcolor = colors
            ticklabelcolor = colors
            axiscolor = colors
            titlecolor = colors
        if direction is not None:
            axis.tick_params(
                axis = which, which = 'both', direction = direction
            )
        if majorlength is not None:
            axis.tick_params(
                axis = which, which = 'major', length = majorlength
            )
        if minorlength is not None:
            axis.tick_params(
                axis = which, which = 'minor', length = minorlength
            )
        if tickwidth is not None:
            axis.tick_params(
                axis = which, which = 'both', width = tickwidth
            )
        if tickcolor is not None:
            axis.tick_params(
                axis = which, which = 'both', color = tickcolor
            )
        if ticklabelsize is not None:
            axis.tick_params(
                axis = which, which = 'both', labelsize = ticklabelsize
            )
        if ticklabelcolor is not None:
            axis.tick_params(
                axis = which, which = 'both', labelcolor = ticklabelcolor
            )
        if ticklabelpad is not None:
            axis.tick_params(
                axis = which, which = 'both', pad = ticklabelpad
            )
        # axis specific
        if which == 'x' or which == 'both':
            if ticksbottom is not None:
                axis.tick_params(
                    axis = 'x', which = 'both', bottom = ticksbottom
                )
            if tickstop is not None:
                axis.tick_params(
                    axis = 'x', which = 'both', top = tickstop
                )
            if ticklabeltop is not None:
                axis.tick_params(
                    axis = which, which = 'both', labeltop = ticklabeltop
                )
            if ticklabelbottom is not None:
                axis.tick_params(
                    axis = which, which = 'both', labelbottom = ticklabelbottom
                )
            if isinstance(labelsize, int):
                # set the fontsize of the x-axis label
                axis.xaxis.get_label().set_fontsize(labelsize)
            if isinstance(xlabelloc, str):
                # set the position of the x-axis label
                axis.xaxis.set_label_position(xlabelloc)
            if isinstance(labelcolor, str):
                # set the color of the x-axis label
                axis.xaxis.label.set_color(labelcolor)
            if axiscolor is not None:
                for place in ['top', 'bottom']:
                    # set color of x-axis lines
                    axis.spines[place].set_color(axiscolor)
            if axiswidth is not None:
                for place in ['top', 'bottom']:
                    # set the width of the x-axis lines
                    axis.spines[place].set_linewidth(axiswidth)
            if not ticks:
                # remove all ticks on the x-axis
                #axis.minorticks_off()
                axis.tick_params(
                    axis = 'x', which = 'both', top = False, bottom = False
                )
            if not ticklabels:
                # remove all tick labels on the x-axis
                axis.tick_params(
                    axis = 'x',
                    which = 'both',
                    labelbottom = False,
                    labeltop = False
                )
            if not labels:
                # remove label of the x-axis
                axis.set_xlabel('')
        if which == 'y' or which == 'both':
            if ticksleft is not None:
                axis.tick_params(
                    axis = 'y', which = 'both', left = ticksleft
                )
            if ticksright is not None:
                axis.tick_params(
                    axis = 'y', which = 'both', right = ticksright
                )
            if ticklabelleft is not None:
                axis.tick_params(
                    axis = 'y', which = 'both', labelleft = ticklabelleft
                )
            if ticklabelright is not None:
                axis.tick_params(
                    axis = 'y', which = 'both', labelright = ticklabelright
                )
            if isinstance(labelsize, int):
                # set the fontsize of the y-axis label
                axis.yaxis.get_label().set_fontsize(labelsize)
            if isinstance(ylabelloc, str):
                # set the position of the y-axis label
                axis.yaxis.set_label_position(ylabelloc)
            if isinstance(labelcolor, str):
                # set the color of the y-axis label
                axis.yaxis.label.set_color(labelcolor)
            if axiscolor is not None:
                for place in ['left', 'right']:
                    # set color of y-axis lines
                    axis.spines[place].set_color(axiscolor)
            if axiswidth is not None:
                for place in ['left', 'right']:
                    # set the width of the y-axis lines
                    axis.spines[place].set_linewidth(axiswidth)
            if not ticks:
                # remove all ticks on the y-axis
                #axis.minorticks_off()
                axis.tick_params(
                    axis = 'y', which = 'both', left = False, right = False
                )
            if not ticklabels:
                # remove all tick labels on the y-axis
                axis.tick_params(
                    axis = 'y',
                    which = 'both',
                    labelleft = False,
                    labelright = False
                )
            if not labels:
                # remove label of the y-axis
                axis.set_ylabel('')
        # global
        if backgroundcolor is not None:
            # set the color of the axis background
            axis.set_facecolor(backgroundcolor)
        if isinstance(titlesize, int):
            # set the fontsize of the axis title
            axis.title.set_size(titlesize)
        if isinstance(titlecolor, str):
            # set the color of the axis title
            axis.title.set_color(titlecolor)


def new(figsize: tuple[float, float] = None,
        facecolor: str = 'white',
        nrows: int = 1,
        ncols: int = 1,
        width_ratios: float | None = None,
        height_ratios: float | None = None,
        sharex: bool = False,
        sharey: bool = False,
        wspace: float | None = None,
        hspace: float | None = None,
        w_pad: float | None = None,
        h_pad: float | None = None,
        **kwargs
        ) -> tuple[plt.Figure, mpl.axes.Axes | np.ndarray]:
    r'''
    Create a new matplotlib.pyplot figure (uses matplotlib.pyplot.subplots).

    Parameters
    ----------
    figsize : tuple, optional
        Size of the figure in inches.
        The default, None, reverts to the default amep or
        matplotlib figsize. 
    facecolor : str, optional
        Background color of the figure. The default is 'white'.
    nrows : int, optional
        Number of rows. The default is 1.
    ncols : int, optional
        Number of columns. The default is 1.
    width_ratios : list, optional
        Size ratios of the columns (list of int). The default is None.
    height_ratios : list, optional
        Size ratios of the rows (list of int). The default is None.
    sharex : bool or {'none', 'all', 'row', 'col'}, optional
        Share the same x axis. The default is False.
    sharey : bool or {'none', 'all', 'row', 'col'}, optional
        Share the same y axis. The default is False.
    wspace : float, optional
        Space between the columns. The default is None.
    hspace : float, optional
        Space between the rows. The default is None.
    w_pad : float or None, optional
        Padding around the axes elements in inches. The default is None.
    h_pad : float or None, optional
        Padding around the axes elements in inches. The default is None.
    ** kwargs : keywordarguments
        for dpi and similar. Parsed to matplotlib plt.subplots.

    Returns
    -------
    fig : Figure
        matplotlib.pyplot figure object.
    axes : AxesSubplot
        matplotlib.pyplot axes objects..

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> fig, axs = amep.plot.new()
    >>> x = np.linspace(0,10,1000)
    >>> axs.plot(x, np.sin(x))

    '''
    gridspec = {}
    if width_ratios:
        gridspec['width_ratios'] = width_ratios
    if height_ratios:
        gridspec['height_ratios'] = height_ratios
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        gridspec_kw=gridspec,
        **kwargs
    )
    fig.patch.set_facecolor(facecolor)
    # check if the AMEP default layout engine is used
    if isinstance(
        fig.get_layout_engine(),
        mpl.layout_engine.ConstrainedLayoutEngine
    ):
        fig.get_layout_engine().set(
            w_pad = w_pad,
            h_pad = h_pad,
            hspace = hspace,
            wspace = wspace
        )
    else:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig, axes


def add_colorbar(
        fig: mpl.figure.Figure, axis: mpl.axes.Axes,
        mappable: mpl.cm.ScalarMappable,
        cax_extent: list | None = None,
        label: str = '', **kwargs) -> mpl.axes.Axes:
    r'''Add a colorbar to an existing plot.

    This requires to add a new axis in which the colorbar is shown.

    Parameters
    ----------
    fig : Figure
        Matplotlib.pyplot figure object.
    axis : AxisSubplot
        Matplotlib.pyplot AxisSubplot object to which the colorbar belongs to.
    mappable : mappable
        Matplotlib.pyplot mappable (e.g. a scatter plot or PatchCollection).
    cax_extent : np.ndarray | list | tuple | None, optional
        Relative coordinates of the axis on which the colorbar should be
        plotted starting from the lower left corner:
        `[x_0, y_0, d_x, d_y]`

        x_0 : float
            x coordinate of the lower left corner of the inset figure.
        y_0 : float
            y coordinate of the lower left corner of the inset figure.
        d_x : float
            Width of the inset figure.
        d_y : float
            Height of the inset figure.
            
        If None, the colorbar is bound to the given axis. The default
        is None.
    label : str, optional
        Axis label. The default is ''.
    **kwargs
        All other keyword arguments are forwared to `fig.colorbar`.

    Returns
    -------
    cax : AxesSubplot
        Colorbar axis.
        
    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
    >>> frame = traj[-1]
    >>> X,Y = frame.grid
    >>> C = frame.data('c')
    >>> fig, axs = amep.plot.new(figsize=(5.5,3), ncols=2)
    >>> field1 = amep.plot.field(axs[0], C, X, Y, cmap='viridis')
    >>> cax1 = amep.plot.add_colorbar(
    ...     fig, axs[0], field1, label=r'$c(x,y)$'
    ... )
    >>> field2 = amep.plot.field(axs[1], C, X, Y, cmap='viridis')
    >>> cax2 = amep.plot.add_colorbar(
    ...     fig, axs[1], field2, label=r'$c(x,y)$',
    ...     orientation='horizontal'
    ... )
    >>> axs[0].set_xlabel(r'$x$')
    >>> axs[0].set_ylabel(r'$y$')
    >>> axs[1].set_xlabel(r'$x$')
    >>> axs[1].set_ylabel(r'$y$')
    >>> fig.savefig('./figures/plot/plot-add_colorbar_1.png')    
    >>> 
    
    .. image:: /_static/images/plot/plot-add_colorbar_1.png
      :width: 600
      :align: center
      
      
    >>> fig, axs = amep.plot.new(figsize=(6,3), ncols=2)
    >>> field1 = amep.plot.field(axs[0], C, X, Y, cmap='viridis')
    >>> field2 = amep.plot.field(axs[1], C, X, Y, cmap='viridis')
    >>> cax = amep.plot.add_colorbar(
    ...     fig, axs[1], field2, cax_extent = [1.0, 0.25, 0.02, 0.6],
    ...     label=r'$c(x,y)$'
    ... )
    >>> axs[0].set_xlabel(r'$x$')
    >>> axs[0].set_ylabel(r'$y$')
    >>> axs[1].set_xlabel(r'$x$')
    >>> axs[1].set_ylabel(r'$y$')
    >>> fig.savefig('./figures/plot/plot-add_colorbar_2.png')
    >>> 

    .. image:: /_static/images/plot/plot-add_colorbar_2.png
      :width: 600
      :align: center
    
    '''
    if cax_extent is not None:
        cax = fig.add_axes(cax_extent)
    else:
        cax = None
    c_b = fig.colorbar(
        mappable,
        ax = axis,
        cax = cax,
        **kwargs
    )
    c_b.set_label(label)
    return c_b.ax


def add_inset(
        axis: mpl.axes.Axes,
        indicator_extent: np.ndarray | list | tuple,
        inset_extent: np.ndarray | list | tuple,
        connections: np.ndarray | tuple[tuple[int,int],...] = ((3, 3), (1, 1)),
        arrow_style: str = '-', show_box: bool = True,
        show_connections: bool = True, **kwargs) -> mpl.axes.Axes:
    r'''
    Add an inset plot to the existing plot (axis).

    If used in conjunction with plot.particles the keyword
    `set_ax_limits` has to be set to `False`.

    Parameters
    ----------
    axis : AxisSubplot
        Matplotlib.pyplot AxisSubplot object.
    indicator_extent : np.ndarray | list | tuple
        Absolute coordinates of the area to be enlarged inside the initial plot
        starting from the lower left corner:
        `[x_0, y_0, d_x, d_y]`

        x_0 : float
            x coordinate of the lower left corner of the inset figure.
        y_0 : float
            y coordinate of the lower left corner of the inset figure.
        d_x : float
            Width of the inset figure.
        d_y : float
            Height of the inset figure.
    inset_extent: np.ndarray | list | tuple
        The relative coordinates of the patch the zoomlines are pointing too.
        `[x_0, y_0, d_x, d_y]`
        compare `indicator_extent`
    connections : np.ndarray | tuple[tuple[int,int],...], optional
        Defines the corners to be connected for the zoomlines.
        `((connection_1_start, connection_1_end), (connection_2_start, connection_2_end))`
        The lower left corner has the index 0, continuing counter-clockwise.
    arrow_style : str, optional
        It is used for styling the connection arrow.
        matplotlib.patches.ArrowStyle
        Its default type is '-'.
    show_box : bool, optional
        If True, the inset box is shown. The default is True.
    show_connections : bool, optional
        If True, the connections lines are shown. The default is True.
    **kwargs
        Addtional keyword arguments are forwarded to the Matplotlib patches
        [Rectangle](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html) 
        and [ConnectionPatch](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.ConnectionPatch.html).

    Returns
    -------
    inset : AxisSubplot
        Matplotlib.pyplot AxisSubplot object of the inset figure.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(0,10)
    >>> y = x**2
    >>> fig, axs = amep.plot.new()
    >>> axs.plot(x, y)
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$f(x)=x^2$")
    >>> inset = amep.plot.add_inset(
    ...     axs, [4, 10, 2, 20], [0.15, 0.5, 0.3, 0.3],
    ...     connections = ((0, 0), (2, 2))
    ... )
    >>> inset.plot(x, y)
    >>> inset.set_xlabel(r"$x$")
    >>> inset.set_ylabel(r"$f(x)=x^2$")
    >>> fig.savefig("./figures/plot/plot-add_inset_1.png")
    >>>
    
    .. image:: /_static/images/plot/plot-add_inset_1.png
      :width: 400
      :align: center
      
      
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> fig, axs = amep.plot.new(figsize=(3.6, 3))
    >>> mp = amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius(),
    ...     values=frame.ids(), cmap="rainbow"
    ... )
    >>> cax = amep.plot.add_colorbar(fig, axs, mp, label="id")
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> inset = amep.plot.add_inset(
    ...     axs, [50, 40, 5, 5], [0.15, 0.15, 0.3, 0.3],
    ...     connections=((3, 3), (1, 1)), ls="--"
    ... )
    >>> amep.plot.particles(
    ...     inset, frame.coords(), frame.box, frame.radius(),
    ...     values=frame.ids(), cmap="rainbow", set_ax_limits=False
    ... )
    >>> amep.plot.format_axis(inset, ticklabels=False, ticks=False)
    >>> fig.savefig("./figures/plot/plot-add_inset_2.png")
    >>> 

    .. image:: /_static/images/plot/plot-add_inset_2.png
      :width: 400
      :align: center

    '''
    # coordinate translation only small hack ready to be re-factored
    # translate corner values 0-3 to relative coordinates
    CORNERS = [[0,0], [1,0], [1,1], [0,1]]
    connA_start = CORNERS[connections[0][0]]
    connA_end = CORNERS[connections[0][1]]
    connB_start = CORNERS[connections[1][0]]
    connB_end = CORNERS[connections[1][1]]

    # translate new coordinates to old format, ready to be re-factored
    inset = axis.inset_axes(inset_extent)
    picextent = indicator_extent

    width = picextent[2]
    height = picextent[3]

    inset.set_xlim(picextent[0], picextent[0]+width)
    inset.set_ylim(picextent[1], picextent[1]+height)

    rect = Rectangle(
        [picextent[0], picextent[1]], width=width, height=height,
        transform=axis.transData, fill=None, **kwargs
    )
    if show_box:
        axis.add_patch(rect)

    # create connection lines between rectangle and inset axis
    connA = ConnectionPatch(
        xyA=connA_start, coordsA=rect.get_transform(),
        xyB=connA_end, coordsB=inset.transAxes,
        arrowstyle=arrow_style, **kwargs
    )
    connB = ConnectionPatch(
        xyA=connB_start, coordsA=rect.get_transform(),
        xyB=connB_end, coordsB=inset.transAxes,
        arrowstyle=arrow_style, **kwargs
    )
    
    if  show_connections:
        axis.add_artist(connA)
        axis.add_artist(connB)

    return inset

def box(axis: mpl.axes.Axes, box_boundary: np.ndarray, **kwargs) -> None:
    r'''
    Adds the simulation box to the given axis.

    Parameters
    ----------
    axis : AxisSubplot
        Matplotlib.pyplot AxisSubplot object.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    **kwargs : 
        Forwarded to axis.plot.

    Returns
    -------
    None.
    '''
    warnings.warn("The function 'plot.box' will be removed in version 2.0.0. Please use the function 'plot.box_boundary' instead.", PendingDeprecationWarning)
    defaultKwargs = {'c': 'k', 'ls': '-', 'marker': ''}
    if 'color' in kwargs:
        # either 'c' or 'color' can be given, not both
        kwargs['c'] = kwargs['color']
        del kwargs['color']
    if 'linestyle' in kwargs:
        kwargs['ls'] = kwargs['linestyle']
        del kwargs['linestyle']
    kwargs = defaultKwargs | kwargs
    axis.plot([box_boundary[0,0],box_boundary[0,0]], box_boundary[1], **kwargs)
    axis.plot([box_boundary[0,1],box_boundary[0,1]], box_boundary[1], **kwargs)
    axis.plot(box_boundary[0], [box_boundary[1,0],box_boundary[1,0]], **kwargs)
    axis.plot(box_boundary[0], [box_boundary[1,1],box_boundary[1,1]], **kwargs)


def box_boundary(axis: mpl.axes.Axes, box_boundary: np.ndarray, **kwargs) -> None:
    r'''
    Adds the simulation box to the given axis.

    Parameters
    ----------
    axis : AxisSubplot
        Matplotlib.pyplot AxisSubplot object.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    **kwargs : 
        Forwarded to axis.plot.

    Returns
    -------
    None.
    '''
    defaultKwargs = {'c': 'k', 'ls': '-', 'marker': ''}
    if 'color' in kwargs:
        # either 'c' or 'color' can be given, not both
        kwargs['c'] = kwargs['color']
        del kwargs['color']
    if 'linestyle' in kwargs:
        kwargs['ls'] = kwargs['linestyle']
        del kwargs['linestyle']
    kwargs = defaultKwargs | kwargs
    axis.plot([box_boundary[0,0],box_boundary[0,0]], box_boundary[1], **kwargs)
    axis.plot([box_boundary[0,1],box_boundary[0,1]], box_boundary[1], **kwargs)
    axis.plot(box_boundary[0], [box_boundary[1,0],box_boundary[1,0]], **kwargs)
    axis.plot(box_boundary[0], [box_boundary[1,1],box_boundary[1,1]], **kwargs)



def particles(
        ax: mpl.axes.Axes, coords: np.ndarray, box_boundary: np.ndarray,
        radius: np.ndarray | float, scalefactor: float = 1.0,
        values: np.ndarray | None = None,
        cmap: list | str = 'viridis', set_ax_limits: bool = True,
        vmin: float | None = None, vmax: float | None = None,
        cscale: str = 'lin', verbose: bool = False, **kwargs) -> None:
    r'''
    Visualize particles as circles on a matplotlib axes object.

    It works only in 2D and considers x and y coordinates for the plot.
    
    Notes
    -----
    The particles are only visualized in their proper size (radius) if 
    `linewidth=0`. In case you want to color the edge of the particles, set 
    linewidth to a non-zero float and pass `edgecolor` with your desired color
    for the edge of the particles. This will cause the particles to overlap 
    more than they are supposed to. Therefore, you might consider 
    counter-acting this by choosing a suitable value for `scalefactor`.

    Parameters
    ----------
    ax : AxisSubplot
        The matplotlib axes object where the particles will be plotted.
    coords : np.ndarray
        An array of coordinates for the particles.
    box_boundary : np.ndarray of shape (3,2)
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
    radius : float or np.ndarray
        The radius of the particles.
    scalefactor : float, optional
        Scales the size of the particles by this factor.
        The default is 1.0.
    values : np.ndarray or None, optional
        Values used to color the particles. The default is None.
    cmap: list or str, optional
        A list representing the colormap to use for the particles or
        the name of a matplotlib colormap. The default is 'viridis'.
    set_ax_limits: bool, optional
        If `True` axis limits are set to box size.
        Set to `False` if combined with inset functions!
        The default is True.
    vmin : float or None, optional
        Lower limit for coloring the particles. The default is None.
    vmax : float or None, optional
        Upper limit for coloring the particles. The default is None.
    cscale : str, optional
        Color scale. Use `'lin'` for a linear scale and `'log'` for a 
        logarithmic scale. The default is `'lin'`.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.
    **kwargs:
        Keyword arguments are forwarded to
        matplotlib.collections.PatchCollection.

    Returns
    -------
    mpl.collections.PatchCollection

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> frame = traj[-1]
    >>> fig, axs = amep.plot.new(figsize=(3, 3))
    >>> amep.plot.particles(
    ...     axs, frame.coords(), frame.box, frame.radius()
    ... )
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/plot/plot-particles.png")
    >>> 
    
    .. image:: /_static/images/plot/plot-particles.png
      :width: 400
      :align: center

    '''
    if coords.ndim == 1:
        coords = coords[None, :]
    x = coords[:, 0]
    y = coords[:, 1]
    if not np.all(coords[:,2]==coords[0,2]) and verbose:
        _log.info(
            "Coordinates are 3-dimensional. The plot particles function "\
            "only plots x and y coordinates."
        )
    # fix linewith to default = 0 (otherwise particle patches have incorrect radius)
    if not ( "lw" in kwargs or "linewidth" in kwargs or "linewidths" in kwargs) :
        if ( "ec" in kwargs or "edgecolor" in kwargs or "edgecolors" in kwargs) :
            kwargs["linewidth"] = 1
        else:
            kwargs["linewidth"] = 0
    radius = radius*scalefactor
    # convert to array if float is given
    if isinstance(radius, float):
        radius = np.ones(len(coords))*radius
    # check values
    values_given = True
    if values is None:
        values = np.ones(len(coords))
        values_given = False
    patches = []
    new_values = []
    for x1, y1, r, v in zip(x, y, radius, values):
        circle = Circle((x1, y1), r)
        patches.append(circle)
        new_values.append(v)
        if abs(y1 - box_boundary[1, 0]) < r:
            circle2 = Circle((x1,
                              (y1 + (box_boundary[1, 1] - box_boundary[1, 0]))
                              ), r)
            patches.append(circle2)
            new_values.append(v)
        if abs(y1 - box_boundary[1, 1]) < r:
            circle3 = Circle((x1,
                              (y1 - (box_boundary[1, 1] - box_boundary[1, 0]))
                              ), r)
            patches.append(circle3)
            new_values.append(v)
        if abs(x1 - box_boundary[0, 0]) < r:
            circle4 = Circle((x1 + (box_boundary[0, 1] - box_boundary[0, 0]),
                              y1
                              ), r)
            patches.append(circle4)
            new_values.append(v)
        if abs(x1 - box_boundary[0, 1]) < r:
            circle5 = Circle((x1 - (box_boundary[0, 1] - box_boundary[0, 0]),
                              y1
                              ), r)
            patches.append(circle5)
            new_values.append(v)

    if type(cmap) == str:
        cmap = mpl.colormaps[cmap]
    elif type(cmap) == list:
        cmap = ListedColormap(cmap)
    else:
        if verbose:
            _log.info(
                f"cmap has the wrong type {type(cmap)}. "\
                "Use default 'viridis'."
            )
        cmap = mpl.colormaps['viridis']
    if values_given:
        if vmin is None:
            vmin = np.min(values)
        if vmax is None:
            vmax = np.max(values)
    
        if cscale == 'lin':
            norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
        elif cscale == 'log':
            norm = mpl.colors.LogNorm(vmin = vmin, vmax = vmax)
        else:
            raise ValueError(
                f"""Invalid value {cscale} for cscale. Use 'lin' or 'log'.
                """
            )
        p = PatchCollection(patches, cmap = cmap, norm = norm, **kwargs)
        p.set_array(new_values)
    else:
        p = PatchCollection(patches, **kwargs)
    ax.add_collection(p)
    if set_ax_limits:
        ax.set_xlim(box_boundary[0, 0], box_boundary[0, 1])
        ax.set_ylim(box_boundary[1, 0], box_boundary[1, 1])
    return p


def field(
        ax: mpl.axes.Axes, density: np.ndarray, X: np.ndarray, Y: np.ndarray,
        cmap: list | str = 'plasma', box_boundary: np.ndarray | None = None,
        vmin: float | None = None, vmax: float | None = None,
        cscale: str = 'lin', verbose: bool = False, **kwargs
        ) -> mpl.collections.QuadMesh:
    r'''Visualize a scalar field on a matplotlib axes object.

    It works only in 2D.

    Parameters
    ----------
    ax : AxisSubplot
        The matplotlib axes object where the field will be plotted.
    density : np.ndarray
        An array of scalar values for the field.
    X : np.ndarray
        An array of x coordinates for the field.
    Y : np.ndarray
        An array of y coordinates for the field.
    cmap: list or str, optional
        A list representing the colormap to use for the particles or
        the name of a matplotlib colormap. The default is 'viridis'.
    box_boundary : np.ndarray of shape (3,2) | None, optional
        Boundary of the simulation box in the form of
        `np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])`.
        Needed if `set_ax_limits = True`.
    vmin : float or None, optional
        Lower limit for the colormap. The default is None.
    vmax : float or None, optional
        Upper limit for the colormap. The default is None.
    cscale : str, optional
        Color scale. Use `'lin'` for a linear scale and `'log'` for a 
        logarithmic scale. The default is `'lin'`.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.
    **kwargs :
        Keyword arguments are forwarded to matplotlib.pyplot.pcolormesh.

    Returns
    -------
    mpl.collections.QuadMesh

    Examples
    --------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
    >>> frame = traj[4]
    >>> fig, axs = amep.plot.new(figsize=(3.6,3))
    >>> mp = amep.plot.field(
    ...     axs, frame.data("c"), *frame.grid
    ... )
    >>> cax = amep.plot.add_colorbar(fig, axs, mp, label=r"$c(x,y)$")
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/plot/plot-field.png")
    >>> 

    .. image:: /_static/images/plot/plot-field.png
      :width: 400
      :align: center
    
    '''
    # get norm for colors
    if vmin is None:
        vmin = np.min(density)
    if vmax is None:
        vmax = np.max(density)

    # check colormap
    if type(cmap) == str:
        cmap = mpl.colormaps[cmap]
    elif type(cmap) == list:
        cmap = ListedColormap(cmap)
    else:
        if verbose:
            _log.info(
                f"cmap has the wrong type {type(cmap)}. "\
                "Use default 'viridis'."
            )
        cmap = mpl.colormaps['viridis']
    if cscale == 'lin':
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    elif cscale == 'log':
        norm = mpl.colors.LogNorm(vmin = vmin, vmax = vmax)
    else:
        raise ValueError(
            f"""Invalid value {cscale} for cscale. Use 'lin' or 'log'.
            """
        )
    if box_boundary is not None:
        ax.set_xlim(box_boundary[0, 0], box_boundary[0, 1])
        ax.set_ylim(box_boundary[1, 0], box_boundary[1, 1])
    return ax.pcolormesh(X, Y, density, cmap = cmap, norm = norm, **kwargs)


def colored_line(
        axis: mpl.axes.Axes, x_vals: np.ndarray, y_vals: np.ndarray,
        cols: np.ndarray | None = None, cmap: str | None = None,
        vmin: float | None = None, vmax: float | None = None,
        norm = None, linewidth: float = 2) -> mpl.collections.Collection:
    r'''
    Add a line to the given axis.

    The line is colored via the given colormap and the c values.

    Parameters
    ----------
    axis : AxisSubplot
        Matplotlib.pyplot AxisSubplot object..
    x : np.ndarray
        Array of x-values.
    y : np.ndarray
        Array of y-values.
    c : np.ndarray, optional
        Coloring values. The default is None.
    cmap : str, optional
        Colormap name. Available colormaps can be viewed with
        amep.plot.colormaps(). The default is None.
    vmin : float, optional
        Minimum value for colormap normalization. The default is None.
    vmax : float, optional
        Maximum value for colormap normalization. The default is None.
    norm : Norm, optional
        Normalization. The default is None.
    linewidth : int, optional
        Line width. The default is 2.

    Returns
    -------
    line : mpl.collections.Collection
    
    Examples
    --------
    >>> import amep
    >>> fig, axs = amep.plot.new()
    >>> x = np.linspace(0,5,1000)
    >>> y = np.sin(10*x)
    >>> line = amep.plot.colored_line(axs, x, y)
    >>> axs.set_xlim(0, 5)
    >>> axs.set_ylim(-1.1, 1.1)
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$\sin(10x)$')
    >>> fig.savefig('./figures/plot/plot-colored_line.png')
    >>> 

    .. image:: /_static/images/plot/plot-colored_line.png
      :width: 400
      :align: center

    '''
    points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if cols is None:
        cols = y_vals
    if vmin is None:
        vmin = np.min(cols)
    if vmax is None:
        vmax = np.max(cols)
    if norm is None:
        norm = plt.Normalize(vmin, vmax)

    l_c = LineCollection(segments, cmap=cmap, norm=norm)
    l_c.set_array(cols)
    l_c.set_linewidth(linewidth)
    line = axis.add_collection(l_c)

    return line


def linear_mappable(
        cmap: str | mpl.colors.Colormap | None,
        vmin: float, vmax: float, cscale: str = 'lin') -> mpl.cm.ScalarMappable:
    r'''Generate a scalar mappable colormap.

    Used to color lines according to some value
    v between vmin and vmax with color=sm.to_rgba(v).

    Parameters
    ----------
    cmap : str or mpl.colors.Colormap or None
        Colormap object or name of the colormap. If None, the default colormap
        is used as specified in the used style sheet.
    vmin : float
        Minimum value.
    vmax : float
        Maximum value.
    cscale : str, optional
        Color scale. Use `'lin'` for a linear scale and `'log'` for a logarithmic
        scale. The default is `'lin'`.

    Returns
    -------
    sm : mpl.cm.ScalarMappable
        Object that maps a float to an rgba value.

    Examples
    --------
    >>> import amep
    >>> import numpy as np
    >>> x = np.linspace(0,10,100)
    >>> z = np.linspace(0,2,20)
    >>> sm = amep.plot.linear_mappable('viridis', 0, 2)
    >>> fig, axs = amep.plot.new()
    >>> for i in range(len(z)):
    ...     axs.plot(x, x+i, c=sm.to_rgba(z[i]), marker=''
    ... )
    >>> cax = amep.plot.add_colorbar(fig, axs, sm, label=r'$z$')
    >>> axs.set_xlabel(r'$x$')
    >>> axs.set_ylabel(r'$y$')
    >>> fig.savefig('./figures/plot/plot-linear_mappable.png')
    >>>
    
    .. image:: /_static/images/plot/plot-linear_mappable.png
      :width: 400
      :align: center

    '''
    if cscale == 'lin':
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    elif cscale == 'log':
        norm = mpl.colors.LogNorm(vmin = vmin, vmax = vmax)
    else:
        raise ValueError(
            f"""Invalid value {cscale} for cscale. Use 'lin' or 'log'."""
        )
    cmap = plt.get_cmap(cmap)
    s_m = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
    return s_m


def draw_box(
        fig: plt.Figure,
        box_extent: np.ndarray | list | tuple,
        text: str | None = None,
        edgecolor: str = "k", facecolor: str = "w",
        linestyle: str = '--', linewidth: float = 1.0,
        edgealpha: float = 1.0, facealpha: float = 0.0,
        boxstyle: str = 'square', pad: float = 0.0,
        dxtext: float = 0.0, dytext: float = 0.0,
        ha="center", va="bottom", **kwargs
        ):
    r'''
    Draws a box.

    Parameters
    ----------
    fig : plt.Figure
        Figure to which the box is added.
    box_extent : np.ndarray | list | tuple
        relative (to the figure) coordinates of the area to be enlarged
        inside the initial plot starting from the lower left corner:
        `[x_0, y_0, d_x, d_y]`

        x_0 : float
            x coordinate of the lower left corner of the inset figure.
        y_0 : float
            y coordinate of the lower left corner of the inset figure.
        d_x : float
            Width of the inset figure.
        d_y : float
            Height of the inset figure.

    text : str, optional
        Text to be written above the drawn box. The location of the 
        box can be changed with dxtext, dytext and the arguments
        forwarded to fig.txt().
    edgecolor : str|None, optional
        Color of the box boundary. The default is None.
    facecolor : str|None, optional
        Filling color of the box. The default is None.
    linestyle : str, optional
        Style of the box boundary. Possible linestyles are '-', ':',
        '--', '-.'. See also `amep.plot.linestyles()`. The default is '--'.
    linewidth : float, optional
        Width of the box boundary. The default is 1.0.
    edgealpha : float, optional
        Transarency of the box boundary. The default is 1.0.
    facealpha : float, optional
        Transparency of the box filling. The default is 1.0.
    boxstyle : str, optional
        Style of the box. Possible styles are 'circle', 'darrow',
        'larrow', 'rarrow', 'round', 'round4', 'roundtooth', 'sawtooth',
        'square' (see also https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html).
        The default is 'square'.
    pad : float, optional
        The amount of padding around the original box.. The default is 0.0.
    dxtext : float, optional
        x-offset of the text from the centre above the box. Relative
        to the box coordinates. The default is 0.
    dytext : float, optional
        y-offset of the text from the centre above the box. Relative
        to the box coordinates. The default is 0.
    ha : str, optional
        keyword argument passed to fig.text(). horizontal alignment.
    va : str, optional
        keyword argument passed to fig.text(). vertical alignment.
    **kwargs:
        keyword arguments passed to fig.text() such as fontsize, color/c, ...

    Returns
    -------
    None.

    Examples
    --------
    >>> import amep
    >>> fig, axs=amep.plot.new()
    >>> axs.plot([0,1,2,3], [0,2,1,1])
    >>> amep.plot.draw_box(
    ...     fig, box_extent=[0, 0, 0.45, 1.01],
    ...     text="inside box1", fontsize=15, c="r",
    ...     facecolor="tab:orange", facealpha=0.2,
    ...     dxtext=.12, dytext=-.15, ha="right"
    ... )
    >>> amep.plot.draw_box(
    ...     fig, box_extent=[0.5, 0, 0.5, 1.01],
    ...     text="box2", facecolor="tab:blue", facealpha=0.2,
    ...     edgecolor="tab:green", linestyle=":"
    ... )
    >>> fig.savefig('./figures/plot/plot-draw_box.png')
    >>> 
    
    .. image:: /_static/images/plot/plot-draw_box.png
      :width: 400
      :align: center

    '''
    x, y, width, height = box_extent

    facecolor = to_rgba(facecolor, facealpha)
    edgecolor = to_rgba(edgecolor, edgealpha)

    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=boxstyle+', pad=%1.2f' % pad,
                         ec=edgecolor,
                         fc=facecolor, ls=linestyle, lw=linewidth,
                         transform=fig.transFigure
                         )
    fig.add_artist(box)

    if text is not None:
        fig.text(x+width/2+dxtext, y+height+dytext, text, ha=ha, va=va, **kwargs)


def animate_trajectory(
        trajectory: FieldTrajectory | ParticleTrajectory,
        outfile: Path | str, ptype: int | None = None,
        ftype: str | None = None, xlabel: str = '', ylabel: str = '',
        cbar_label: str = '', cbar: bool = True,
        formatter: Callable[[mpl.axes.Axes,], None] | None = None,
        painter: object | None = None,
        title: str = '', figsize: tuple[float, float] | None = None,
        start: float = 0.0, stop: float = 1.0, nth: int = 1, fps: int = 10,
        verbose: bool = False, **kwargs) -> None:
    r'''Create a video from a trajectory.

    Quick video writeout. Just takes a trajectory and animates it.
    For further customization of videos, check the `amep.plot.create_video`
    method.

    Note for Field: The box must have the same size for the whole trajectory.

    Parameters
    ----------

    trajectory: FieldTrajectory or ParticleTrajectory
        An AMEP-Trajectory object containing the data to be animated.
        If you give a ParticleTrajectory your dataset needs to contain
        radii for the particles under the key "radius". If you provide a
        FieldTrajectory, the data must be 2D.
    outfile: str or Path
        A path for the animation to be written to.
    ptype : int or None, optional
        The particle types to be visualized. This value is only used if a
        ParticleTrajectory is provided. The default is None.
    ftype : str or None, optional
        The field to be visualized. This value is only used if a
        FieldTrajectory is provided. The default is None.
    xlabel : str, optional
        Label for the x axis. The default is ''.
    ylabel : str, optional
        Label for the y axis. The default is ''.
    cbar_label : str, optional
        Label for the colorbar. The default is ''.
    cbar : bool, optional
        If True, a colorbar is shown. If ParticleTrajectory is provided, the
        colorbar is only shown if also a painter function is provided.
        The default is True.
    formatter : object or None, optional
        A function that takes an axis as input and formats the given axis.
        For example, use `formatter=lambda x: amep.plot.format_axis(x)` to
        format the axis of the video frames. The default is None.
    painter : object or None, optional
        A function that calculates the values used to color the particles. This
        argument is only considered if a ParticleTrajectory is provided. The
        function must take two positional arguments, first, a BaseFrame object,
        and second, a particle type (int or None). For example, to color the
        particles by their ID, use `painter=lambda f,p: f.ids(ptype=p)`. The
        default is None.
    figsize : tuple[float, float]
        Figure size for the video.

        Note:
            If the video changes in size during the animation, adapt the
            figure size. (Colorbars with long (changing)
            axis labels need padding.)

        Default is `(7,5)`.
    start : float, optional
        Fraction of the trajectory at which to start animate the data. Must be
        smaller than `stop`. The default is 0.0.
    stop : float, optional
        Fraction of the trajectory at which to stop animate the data. Must be
        larger than `start`. The default is 1.0.
    nth : int, optional
        Use each nth frame to make the animate. The default is 1.
    fps: int, optional
        The frames per second of the video. The default is 10.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.
    **kwargs
        All additional keyword arguments are forwared to `amep.plot.particles`
        if a ParticleTrajectory is provided and to `amep.plot.field` if a
        FieldTrajectory is provided.

    Returns
    -------
    None

    Example
    -------
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
    >>> amep.plot.animate_trajectory(
    ...     traj, "./figures/plot/plot-animate_trajectory_1.gif",
    ...     formatter=lambda x: amep.plot.format_axis(x, direction="out"),
    ...     vmin=0.0, vmax=3.0, ftype='c', xlabel=r"$x$", ylabel=r"$y$",
    ...     cbar_label=r"$c(x,y)$", fps=5
    ... )
    >>>
    
    .. image:: /_static/images/plot/plot-animate_trajectory_1.gif
      :width: 400
      :align: center


    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> amep.plot.animate_trajectory(
    ...     traj, "./figures/plot/plot-animate_trajectory_2.gif",
    ...     painter=lambda x,p: (x.velocities(ptype=p)**2).sum(axis=1),
    ...     xlabel='x', ylabel='y', cbar_label=r'$|\vec{v}|^2$',
    ...     vmin=1e0, vmax=1e6, cscale="log", fps=5
    ... )
    >>> 

    .. image:: /_static/images/plot/plot-animate_trajectory_2.gif
      :width: 400
      :align: center

    '''
    # check input
    if start >= stop:
        raise ValueError('start must be smaller than stop.')
    if not 0.0 < stop <= 1.0:
        raise ValueError(
            'stop must be greater than zero and smaller than or equal to 1.0.'
        )
    if not 0.0 <= start < 1.0:
        raise ValueError(
            'start must be smaller than 1.0 and larger than or equal to 0.0.'
        )
    # initialize figure
    fig, axe = new(figsize=figsize)
    # Add two small invisible rectangles to fix the size of the figure 
    # during the animation. tight_layout adjusts the size to those.
    # If figure size changes during the animation, the initial figure size
    # needs to be adapted.
    for i in range(2):
        axe.add_patch(mpl.patches.Rectangle(
            (i, i),
            width=0.01*(-1)**i,
            height=0.01*(-1)**i,
            transform=fig.transFigure,
            clip_on=False,
            alpha=0
        ))
    fig.set_layout_engine("tight", pad=0)

    axe.set_xlabel(xlabel)
    axe.set_ylabel(ylabel)
    axe.set_title(title)
    axe.set_aspect('equal', 'box')
    if formatter is not None:
        formatter(axe)
    # plot fields
    if isinstance(trajectory, FieldTrajectory):
        # check field
        if ftype is None:
            ftype = trajectory[0].keys[0]
            warnings.warn(
                "amep.plot.animate_trajectory: A FieldTrajectory is provided "\
                "without the field to plot. "\
                f"Plotting {trajectory[0].keys[0]}."
            )
        elif ftype not in trajectory[0].keys:
            raise KeyError(
                f"amep.plot.animate_trajectory: Invalid key {ftype}. Please "\
                "use one of {trajectory[0].keys}."
            )
        if trajectory[0].data(ftype).ndim != 2:
            raise ValueError(
                "amep.plot.animate_trajectory: The given data has "\
                f"{trajectory[0].data().ndim} spatial dimensions. "\
                "Visualization is only possible for 2D data."
            )
        else:
            image = field(
                axe, trajectory[0].data(ftype), trajectory[0].grid[0],
                trajectory[0].grid[1], box_boundary=trajectory[0].box,
                **kwargs
            )
            if cbar:
                cax = fig.colorbar(image, label=cbar_label)

            def animate(index: int) -> list[mpl.image.AxesImage]:
                '''Draw a field at each timestep.'''
                now = trajectory[index].data(ftype)
                image.set_array(now)
                if 'vmin' in kwargs:
                    vmin = kwargs['vmin']
                else:
                    vmin = np.min(now)
                if 'vmax' in kwargs:
                    vmax = kwargs['vmax']
                else:
                    vmax = np.max(now)
                image.set_clim(vmin, vmax)
                return [image]
    # plot particles
    elif isinstance(trajectory, ParticleTrajectory):
        # get colormap
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
        else:
            cmap = None
        # plot colorbar
        if cbar and painter is not None:
            colorbar = fig.colorbar(
                mappable=linear_mappable(cmap, 0.0, 1.0),
                ax = axe, label = cbar_label
            )

        def animate(index: int) -> list:
            '''Draw the particles at each timestep.'''
            # clear the axis to draw the new frame
            axe.clear()
            # add axis labels
            axe.set_xlabel(xlabel)
            axe.set_ylabel(ylabel)
            # generate values for coloring the particles
            if painter is None:
                values = None
            else:
                values = painter(trajectory[index], ptype)
            # plot particles
            patches = particles(
                axe,
                trajectory[index].coords(ptype=ptype),
                trajectory[index].box,
                trajectory[index].data("radius", ptype=ptype),
                values = values,
                **kwargs
            )
            # update colorbar
            if cbar and values is not None:
                colorbar.update_normal(patches)
            
            # format axis if formatter is provided
            if formatter is not None:
                formatter(axe)
            return [patches]
    else:
        raise ValueError(
            f'''amep.plot.animate_trajectory: Invalid trajectory type
            {type(trajectory)}. Please provide either a ParticleTrajectory
            or a FieldTrajectory object.'''
        )
    indices = np.arange(
        int(start*trajectory.nframes),
        int(stop*trajectory.nframes),
        nth
    )
    anim = FuncAnimation(fig, animate, frames = indices)
    with tqdm(total = len(indices)) as pbar:
        anim.save(
            outfile,
            fps = fps,
            progress_callback = lambda frameindex, nframes: pbar.update()
        )
    plt.close(fig)


def create_video(
        fig: mpl.figure.Figure, update_frame_func: object,
        data: Iterable,
        output_filename: str, fps: int = 10,
        output_codec: str = mpl.rcParams["animation.codec"],
        bitrate: int = -1,
        layout_engine: str = "constrained") -> None:
    r'''Create a video from frames generated by a make_frame function.

    Forwards the matlplotlib FuncAnimation for simpler use with AMEP.
    We strongly recommend to use animate_trajectory function
    instead of doing this here at low level. If you need a lower level
    animation use matplotlibs functions directly. This function is just
    meant for internal use.

    Parameters
    ----------
    fig : mpl.figure.Figure
        Figure object to update with the `update_frame_func`.
    update_frame_func : Callable
        A function that generates frames for the video.
    data : Iterable
        The data to be plotted at each frame.
    fps: int, optional
        The frames per second of the video. The default is 10.
    output_filename : str
        The filename of the output video file. The extension 
        is used to determine the video format.
    output_codec : str, optional
        The codec to use for the MP4 video.
        only applicable if output_format is "mp4".
    bitrate : int, optional
        The bitrate of the MP4 video.
        only applicable if output_format is "mp4".

    Returns
    -------
    None.

    Examples
    --------
    In this example, we first calculate the local packing fraction from a
    Voronoi tesselation for each frame and the then create a video of it.
    
    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> ld = amep.evaluate.LDdist(
    ...     traj, nav=traj.nframes,
    ...     mode="packing", use_voronoi=True
    ... )
    >>> fig, axs = amep.plot.new(layout="constrained")
    >>> def update_frame(data):
    ...     axs.clear()
    ...     plot = axs.plot(data[1], data[0])
    ...     axs.set_xlabel(r'$\varphi$')
    ...     axs.set_ylabel(r'$p(\varphi)$')
    ...     axs.set_ylim(-0.05, 10)
    ...     axs.set_xlim(0, 1.2)
    ...     return [plot,]
    >>> amep.plot.create_video(
    ...     fig, update_frame, ld.frames, 
    ...     "./figures/plot/plot-create_video.gif", fps=5,
    ... )
    >>>     
    
    .. image:: /_static/images/plot/plot-create_video.gif
      :width: 400
      :align: center

    '''
    # set layout engine
    fig.set_layout_engine(layout_engine)
    
    # animation = VideoClip(make_frame_func, duration=duration)
    anim = FuncAnimation(fig, update_frame_func, frames=data)

    with tqdm(total = len(data)) as pbar:
        anim.save(
            output_filename,
            codec = output_codec,
            fps = fps,
            bitrate = bitrate,
            progress_callback = lambda i, n: pbar.update()
        )


def voronoi(axs: mpl.axes.Axes, vor: Voronoi, **kwargs):
    r'''
    Plots the Voronoi tessellations from a Voronoi object calculated with scipy
    function voronoi_plot_2d [1]_.


    Notes
    -----

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.voronoi_plot_2d.html


    Parameters
    ----------
    axs : AxisSubplot
        Matplotlib.pyplot AxisSubplot object.
    voronoi : Voronoi object
        Voronoi object to plot the Voronoi tessellation of.
    **kwargs : Keyword arguments for the plot.
        show_points, show_vertices, line_colors, line_width, line_alpha, point_size
        Compare [1]_

    Returns
    -------
    matplotlib.figure.Figure instance
        matplotlib figure
        
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
    ...     axs, vor, show_vertices=False, line_colors="orange",
    ...     line_width=1, line_alpha=0.6, point_size=1, c="b"
    ... )
    >>> amep.plot.box(axs, frame.box)
    >>> axs.set_xlabel(r"$x$")
    >>> axs.set_ylabel(r"$y$")
    >>> fig.savefig("./figures/plot/plot-voronoi.png")
    >>> 
    
    .. image:: /_static/images/plot/plot-voronoi.png
      :width: 400
      :align: center
      
    '''
    # Plot the Voronoi diagram
    if len(vor.points[0])==2:
        return voronoi_plot_2d(vor, ax=axs, **kwargs)
    else:
        raise Exception("amep.plot.voronoi: Cannot plot 3d data.")
    plt.show()

def draw_arrow(fig, x: float, y: float, dx: float, dy: float, **kwargs):
    r"""Draws an arrow on a Matplotlib figure.

    This function uses the `FancyArrow` class to draw an arrow on a Matplotlib 
    figure at a specified position, with a given displacement. The arrow is 
    added directly to the figure object.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object on which the arrow will be drawn.
    x : float
        The starting x-coordinate of the arrow, in figure coordinates (0 to 1).
    y : float
        The starting y-coordinate of the arrow, in figure coordinates (0 to 1).
    dx : float
        The horizontal displacement (change in x) of the arrow, in figure coordinates.
    dy : float
        The vertical displacement (change in y) of the arrow, in figure coordinates.
    **kwargs
        Additional keyword arguments passed to `matplotlib.patches.FancyArrow`, 
        such as `color`, `width`, `head_width`, and `head_length`.

    Returns
    -------
    None.

    Notes
    -----
    The arrow's position and size are specified in figure coordinates. Figure 
    coordinates range from 0 to 1, where (0, 0) represents the bottom-left 
    corner and (1, 1) represents the top-right corner of the figure.

    Examples
    --------
    >>> fig, axs = amep.plot.new(figsize=(3, 3))
    >>> x=0.2
    >>> y=0.2
    >>> dx=0.5
    >>> dy=0.5
    >>> amep.plot.draw_arrow(
    ...     fig, x, y, dx, dy, color="blue", alpha=0.8, width=0.05,
    ...     head_width=0.1, head_length=0.03
    ... )
    >>> 

    """
    arrow = FancyArrow(
        x, y, dx, dy, transform=fig.transFigure,
        length_includes_head=True, **kwargs
    )
    fig.add_artist(arrow) 
