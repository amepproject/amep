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
Base Classes
============

.. module:: amep.base

The AMEP module :mod:`amep.base` contains all basic classes used in the 
backend of AMEP.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import shutil
import warnings
import inspect
import logging

from typing import Collection, Iterable, Sequence
from  io import StringIO
from contextlib import redirect_stdout
from tqdm import TqdmExperimentalWarning

import h5py
import numpy as np
import scipy.odr as ODR

from pathlib import Path
from datetime import datetime
from ._version import __version__

warnings.simplefilter('always', UserWarning)
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
# =============================================================================
# UTILITIES
# =============================================================================
def check_path(path: str, extension: str) -> tuple[str, str]:
    r"""
    Checks if the directory of a given path exists and separates between
    directory and file name.

    Parameters
    ----------
    path : str
        Path to check.
    extension : str
        File extension of the given path.

    Raises
    ------
    ValueError
        Raised if an invalid file extension has been identified.
    FileNotFoundError
        Raised if the directory does not exist.

    Returns
    -------
    directory : str
        Directory of the given path.
    filename : str
        File name in the given path.

    """
    # normalize path
    path = os.path.normpath(path)
    
    # get extension
    _, file_extension = os.path.splitext(path)
    
    # check extension
    if file_extension == extension:
        # split into directory and filename
        directory, filename = os.path.split(path)
        # set directory to current working directory if empty
        if directory == '':
            directory = os.getcwd()
    elif file_extension == '':
        directory = os.path.normpath(path)
        filename  = ''
    else:
        raise ValueError(
            f'''Incorrect file extension. Got {file_extension} instead
            of {extension}.'''
        )

    if os.path.exists(directory):
        return directory, filename
    else:
        raise FileNotFoundError(f'No such directory: {directory}')


# =============================================================================
# CONSTANTS
# =============================================================================
ROOTGROUPS = [
    'params',
    'scripts',
    'info',
    'frames',
    'amep'
]
TRAJFILENAME = 'traj.h5amep'
COMPRESSION = 'lzf'
SHUFFLE = True
FLETCHER = True
DTYPE = np.float32
KEYS = {
    'coords': ['x', 'y', 'z'],
    'uwcoords': ['xu', 'yu', 'zu'],
    'njcoords': ['njx', 'njy', 'njz'],
    'forces': ['fx', 'fy', 'fz'],
    'omegas': ['omegax', 'omegay', 'omegaz'],
    'orientations': ['mux', 'muy', 'muz'],
    'velocities': ['vx', 'vy', 'vz'],
    'angmom': ['angmomx', 'angmomy', 'angmomz']
}
KEYASSIGN    = {            # ['key',Index]
    'x' : ['coords',0],
    'y' : ['coords',1],
    'z' : ['coords',2],
    'xu' : ['uwcoords',0],
    'yu' : ['uwcoords',1],
    'zu' : ['uwcoords',2],
    'njx' : ['njcoords',0],
    'njy' : ['njcoords',1],
    'njz' : ['njcoords',2],
    'fx' : ['forces',0],
    'fy' : ['forces',1],
    'fz' : ['forces',2],
    'omegax' : ['omegas',0],
    'omegay' : ['omegas',1],
    'omegaz' : ['omegas',2],
    'mux' : ['orientations',0],
    'muy' : ['orientations',1],
    'muz' : ['orientations',2],
    'vx' : ['velocities',0],
    'vy' : ['velocities',1],
    'vz' : ['velocities',2],
    'angmomx': ['angmom', 0],
    'angmomy': ['angmom', 1],
    'angmomz': ['angmom', 2]
}
GRIDKEYS = [
    'X',
    'Y',
    'Z'
]
LOADMODES = [
    'lammps',
    'h5amep',
    'field',
    'hoomd',
    'gromacs'
]
# maximum RAM usage in GB per CPU (used for parallelized methods)
MAXMEM = 1

# logger format and level
LOGGERFORMAT = "%(levelname)s:%(name)s.%(funcName)s: %(message)s"
LOGGINGLEVEL = "INFO"


# =============================================================================
# LOGGER
# =============================================================================
# set default format
logging.basicConfig(format=LOGGERFORMAT)

def get_module_logger(mod_name):
    r"""
    Creates a module logger.

    Parameters
    ----------
    mod_name : str
        Module name. Always use `__name__`.

    Returns
    -------
    logger : logging.Logger
        Logger object.

    """
    # create logger
    logger = logging.getLogger(mod_name)

    # set logging level
    logger.setLevel(LOGGINGLEVEL)

    return logger


def get_class_logger(mod_name, class_name):
    r"""
    Creates a class logger.

    Parameters
    ----------
    mod_name : str
        Module name. Always use `__name__`.
    class_name : str
        Class name. Always use `self.__class__.__name__`.

    Returns
    -------
    logger : logging.Logger
        Logger object.

    """
    # create logger
    logger = logging.getLogger(mod_name + "." + class_name)

    # set logging level
    logger.setLevel(LOGGINGLEVEL)

    return logger


# =============================================================================
# READER BASE CLASS
# =============================================================================
class BaseReader:
    """
    Base class to read simulation data from output files and to convert it into
    the hdf5 file format.
    """
    def __init__(
            self, savedir: str, start: float, stop: float,
            nth: int, filename: str) -> None:
        r"""
        Initializes a BaseReader object.

        Parameters
        ----------
        savedir : str
            Directory in which the .h5amep file is created.
        start : int
            Start reading the trajectory data from this fraction of the
            trajectory.
        stop : int
            Stop reading the trajectory data from this fraction of the 
            trajectory.
        nth : int
            Read each nth frame.
        filename : str
            Name of the trajectory file that is created. Needs to be an
            .h5amep file.

        Returns
        -------
        None

        """
        # filename of the hdf5 trajectory file
        self.filename = filename
        # check if temporary file exists and delete before creating a new one
        if "#temp#" in filename and os.path.exists(os.path.join(savedir, self.filename)):
            os.remove(os.path.join(savedir, self.filename))
        # create hdf5 file with default groups
        with h5py.File(os.path.join(savedir, self.filename), 'a') as root:
            # ROOT Level (define groups)
            for g in ROOTGROUPS:
                if g not in root.keys():
                    root.create_group(g)

            # add amep version to h5amep file if file is new
            if 'version' not in root['amep'].attrs.keys():
                root['amep'].attrs['version'] = __version__

        self.savedir = savedir
        # check loading configuration
        self.start = 0.0
        if start is not None:
            if 0.0 <= start < 1.0:
                self.start = start
        self.stop = 1.0
        if stop is not None:
            if 0.0 < stop <= 1.0 and stop > start:
                self.stop = stop

        if nth is not None:
            self.nth = int(nth)
        else:
            self.nth = 1

    @property
    def start(self):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            x = root['params'].attrs['start']
        return x

    @start.setter
    def start(self,x):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            root['params'].attrs['start'] = x

    @property
    def stop(self):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            x = root['params'].attrs['stop']
        return x
    @stop.setter
    def stop(self,x):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            root['params'].attrs['stop'] = x

    @property
    def nth(self):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            x = root['params'].attrs['nth']
        return x
    @nth.setter
    def nth(self,x):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            root['params'].attrs['nth'] = x

    @property
    def savedir(self):
        return self.__savedir
    @savedir.setter
    def savedir(self,x):
        self.__savedir = x

    @property
    def dt(self):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            x = root['params'].attrs['dt']
        return x
    @dt.setter
    def dt(self,x):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            root['params'].attrs['dt'] = x
        self.times = self.steps*x

    @property
    def d(self):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            x = root['params'].attrs['d']
        return x
    @d.setter
    def d(self,x):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            root['params'].attrs['d'] = x

    @property
    def steps(self) -> np.ndarray:
        """The array of simulation steps for each saved frame.

        The simulation steps are also used
        to index the frame data in the HDF5-File.
        """
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            return root['frames']['steps'][:]

    @steps.setter
    def steps(self, vals: Collection[int]):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            if 'steps' not in root['frames'].keys():
                root['frames'].create_dataset(
                    'steps',
                    (len(vals),),
                    data=vals,
                    dtype=int,
                    compression=COMPRESSION,
                    shuffle=SHUFFLE,
                    fletcher32=FLETCHER,
                    maxshape=(None,)
                )
            else:
                root['frames']['steps'][:] = vals

    @property
    def times(self):
        """The array of physical times for each saved frame."""
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            vals = root['frames']['times'][:]
        return vals

    @times.setter
    def times(self, vals: Collection[float]):
        with h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            if 'times' not in root['frames'].keys():
                root['frames'].create_dataset(
                    'times',
                    (len(vals),),
                    data=vals,
                    dtype=float,
                    compression=COMPRESSION,
                    shuffle=SHUFFLE,
                    fletcher32=FLETCHER,
                    maxshape=(None,)
                )
            else:
                root['frames']['times'][:] = vals

    @property
    def filename(self) -> Path:
        """The path to the HDF5-file that contains the writers data."""
        return self.__filename

    @filename.setter
    def filename(self, val: Path | str):
        self.__filename = Path(val)


# =============================================================================
# FRAME BASE CLASS
# =============================================================================
class BaseFrame:
    """
    Per particle simulation data at one time step (a single frame).
    """
    def __init__(self, reader, index):
        r"""
        Basic data frame containing simulation data of particle-based
        simulations for one time step.

        Parameters
        ----------
        reader : 
            AMEP reader object.
        index : int
            Frame index.

        Returns
        -------
        None.

        """
        self.__reader = reader
        self.__index  = index

        # get step
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            self.__step = root['frames']['steps'][self.__index]
    def __eq__(self,other):
        return self.__reader==other.__reader and self.__index==other.__index

    @property
    def step(self):
        '''
        Time step of the frame.

        Returns
        -------
        int
            Time step of the frames.

        '''
        return self.__step
    @property
    def time(self) -> float:
        """The physical time of the frame."""
        try:
            return self.__reader.times[self.__index]
        except:
            return self.__step*self.__reader.dt
    @property
    def center(self):
        '''
        Center of the simulation box.

        Returns
        -------
        x : np.ndarray
            Center of the simulation box.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            cen = root['frames'][str(self.__step)]['center'][:]
        return cen
    @property
    def dim(self):
        '''
        Spatial dimension of the simnulation.

        Returns
        -------
        x : int
            Spatial dimension.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            dimension = root['frames'][str(self.__step)].attrs['d']
        return dimension
    def n(self, ptype=None):
        '''
        Total number of particles.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None (all particles).
            
        Returns
        -------
        int
            Total particle number of the given particle type.
        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            types = root['frames'][str(self.__step)]['type'][:]

        if ptype in self.ptypes:
            return np.where(types==ptype)[0].shape[0]
        return types.shape[0]
    def coords(self, **kwargs):
        '''
        All coordinates of all particles or of all particles of
        a specific particle type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Coordinate frame of particle coordinates.
        '''
        return self.__read_data('coords', **kwargs)
    def nojump_coords(self, **kwargs):
        '''
        Returns the nojump coordinates of all particles or of 
        all particles of a specific particle type.
        These coordinates are only available, when the nojump method of the 
        trajectory which contains the frames was called.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Coordinate frame of particle coordinates.
        '''
        return self.__read_data('njcoords', **kwargs)
    def unwrapped_coords(self, **kwargs) -> np.ndarray:
        """
        Returns the unwrapped coordinates of all particles or of 
        all particles of a specific particle type.

        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.

        Returns
        -------
        np.ndarray
            Coordinate frame of unwrapped particle coordinates.

        """
        return self.__read_data("uwcoords",**kwargs)
    def velocities(self, **kwargs) -> np.ndarray:
        '''
        Returns the velocities of all particles or of 
        all particles of a specific particle type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Coordinate frame of particle velocities.
        '''
        return self.__read_data('velocities', **kwargs)
    def forces(self, **kwargs) -> np.ndarray:
        '''
        Returns the forces acting on each particle or 
        on particles of a specific particle type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Coordinate frame of particle forces.
        '''
        return self.__read_data('forces', **kwargs)
    def orientations(self, **kwargs) -> np.ndarray:
        '''
        Returns the orientation vectors of all particles or of
        all particles of a specific particle type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Coordinate frame of particle orientation vectors.
        '''
        return self.__read_data('orientations', **kwargs)
    def omegas(self, **kwargs) -> np.ndarray:
        """
        Returns the angular velocities of all particles or of 
        all particles of a specific particle type.

        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.

        Returns
        -------
        np.ndarray
            Coordinate frame of particle angular velocity vectors.
        """
        return self.__read_data('omegas', **kwargs)
    def torque(self, **kwargs) -> np.ndarray:
        '''
        Returns the torque acting on each particle or on 
        each particle of a specific particle type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Coordinate frame of particle torque vectors.
        '''
        return self.__read_data('torque', **kwargs)
    def radius(self, **kwargs) -> np.ndarray:
        '''
        Returns the radius of all particles or of
        all particles of a specific type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Array of particle radii.
        '''
        return self.__read_data('radius',**kwargs)
    def mass(self, **kwargs) -> np.ndarray:
        '''
        Returns the mass of all particles or of
        all particles of a specific type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Array of particle masses.
        '''
        return self.__read_data('mass',**kwargs)
    def angmom(self, **kwargs) -> np.ndarray:
        '''
        Returns the angular momentum of all particles or of
        all particles of a specific type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Array of particle angular momenta.
        '''
        return self.__read_data('angmom',**kwargs)
    def types(self, **kwargs) -> np.ndarray:
        '''
        Returns the particle type of each or of
        each particle of a specific type.
        
        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.
            
        Returns
        -------
        np.ndarray
            Array of particle types.
        '''
        return self.__read_data('type',**kwargs)
    def ids(self, **kwargs) -> np.ndarray:
        '''
        Returns the particle indices of all particles or of
        all particles of a specific particle type.

        Parameters
        ----------
        ptype : int, optional
            Particle type. The default is None.

        Returns
        -------
        np.ndarray
            Particle indices.

        '''
        return self.__read_data('id', **kwargs)
    @property
    def keys(self) -> list:
        '''
        Returns a list of all available data keys.

        Returns
        -------
        datakeys : list
            List of str containing all available data keys.
        '''
        datakeys = []

        # open trajectory file
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:

            # get all available (combined) keys
            rootkeys = list(root['frames'][str(self.__step)].keys())

        # convert combined keys to single keys
        for key in rootkeys:
            if key in KEYS:
                datakeys.extend(KEYS[key])
            elif key not in ['center', 'box']:
                datakeys.append(key)

        return datakeys
    def data(
            self, *args: str | tuple[list[str], ...], ptype: int | None = None, zerofill: bool = False,
            return_keys: bool = False) -> tuple[list, np.ndarray]:
        r'''
        Returns the entire data frame for all particles or for
        all particles of a specific particle type.
        
        Notes
        -----
        One can use _one_ wildcard character asterisk ("*") to load all data
        sets matching that name. For example, "name*" returns all datasets
        that start with "name", "value[*]" returns all datasets with any
        number of characters in between the square brackets i.e. "value[1]",
        "value[2]", ..., "value[any text with any length]".

        Duplicate keys are removed.
        
        Parameters
        ----------
        *args : str | tuple[list[str], ...]
            Parameter keys. One wildcard character asterisk can be used, see
            note above. Either multiple strings or lists of strings are
            allowed, a combination should not be used.
        ptype : int | list, optional
            Particle type. Is internally converted to a list and all matching
            ptypes are returned. The default is None.
        zerofill : bool, optional
            If True, empty strings are replaced by a column of zeros in the
            returned data array. The default is False.
        returnkeys : bool, optional
            If True, a list of the data keys is returned together with the
            data. The default is False.

        Returns
        -------
        datakeys : list
            List of str containing the column keys/names.
            This is only returned if returnkeys=True.
        data : np.ndarray
            Data array of shape (N, len(keys)).
        '''
        data = None
        datakeys = []

        # allow lists of keys as input
        islist=False
        listresult=[]
        for arg in args:
            if isinstance(arg, (list, np.ndarray)):
                islist=True
                listresult.append(self.data(*arg, ptype = ptype, zerofill = zerofill, return_keys = return_keys))
        if islist:
            if len(args)==1:
                return listresult[0]
            return listresult
            
        # return all data if no arguments are given
        if len(args)==0:
            args = self.keys
        else:
            # Transform list of all given keys by allowing semi-wildcard matches
            # One asterisk * is allowed.
            extended_keys = []

            for arg in args:
                found_key = False
                argsplit=arg.split("*")
                if len(argsplit)>2:
                    raise KeyError(f"Only one '*' allowed. You supplied {arg}.")
                if len(argsplit)==1:
                    if arg in self.keys:
                        extended_keys.append(arg)
                        found_key = True
                else: # if a wildcard character asterisk (*) is used
                    for key in self.keys:
                        if key.startswith(argsplit[0]) and key.endswith(argsplit[1]) and len(arg)-1<=len(key):
                            extended_keys.append(key)
                            found_key = True
                if not found_key:
                    raise KeyError(
                        f"The key \"{arg}\" does not exist in the frame, returning no data!"
                    )
            # remove duplicates
            args = np.array(extended_keys)[np.sort(np.unique(extended_keys, return_index=True)[1])]

        # loop through given keys
        for i,key in enumerate(args):

            if key in KEYASSIGN:
                # load partially from file
                d = self.__read_data(KEYASSIGN[key][0], ptype=ptype)[:,KEYASSIGN[key][1]]
            elif key == '' and zerofill:
                # add a column of zeroes
                d = np.zeros(self.n(ptype=ptype))
            elif key == '' and not zerofill:
                # don't add column of zeros and print warning
                warnings.warn(
                    "Empty string detected with zerofill=False. "\
                    "Empty string will be ignored."
                )
                d = None
            else:
                # check for wildcard character
                if key.endswith('*'):
                    # generate list of corresponding keys
                    ks = []
                    for k in self.keys:
                        if k.startswith(key.split('*')[0]):
                            ks.append(k)
                    # load data from file by calling data
                    if ks != []:
                        d = self.data(*ks, ptype=ptype)
                    else:
                        d = None
                else:            
                    # load directly from file
                    d = self.__read_data(key, ptype=ptype)

            # append loaded data to data array
            if d is not None:
                if data is None:
                    data = np.copy(d)
                elif data.ndim == 1 and d.ndim == 1:
                    data = np.hstack((data[:,None],d[:,None]))
                elif data.ndim == 1 and d.ndim == 2:
                    data = np.hstack((data[:,None],d))
                elif data.ndim == 2 and d.ndim == 1:
                    data = np.hstack((data,d[:,None]))
                else:
                    data = np.hstack((data,d))

            # get list of keys
            if key in KEYS:
                datakeys.extend(KEYS[key])
            else:
                datakeys.append(key)

        if return_keys:
            return datakeys, data
        return data

    def __read_data(
            self, key: str, ptype: int | list | None = None,
            pid: int | Sequence | None = None) -> np.ndarray:
        r'''
        Reads a dataset from the HDF5 file either for all particles
        or for all particles of a specific particle type.

        Parameters
        ----------
        key : str
            Key.
        ptype : int or list or None, optional
            Particle type(s). If None, the data is returned for all particle
            types. The default is None.
        pid : int or list or None, optional
            Particle ID. Returns the data only for particles with the given
            ID(s). If None, the data is returned for all particles with the
            given particle type(s). If not None, `ptype` is ignored.
            The default is None.
        Returns
        -------
        data : np.ndarray
            Dataset.

        '''
        # open the file in read-only mode
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:

            if pid:
                data_ids = root['frames'][str(self.__step)]['id'][:]
                if not isinstance(pid, Sequence):
                    pid = [pid]
                id_list = [int(np.where(data_ids == part_id)[0])
                           for part_id in pid]
                if key in root['frames'][str(self.__step)].keys():
                    data = root['frames'][str(self.__step)][key][id_list]
                    return data

            # get particle types
            types = root['frames'][str(self.__step)]['type'][:]
            # check if a dataset with the given key exists and read the data
            if key in root['frames'][str(self.__step)].keys():
                data = root['frames'][str(self.__step)][key][:]
                # If no ptype is provided return all data
                if not ptype:
                    return data
                # Transform ptype so it is a list of all ptypes
                if not isinstance(ptype, Sequence):
                    ptype = [ptype]
                mask = np.zeros(data.shape[0],)
                # check particle type
                mask += sum(types == single_ptype for single_ptype
                            in ptype if single_ptype in self.ptypes)
                if any(single_ptype not in self.ptypes for
                       single_ptype in ptype):
                    for single_ptype in ptype:
                        if single_ptype not in self.ptypes:
                            warnings.warn(
                                f"The specified particle type {single_ptype} "
                                "does not exist. Returning data without type "
                                f"{single_ptype}."
                            )
                return data[mask.astype(bool)]
            raise KeyError(
                    f"The key {key} does not exist in the frame. "
                    "Returning no data!"
                )

    def add_data(self, key: str, data: np.ndarray) -> None:
        '''
        Adds new data to the frame.

        Parameters
        ----------
        key : str
            Name of the data.
        data : np.ndarray
            Data array of shape (N,x) with N being the total
            number of particles.

        Returns
        -------
        None.

        '''
        N = self.n()
        if data.shape[0]==N and len(data.shape)<=2:
            with h5py.File(
                os.path.join(self.__reader.savedir, self.__reader.filename),
                'a'
            ) as root:
                if key not in root['frames'][str(self.__step)].keys():
                    root['frames'][str(self.__step)].create_dataset(
                        key,
                        data.shape,
                        data = data,
                        dtype = DTYPE,
                        compression = COMPRESSION,
                        shuffle = SHUFFLE,
                        fletcher32 = FLETCHER
                    )
                else:
                    root['frames'][str(self.__step)][key][:] = data
        else:
            raise ValueError('The given data has the wrong shape.')
    @property
    def ptypes(self):
        '''
        Array of all particle types.
        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            types  = root['frames'][str(self.__step)]['type'][:]
        return np.unique(types)
    @property
    def box(self):
        '''
        Box boundary.
        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            boxe = root['frames'][str(self.__step)]['box'][:]
        return boxe
    @property
    def volume(self):
        '''
        Returns the volume (for self.__dim=3) / area (for self.__dim=2)
        of the simulation box.
        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            box = root['frames'][str(self.__step)]['box'][:]
            d   = root['frames'][str(self.__step)].attrs['d']

        if d == 2:
            res = np.prod(np.diff(box).T[0,:2])
        elif d == 3:
            res = np.prod(np.diff(box).T[0])
        else:
            res = None

        return res
    def density(self, ptype=None):
        '''
        Returns the number density.
        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            box = root['frames'][str(self.__step)]['box'][:]
            d   = root['frames'][str(self.__step)].attrs['d']            

        if d == 2:
            res = self.n(ptype=ptype)/np.prod(np.diff(box).T[0,:2])
        elif d == 3:
            res = self.n(ptype=ptype)/np.prod(np.diff(box).T[0])
        else:
            res = None
        return res


# =============================================================================
# FIELD BASE CLASS
# =============================================================================
class BaseField:
    '''
    Continuum field data at one time step (one frame).
    '''
    def __init__(self, reader:BaseReader,index:int):
        self.__reader = reader
        self.__index = index
        #Step is an important indexing tool. So we keep it around as object variable.
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            self.__step = root['frames']['steps'][self.__index]
    @property
    def step(self) -> int:
        '''
        Time step of the frame.

        Returns
        -------
        float
            Time step of the frames.

        '''
        return self.__step
    @property
    def time(self) -> float:
        """The physical time of the frame."""
        return self.__reader.times[self.__index]
    @property
    def center(self)->np.ndarray:
        '''
        Center of the simulation box.

        Returns
        -------
        x : np.ndarray
            Center of the simulation box.

        '''
        return np.mean(self.box, axis=1)
    @property
    def dim(self)->int:
        '''
        Spatial dimension of the simnulation.

        Returns
        -------
        x : int
            Spatial dimension.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            dimension = root['params'].attrs['d']
        return dimension
    @property
    def box(self)->np.ndarray:
        '''
        Box boundaries of the simulation.
        Returns
        -------
        boxe : np.ndarray
            Box boundaries [[<lower bounds>],[<upper bounds>]]

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            boxe = root['frames/grid/box'][:]
        return boxe
    @property
    def volume(self)->float:
        '''
        Returns the volume (for self.__dim=3) / area (for self.__dim=2)
        of the simulation box.
        
        Returns
        -------
        res: float
            Volume or Area of the simulation.
        '''

        if self.dim == 2:
            res = np.prod(np.diff(self.box).T[0,:2])
        elif self.dim == 3:
            res = np.prod(np.diff(self.box).T[0])
        else:
            res = np.diff(self.box).T[0,:1][0]
        return res
    @property
    def grid(self):
        '''
        Coordinates of the grid points.

        Parameters
        ----------
        None.

        Returns
        -------
        np.ndarray
            Meshgrid of positions of grid points.
        '''
        data = []
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            grid_shape = root['frames/grid/shape'][:]
            keys = list(root['frames/grid'].keys())
            for key in keys:
                if key in GRIDKEYS:
                    d = root['frames/grid'][key][:]
                    data.append(d.reshape(grid_shape))
        return data
    @property
    def keys(self):
        '''
        Returns a list of all available data keys.

        Returns
        -------
        datakeys : list
            List of str containing all available data keys.
        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            keys = list(root['frames'][str(self.__step)].keys())
        return keys
    def data(self, *args, returnkeys=False):
        '''
        The field values at each gridpoint,
        as a array of the values at each point.

        Parameters
        ----------
        *args : str
            Keys of the data to be returned.
        returnkeys : bool, optional
            If True, the keys are returned. The default is False.

        Returns
        -------
        np.ndarray
            The field values at each grid point
        '''
        # return all data if no arguments are given
        if len(args)==0:
            args = self.keys
            
        data = []
            
        # load data
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            
            shape = root['frames/grid/shape'][:]
            
            # loop through given keys
            for i,key in enumerate(args):
                d = root['frames'][str(self.__step)][key][:]
                d = d.reshape(shape)
                data.append(d)
        
        if len(data) == 1:
            data = data[0]
        else:
            data = np.array(data)

        if returnkeys:
            return data, args
        return data

# =============================================================================
# TRAJECTORY BASE CLASS
# =============================================================================
class BaseTrajectory:
    '''
    Trajectory base object.
    '''
    def __init__(self, reader):
        r'''
        Creates a trajectory object containing data frames for multiple
        time steps.

        Parameters
        ----------
        reader : BaseReader
            Reader of the data.

        Returns
        -------
        None.

        '''
        self.__reader = reader

    def __getitem__(self, item: int | slice | Iterable[int]
                    ) -> BaseFrame | BaseField | list[BaseField | BaseFrame]:
        """Get an individual frame or field of a simulation.

        Supports slicing as well as iterables of valid integer indices.
        The return type depends on the type of the trajectory.
        Also it depends if only one frame or a collection of frames is requested.
        If a collection of frames is requested a list of frames is returned.

        Parameters
        ----------
        item : int | slice | Iterable[int]

        Returns
        -------
        BaseFrame | BaseField | list[BaseField | BaseFrame] 
        """
        if isinstance(item, slice):
            sli = range(*item.indices(len(self.__reader.steps)))
            if self.type == "field":
                return [BaseField(self.__reader, index) for index in sli]
            # Any of these returns defaults to particle Frames.
            # If we get more types of trejaectories we have to add them here
            # with an if statement as above.
            return [BaseFrame(self.__reader, index) for index in sli]
        if isinstance(item, Iterable):
            if self.type == "field":
                return [BaseField(self.__reader, index) for index in item]
            return [BaseFrame(self.__reader, index) for index in item]
        if isinstance(item, (int, np.integer)):
            if self.type == "field":
                return BaseField(self.__reader, item)
            return BaseFrame(self.__reader, item)
        raise KeyError('''BaseTrajectory: Invalid key. Only integer values,
        1D lists and arrays, and slices are allowed.'''
                       )

    def __iter__(self):
        """Iterate over all frames of the trajectory."""
        for i, _ in enumerate(self.__reader.steps):
            yield self[i]

    def __next__(self):
        pass

    def __len__(self):
        return len(self.__reader.steps)

    def add_author_info(
            self, author: str, key: str, value: int | float | str) -> None:
        '''
        Adds author information for the given author to the trajectory.

        Parameters
        ----------
        author : str
            Author name.
        key : str
            Name or category of the information to be added.
        value : int or float or str
            Information to be added.

        Returns
        -------
        None.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'a'
        ) as root:
            if 'authors' not in root['info'].keys():
                root['info'].create_group('authors')
            if author not in root['info']['authors'].keys():
                root['info']['authors'].create_group(author)
            root['info']['authors'][author].attrs[key] = value

    def get_author_info(self, author: str) -> dict:
        r'''
        Returns all information for the given author.

        Parameters
        ----------
        author : str
            Author name.

        Returns
        -------
        p : dict
            Author information.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            # check if author information is available
            if 'authors' in root['info'].keys():
                p = dict(a for a in root['info']['authors'][author].attrs.items())
                return p
            return {}

    def delete_author_info(self, author: str, key: str | None = None) -> None:
        r'''
        Deletes all information (key=None) or specific information given by
        the key keyword of the given author.

        Parameters
        ----------
        author : str
            Author name.
        key : str or None, optional
            Info that should be deleted. If None, all info is deleted.
            The default is None.

        Returns
        -------
        None.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'a'
        ) as root:
            if key is None:
                del root['info']['authors'][author]
            elif type(key)==str:
                root['info']['authors'][author].attrs.__delitem__(key)

    @property
    def authors(self) -> list[str]:
        r'''
        Returns a list of all author names.

        Returns
        -------
        keys : list
            List of author names.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            # check if author information is available
            if 'authors' in root['info'].keys():
                keys = list(root['info']['authors'].keys())
                return keys
            return []

    def add_software_info(self, key: str, value: str | int | float) -> None:
        r'''
        Add software information to the hdf5 trajectory file.

        Parameters
        ----------
        key : str
            Name of the parameter.
        value : str or int or float
            Value of the parameter.

        Returns
        -------
        None.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'a'
        ) as root:
            if 'software' not in root['info'].keys():
                root['info'].create_group('software')
            root['info']['software'].attrs[key] = value

    def delete_software_info(self, key: str | None = None) -> None:
        r'''
        Deletes all software information (key=None) or specific information
        given by the key keyword.

        Parameters
        ----------
        key : str or None, optional
            Information that should be deleted. If None, all information is
            deleted. The default is None.

        Returns
        -------
        None.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'a'
        ) as root:
            if key is None:
                for key in list(root['info']['software'].attrs.keys()):
                    root['info']['software'].attrs.__delitem__(key)
            else:
                root['info']['software'].attrs.__delitem__(key)

    @property
    def software(self) -> dict:
        r'''
        Returns all software information.

        Returns
        -------
        keys : dict
            Software information.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            # check if author information is available
            if 'software' in root['info'].keys():
                p = dict(a for a in root['info']['software'].attrs.items())
                return p
            return {}
    def add_script(self, path: Path | str) -> None:
        r'''
        Adds a script in text format to the hdf5 file.

        Parameters
        ----------
        path : str or Path
            Path of the script to add.

        Returns
        -------
        None.

        '''
        # get file name from path
        if isinstance(path, str):
            #filename = path.split('/')[-1]
            path = os.path.normpath(path)
            filename = path.split(os.sep)[-1]
        elif isinstance(path, Path):
            filename = path.name
        else:
            raise TypeError
        # read data from the file
        with open(path, 'r') as f:
            lines = f.readlines()
            
        # store data in the hdf5 trajectory file
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'a'
        ) as root:
            if filename not in root['scripts'].keys():
                root['scripts'].create_dataset(filename,
                                               data=lines,
                                               compression=COMPRESSION,
                                               shuffle=SHUFFLE,
                                               fletcher32=False,
                                               maxshape=(None,))
            else:
                root['scripts'][filename].resize((len(lines),))
                root['scripts'][filename][:] = lines
    def get_script(
            self, filename: str, store: bool = False,
            directory: str = '.') -> list:
        r'''
        Returns a stored script as a list of lines and stores it as a file
        in the given directory if store is True.

        Parameters
        ----------
        filename : str
            File name.
        store : bool, optional
            If True, the script is stored in a file. The default is False.
        directory : str, optional
            Directory in which the file is stored if store is True.
            The default is '.'.

        Returns
        -------
        decoded : list
            List of lines.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            lines = root['scripts'][filename][:]
        # decode
        decoded = [l.decode('utf-8') for l in lines]
        if store:
            with open(os.path.join(directory, filename), 'w') as f:
                f.writelines(decoded)
        return decoded
    def delete_script(self, filename: str) -> None:
        r'''
        Deletes the script of the given filename.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        None.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'a'
        ) as root:
            del root['scripts'][filename]
    @property
    def scripts(self) -> list:
        r'''
        Returns a list of all scripts (filenames) that are stored in the 
        trajectory object.

        Returns
        -------
        keys : list
            List of filenames.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            keys = list(root['scripts'].keys())
        return keys
    @property
    def params(self) -> dict:
        r'''
        Returns all parameters stored within the trajectory as a dictionary.
        
        Returns
        -------
        p : dict
            Parameter dictionary.
        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            p = dict(a for a in root['params'].attrs.items())
        return p
    def add_param(self, param: str, value: int | float | str) -> None:
        r'''
        Adds a parameter to the trajectory.

        Parameters
        ----------
        param : str
            Parameter.
        value : str or float or int
            Value of the parameter.

        Raises
        ------
        ValueError
            Raises an error if param is not of type str.

        Returns
        -------
        None.

        '''
        if isinstance(param, str):
            with h5py.File(
                os.path.join(self.__reader.savedir, self.__reader.filename),
                'a'
            ) as root:
                root['params'].attrs[param] = value
        else:
            raise ValueError('param is not of type str.')
    def delete_param(self, param: str) -> None:
        r'''
        Deletes the given parameter.

        Parameters
        ----------
        param : str
            Parameter.

        Raises
        ------
        ValueError
            Raises an error if param is not of type str.

        Returns
        -------
        None.

        '''
        if isinstance(param, str):
            with h5py.File(
                os.path.join(self.__reader.savedir, self.__reader.filename),
                'a'
            ) as root:
                root['params'].attrs.__delitem__(param)
        else:
            raise ValueError('param is not of type str.')
    @property
    def info(self) -> dict:
        r'''
        Returns all stored metadata information as a dictionary.

        Returns
        -------
        dict
            Metadata.
        '''
        # create empty dictionary
        info = {}
        # add software information
        info['software'] = self.software
        # add author information (one dict per author)
        info['authors'] = {}
        for author in self.authors:
            info['authors'][author] = self.get_author_info(author)
        # add parameters
        info['params'] = self.params
        return info
    @property
    def type(self)->str:
        '''
        Returns the data type (`'particles'` or `'field'`).

        Returns
        -------
        str
            Data type.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            out = root.attrs["type"]
        return out
    @property
    def reader(self):
        return self.__reader
    @property
    def start(self):
        '''
        Start fraction.
        '''
        return self.__reader.start
    @property
    def stop(self):
        '''
        Stop fraction.
        '''
        return self.__reader.stop
    @property
    def nth(self):
        '''
        Every which dump file was loaded.
        '''
        return self.__reader.nth
    @property
    def nframes(self):
        '''
        Total number of loaded frames.
        '''
        return len(self)
    @property
    def steps(self):
        '''
        Array of the number of time steps for each frame.
        '''
        return self.__reader.steps
    @property
    def times(self):
        '''
        Physical times.
        '''
        try:
            return self.__reader.times
        except:
            return self.steps*self.__reader.dt
    @property
    def dt(self):
        '''
        Size of the time step used for the simulation.
        '''
        return self.__reader.dt
    @dt.setter
    def dt(self, x):
        if type(x) == float:
            self.__reader.dt = x
    @property
    def dim(self) -> int:
        '''
        Spatial dimension of the simnulation.

        Returns
        -------
        x : int
            Spatial dimension.

        '''
        return self.__reader.d
    @property
    def savedir(self):
        '''
        Trajectory file directory.
        '''
        return self.__reader.savedir
    @savedir.setter
    def savedir(self,x):
        if type(x)==str:
            self.__reader.savedir = x
    @property
    def version(self):
        '''
        AMEP version with which the trajectory file has been created.

        Returns
        -------
        ver : str
            Version.

        '''
        with h5py.File(
            os.path.join(self.__reader.savedir, self.__reader.filename), 'r'
        ) as root:
            ver = root['amep'].attrs['version']
        return ver

# =============================================================================
# EVALUATION BASE CLASS
# =============================================================================
class BaseEvaluation:
    """
    Evaluation base class.
    """
    def __init__(self) -> None:
        r"""
        Evaluation base object containing general attributes needed for
        evaluation functions applied to simulation data.

        Returns
        -------
        None.
        
        """
        # evaluation name
        self.__name = ''

    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self) -> list[str]:
        """The keys to the evaluation object.
        
        Used so Evaluation-objects can be used as dictionaries.
        """
        return [name for (name, _) in inspect.getmembers(
            type(self), lambda x: isinstance(x, property)
        )]

    def values(self):
        return [self.__getitem__(key) for key in self.keys()]

    def items(self):
        return [i for i in zip(self.keys(), self.values())]

    def save(
            self, path: str, backup: bool = True,
            database: bool = False, name: str | None = None) -> None:
        r"""
        Stores the evaluation result in an HDF5 file.
        
        Parameters
        ----------
        path : str
            Path of the `'.h5'` file in which the data should be stored. If 
            only a directory is given, the filename is chosen as `self.name`.
            Raises an error if the given directory does not exist or if the
            file extension is not `'.h5'`.
        backup : bool, optional
            If True, an already existing file is backed up and not overwritten.
            This keyword is ignored if `database=True`. The default is True.
        database : bool, optional
            If True, the results are appended to the given `'.h5'` file if it 
            already exists. If False, a new file is created and the old is
            backed up. If False and the given `'.h5'` file contains multiple
            evaluation results, an error is raised. In this case, `database` 
            has to be set to `True`. The default is False.
        name : str or None, optional
            Name under which the data should be stored in the HDF5 file. If 
            None, self.name is used. The default is None.
            
        Returns
        -------
        None.
        
        """
        # check path
        directory, filename = check_path(path, '.h5')
        
        # set filename to evaluation name if not given
        if filename == '':
            filename = self.name + '.h5'
        
        # get current path
        current_path = os.path.join(directory, filename)
        
        # check name
        if name is None:
            name = self.name
        
        # create database object
        db = BaseDatabase(current_path)         
            
        if db.keys() == []:
            # just add the evaluation to the database if the database is empty
            db.add(name, self)
        else:
            if name in db.keys():
                if len(db.keys()) > 1:
                    if database:
                        if backup:
                            # create a copy of the old file as a backup
                            current_time = datetime.now().strftime(
                                '%Y-%m-%d_%H-%M-%S'
                            )
                            backup_path = os.path.join(
                                directory, '#'\
                                + filename.split('.h5')[0]\
                                + '_' + current_time + '.h5'
                            )
                            shutil.copy(current_path, backup_path)
                        # overwrite the data
                        db.delete(name)
                        db.add(name, self)
                    else:
                        # database=True required to add to a non-empty .h5 file
                        raise RuntimeError(
                            f"The given file {current_path} already contains "\
                            "evaluation results. You can add the current "\
                            "evaluation to this file by setting "\
                            "database=True. If you do not want to add this "\
                            "evaluation to the given file, please change the "\
                            "filename."
                        )
                else:
                    if backup:
                        # create a copy of the old file as a backup
                        current_time = datetime.now().strftime(
                            '%Y-%m-%d_%H-%M-%S'
                        )
                        backup_path = os.path.join(
                            directory, '#'\
                            + filename.split('.h5')[0]\
                            + '_' + current_time + '.h5'
                        )
                        shutil.copy(current_path, backup_path)
                    # overwrite the data
                    db.delete(name)
                    db.add(name, self)
                
            else:
                # just add the evaluation if one with the same name does not
                # already exists
                if database:
                    # add to the file
                    db.add(name, self)
                else:
                    # database=True required to add to a non-empty .h5 file
                    raise RuntimeError(
                        f"The given file {current_path} already contains "\
                        "evaluation results. You can add the current "\
                        "evaluation to this file by setting database=True. "\
                        "If you do not want to add this evaluation to the "\
                        "given file, please change the filename."
                    )

    @property
    def name(self) -> str:
        return self.__name
    @name.setter
    def name(self, x: str) -> None:
        if type(x) == str:
            self.__name = x
            
            

# =============================================================================
# EVALUATION-DATA BASE CLASS
# =============================================================================
class BaseEvalData:
    """
    Evaluation data base class (for loaded evaluation data).
    """
    def __init__(self, path: str, group: str | None = None) -> None:
        r'''
        Evaluation-data base class for accessing evaluation data from an HDF5
        file.

        Parameters
        ----------
        path : str
            Path of the HDF5 file.
        group : str or None, optional
            Group (name of the evaluation) of the given HDF5 file. The default
            is None.

        Returns
        -------
        None.

        '''
        self.__path  = path
        self.__group = group
        
        with h5py.File(self.__path, 'r') as root:
            # for backwards compatibility
            if not 'type' in root.attrs.keys():
                self.__datakeys = list(root.keys())
                self.__attrkeys = list(root.attrs.keys())
                self.__keys = self.__datakeys.copy()
                self.__keys.extend(self.__attrkeys)
            elif self.__group in root.keys():
                self.__datakeys = list(root[self.__group].keys())
                self.__attrkeys = list(root[self.__group].attrs.keys())
                self.__keys = self.__datakeys.copy()
                self.__keys.extend(self.__attrkeys)
            else:
                raise KeyError(f'Evaluation {self.__group} does not exist.')
        
    def __getattr__(self, item):
        with h5py.File(self.__path, 'r') as root:
            if self.__group is None:
                if item in self.__datakeys:
                    return root[item][:]
                elif item in self.__attrkeys:
                    return root.attrs[item]
                else:
                    raise KeyError(f'Invalid key. Choose one of {self.__keys}')
            else:
                if item in self.__datakeys:
                    return root[self.__group][item][:]
                elif item in self.__attrkeys:
                    return root[self.__group].attrs[item]
                else:
                    raise KeyError(f'Invalid key. Choose one of {self.__keys}')
                
    def __getitem__(self, item):
        return self.__getattr__(item)
                
    def keys(self):
        return self.__keys

    def values(self):
        return [self.__getitem__(key) for key in self.keys()]

    def items(self):
        return [i for i in zip(self.keys(), self.values())]
    
    @property
    def path(self) -> str:
        return self.__path
    
    @property
    def group(self) -> str:
        return self.__group
    
    
# =============================================================================
# DATABASE BASE CLASS
# =============================================================================    
class BaseDatabase:
    """
    Evaluation database base class for storing and loading multiple evaluation
    results stored in a single HDF5 file.
    """
    def __init__(self, path: str) -> None:
        r'''
        Loads/creates a database HDF5 file for storing data of multiple
        evaluate objects in a single file.

        Parameters
        ----------
        path : str
            Path of the HDF5 file. Raises an error if the path does not contain
            a filename with file extension `'.h5'`.

        Returns
        -------
        None.
        
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> msd = amep.evaluate.MSD(traj)
        >>> sf2d = amep.evaluate.SF2d(traj, nav=2)
        >>> db = amep.base.BaseDatabase('./eval/database.h5')
        >>> db.add('msd', msd)
        >>> db.add('sf2d', sf2d)
        >>> print(db.keys())
        ['msd', 'sf2d']
        >>> print(db.msd)
        <amep.base.BaseEvalData object at 0x0000025B7CB439A0>
        >>> db.delete('msd')
        >>> print(db.keys())
        ['sf2d']
        >>> print(db.items())
        [('sf2d', <amep.base.BaseEvalData object at 0x0000025B7A0A6E30>)]
        >>> 
        '''
        # check the given path
        directory, filename = check_path(path, '.h5')
        if filename == '':
            raise ValueError(
                f"""The given path {path} does not contain any filename. Please
                ensure that the full path to a file with file extension .h5 is
                given."""
            )
        # set path attribute
        self.__path = path
        
        if not os.path.exists(self.__path):
            # create new empty file
            with h5py.File(self.__path, 'w') as root:
                # add amep version as attribute
                root.attrs['version'] = __version__
                # add type indicator as attribute
                root.attrs['type'] = 'database'
        
    def __getitem__(self, item: str):
        if item in self.keys():
            return BaseEvalData(self.__path, group=item)
        else:
            raise KeyError(f'Invalid key. Choose one of {self.keys()}')
    
    def __getattr__(self, item):
        return self.__getitem__(item)
        
    def keys(self) -> list:
        with h5py.File(self.__path, 'r') as root:
            k = list(root.keys())
        return k
    
    def values(self) -> list:
        return [self.__getitem__(key) for key in self.keys()]

    def items(self) -> list:
        return [i for i in zip(self.keys(), self.values())]
        
    def add(self, name: str, evaluation: BaseEvaluation) -> None:
        r'''
        Adds the data of an evaluate object as a new group to the database
        HDF5 file.

        Parameters
        ----------
        name : str
            Name of the group/data.
        evaluation : BaseEvaluation
            Evaluate object of which the data should be stored in the database
            HDF5 file.

        Returns
        -------
        None.

        '''
        with h5py.File(self.__path, 'a') as root:
            group = root.require_group(name)
            for key in evaluation.keys():
                if isinstance(evaluation[key], np.ndarray):
                    group.require_dataset(
                        key, evaluation[key].shape, data=evaluation[key],
                        compression=COMPRESSION, shuffle=SHUFFLE,
                        fletcher32=FLETCHER, dtype=DTYPE)
                else:
                    if key=='ptype' and evaluation[key] is None:
                        group.attrs[key] = 'None'
                    else:
                        group.attrs[key] = evaluation[key]
    
    def delete(self, name: str) -> None:
        r'''
        Deletes a group/data from the database HDF5 file.

        Parameters
        ----------
        name : str
            Name of the group that should be deleted.

        Returns
        -------
        None.

        '''
        if name in self.keys():
            with h5py.File(self.__path, 'a') as root:
                del root[name]
        else:
            raise KeyError(
                f"The key {name} does not exist. Available keys are "\
                f"{self.keys()}."
            )
    
    
# =============================================================================
# FUNCTION BASE CLASS
# =============================================================================
class BaseFunction:
    """
    Base class to fit a function to data.
    """
    def __init__(self, nparams: int) -> None:
        r'''
        Creats a BaseFunction object.
        
        Parameters
        ----------
        nparams : int
            Number of parameters of the fit function.
        '''

        self.__name = ''
        self.__output = None

        self.__nparams = nparams
        self.__keys    = ['p%s' %i for i in range(self.__nparams)]
        self.__params  = np.zeros(nparams)
        self.__errors  = np.zeros(nparams)

    def f(self, p, x):
        return p[0]*x + p[1]

    def fit(
            self, xdata: np.ndarray, ydata: np.ndarray, p0: list | None = None,
            sigma: np.ndarray | None = None, maxit: int | None = None,
            verbose: bool = False) -> None:
        r'''
        Fits the function self.f to the given data
        by using ODR (orthogonal distance regression).
    
        Parameters
        ----------
        xdata : np.ndarray
            x values.
        ydata : np.ndarray
            y values.
        p0 : list or None, optional
            List of initial values. The default is None.
        sigma : np.ndarray or None, optional
            Absolute error for each data point. The default is None.
        maxit : int, optional
            Maximum number of iterations. The default is 200.
        verbose : bool, optional
            If True, the main results are printed. The default is False.
            
        Returns
        -------
        None.
        '''
        if p0 is None:
            p0 = np.ones(self.__nparams)

        if maxit is None:
            maxit = 200*self.__nparams

        # odr fit
        model = ODR.Model(self.f)
        data  = ODR.Data(xdata, ydata)
        myodr = ODR.ODR(data, model, beta0=p0, maxit=maxit, delta0=sigma)
        
        # set odr's job nr (if not set an error occurs when sigma is not None)
        myodr.set_job()
        
        # get fit results
        self.__output = myodr.run()
        
        # print results
        if verbose:
            self.__output.pprint()
        
        # check if iteration limit was reached
        # (checks if fit was successful)
        with StringIO() as buf, redirect_stdout(buf):
            self.__output.pprint()
            output = buf.getvalue()
        if "Iteration limit reached" in output:
            warnings.warn(
                "Iteration limit reached! Please specify initial p0 or "\
                "increase maxit."
            )
        # get fit parameters and fit errors
        self.__params = self.__output.beta
        self.__errors = self.__output.sd_beta

    def generate(self, x: np.ndarray, p: list | None = None) -> np.ndarray:
        r'''
        Returns the y values for given x values.

        Parameters
        ----------
        x : np.ndarray
            x values.
        p : list, optional
            List of parameters. The default is None.

        Returns
        -------
        y : np.ndarray
            f(x)

        '''
        if p is None:
            p = self.params
            
        return self.f(p, x)

    @property
    def params(self) -> np.ndarray:
        r"""
        Returns an array of the optimal fit parameters.

        Returns
        -------
        np.ndarray
            Fit parameter values.

        """
        return self.__params
    
    @property
    def errors(self) -> np.ndarray:
        r"""
        Returns the fit errors for each parameter as an array.

        Returns
        -------
        np.ndarray
            Fit errors.

        """
        return self.__errors
    
    @property
    def results(self) -> dict:
        r"""
        Returns the dictionary of fit results including parameter names, 
        parameter values, and fit errors.

        Returns
        -------
        dict
            Fit results.

        """
        results = {}
        for i,p in enumerate(zip(self.params, self.errors)):
            results[self.__keys[i]] = p
        return results

    @property
    def output(self):
        return self.__output
    
    @property
    def nparams(self):
        return self.__nparams
 
    @property
    def name(self):
        return self.__name
    @name.setter
    def name(self,x):
        if type(x) == str:
            self.__name = x

    @property
    def keys(self):
        return self.__keys
    @keys.setter
    def keys(self, x):
        self.__keys = x
