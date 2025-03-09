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
Reading Simulation Data
=======================

.. module:: amep.reader

The AMEP module :mod:`amep.reader` as an internal module that provides 
different reader classes for reading simulation data created with various 
simulation software. This module is part of the backend of AMEP.

"""
#=============================================================================
# IMPORT MODULES
#=============================================================================
import os
import glob
import csv
import warnings
import h5py
import string
import numpy as np
from gsd import hoomd
from chemfiles import Trajectory

from tqdm.autonotebook import tqdm
from .base import BaseReader, TRAJFILENAME, COMPRESSION, SHUFFLE, FLETCHER
from .base import DTYPE, get_class_logger
from .utils import quaternion_rotate, quaternion_conjugate, quaternion_multiply

warnings.simplefilter('always', UserWarning)
warnings.simplefilter('always', DeprecationWarning)

#=============================================================================
# UTILITIES
#=============================================================================
class FileError(Exception):
    """An error raised when wrong files are fed into a reader."""

# =============================================================================
# BASIC DATA READERS
# =============================================================================
class H5amepReader(BaseReader):
    '''Reads simulation data from an HDF5 trajectory file.
    '''

    def __init__(self, directory, trajfile=TRAJFILENAME, **kwargs):
        r'''
        Reader for an HDF5 trajectory file of name filename
        in the given directory.

        Parameters
        ----------
        directory : str
            Simulation directory.
        trajfile : str, optional
            File name of the HDF5 file. The default is TRAJFILENAME.
        **kwargs
            Additional keyword arguments are ignored.

        Returns
        -------
        None.

        '''
        self.__version = None
        
        path = os.path.join(directory, trajfile)
        if not os.path.exists(path):
            raise FileError(f'File {path} does not exist.')
        else:
            pass

        correct_file = True
        if trajfile.endswith('.h5amep'):
            with h5py.File(os.path.join(directory, trajfile), 'r') as root:
                if 'params' in root.keys():
                    start = root['params'].attrs['start']
                    stop  = root['params'].attrs['stop']
                    nth   = root['params'].attrs['nth']
                else:
                    correct_file = False
                if 'amep' in root.keys():
                    self.__version = root['amep'].attrs['version']
                else:
                    correct_file = False

            if correct_file:
                super().__init__(directory, start, stop, nth, trajfile)
            else:
                FileError('Wrong file format.')
        else:
            raise FileError('Not a .h5amep file.')
    @property
    def type(self):
        r"""The type of trajectory. Field or Particles."""
        with h5py.File(os.path.join(self.savedir,self.filename),"r") as root:
            return root.attrs["type"]
    @property
    def version(self):
        r'''
        AMEP version with which the h5amep file has been created.

        Returns
        -------
        str
            Version.

        '''
        return self.__version


class LammpsReader(BaseReader):
    '''Reads LAMMPS simulation data and writes it to an hdf5 file.
    '''

    def __init__(
            self, directory: str, savedir: str, start: float = 0.0,
            stop: float = 1.0, nth: int = 1, dumps: str = 'dump*.txt',
            trajfile: str = TRAJFILENAME, deleteold: bool = False,
            verbose: bool = False) -> None:
        r'''
        Reader for LAMMPS simulation data in dump-file format. Requires
        the log.lammps file to read certain information about the
        simulation parameters.

        Parameters
        ----------
        directory : str
            Simulation directory.
        savedir : str
            Directory in which the trajectory file will be stored.
        start : float, optional
            Start reading data from this fraction of the whole trajectory.
            The default is 0.
        stop : float, optional
            Stop reading data from this fraction of the whole trajectory.
            The default is 1.
        nth : int, optional
            Read each nth dump file. The default is None.
        dumps : str, optional
            File name of the dump files. The default is 'dump*.txt'.
        trajfile : str, optional
            Name of the hdf5 trajectory file that is created when an object of
            this class is initialized. The default is TRAJFILENAME.
        deleteold : bool, optional
            If True, an existing old h5amep trajectory file #<trajfile>
            will be removed. The default is False.
        verbose : bool, optional
            If True, runtime information is printed. The default is False.

        Returns
        -------
        None.

        '''
        # init class logger
        self.__log = get_class_logger(__name__, self.__class__.__name__)
        
        # back up trajectory file
        if os.path.exists(os.path.join(savedir, trajfile)):
            # rename trajectory to backup filename "#+<trajfile>"
            if os.path.exists(os.path.join(savedir, "#"+trajfile)):
                if verbose:
                    self.__log.info(
                        f"Existing old file #{trajfile} will be replaced."
                    )
                os.remove(os.path.join(savedir, "#"+trajfile))
            os.rename(
                os.path.join(savedir, trajfile),
                os.path.join(savedir, "#"+trajfile)
            )
            if verbose:
                self.__log.info(
                    f"Renamed {trajfile} to #{trajfile}."
                )
        # super(LammpsReader, self).__init__(directory, start, stop, nth, "#temp#"+trajfile)
        super().__init__(os.path.abspath(savedir), start, stop, nth, "#temp#"+trajfile)
        self.directory = directory

        try:
            self.__dumps = dumps

            # get all dump files
            files = sorted(glob.glob(os.path.join(self.directory, self.__dumps)), key=self.__sorter)
            if len(files)==0:
                raise Exception(f"LammpsReader: no dump files {dumps} in directory {directory}.")

            first = int(self.start*len(files))  # first dump file index
            last  = int(self.stop*len(files))   # last dump file index

            # dump files to load
            files = files[first:last:self.nth]

            # total number of files
            self.__nfiles = len(files)

            # array of number of time steps
            steps = np.zeros(self.__nfiles, dtype=int)

            # load data from dump files
            with h5py.File(os.path.join(self.savedir, "#temp#"+trajfile), 'a') as root:
                # Set type of h5amep file to particle
                root.attrs["type"] = "particle"

                # If no ID is specified only one warning is issued to the user.
                # Trigger idwarning is set to false:
                idwarning=False

                # loop through all dump files
                for n, file in enumerate(tqdm(files)):

                    with open(file, encoding="utf-8") as f:

                        # read all lines of the dump file
                        lines = f.readlines()

                        # check if last dump file is incomplete
                        # Scenario: job on cluster was canceled during writing dump file
                        if n == self.__nfiles-1:
                            if len(lines) < 3 or len(lines) != int(lines[3])+9:
                                warnings.warn(
                                    "amep.reader.LammpsReader: The last dump "\
                                    "file is incomplete. Onmitting last file."
                                )
                                steps = steps[:-1]
                                break

                        # get number of time steps
                        step = int(lines[1])

                        # get total number of atoms
                        N = int(lines[3])

                        if len(lines) != N+9:
                            raise Exception(
                                f"amep.reader.LammpsReader: Dump file {step} "\
                                "incomplete."
                            )

                        # get total number of parameters
                        nparams = len(np.fromstring(lines[9], sep=' '))

                        # get parameter keys
                        keys = lines[8].split(' ')[2:]
                        keys[-1] = keys[-1].rstrip('\n')

                        # remove last key if empty
                        if keys[-1] == '':
                            keys = keys[:-1]

                        # create new group for the given step in the 'frames' group
                        if str(step) not in root['frames'].keys():
                            frame = root['frames'].create_group(str(step))
                        else:
                            frame = root['frames'][str(step)]

                        # get boundaries of the simulation box
                        box = np.zeros((3, 2))
                        for l, line in enumerate(lines[5:8]):
                            box[l] = np.fromstring(line, sep=' ')

                        # add box to hdf5 file
                        if 'box' not in frame.keys():
                            frame.create_dataset('box',
                                                 (3, 2),
                                                 data=box,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['box'][:] = box

                        # get center of the simulation box
                        center = (box[:, 1]+box[:, 0])/2

                        # add center to hdf5 file
                        if 'center' not in frame.keys():
                            frame.create_dataset('center',
                                                 (3,),
                                                 data=center,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['center'][:] = center

                        # extract the data
                        data = np.zeros((N, nparams), dtype=DTYPE)
                        # ignore other lines (N+9 and larger)
                        # (LAMMPS sometimes adds an empty line at the end
                        #  which would cause problems)
                        for l, line in enumerate(lines[9:N+9]):
                            data[l] = np.fromstring(line, sep=' ')

                        # add data to hdf5 file
                        coords       = np.zeros((N, 3), dtype=DTYPE)
                        uwcoords     = np.zeros((N, 3), dtype=DTYPE)  # unwrapped
                        velocities   = np.zeros((N, 3), dtype=DTYPE)
                        forces       = np.zeros((N, 3), dtype=DTYPE)
                        orientations = np.zeros((N, 3), dtype=DTYPE)
                        omegas       = np.zeros((N, 3), dtype=DTYPE)
                        torque       = np.zeros((N, 3), dtype=DTYPE)
                        angmom       = np.zeros((N, 3), dtype=DTYPE)

                        # sort data by id if available
                        if "id" in keys:
                            sorting = np.argsort(data[:, keys.index("id")])
                            data = data[sorting, :]
                        else:
                            # Warn user if data could not be sorted.
                            idwarning = True
                            # side-note from the documentation:
                            # """Repetitions of a particular warning for the same
                            # source location are typically suppressed."""

                        unwrapped_available = False
                        d = 0
                        for i, key in enumerate(keys):
                            if key == 'x':
                                coords[:, 0] = data[:, i]
                                if not np.all((data[:, i] == 0.0)):
                                    d += 1
                            elif key == 'y':
                                coords[:, 1] = data[:, i]
                                if not np.all((data[:, i] == 0.0)):
                                    d += 1
                            elif key == 'z':
                                coords[:, 2] = data[:, i]
                                if not np.all((data[:, i] == 0.0)):
                                    d += 1
                            elif key == 'xu':
                                uwcoords[:, 0] = data[:, i]
                                unwrapped_available = True
                            elif key == 'yu':
                                uwcoords[:, 1] = data[:, i]
                                unwrapped_available = True
                            elif key == 'zu':
                                uwcoords[:, 2] = data[:, i]
                                unwrapped_available = True
                            elif key == 'vx':
                                velocities[:, 0] = data[:, i]
                            elif key == 'vy':
                                velocities[:, 1] = data[:, i]
                            elif key == 'vz':
                                velocities[:, 2] = data[:, i]
                            elif key == 'fx':
                                forces[:, 0] = data[:, i]
                            elif key == 'fy':
                                forces[:, 1] = data[:, i]
                            elif key == 'fz':
                                forces[:, 2] = data[:, i]
                            elif key == 'mux':
                                orientations[:, 0] = data[:, i]
                            elif key == 'muy':
                                orientations[:, 1] = data[:, i]
                            elif key == 'muz':
                                orientations[:, 2] = data[:, i]
                            elif key == 'omegax':
                                omegas[:, 0] = data[:, i]
                            elif key == 'omegay':
                                omegas[:, 1] = data[:, i]
                            elif key == 'omegaz':
                                omegas[:, 2] = data[:, i]
                            elif key == 'tqx':
                                torque[:, 0] = data[:, i]
                            elif key == 'tqy':
                                torque[:, 1] = data[:, i]
                            elif key == 'tqz':
                                torque[:, 2] = data[:, i]
                            elif key == 'angmomx':
                                angmom[:, 0] = data[:, i]
                            elif key == 'angmomy':
                                angmom[:, 1] = data[:, i]
                            elif key == 'angmomz':
                                angmom[:, 2] = data[:, i]
                            elif key == 'type':
                                if key not in frame.keys():
                                    frame.create_dataset(key,
                                                         (N,),
                                                         data=data[:, i],
                                                         dtype=int,
                                                         compression=COMPRESSION,
                                                         shuffle=SHUFFLE,
                                                         fletcher32=FLETCHER)
                                else:
                                    frame[key][:] = data[:, i]
                            elif key == 'id':
                                if key not in frame.keys():
                                    frame.create_dataset(key,
                                                         (N,),
                                                         data=data[:, i],
                                                         dtype=int,
                                                         compression=COMPRESSION,
                                                         shuffle=SHUFFLE,
                                                         fletcher32=FLETCHER)
                                else:
                                    frame[key][:] = data[:, i]
                            else:
                                if key not in frame.keys():
                                    frame.create_dataset(key,
                                                         (N,),
                                                         data=data[:, i],
                                                         dtype=DTYPE,
                                                         compression=COMPRESSION,
                                                         shuffle=SHUFFLE,
                                                         fletcher32=FLETCHER)
                                else:
                                    frame[key][:] = data[:, i]

                        if 'coords' not in frame.keys():
                            frame.create_dataset('coords',
                                                 (N, 3),
                                                 data=coords,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['coords'][:] = coords

                        if 'uwcoords' not in frame.keys() and unwrapped_available:
                            frame.create_dataset('uwcoords',
                                                 (N, 3),
                                                 data=uwcoords,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        elif 'uwcoords' in frame.keys() and unwrapped_available:
                            frame['uwcoords'][:] = uwcoords

                        if 'velocities' not in frame.keys():
                            frame.create_dataset('velocities',
                                                 (N, 3),
                                                 data=velocities,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['velocities'][:] = velocities

                        if 'forces' not in frame.keys():
                            frame.create_dataset('forces',
                                                 (N, 3),
                                                 data=forces,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['forces'][:] = forces

                        if 'orientations' not in frame.keys():
                            frame.create_dataset('orientations',
                                                 (N, 3),
                                                 data=orientations,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['orientations'][:] = orientations

                        if 'omegas' not in frame.keys():
                            frame.create_dataset('omegas',
                                                 (N, 3),
                                                 data=omegas,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['omegas'][:] = omegas

                        if 'torque' not in frame.keys():
                            frame.create_dataset('torque',
                                                 (N, 3),
                                                 data=torque,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['torque'][:] = torque

                        if 'angmom' not in frame.keys():
                            frame.create_dataset('angmom',
                                                 (N, 3),
                                                 data=angmom,
                                                 dtype=DTYPE,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)
                        else:
                            frame['angmom'][:] = angmom

                        # spatial dimension
                        frame.attrs['d'] = d

                        # add type if not specified
                        if 'type' not in frame.keys():
                            frame.create_dataset('type',
                                                 (N,),
                                                 data=np.ones(N),
                                                 dtype=int,
                                                 compression=COMPRESSION,
                                                 shuffle=SHUFFLE,
                                                 fletcher32=FLETCHER)

                        steps[n] = step

                        del lines

                if idwarning:
                    # Warn user if data could not be sorted.
                    self.__log.warning(
                        "No IDs specified. Data may be unsorted between "\
                        "individual frames."
                    )

            self.steps = steps
            # get time step from log file (this also sets self.times)
            self.dt = self.__get_timestep_from_logfile()
            self.d = d
        except Exception as e:
            os.remove(os.path.join(savedir, "#temp#"+trajfile))
            # print("LammpsReader: loading dump files failed.")
            if "LammpsReader: no dump files in this directory." in e.args[0]:
                # print(e)
                # we should not raise an error in this case.
                pass
            # print(f"LammpsReader: error while loading dump files. {e}")
            raise
        else:
            # if no exception. rename #temp#<trajfile> and delete #<trajfile>.
            if os.path.exists(os.path.join(savedir, trajfile)) and verbose:
                # This case should not occur if AMEP is used properly.
                # Possible scenario:
                # two instances of AMEP reading and writing in the same directory.
                self.__log.info(
                    f"File {trajfile} already exists. "\
                    "Overwriting existing file."
                )
            os.rename(os.path.join(savedir, "#temp#"+trajfile), os.path.join(savedir, trajfile))
            self.filename = trajfile

            # delete old trajectory file #<trajfile> if specified by user.
            if deleteold and os.path.exists(os.path.join(savedir, "#"+trajfile)):
                os.remove(os.path.join(savedir, "#"+trajfile))
                if verbose:
                    self.__log.info(
                        f"Deleted old trajectory file #{trajfile}"
                    )
        finally:
            pass


    def __sorter(self, item):
        r'''
        Returns the time step of a dump file that is given
        in the filename of the dump file.

        INPUT:
            item: dump-file name (str)

        OUTPUT:
            time step (float)
        '''
        key = self.__dumps.split('*')[0]
        basedir, basename = os.path.split(item)
        if key=='':
            return float(basename.split('.')[0])
        return float(basename.split(key)[1].split('.')[0])


    def __get_timestep_from_logfile(self):
        r'''
        Reads the time step from the log.lammps file.

        Returns
        -------
        dt : float
            Time step.
        '''
        if os.path.exists(os.path.join(self.directory, 'log.lammps')):
            with open(os.path.join(
                    self.directory, 'log.lammps'
                ), encoding="utf-8") as f:
                lines = f.readlines()
                relevant = [line for line in lines if line.startswith('timestep')]
                if len(relevant)>=1:
                    dt = float(relevant[-1].split('timestep')[-1])
                else:
                    dt = 1
                    self.__log.warning(
                        "No timestep mentioned in logfile. Using dt=1. "\
                        "The timestep can be set manually by traj.dt=<dt>."
                    )
                if len(relevant)>1:
                    self.__log.warning(
                        "More than one timestep found. Using last mention, "\
                        f"dt={dt}. "\
                        "The timestep can be set manually by traj.dt=<dt>."
                    )
        else:
            self.__log.warning(
                "No log file found. Using dt=1. "\
                "The timestep can be set manually by traj.dt=<dt>."
            )
            dt = 1

        return dt


class ContinuumReader(BaseReader):
    """A reader for continuum data"""
    def __init__(
            self, directory: str, savedir: str, start: float = 0.0,
            stop: float = 1.0, nth: int = 1, trajfile: str = TRAJFILENAME,
            deleteold: bool = False, dumps: str = 'dump*.txt',
            gridfile: str = 'grid.txt', delimiter: str = " ",
            timestep: float | None = None, verbose: bool = False) -> None:
        r'''
        Reader for continuum simulation data.

        Parameters
        ----------
        directory : str
            Simulation directory.
        savedir : str
            Directory in which the trajectory file will be stored.
        start : float, optional
            Start reading data from this fraction of the whole trajectory.
            The default is 0.
        stop : float, optional
            Stop reading data from this fraction of the whole trajectory.
            The default is 1.
        nth : int, optional
            Read each nth dump file. The default is None.
        dumps : str, optional
            File name of the dump files. The default is 'dump*.txt'.
        trajfile : str, optional
            Name of the hdf5 trajectory file that is created when an object of
            this class is initialized. The default is TRAJFILENAME.
        deleteold : bool, optional
            If True, an existing old h5amep trajectory file #<trajfile>
            will be removed. The default is False.
        delimiter : str, optional
            Delimiter used in the data files. The default is ' '.
        timestep : float or None, optional
            Timestep size used in the simulation. The default is None.
        gridfile : str, optional
            File name that contains the information about the grid used in the
            simulation. The default is `'grid.txt'`.
        verbose : bool, optional
            If True, runtime information is printed. The default is False.

        Returns
        -------
        None.

        '''
        # init class logger
        self.__log = get_class_logger(__name__, self.__class__.__name__)

        # check if trajectory file already exists
        # and if exists create a backup
        if os.path.exists(os.path.join(savedir, trajfile)):
            # rename trajectory to backup filename "#+<trajfile>"
            if os.path.exists(os.path.join(savedir, "#"+trajfile)):
                if verbose:
                    self.__log.info(
                        f"Existing old file #{trajfile} will be replaced."
                    )
                os.remove(os.path.join(savedir, "#"+trajfile))
            os.rename(
                os.path.join(savedir, trajfile),
                os.path.join(savedir, "#" + trajfile)
            )
            if verbose:
                self.__log.info(
                    f"Renamed {trajfile} to #{trajfile}."
                )
        super().__init__(os.path.abspath(savedir),
                         start, stop, nth, f"#temp#{trajfile}")
        self.directory = directory

        self.__dumps = dumps
        self.__gridfile = gridfile
        self.__delimiter_warning = False
        self.__time_warning = False
        self.__verbose = verbose

        try:
            # get data files
            fields = sorted(glob.glob(os.path.join(self.directory,
                                                   self.__dumps)),
                            key=self.__sorter)

            if len(fields) == 0:
                raise RuntimeError(
                    f"amep.reader.ContinuumReader: No dump files {dumps} "\
                    f"in directory {directory}."
                )

            # determine first and last file index of files to be loaded   
            first = int(self.start*len(fields))  # first dump file index
            last  = int(self.stop*len(fields))   # last dump file index

            # dump files to load
            fields = fields[first:last:self.nth]
            
            # array of number of time steps
            steps = np.zeros(len(fields), dtype=int)
            times = np.zeros(len(fields), dtype=float)

            with h5py.File(os.path.join(self.savedir, self.filename),
                           "a") as root:
                # set type
                root.attrs["type"] = "field"

            d = self.__set_grid(os.path.join(self.directory, gridfile), delimiter)
            for index, field_file in enumerate(tqdm(fields)):
                step, time = self.__set_field(field_file, index, delimiter)
                steps[index] = step
                if time == 0.0:
                    # use dt=1
                    time = step
                times[index] = time
            self.steps = steps
            # set time step
            if timestep is not None:
                self.dt = timestep
            else:
                self.times = times
                self.__log.warning(
                    "Timestep not set. "\
                    "Use default timestep of 1.0. This value is ignored if the "\
                    "'TIME' is specified in your data files. Please check the "\
                    "required continuum data format for more details. "\
                    "The timestep can be set manually by traj.dt=<dt>."
                )
            # set dimension
            self.d = d
            
        except Exception as err:
            os.remove(os.path.join(savedir, "#temp#"+trajfile))
            if "ContinuumReader: no dump files in this directory." in err.args[0]:
                # we should not raise an error in this case.
                pass
            raise
        else:
            # if no exception. rename #temp#<trajfile> and delete #<trajfile>.
            if os.path.exists(os.path.join(savedir, trajfile)) and verbose:
                # This case should not occur if AMEP is used properly.
                # Possible scenario:
                # two instances of AMEP reading and writing in the same directory.
                self.__log.info(
                    f"ContinuumReader: file {trajfile} already exists. "\
                    "Overwriting existing file."
                )
            os.rename(
                os.path.join(savedir, "#temp#"+trajfile),
                os.path.join(savedir, trajfile)
            )
            self.filename = trajfile

            # delete old trajectory file #<trajfile> if specified by user.
            if deleteold and os.path.exists(os.path.join(savedir, "#"+trajfile)):
                os.remove(os.path.join(savedir, "#"+trajfile))
                if verbose:
                    self.__log.info(
                        f"Deleted old trajectory file #{trajfile}"
                    )
        finally:
            pass

    def __extract_data(self, fil: str):
        pass

    def __set_grid(self, path: str, delimiter: str):
        r"""Takes a grid file and writes all parameters """
        # check if given grid file exists
        if not os.path.exists(path):
            raise FileNotFoundError(
                "amep.reader.ContinuumReader: No such file or directory: "\
                f"{path}. Please check the path of your grid file."
            )
        
        with open(path, "r", encoding="utf-8") as rfile, h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            
            grid = root["frames"].require_group("grid")

            coordinates = False
            box = False
            box_boundary = np.zeros((3,2))
            shape = False
            any_shape = False
            data = []
            
            # check delimiter
            delimiters: list | None = self.__detect_delimiter(rfile.read())
            if delimiters is None and self.__verbose:
                self.__log.warning(
                    "Delimiter detection in "\
                    f"{path} failed. File {path} might have the wrong format."
                )
            elif delimiter not in delimiters and self.__verbose:
                self.__log.warning(
                    f"The given delimiter '{delimiter}' is not in the list "\
                    f"of detected delimiters {delimiters} found in file "\
                    f"{path}. You might have to change the "\
                    f"delimiter keyword to one of {delimiters}."
                )

            # set reader cursor to start of file again
            rfile.seek(0)
            
            # read data from file line by line
            for line in csv.reader(rfile, delimiter=delimiter):
                match line:
                    case ["COORDINATES:", *keys]:
                        # box = False
                        coordinates = True
                    case ["BOX:"]:
                        box = True
                        boxcount=0
                    case ['SHAPE:']:
                        shape = True
                        any_shape = True
                    case _:
                        if box and boxcount < 3:
                            line = np.array(line, dtype=float)
                            box_boundary[boxcount] = line
                            boxcount += 1
                            if boxcount > 3:
                                box = False
                        elif coordinates:
                            data.append(line)
                        elif shape:
                            shape_data = np.array(line, int)
                            shape_data = shape_data[shape_data != 0]
                            if 'shape' not in grid.keys():
                                grid.create_dataset(
                                    'shape',
                                    data=shape_data,
                                    dtype=int,
                                    compression=COMPRESSION,
                                    shuffle=SHUFFLE,
                                    fletcher32=FLETCHER
                                )
                            else:
                                grid['shape'][:] = shape_data
                            shape = False

            # check data availability
            if not any_shape:
                raise RuntimeError(
                    f"amep.reader.ContinuumReader: The shape information is "\
                    f"missing in file {path}. Please check the required "\
                    "format and provide the shape of the data in the grid file."
                )
            if not coordinates:
                raise RuntimeError(
                    "amep.reader.ContinuumReader: The coordinates are "\
                    f"missing in file {path}. Please check the required format "\
                    "and provide the coordinates of the grid in the grid file."
                )
            if not box:
                raise RuntimeError(
                    "amep.reader.ContinuumReader: The box is not given in "\
                    f"file {path}. Please check the required format and "\
                    "provide the boundaries of the simulation box in the "\
                    "grid.txt file."
                )
            # add box to h5amep file
            if 'box' not in grid.keys():
                grid.create_dataset(
                    'box',
                    data=box_boundary,
                    dtype=DTYPE,
                    compression=COMPRESSION,
                    shuffle=SHUFFLE,
                    fletcher32=FLETCHER
                )
            else:
                grid['box'][:] = box_boundary
            # add grid coordinates to h5amep file
            data = np.asarray(data, dtype=float)
            d = 0  # spatial dimension
            for i, key in enumerate(keys):
                if not data[:,i].any() == 0:
                    if key not in grid.keys():
                        grid.create_dataset(
                            key,
                            data=data[:,i],
                            dtype=DTYPE,
                            compression=COMPRESSION,
                            shuffle=SHUFFLE,
                            fletcher32=FLETCHER
                        )
                    else:
                        grid[key][:] = data[:,i]
                    d += 1
        return d

    def __set_field(self, path: str, time_index: int,
                    delimiter: str) -> tuple[int, float]:
        with open(path, "r", encoding="utf-8") as rfile, h5py.File(os.path.join(self.savedir, self.filename), 'a') as root:
            data = False
            timestep = False
            any_time_step = False
            time = False
            any_time = False
            now_step = 0
            now_phys = 0.0
            linecount = 0

            # check delimiter
            if not self.__delimiter_warning:
                delimiters: list | None = self.__detect_delimiter(rfile.read())
                if delimiters is None and self.__verbose:
                    self.__log.warning(
                        "Delimiter detection in "\
                        f"{path} failed. File {path} seems to have the wrong "\
                        "format. You can ignore this warning if your data "\
                        "file contains only one column of data."
                    )
                    self.__delimiter_warning = True
                elif delimiter not in delimiters and self.__verbose:
                    self.__log.warning(
                        "The given delimiter "\
                        f"'{delimiter}' is not in the list of detected "\
                        f"delimiters {delimiters} found in file {path}. "\
                        "You might have to change the delimiter keyword to "\
                        f"one of {delimiters}. You can ignore this warning "\
                        "if your data file contains only one column of data."
                    )
                    self.__delimiter_warning = True
                
            # set reader cursor to start of file again
            rfile.seek(0)
            
            # read data line by line
            this_reader = csv.reader(rfile, delimiter=delimiter)
            for line in this_reader:
                match line:
                    case ["TIMESTEP:"]:
                        timestep = True
                        any_time_step = True
                        linecount += 1
                    case ["TIME:"]:
                        time = True
                        any_time = True
                        linecount += 1
                    case ["DATA:", *keys]:
                        data = True
                        root['frames'].require_group(str(now_step))
                        linecount += 1
                        break
                    case _:
                        if timestep:
                            now_step = int(line[0])
                            timestep = False
                            linecount += 1
                        elif time:
                            now_phys = float(line[0])
                            time = False
                            linecount += 1

            # check data availability
            if not data:
                raise RuntimeError(
                    "amep.reader.ContinuumReader: The data is not given in "\
                    f"file {path} or file {path} has the wrong format. Please "\
                    "check the required format and provide the data the dump "\
                    "file."
                )
            if not any_time_step:
                raise RuntimeError(
                    "amep.reader.ContinuumReader: The time step is missing in "\
                    f"file {path}. Please check the required format "\
                    "and provide the time step in the dump file."
                )
            # TODO: Deprecate the field data format without
            # explicit physical time.
            if not any_time and not self.__time_warning:
                warnings.warn(
                    "You want to import continuum data with no physical time. "\
                    "This possibility will soon be deprecated. Please look up "\
                    "the correct data format in the Documentation.",
                    DeprecationWarning
                )
                self.__time_warning = True

            data_array = np.array(list(this_reader), dtype=float)
            for i, key in enumerate(keys):
                if key not in root['frames'][str(now_step)].keys():
                    root['frames'][str(now_step)].create_dataset(
                        key,
                        data=data_array[:, i],
                        dtype=DTYPE,
                        compression=COMPRESSION,
                        shuffle=SHUFFLE,
                        fletcher32=FLETCHER
                    )
                else:
                    root['frames'][str(now_step)][key][:] = data_array[:, i]
        return now_step, now_phys

    def __sorter(self, item):
        r'''
        Returns the time step of a dump file that is given
        in the filename of the dump file.

        INPUT:
            item: dump-file name (str)

        OUTPUT:
            time step (float)
        '''
        key = self.__dumps.split('*')[0]
        basedir, basename = os.path.split(item)
        if key == '':
            return float(basename.split('.')[0])
        return float(basename.split(key)[1].split('.')[0].strip("_"))

    def __detect_delimiter(
            self, text: str,
            delimiters: list[str] = [',', ';', '|', ' ', '\t', ':']
            ) -> list[str] | None:
        r"""
        Detects the delimiter used in the given text.

        Parameters
        ----------
        text : str
            Text of which the delimiter should be identified.
        delimiters : list of str, optional
            Possible delimiters to check for. The default is
            `[',', ';', '|', ' ', '\t', ':']`.

        Returns
        -------
        list or None
            Detected/default delimiter(s).

        """
        # invalid delimiters (e.g., letters, numbers, dot)
        blacklist = frozenset(
            string.ascii_letters + string.digits + '.' + '\n' + '-'
        )
        # dictionary of counted delimiters
        counterdict = {}
        for delimiter in delimiters:
            counterdict[delimiter] = 0
        
        # count delimiters in the given text
        for char in text:
            if char not in blacklist:
                for delimiter in delimiters:
                    if char == delimiter:
                        counterdict[delimiter] += 1
                    elif char in counterdict:
                        counterdict[char] += 1
                    else:
                        counterdict[char] = 1
                        
        # get most likely delimiter(s)
        delimiter = []
        maxcount = max(counterdict.values())
        for key, val in counterdict.items():
            if val == maxcount and val > 1:
                delimiter.append(key)
        
        if delimiter == []:
            return None
        return delimiter


class HOOMDReader(BaseReader):
    '''Reads hoomd-blue GSD files and writes the containing data to an hdf5 file.
    '''

    def __init__(
            self, directory: str, savedir: str, start: float = 0.0,
            stop: float = 1.0, nth: int = 1, filename: str = 'trajectory.gsd',
            dt: float = 1.0, e0: np.ndarray = [1,0,0],
            trajfile: str = TRAJFILENAME, deleteold: bool = False,
            verbose: bool = False) -> None:
        r'''
        Reader for simulation data in the hoomd-blue gsd file format.

        Note: The gsd file format does not save the (physical) time.
            Only the timestep is saved. By supplying the simulation 
            timestep `dt`, the (physical) time will be calculated.
            The default for `dt` is 1.

        Parameters
        ----------
        directory : str
            Simulation directory.
        savedir : str
            Directory in which the trajectory file will be stored.
        start : float, optional
            Start reading data from this fraction of the whole trajectory.
            The default is 0.
        stop : float, optional
            Stop reading data from this fraction of the whole trajectory.
            The default is 1.
        nth : int, optional
            Read each nth dump file. The default is None.
        filename : str, optional
            File name of the gsd trajectory file. The default is 'trajectory.gsd'.
        dt : float, optional
            Timestep of the simulation. The GSD file format does only save
            the simulation step number but not the (physical) time.
            The default is 1.
        e0 : np.ndarray, optional
            Orientation vector of the particles. hoomd-blue does only save
            rotations instead of orientations, thus an (initial) orientation
            vector is needed. This can, for example, be the self-propulsion
            direction of the particles. shape=(3,) or (N,3,).
            The default is [1,0,0].
        trajfile : str, optional
            Name of the hdf5 trajectory file that is created when an object of
            this class is initialized. The default is TRAJFILENAME.
        deleteold : bool, optional
            If True, an existing old h5amep trajectory file #<trajfile>
            will be removed. The default is False.
        verbose : bool, optional
            If True, runtime information is printed. The default is False.

        Returns
        -------
        None.

        '''
        # init class logger
        self.__log = get_class_logger(__name__, self.__class__.__name__)
        
        # back up trajectory file
        if os.path.exists(os.path.join(savedir, trajfile)):
            # rename trajectory to backup filename "#+<trajfile>"
            if os.path.exists(os.path.join(savedir, "#"+trajfile)):
                if verbose:
                    self.__log.info(
                        f"Existing old file #{trajfile} will be replaced."
                    )
                os.remove(os.path.join(savedir, "#"+trajfile))
            os.rename(
                os.path.join(savedir, trajfile),
                os.path.join(savedir, "#"+trajfile)
            )
            if verbose:
                self.__log.info(
                    f"Renamed {trajfile} to #{trajfile}."
                )
        # super(LammpsReader, self).__init__(directory, start, stop, nth, "#temp#"+trajfile)
        super().__init__(os.path.abspath(savedir), start, stop, nth, "#temp#"+trajfile)
        self.directory = directory

        try:
            # self.__filename = filename # important: self.filename ≠ self.__filename

            # load gsd file
            gsd_file = hoomd.open(os.path.join(directory,filename))

            first = int(self.start*len(gsd_file))  # first frame index
            last  = int(self.stop*len(gsd_file))   # last frame index
            
            # select part of gsd.hoomd.HOOMDTrajectory ( = gsd.hoomd._HOOMDTrajectoryView )
            gsd_file = gsd_file[first:last:self.nth]

            # total number of files
            self.__nframes = len(gsd_file)

            # array of number of time steps
            steps = np.zeros(self.__nframes, dtype=int)

            # load data from dump files
            with h5py.File(os.path.join(self.savedir, "#temp#"+trajfile), 'a') as root:
                # Set type of h5amep file to particle
                root.attrs["type"] = "particle"

                # If no ID is specified only one warning is issued to the user.
                # Trigger idwarning is set to false:
                idwarning=False

                # loop through all dump files
                for n, gsd_frame in enumerate(tqdm(gsd_file)):

                    # get number of time steps
                    step = gsd_frame.configuration.step

                    # get total number of atoms
                    N = gsd_frame.particles.N

                    # create new group for the given step in the 'frames' group
                    if str(step) not in root['frames'].keys():
                        frame = root['frames'].create_group(str(step))
                    else:
                        frame = root['frames'][str(step)]

                    # get boundaries of the simulation box
                    gsd_box = gsd_frame.configuration.box
                    if np.any(gsd_box[3:] != 0):
                        raise Exception(f"GSDReader: box tilt factors {gsd_box[3:]} must be 0.")
                    box=np.zeros((3,2))
                    box[:,0]=-gsd_box[:3]/2
                    box[:,1]=gsd_box[:3]/2 # hoomd-gsd box always centered around 0
                    if 'box' not in frame.keys():
                        frame.create_dataset('box',
                                             (3, 2),
                                             data=box,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['box'][:] = box

                    # get center of the simulation box
                    center = (box[:, 1]+box[:, 0])/2
                    if 'center' not in frame.keys():
                        frame.create_dataset('center',
                                             (3,),
                                             data=center,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['center'][:] = center

                    # add data to hdf5 file
                    forces       = np.zeros((N, 3), dtype=DTYPE) # gsd stores no forces
                    if 'forces' not in frame.keys():
                        frame.create_dataset('forces',
                                             (N, 3),
                                             data=forces,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['forces'][:] = forces

                    torque       = np.zeros((N, 3), dtype=DTYPE) # gsd stores no forces
                    if 'torque' not in frame.keys():
                        frame.create_dataset('torque',
                                             (N, 3),
                                             data=torque,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['torque'][:] = torque

                    # coordinates
                    coords       = np.array(gsd_frame.particles.position)
                    if 'coords' not in frame.keys():
                        frame.create_dataset('coords',
                                             (N, 3),
                                             data=coords,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['coords'][:] = coords

                    # velocities
                    velocities   = np.array(gsd_frame.particles.velocity)
                    if 'velocities' not in frame.keys():
                        frame.create_dataset('velocities',
                                             (N, 3),
                                             data=velocities,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['velocities'][:] = velocities

                    # type ids of the particles
                    type_ids     = np.array(gsd_frame.particles.typeid)
                    if "type" not in frame.keys():
                        frame.create_dataset("type",
                                             (N,),
                                             data=type_ids,
                                             dtype=int,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame["type"][:] = data[:, i]

                    # type names of the particles
                    type_names = np.array(gsd_frame.particles.types)[np.array(gsd_frame.particles.typeid)]
                    type_names = [str(a) for a in type_names]
                    # delete first if exists. length of string might have to be updated!
                    if "type_name" in frame.keys():
                        del frame["type_name"]
                    if "type_name" not in frame.keys():
                        frame.create_dataset("type_name",
                                             (N,),
                                             data=type_names,
                                             dtype=h5py.string_dtype('utf-8', len(max(type_names, key=len))),
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)

                    # dimensions of the simulation
                    d = gsd_frame.configuration.dimensions
                    frame.attrs['d'] = d

                    # unwrapped coordinates/particle image
                    gsd_image=gsd_frame.particles.image
                    x=coords[:,0]+gsd_image[:,0]*gsd_box[0]+gsd_box[3]*gsd_image[:,1]*gsd_box[1]+gsd_box[4]*gsd_image[:,2]*gsd_box[2]
                    y=coords[:,1]+gsd_image[:,1]*gsd_box[1]+gsd_box[5]*gsd_image[:,2]*gsd_box[2]
                    z=coords[:,2]+gsd_image[:,2]*gsd_box[2]
                    uwcoords = np.stack([x,y,z]).T
                    if 'uwcoords' not in frame.keys():
                        frame.create_dataset('uwcoords',
                                             (N, 3),
                                             data=uwcoords,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    elif 'uwcoords' in frame.keys():
                        frame['uwcoords'][:] = uwcoords

                    # orientations and orientation angle quat_theta from orientation-quaternions
                    quat_orientations   = np.array(gsd_frame.particles.orientation)
                    # orientations        = quat_orientations[:,1:]/np.linalg.norm(quat_orientations[:,1:], axis=1)[:, None] # old orientation calculation - but hoomd-blue only saves rotations instead of absolute orientations.
                    orientations        = quaternion_rotate(quat_orientations, e0)
                    orientations        /= np.linalg.norm(orientations, axis=1)[:,None]
                    if 'orientations' not in frame.keys():
                        frame.create_dataset('orientations',
                                             (N, 3),
                                             data=orientations,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['orientations'][:] = orientations
                    # quat_thetas         = 2*np.arctan2(np.linalg.norm(quat_orientations[:,1:], axis=1), quat_orientations[:,0])
                    # if 'quat_theta' not in frame.keys():
                    #     frame.create_dataset('quat_theta',
                    #                          (N,),
                    #                          data=quat_thetas,
                    #                          dtype=DTYPE,
                    #                          compression=COMPRESSION,
                    #                          shuffle=SHUFFLE,
                    #                          fletcher32=FLETCHER)
                    # else:
                    #     frame['quat_theta'][:] = quat_thetas

                    # moment of inertia of the particles
                    moment_inertias = np.array(gsd_frame.particles.moment_inertia)
                    if 'moment_inertia' not in frame.keys():
                        frame.create_dataset('moment_inertia',
                                             (N, 3),
                                             data=moment_inertias,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['moment_inertia'][:] = moment_inertias

                    # angular momentum of the particles from quaternions
                    # see: https://hoomd-blue.readthedocs.io/en/v2.9.3/aniso.html
                    quat_angmoms    = np.array(gsd_frame.particles.angmom)
                    angmoms_principal = 0.5*quaternion_multiply(quaternion_conjugate(quat_orientations), quat_angmoms)[:,1:]
                    angmoms          = quaternion_rotate(quat_orientations, angmoms_principal)
                    if 'angmom' not in frame.keys():
                        frame.create_dataset('angmom',
                                             (N, 3),
                                             data=angmoms,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['angmom'][:] = angmoms
                    # angmom_thetas   = quat_angmoms[:,0]
                    # if 'angmom_theta' not in frame.keys():
                    #     frame.create_dataset('angmom_theta',
                    #                          (N, ),
                    #                          data=angmom_thetas,
                    #                          dtype=DTYPE,
                    #                          compression=COMPRESSION,
                    #                          shuffle=SHUFFLE,
                    #                          fletcher32=FLETCHER)
                    # else:
                    #     frame['angmom_theta'][:] = angmom_thetas
                    omegas_principal   = angmoms_principal / moment_inertias
                    omegas             = quaternion_rotate(quat_orientations, omegas_principal)
                    if 'omegas' not in frame.keys():
                        frame.create_dataset('omegas',
                                             (N, 3),
                                             data=omegas,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['omegas'][:] = omegas

                    # radius of the particles
                    # AMEP works with radius instead of diameters
                    radii       = np.array(gsd_frame.particles.diameter)/2
                    if 'radius' not in frame.keys():
                        frame.create_dataset('radius',
                                             (N,),
                                             data=radii,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['radius'][:] = radii

                    # masses of the particles
                    masses          = np.array(gsd_frame.particles.mass)
                    if 'mass' not in frame.keys():
                        frame.create_dataset('mass',
                                             (N,),
                                             data=masses,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['mass'][:] = masses

                    # masses of the particles
                    charges         = np.array(gsd_frame.particles.charge)
                    if 'charge' not in frame.keys():
                        frame.create_dataset('charge',
                                             (N,),
                                             data=charges,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['charge'][:] = charges

                    # masses of the particles
                    bodies          = np.array(gsd_frame.particles.body)
                    if 'body' not in frame.keys():
                        frame.create_dataset('body',
                                             (N,),
                                             data=bodies,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['body'][:] = bodies

                    # do not save:
                    # print("type_shapes", gsd_frame.particles.type_shapes) # too complicated to save?

                    steps[n] = step

            self.steps = steps
            self.dt = dt
            self.d = d
        except Exception as e:
            os.remove(os.path.join(savedir, "#temp#"+trajfile))
            raise
        else:
            # if no exception. rename #temp#<trajfile> and delete #<trajfile>.
            if os.path.exists(os.path.join(savedir, trajfile)) and verbose:
                # This case should not occur if AMEP is used properly.
                # Possible scenario:
                # two instances of AMEP reading and writing in the same directory.
                self.__log.info(
                    f"File {trajfile} already exists. "\
                    "Overwriting existing file."
                )
            os.rename(os.path.join(savedir, "#temp#"+trajfile), os.path.join(savedir, trajfile))
            self.filename = trajfile

            # delete old trajectory file #<trajfile> if specified by user.
            if deleteold and os.path.exists(os.path.join(savedir, "#"+trajfile)):
                os.remove(os.path.join(savedir, "#"+trajfile))
                if verbose:
                    self.__log.info(
                        f"Deleted old trajectory file #{trajfile}"
                    )
        finally:
            pass


    def __sorter(self, item):
        r'''
        Returns the time step of a dump file that is given
        in the filename of the dump file.

        INPUT:
            item: dump-file name (str)

        OUTPUT:
            time step (float)
        '''
        key = self.__dumps.split('*')[0]
        basedir, basename = os.path.split(item)
        if key=='':
            return float(basename.split('.')[0])
        return float(basename.split(key)[1].split('.')[0])



class GROMACSReader(BaseReader):
    '''Reads GROMACS files and writes the containing data to an hdf5 file.

    Necessary files: The trajectory (or one frame) in the file format of .trr 
    or .gro or similar (see chemfiles documentation https://chemfiles.org/ ) 
    as well as a topology in the format of .tpr or similar are needed.

    Note: The molecule identification uses the bonds and a maximum molecule
        search size `maxmss` with a default value of 20. If the molecules are
        not identified correctly, try increasing this value (up to the number
        of atoms of the larges molecule for example).

        For very small trajectory files (such as the example file), the
        h5amep format is larger than the GROMACS files. This evens out for
        larger simulations.
    '''

    def __init__(
            self, directory: str, savedir: str, start: float = 0.0,
            stop: float = 1.0, nth: int = 1, filename: str = 'traj.trr',
            topology: str = 'topol.tpr', maxmss: int = 20,
            dt: float = 1.0,
            trajfile: str = TRAJFILENAME, deleteold: bool = False,
            verbose: bool = False) -> None:
        r'''
        Reader for simulation data from GROMACS molecular dynamics simulations.

        TODO in progress

        Parameters
        ----------
        directory : str
            Simulation directory.
        savedir : str
            Directory in which the trajectory file will be stored.
        start : float, optional
            Start reading data from this fraction of the whole trajectory.
            Must be in [0,1). The default is 0.
        stop : float, optional
            Stop reading data from this fraction of the whole trajectory.
            Must be in (0,1]. The default is 1.
        nth : int, optional
            Read each nth dump file. The default is None.
        filename : str, optional
            File name of the GROMACS trajectory file. The default is 'traj.trr'.
        topology : str, optional
            File name of the GROMACS topology file. The default is 'topol.tpr'.
        maxmss : int
            Maximum molecule search size. If the molecules are not identified 
            correctly, try increasing this value up to the number of atoms 
            of the larges molecule for example. Because GROMACS is storing
            connected atoms in proximity to each other, smaller values most
            often work as well.
            The default is 20.
        dt : float, optional
            Timestep of the simulation. The GSD file format does only save
            the simulation step number but not the (physical) time.
            The default is 1.
        trajfile : str, optional
            Name of the hdf5 trajectory file that is created when an object of
            this class is initialized. The default is TRAJFILENAME.
        deleteold : bool, optional
            If True, an existing old h5amep trajectory file #<trajfile>
            will be removed. The default is False.
        verbose : bool, optional
            If True, runtime information is printed. The default is False.

        Returns
        -------
        None.

        '''
        # init class logger
        self.__log = get_class_logger(__name__, self.__class__.__name__)
        
        # back up trajectory file
        if os.path.exists(os.path.join(savedir, trajfile)):
            # rename trajectory to backup filename "#+<trajfile>"
            if os.path.exists(os.path.join(savedir, "#"+trajfile)):
                if verbose:
                    self.__log.info(
                        f"Existing old file #{trajfile} will be replaced."
                    )
                os.remove(os.path.join(savedir, "#"+trajfile))
            os.rename(
                os.path.join(savedir, trajfile),
                os.path.join(savedir, "#"+trajfile)
            )
            if verbose:
                self.__log.info(
                    f"Renamed {trajfile} to #{trajfile}."
                )
        # super(LammpsReader, self).__init__(directory, start, stop, nth, "#temp#"+trajfile)
        super().__init__(os.path.abspath(savedir), start, stop, nth, "#temp#"+trajfile)
        self.directory = directory

        try:
            # self.__filename = filename # important: self.filename ≠ self.__filename

            # load gromacs trajectory
            gromacs_file = Trajectory(os.path.join(directory,filename))
            gromacs_file.set_topology(os.path.join(directory,topology))

            first = int(self.start*gromacs_file.nsteps)  # first frame index
            last  = int(self.stop*gromacs_file.nsteps)   # last frame index
            
            # indices used for the trajectory:
            gromacs_indices = np.arange(first,last,nth)

            # total number of files
            self.__nframes = len(gromacs_indices)

            # array of number of time steps
            steps = np.zeros(self.__nframes, dtype=int)
            times = np.zeros(self.__nframes, dtype=int)

            # load data from dump files
            with h5py.File(os.path.join(self.savedir, "#temp#"+trajfile), 'a') as root:
                # Set type of h5amep file to particle
                root.attrs["type"] = "particle"

                # If no ID is specified only one warning is issued to the user.
                # Trigger idwarning is set to false:
                idwarning=False

                # save molecule_ids after first identification:
                molecule_ids = None

                # loop through all dump files
                for n, gromacs_index in enumerate(tqdm(gromacs_indices)):

                    gromacs_frame = gromacs_file.read_step(gromacs_index)
                    
                    # get time steps
                    step = gromacs_frame.step
                    time = gromacs_frame["time"]

                    # get total number of atoms
                    N = len(gromacs_frame.atoms)

                    # create new group for the given step in the 'frames' group
                    if str(step) not in root['frames'].keys():
                        frame = root['frames'].create_group(str(step))
                    else:
                        frame = root['frames'][str(step)]

                    # get boundaries of the simulation box
                    gromacs_box = gromacs_frame.cell
                    if np.any(np.array(gromacs_box.angles) != 90):
                        raise Exception(f"GROMACSReader: box angles {np.array(gromacs_box.angles)} must be 90°.")
                    box=np.zeros((3,2))
                    box[:,0]=np.array([0,0,0])
                    box[:,1]=np.array(gromacs_box.lengths) # gromax box always starting at 0
                    if 'box' not in frame.keys():
                        frame.create_dataset('box',
                                             (3, 2),
                                             data=box,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['box'][:] = box

                    # get center of the simulation box
                    center = (box[:, 1]+box[:, 0])/2
                    if 'center' not in frame.keys():
                        frame.create_dataset('center',
                                             (3,),
                                             data=center,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['center'][:] = center

                    # add data to hdf5 file
                    forces       = np.zeros((N, 3), dtype=DTYPE) # chemfiles reads no forces
                    if 'forces' not in frame.keys():
                        frame.create_dataset('forces',
                                             (N, 3),
                                             data=forces,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['forces'][:] = forces

                    torque       = np.zeros((N, 3), dtype=DTYPE) # gromacs stores no forces
                    if 'torque' not in frame.keys():
                        frame.create_dataset('torque',
                                             (N, 3),
                                             data=torque,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['torque'][:] = torque

                    # coordinates
                    if gromacs_frame["has_positions"]:
                        coords       = np.array(gromacs_frame.positions)
                    else:
                        coords       = np.zeros((N, 3), dtype=DTYPE)
                    if 'coords' not in frame.keys():
                        frame.create_dataset('coords',
                                             (N, 3),
                                             data=coords,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['coords'][:] = coords

                    # velocities
                    if gromacs_frame.has_velocities():
                        velocities   = np.array(gromacs_frame.velocities)
                    else:
                        velocities   = np.zeros((N, 3), dtype=DTYPE)
                    if 'velocities' not in frame.keys():
                        frame.create_dataset('velocities',
                                             (N, 3),
                                             data=velocities,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['velocities'][:] = velocities

                    # type ids of the particles
                    charges          = np.zeros((N,), dtype=DTYPE)
                    masses           = np.zeros((N,), dtype=DTYPE)
                    type_names       = []
                    atom_names       = []
                    residue_names    = []
                    residue_ids      = np.zeros((N,), dtype=int)
                    frame_topology=gromacs_frame.topology
                    for i, atom in enumerate(gromacs_frame.atoms):
                        masses[i] = atom.mass
                        charges[i] = atom.charge
                        type_names.append(atom["ff_type"])
                        atom_names.append(atom.name)
                        atomresidue=frame_topology.residue_for_atom(i)
                        residue_names.append(atomresidue.name)
                        residue_ids[i] = atomresidue.id

                    all_atom_types=np.unique(type_names)
                    type_ids     = np.array([np.where(all_atom_types==type_name)[0][0] for type_name in type_names])
                    if "type" not in frame.keys():
                        frame.create_dataset("type",
                                             (N,),
                                             data=type_ids,
                                             dtype=int,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame["type"][:] = data[:, i]

                    # type names of the particles
                    # delete first if exists. length of string might have to be updated!
                    if "type_name" in frame.keys():
                        del frame["type_name"]
                    if "type_name" not in frame.keys():
                        frame.create_dataset("type_name",
                                             (N,),
                                             data=type_names,
                                             dtype=h5py.string_dtype('utf-8', len(max(type_names, key=len))),
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)

                    if "atom_name" in frame.keys():
                        del frame["atom_name"]
                    if "atom_name" not in frame.keys():
                        frame.create_dataset("atom_name",
                                             (N,),
                                             data=atom_names,
                                             dtype=h5py.string_dtype('utf-8', len(max(type_names, key=len))),
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)

                    if "residue_name" in frame.keys():
                        del frame["residue_name"]
                    if "residue_name" not in frame.keys():
                        frame.create_dataset("residue_name",
                                             (N,),
                                             data=residue_names,
                                             dtype=h5py.string_dtype('utf-8', len(max(type_names, key=len))),
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)

                    if 'residue_id' not in frame.keys():
                        frame.create_dataset('residue_id',
                                             (N,),
                                             data=residue_ids,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['residue_id'][:] = residue_ids

                    # determine molecule ids from bonds
                    if molecule_ids is None:
                        molecule_ids     = np.zeros((N,), dtype=int)
                        # determine molecule ids from bonds
                        for i in range(1):
                            bonds=gromacs_frame.topology.bonds
                            molecules = [list(bonds[0])]
                            bonds=bonds[1:]
                            i=0
                            while len(bonds)>0:
                                bond = bonds[i]
                                molecule=molecules[-1]
                                if bond[0] in molecule or bond[1] in molecule:
                                    if i==0:
                                        bonds=bonds[1:]
                                    else:
                                        bonds=np.concatenate((bonds[:i], bonds[i+1:]))
                                    if bond[0] not in molecule:
                                        molecule.append(bond[0])
                                    if bond[1] not in molecule:
                                        molecule.append(bond[1])
                                    i=-1
                                i+=1
                                # maxmss : maximum molecule search size
                                # reduces computation time significantly
                                if (i==len(bonds) and i!=0) or (i>=maxmss):
                                    i=0
                                    molecules.append(list(bonds[i]))
                                    bonds=bonds[1:]
                    #         extend by single-atom molecules
                    #         necessary for unified-atom simulations of water for example
                    # #         from collections.abc import Iterable
                    #         def flatten(arg):
                    #             if not isinstance(arg, (list, np.ndarray)): # if not list
                    # #             if not isinstance(arg, Iterable): # if not list
                    #                 return [arg]
                    #             return [x for sub in arg for x in flatten(sub)]
                    # #         flattened=flatten(molecules)
                            flattened=[atoms for molecule in molecules for atoms in molecule]
                            if N!=len(flattened):
                                missing_atoms=np.delete(np.arange(0,N), flattened)
                                for atom in missing_atoms:
                                    molecules.append([atom])
                            for molecule_id, molecule in enumerate(molecules):
                                for atom in molecule:
                                    molecule_ids[atom] = molecule_id
                    # store molecule_ids
                    if 'molecule_id' not in frame.keys():
                        frame.create_dataset('molecule_id',
                                             (N,),
                                             data=molecule_ids,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['molecule_id'][:] = molecule_ids

                    # unwrapped coordinates/particle image
                    uwcoords = np.zeros((N, 3), dtype=DTYPE)
                    if 'uwcoords' not in frame.keys():
                        frame.create_dataset('uwcoords',
                                             (N, 3),
                                             data=uwcoords,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    elif 'uwcoords' in frame.keys():
                        frame['uwcoords'][:] = uwcoords

                    # orientations and orientation angle quat_theta from orientation-quaternions
                    orientations        = np.zeros((N, 3), dtype=DTYPE)
                    if 'orientations' not in frame.keys():
                        frame.create_dataset('orientations',
                                             (N, 3),
                                             data=orientations,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['orientations'][:] = orientations

                    # angular momentum
                    angmoms   = np.zeros((N, 3), dtype=DTYPE)
                    if 'angmom' not in frame.keys():
                        frame.create_dataset('angmom',
                                             (N, 3),
                                             data=angmoms,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['angmom'][:] = angmoms

                    # omegas
                    omegas       = np.zeros((N, 3), dtype=DTYPE)
                    if 'omegas' not in frame.keys():
                        frame.create_dataset('omegas',
                                             (N, 3),
                                             data=omegas,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['omegas'][:] = omegas

                    # diameters of the particles
                    diameters       = np.zeros((N,), dtype=DTYPE)
                    if 'diameter' not in frame.keys():
                        frame.create_dataset('diameter',
                                             (N,),
                                             data=diameters,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['diameter'][:] = diameters

                    # masses of the particles
                    if 'mass' not in frame.keys():
                        frame.create_dataset('mass',
                                             (N,),
                                             data=masses,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['mass'][:] = masses

                    # charges of the particles
                    if 'charge' not in frame.keys():
                        frame.create_dataset('charge',
                                             (N,),
                                             data=charges,
                                             dtype=DTYPE,
                                             compression=COMPRESSION,
                                             shuffle=SHUFFLE,
                                             fletcher32=FLETCHER)
                    else:
                        frame['charge'][:] = charges

                    # dimensions of the simulation
                    d = 3
                    frame.attrs['d'] = d
                    steps[n] = step
                    times[n] = time

            self.steps = steps
            self.times = times
            self.dt = dt
            self.d = d

            # close file
            gromacs_file.close()
        except Exception as e:
            os.remove(os.path.join(savedir, "#temp#"+trajfile))
            raise
        else:
            # if no exception. rename #temp#<trajfile> and delete #<trajfile>.
            if os.path.exists(os.path.join(savedir, trajfile)) and verbose:
                # This case should not occur if AMEP is used properly.
                # Possible scenario:
                # two instances of AMEP reading and writing in the same directory.
                self.__log.info(
                    f"File {trajfile} already exists. "\
                    "Overwriting existing file."
                )
            os.rename(os.path.join(savedir, "#temp#"+trajfile), os.path.join(savedir, trajfile))
            self.filename = trajfile

            # delete old trajectory file #<trajfile> if specified by user.
            if deleteold and os.path.exists(os.path.join(savedir, "#"+trajfile)):
                os.remove(os.path.join(savedir, "#"+trajfile))
                if verbose:
                    self.__log.info(
                        f"Deleted old trajectory file #{trajfile}"
                    )
        finally:
            pass


    def __sorter(self, item):
        r'''
        Returns the time step of a dump file that is given
        in the filename of the dump file.

        INPUT:
            item: dump-file name (str)

        OUTPUT:
            time step (float)
        '''
        key = self.__dumps.split('*')[0]
        basedir, basename = os.path.split(item)
        if key=='':
            return float(basename.split('.')[0])
        return float(basename.split(key)[1].split('.')[0])
