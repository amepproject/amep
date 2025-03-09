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
Data Loading
============

.. module:: amep.load

The AMEP module :mod:`amep.load` contains all method required to load 
simulation data or evaluation results.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import os
import h5py

from .reader import LammpsReader, H5amepReader, ContinuumReader, HOOMDReader, GROMACSReader
from .trajectory import ParticleTrajectory,FieldTrajectory
from .base import TRAJFILENAME, BaseEvalData, BaseDatabase, LOADMODES
from .base import check_path, get_module_logger

# logger setup
_log = get_module_logger(__name__)

# =============================================================================
# LOAD TRAJECTORY
# =============================================================================
def traj(
        directory: str, trajfile: str = TRAJFILENAME,
        savedir: str | None = None, mode: str = 'lammps',
        reload: bool = False, deleteold: bool = False, verbose: bool = False,
        **kwargs) -> FieldTrajectory | ParticleTrajectory:
    r"""
    Loads simulation data as trajectory object.

    Some remote storage systems do not allow full access necessary for
    reading, creating and editing HDF5 files used by AMEP (`'.h5amep'`).
    In such cases it is possible to read the simulation data on the remote
    storage system but save the `h5amep` file locally. Below you can find
    an example.

    Parameters
    ----------
    directory : str
        Simulation directory or, if wanting to load a h5amep file, the
        path including the trajectory name and file ending ".h5amep"
        can be given (see example).
    trajfile : str, optional
        Name of the hdf5 trajectory file that is created when an object of
        this class is initialized. The default is TRAJFILENAME.
    savedir : str or None, optional
        Target directory in which the trajectory file will be saved. If None, 
        the simulation directory is used. For `mode='h5amep'`, `savedir`
        is ignored. The default is None.
    mode : str, optional
        Loading mode. Available modes are 'lammps', 'fields', 'hoomd',
        'gromacs' and 'h5amep'.
        The default is 'lammps'.
    reload : bool, optional
        If True, all dump files will be loaded again, otherwise it will be
        tried to load an existing h5amep file. The default is False.
    deleteold : bool, optional
        If True, an existing old h5amep trajectory file #<filename>
        will be removed. The default is False.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.
    **kwargs
        Are forwared to the reader.

    Returns
    -------
    amep.trajectory.ParticleTrajectory or amep.trajectory.FieldTrajectory
        Trajectory object.

    Examples
    --------
    Minimal example for loading lammps trajectory from path `"data"`:

    >>> import amep
    >>> traj = amep.load.traj("/examples/data/lammps/")
    >>> traj = amep.load.traj("/examples/data/lammps/", reload=True) # data will be read again


    Shortcuts to load an h5amep file directly:

    >>> traj = amep.load.traj("/examples/data/lammps.h5amep")
    >>> traj = amep.load.traj(
    ...     "data", trajfile="traj.h5amep", mode="h5amep"
    ... )
    >>> 

    Example for loading GROMACS data:

    >>> path = "examples/data/gromacs/"
    >>> traj=amep.load.traj(directory=path, mode="gromacs", reload=True)

    Example for loading HOOMD data

    >>> path="examples/data/hoomd/"
    >>> traj=amep.load.traj(directory=path, mode="hoomd", reload=True)
    
    Example for loading continuum data:

    >>> traj = amep.load.traj("/examples/data/continuum/", mode="field", dumps="field_*")

    Fix for working with remote files and insufficient access rights:
    This can be solved by saving the h5amep file locally while
    importing/loading the simulation data from the remote directory.

    >>> traj = amep.load.traj(
    ...     "/path/to/sftp:host=192.168.255.255/remote/directory/",
    ...     savedir="/path/to/local/directory/"
    ... )
    >>> 
    
    """
    # check directory
    _, file_extension = os.path.splitext(directory)
    # check if trajfile is already supplied in directory
    if file_extension == '.h5amep':
        # switch to `mode='h5amep'`
        directory, trajfile = check_path(directory, ".h5amep")
        mode = "h5amep"
    elif file_extension == '':
        # check if directory exists
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                    f"The given directory '{directory}' does not exist."
                )
    else:
        # check if it is a file
        if os.path.isfile(directory):
            raise ValueError(
                f"Invalid file extension '{file_extension}'. "\
                "Please use the extension '.h5amep'."  
            )
        # check if it is a directory
        elif not os.path.isdir(directory):
            raise FileNotFoundError(
                    f"The given directory '{directory}' does not exist."
                )
    # check savedir
    if savedir is None:
        savedir = directory
    elif savedir is not None and mode == 'h5amep':
        if verbose:
            _log.info(
                f"savedir='{savedir}' is ignored for mode='{mode}'."
            )
        savedir = directory
    else:
        if os.path.isdir(savedir):
            if verbose:
                _log.info(
                    "amep.load.traj: trajectory file is stored in "\
                    f"savedir={savedir}."
                )
        else:
            raise FileNotFoundError(
                f"The given savedir '{savedir}' does not exist."    
            )
            
    # check trajfile
    if not trajfile.endswith('.h5amep'):
        raise ValueError(
            f"Invalid file extension '{os.path.splitext(trajfile)[1]}' for "\
            "trajfile. Please use the extension '.h5amep'."    
        )
        
    # check if mode is valid
    mode = mode.lower()
    if mode not in LOADMODES:
        raise KeyError(
            f'''amep.load.traj: mode \'{mode}\' does not exist.
                Available modes are {LOADMODES}.'''
        )
        
    # path of the trajectory file
    path = os.path.join(savedir, trajfile)
    
    # check if file already exists and load data from this file if not reload
    if os.path.exists(path) and not reload:
        reader = H5amepReader(
            savedir, trajfile = trajfile, verbose = verbose, **kwargs
        )
        if reader.type == 'field':
            return FieldTrajectory(reader)
        return ParticleTrajectory(reader)
    # load lammps data
    elif mode == 'lammps':
        reader = LammpsReader(
            directory,
            savedir,
            deleteold=deleteold,
            trajfile=trajfile,
            verbose=verbose,
            **kwargs
        )
        return ParticleTrajectory(reader)
    # load h5amep file directly
    elif mode == 'h5amep':
        reader = H5amepReader(
            savedir, trajfile = trajfile, verbose = verbose, **kwargs
        )
        if reader.type == "field":
            return FieldTrajectory(reader)
        return ParticleTrajectory(reader)
    # load continuum field data
    elif mode == 'field':
        reader = ContinuumReader(
            directory,
            savedir,
            trajfile = trajfile,
            deleteold = deleteold,
            verbose = verbose,
            **kwargs
        )
        return FieldTrajectory(reader)
    elif mode == 'hoomd':
        reader = HOOMDReader(
            directory,
            savedir,
            trajfile = trajfile,
            deleteold = deleteold,
            verbose = verbose,
            **kwargs
        )
        return ParticleTrajectory(reader)
    elif mode == 'gromacs':
        reader = GROMACSReader(
            directory,
            savedir,
            trajfile = trajfile,
            deleteold = deleteold,
            verbose = verbose,
            **kwargs
        )
        return ParticleTrajectory(reader)

    # here one has to check both the amep version with which the file has been
    # created (reader.version) and the data type (particles or fields) -
    # the latter is needed to decide whether a ParticleTrajectory or a
    # FieldTrajectory has to be created, the former will be needed in the
    # in the future if the trajectory file format has to be changed for some
    # reason but h5amep files of the old format should be still be able to be
    # loaded and used even with the newest AMEP version (which could be done
    # by defining old Trajectory classes to maintain compatibility with old
    # trajectory files)
    else:
        raise Exception(
            '''amep.load.traj: unknown error. Please contact the AMEP
               developers with a minimal working example for this behavior.'''
        )


# =============================================================================
# LOAD EVALUATION DATA
# =============================================================================
def evaluation(
        path: str, key: None | str = None,
        database: bool = False, verbose: bool = False
        ) -> BaseEvalData | BaseDatabase:
    r"""
    Loads evaluation data stored in an HDF5 file.

    Parameters
    ----------
    path : str
        Path of the HDF5 file.
    key : str or None, optional
        If database = False, the element key of the HDF5 file is returned.
        The default is None.
    database : bool, optional
        If True, a `BaseDatabase` object is returned. If False, a 
        `BaseEvalData` object is returned if the given HDF5 file only contains
        results of one evaluation (otherwise, an error is raised) or if a key
        is supplied. The default is False.
    verbose : bool, optional
        If True, runtime information is printed. The default is False.

    Returns
    -------
    BaseEvalData or BaseDatabase
        Object providing access to the HDF5 data file.
        
    Examples
    --------
    Calculate the MSD, save it, and load the saved data:

    >>> import amep
    >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
    >>> msd = amep.evaluate.MSD(traj)
    >>> msd.save('./eval/msd.h5')
    >>> data = amep.load.evaluation('./eval/msd.h5')
    >>> print(data.keys())
    ['frames', 'indices', 'times', 'avg', 'name', 'ptype']
    >>> 


    Calculate the VACF and save it together with the MSD in one HDF5 file.
    Then, load the results again:

    >>> vacf = amep.evaluate.VACF(traj)
    >>> vacf.save('./eval/db.h5')
    >>> msd.save('./eval/db.h5', database=True)
    >>> db = amep.load.evaluation('./eval/db.h5', database=True)
    >>> print(db.keys())
    ['msd', 'vacf']
    >>> 
    
    
    Load only a specific observable from the HDF5 file (here: the MSD):

    >>> msd = amep.load.evaluation('./eval/db.h5', key='msd')
    >>> print(msd.keys())
    ['frames', 'indices', 'times', 'avg', 'name', 'ptype']
    >>> 
    
    """
    if os.path.exists(path):
        # check if correct type
        with h5py.File(path, 'r') as root:
            if 'type' in root.attrs.keys():
                if root.attrs['type'] != 'database':
                    raise RuntimeError(
                        f"""The file {path} has the wrong format."""
                    )
                else:
                    # create database object
                    db = BaseDatabase(path)
                    if database:
                        return db
                    elif key is None and len(db.keys()) == 1:
                        return BaseEvalData(path, group = db.keys()[0])
                    elif key in db.keys():
                        return BaseEvalData(path, group = key)
                    else:
                        raise RuntimeError(
                            f"The given file {path} contains multiple "\
                            f"evaluations or does not contain the key {key} "\
                            "you supplied. To load this file, use "\
                            "database=True or supply a valid key. Valid keys "\
                            f"are {db.keys()}."
                        )
            else:
                if verbose:
                    _log.info(
                        "It seems you are loading an HDF5 file which has "\
                        "either been created with an older AMEP version or "\
                        "which has the wrong format. It is not guaranteed "\
                        "that loading data from this file is working "\
                        "correctly."    
                    )
                return BaseEvalData(path)
    else:
        raise FileNotFoundError(
            f"""The given file {path} does not exist."""
        )
