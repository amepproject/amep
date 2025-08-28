
from abc import ABC, abstractmethod
from dataclasses import dataclass, field 
import numpy as np
from pathlib import Path

@dataclass
class StorageConfig:
    """Configuration for HDF5 storage settings."""
    COMPRESSION: str = 'lzf'
    SHUFFLE: bool = True
    FLETCHER: bool = True
    DTYPE: type = np.float32
    #max_memory_gb: int = 1
    TRAJFILENAME = 'traj.h5amep'
    ROOTGROUPS  = ['params','scripts','info','frames', 'amep']
  
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


# maximum RAM usage in GB per CPU (used for parallelized methods)
MAXMEM = 1

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


