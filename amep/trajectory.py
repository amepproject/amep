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
Trajectories
============

.. module:: amep.trajectory

The AMEP module :mod:`amep.trajectory` contains all trajectory classes. These 
are containers for a time series of simulation data (particle-based data or 
continuum fields). The following classes are included:

    ParticleTrajetory :
        Time-series of per-particle data such as coordinates, velocities,
        forces, etc.
    FieldTrajectory :
        Time series of a continuum field stored as values on a regular grid.

"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import h5py
import os

import numpy as np

from tqdm.autonotebook import tqdm

from .base import BaseTrajectory

# =============================================================================
# PARTICLE TRAJECTORY
# =============================================================================
class ParticleTrajectory(BaseTrajectory):
    """
    Particle trajectory object. Time-series of per-particle data such as
    coordinates, velocities, forces, etc.
    """
    def __init__(self, reader) -> None:
        r"""
        Creates a trajectory object containing data frames for multiple
        time steps.

        Can be viewed as a Sequence of :class:`.BaseFrame`.

        Parameters
        ----------
        reader : BaseReader
            Reader of the data.

        Returns
        -------
        None.

        """
        super().__init__(reader)
        
        self.__nojump = False

        # add 'particles' group for particle information/properties
        with h5py.File(os.path.join(self.reader.savedir, self.reader.filename), 'a') as root:

            if 'particles' not in root.keys():
                root.create_group('particles')
                
            if 'njx' in self[0].keys:
                self.__nojump = True

    def __get_jumps(self) -> np.ndarray:
        r"""
        Determines when a particle jumps from one side of the simulation box
        to the other due to periodic boundary conditions.
        
        Note
        ----
        Note that this method only gives correct results if the particles
        do not move further than half the box size between two frames.

        Returns
        -------
        jump_data : np.ndarray
            Array containing how much box dimensions each
            particle jumped in each direction at each time step.

        """
        # tuple (for each spatial dimension) of arrays to store the jump data
        jump_data = []

        # loop through all frames
        prev = self[0]
        for i, curr in enumerate(self):

            # get box lenth in each direction
            box = np.diff(curr.box).T[0]

            # determine whether the displacement is larger than the box length
            delta = ((curr.coords() - prev.coords()) / box).round().astype(np.int8)
            prev = curr

            jump_data.append(delta)

        return np.array(jump_data)

    def nojump(self, new: bool = False) -> None:
        r"""
        Generates the nojump coordinates for all particles and all frames
        based on the jump data determined from self.__get_jumps().

        Note
        ----
        Note that this method only gives correct results if the particles
        do not move further than half the box size between two frames.

        Parameters
        ----------
        new : bool, optional
            If True, the nojump coordinates are calculated again and
            existing nojump coordinates are overwritten. If False, nojump
            checks whether nojump coordinates already exist and will only
            calculate them if no nojump coordinates are available. The
            default is False.

        Returns
        -------
        None.

        """
        if self.__nojump == False or new == True:

            # get jump data
            jump_data = self.__get_jumps()

            for i, frame in enumerate(tqdm(self)):

                # box dimensions
                box = np.diff(frame.box).T[0]

                # determine how much the coordinates must be shifted
                delta = np.sum(jump_data[:i+1], axis=0)*box

                # get nojump coordinates
                coords = frame.coords() - delta

                # store nojump coordinates in data frame
                frame.add_data('njcoords', coords)
            
            self.__nojump = True

    def add_particle_info(
            self, ptype: int, key: str, value: int | float | str) -> None:
        r'''
        Adds information for the given particle type to the
        particles group of the trajectory file.

        Parameters
        ----------
        ptype : int
            Particle type.
        key : str
            Name of the parameter.
        value : int/float/str
            Value of the parameter.

        Returns
        -------
        None.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> traj.add_particle_info(1, 'name', 'active')
        >>> traj.add_particle_info(2, 'name', 'passive')
        >>> print(traj.get_particle_info())
        {1: {'name': 'active'}, 2: {'name': 'passive'}}
        >>> traj.delete_particle_info(None)
        >>>

        '''
        if not isinstance(ptype, int):
            raise TypeError("Trajectory.add_particle_info(): The particle type <ptype> must be an integer.")
        with h5py.File(os.path.join(self.reader.savedir, self.reader.filename), 'a') as root:
            if str(ptype) not in root['particles'].keys():
                root['particles'].create_group(str(ptype))
            root['particles'][str(ptype)].attrs[key] = value

    def get_particle_info(self, ptype: int | None = None) -> dict:
        r'''
        Returns all parameters of the given particle type
        as a dictionary.

        Parameters
        ----------
        ptype : int or None, optional
            Particle type. If None, the information of all particle types is
            returned. The default is None.

        Returns
        -------
        p : dict
            Parameters.

        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> traj.add_particle_info(1, 'name', 'active')
        >>> traj.add_particle_info(2, 'name', 'passive')
        >>> print(traj.get_particle_info())
        {1: {'name': 'active'}, 2: {'name': 'passive'}}
        >>> traj.delete_particle_info(None)
        >>>

        '''
        if not isinstance(ptype, int) and ptype is not None:
            raise TypeError("Trajectory.get_particle_info(): The particle type <ptype> must be an integer.")
        with h5py.File(os.path.join(self.reader.savedir, self.reader.filename), 'r') as root:
            if ptype is None:
                p = {}
                for t in list(root['particles'].keys()):
                    p[int(t)] = dict(a for a in root['particles'][t].attrs.items())
            elif str(ptype) in list(root['particles'].keys()):
                p = dict(a for a in root['particles'][str(ptype)].attrs.items())
            else:
                p = None
                raise KeyError(
                    f'''Invalid particle type ptype={ptype}. Available types
                    are {list(root['particles'].keys())}.'''
                )
        return p

    def delete_particle_info(
            self, ptype: int | None, key: str | None = None) -> None:
        r'''
        Deletes specific particle information.

        Parameters
        ----------
        ptype : int or None
            Particle type. If None, the information of all particles is
            deleted.
        key : str, optional
            Parameter to delete. If None, all parameters of the given particle
            type are deleted. The default is None.

        Returns
        -------
        None.
        
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/lammps.h5amep")
        >>> traj.add_particle_info(1, 'name', 'active')
        >>> traj.add_particle_info(2, 'name', 'passive')
        >>> print(traj.get_particle_info())
        {1: {'name': 'active'}, 2: {'name': 'passive'}}
        >>> traj.delete_particle_info(None)
        >>> print(traj.get_particle_info())
        {}
        >>> 
        
        '''
        with h5py.File(os.path.join(self.reader.savedir, self.reader.filename), 'a') as root:
            if ptype is None:
                for p in root['particles'].keys():
                    if key is None:
                        del root['particles'][str(p)]
                    elif isinstance(key, str):
                        root['particles'][str(p)].attrs.__delitem__(key)
            elif isinstance(ptype, int):    
                if key is None:
                    del root['particles'][str(ptype)]
                elif type(key)==str:
                    root['particles'][str(ptype)].attrs.__delitem__(key)
                    
    def animate(self, filename: str, **kwargs) -> None:
        r"""
        Wrapper to `amep.plot.animate_trajectory`.

        Parameters
        ----------
        filename : str
            File in which the video should be saved.
        **kwargs
            All keyword arguments are forwarded to
            `amep.plot.animate_trajectory`.

        Returns
        -------
        None

        """
        from .plot import animate_trajectory
        animate_trajectory(self, filename, **kwargs)


# =============================================================================
# FIELD TRAJECTORY
# =============================================================================
class FieldTrajectory(BaseTrajectory):
    '''
    Field trajectory object.Time-series of per spatial point field values.
    '''

    def __init__(self, reader):
        r'''
        Creates a trajectory object containing data frames for multiple
        time steps.

        Can be viewed as a Sequence of :class:`.BaseField`.

        Parameters
        ----------
        reader : BaseReader
            Reader of the data.

        Returns
        -------
        None.

        '''
        super().__init__(reader)

        # add 'fields' group for field information/properties
        with h5py.File(os.path.join(self.reader.savedir, self.reader.filename), 'a') as root:
            if 'fields' not in root.keys():
                root.create_group('fields')

    def add_field_info(
            self, ftype: str, key: str, value: int | float | str) -> None:
        r'''
        Adds information for the given field to the fields group of the
        trajectory file.

        Parameters
        ----------
        ftype : str
            Field.
        key : str
            Name of the parameter.
        value : int/float/str
            Value of the parameter.

        Returns
        -------
        None.
        
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
        >>> traj.add_field_info('p', 'name', 'particle density')
        >>> traj.add_field_info('c', 'name', 'chemical concentration')
        >>> print(traj.get_field_info())
        {'c': {'name': 'chemical concentration'}, 'p': {'name': 'particle density'}}
        >>> traj.delete_field_info(None)
        >>> 

        '''
        with h5py.File(os.path.join(self.reader.savedir, self.reader.filename), 'a') as root:
            if ftype not in root['fields'].keys():
                root['fields'].create_group(ftype)
            root['fields'][ftype].attrs[key] = value

    def get_field_info(self, ftype: str | None = None) -> dict:
        r'''
        Returns all parameters of the given field as a dictionary.

        Parameters
        ----------
        ftype : str or None, optional
            Field. If None, the information of all fields is returned.
            The default is None.

        Returns
        -------
        p : dict
            Parameters.
            
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
        >>> traj.add_field_info('p', 'name', 'particle density')
        >>> traj.add_field_info('c', 'name', 'chemical concentration')
        >>> print(traj.get_field_info())
        {'c': {'name': 'chemical concentration'}, 'p': {'name': 'particle density'}}
        >>> traj.delete_field_info(None)
        >>> 

        '''
        with h5py.File(os.path.join(self.reader.savedir, self.reader.filename), 'r') as root:
            if ftype is None:
                p = {}
                for f in list(root['fields'].keys()):
                    p[f] = dict(a for a in root['fields'][f].attrs.items())
            elif ftype in list(root['fields'].keys()):
                p = dict(a for a in root['fields'][ftype].attrs.items())
            else:
                p = None
                raise KeyError(
                    f'''Invalid field ftype={ftype}. Available fields
                    are {list(root['fields'].keys())}.'''
                )
        return p

    def delete_field_info(
            self, ftype: str | None, key: str | None = None) -> None:
        r'''
        Deletes specific field information.

        Parameters
        ----------
        ftype : str or None
            Field. If None, the information of all fields is deleted.
        key : str, optional
            Parameter to delete. If None, all parameters of the given field
            are deleted. The default is None.

        Returns
        -------
        None.
        
        Examples
        --------
        >>> import amep
        >>> traj = amep.load.traj("../examples/data/continuum.h5amep")
        >>> traj.add_field_info('p', 'name', 'particle density')
        >>> traj.add_field_info('c', 'name', 'chemical concentration')
        >>> print(traj.get_field_info())
        {'c': {'name': 'chemical concentration'}, 'p': {'name': 'particle density'}}
        >>> traj.delete_field_info(None)
        >>> print(traj.get_field_info())
        {}
        >>> 

        '''
        with h5py.File(os.path.join(self.reader.savedir, self.reader.filename), 'a') as root:
            if ftype is None:
                for f in root['fields'].keys():
                    if key is None:
                        del root['fields'][f]
                    elif isinstance(key, str):
                        root['fields'][f].attrs.__delitem__(key)
            elif isinstance(ftype, str):    
                if key is None:
                    del root['fields'][ftype]
                elif isinstance(key, str):
                    root['fields'][ftype].attrs.__delitem__(key)
                    
    def animate(self, filename: str, **kwargs) -> None:
        r"""
        Wrapper to `amep.plot.animate_trajectory`.

        Parameters
        ----------
        filename : str
            File in which the video should be saved.
        **kwargs
            All keyword arguments are forwarded to
            `amep.plot.animate_trajectory`.

        Returns
        -------
        None

        """
        from .plot import animate_trajectory
        animate_trajectory(self, filename, **kwargs)
