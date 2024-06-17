# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) 2024 Lukas Hecht and the AMEP development team.
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
Test units of the amep.base module.
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
from pathlib import Path
import unittest
import os
import numpy as np
import amep

# =============================================================================
# GLOBAL CONFIG.
# =============================================================================
DATADIR = Path('../examples/data/')
FIELDDIR = DATADIR/"continuum"
RNG = np.random.default_rng(1234)

# for field data creation
BOX = np.array([[0.5, 49.5], [0.5, 49.5], [-0., 0.]])
SHAPE = np.array([50, 50])
X, Y = np.meshgrid(
        np.linspace(BOX[0, 0], BOX[0, 1], SHAPE[0]),
        np.linspace(BOX[1, 0], BOX[1, 1], SHAPE[1]),
        indexing='ij'
)
COORDS = [X.flatten(), Y.flatten()]
TIMESTEPS = np.arange(0, 10001, 1000)
DT = 1.0
FIELDS = ['p', 'c']

# create directories if they do not exist
if not os.path.isdir('./data'):
    os.mkdir('./data')
if not os.path.isdir(FIELDDIR):
    os.mkdir(FIELDDIR)

# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


# =============================================================================
# BASEFIELD TESTS
# =============================================================================
class TestBaseField(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        # generate some test field data
        # create_field_data()

        # load the data
        self.traj = amep.load.traj(
            FIELDDIR,
            mode='field',
            delimiter=' ',
            dumps='field_*.txt',
            reload=True,
            timestep=DT
        )

        # get frame
        self.frame = self.traj[0]


    def test_data(self):
        # check shape and data access
        for key in FIELDS:
            self.assertEqual(
                self.frame.data(key).shape,
                tuple(SHAPE),
                f'''Data {key} has wrong shape {self.frame.data(key).shape}.
                    Require shape {SHAPE}.'''
            )
        data = self.frame.data(FIELDS[0], FIELDS[1])
        self.assertEqual(
            data.shape, 
            (2,) + tuple(SHAPE),
            f'''Loaded data with keys {[FIELDS[0], FIELDS[1]]}
                has invalid shape {data.shape}. Require shape {(3,) + SHAPE}'''
        )

    def test_step(self):
        # step
        for i in range(self.traj.nframes):
            self.assertEqual(
                self.traj[i].step,
                TIMESTEPS[i],
                f'''Incorrect time step: got {self.traj[i].step} instead of
                    {TIMESTEPS[i]} at index {i}.'''
            )
    def test_center(self):
        # center
        center = (BOX[:,0]+BOX[:,1]) / 2.0
        comparison = self.frame.center == center
        self.assertTrue(
            comparison.all(),
            f'''Incorrect center: got {self.frame.center} instead of
                {center}.'''
        )
    def test_dim(self):
        # dim
        self.assertEqual(
            self.frame.dim,
            2,
            f'''Incorrect spatial dimension: got {self.frame.dim}
                instead of 3.'''
        )
    def test_box(self):
        # box
        comparison = self.frame.box == BOX
        self.assertTrue(
            comparison.all(),
            f'''Incorrect simulation box: got {self.frame.box} instead
                of {BOX}.'''
        )

    def test_volume(self):
        # volume
        volume = np.prod(np.diff(BOX[:2]))
        comparison = self.frame.volume == volume
        self.assertTrue(
            comparison.all(),
            f'''Incorrect simulation box volume: got {self.frame.volume}
                instead of {volume} with simulation box {BOX}.'''
        )
    def test_grid(self):
        # grid
        self.assertEqual(
            len(self.frame.grid),
            2,
            f'''Incorrect grid dimension: got {len(self.frame.grid)}
                instead of {len(COORDS)}.'''
        )
        self.assertEqual(
            self.frame.grid[0].shape,
            X.shape,
            f'''Incorrect grid shape: got {self.frame.grid[0].shape}
                instead of {X.shape}.'''
        )
        comparisonx = np.isclose(self.frame.grid[0], X)
        comparisony = np.isclose(self.frame.grid[1], Y)
        self.assertTrue(
            comparisonx.all(),
            f'''Wrong x coordinates. Found {len(X[comparisonx==0])}
                invalid entries out of {len(X.flatten())}.'''
        )
        self.assertTrue(
            comparisony.all(),
            f'''Wrong y coordinates. Found {len(Y[comparisony==0])}
                invalid entries out of {len(Y.flatten())}.'''
        )
        self.assertTrue(
            comparisonz.all(),
            f'''Wrong z coordinates. Found {len(Z[comparisonz==0])}
                invalid entries out of {len(Z.flatten())}.'''
        )
    def test_keys(self):
        # keys
        keys = self.frame.keys
        for key in keys:
            self.assertTrue(
                key in FIELDS,
                f'''Incorrect key {key}. Allowed keys are {FIELDS}.'''
            )


# =============================================================================
# BASETRAJECTORY TESTS
# =============================================================================
class TestBaseTrajectory(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        # generate some test field data
        # create_field_data()
        
        # create a trajectory object
        self.traj = amep.load.traj(
            FIELDDIR,
            mode='field',
            delimiter=' ',
            dumps='field_*.txt',
            reload=True,
            timestep=DT
        )
        
        # add author information
        self.traj.add_author_info(
            'Author A', 'email', 'A@beispiel.de')
        self.traj.add_author_info(
            'Author A', 'affiliation', 'Institute A')
        self.traj.add_author_info(
            'Author B', 'email', 'B@beispiel.de')
        self.traj.add_author_info(
            'Author B', 'affiliation', 'Institute B')
        
        # add software information
        self.traj.add_software_info('version', '4May2023')
        self.traj.add_software_info('web', 'lammps.org')
        self.traj.add_software_info('name', 'LAMMPS')

    def test_get_item(self):
        self.assertTrue(
            isinstance(self.traj[0], amep.base.BaseField),
            f'''traj[0] returns an incorrect item type. Got
            {type(self.traj[0])} instead of amep.base.BaseField.'''
        )
        self.assertTrue(
            isinstance(self.traj[2:], list),
            f'''traj[2:] returns an incorrect type. Got {type(self.traj[2:])}
            instead of list.'''
        )
        self.assertEqual(
            len(self.traj[2:]),
            len(TIMESTEPS)-2,
            f'''traj[2:] has incorrect length. Got {len(self.traj[2:])}
            instead of {len(TIMESTEPS)-2}.'''    
        )
        self.assertTrue(
            isinstance(self.traj[[0,2]], list),
            f'''traj[[0,2]] returns an incorrect type. Got {type(self.traj[[0,2]])}
            instead of list.'''
        )
        self.assertEqual(
            len(self.traj[[0,2]]),
            2,
            f'''traj[[0,2]] has incorrect length. Got {len(self.traj[[0,2]])}
            instead of 2.'''    
        )
        for frame,i in zip(self.traj[[0,2]], [0,2]):
            self.assertEqual(
                frame.step,
                TIMESTEPS[i],
                f'''Frame has wrong time step. Got {frame.step} instead of
                {TIMESTEPS[i]}.'''
            )

    def test_add_author_info(self):
        # add info
        self.traj.add_author_info('Author A', 'test', 'test')
        
        # check if info has been stored
        self.assertIn(
            'test',
            self.traj.get_author_info('Author A').keys(),
            f'''New author information is not stored in the trajectory object.
            The new key "test" is not in the stored information given by
            {self.traj.get_author_info('Author A').keys()}.'''
        )
        
        # delete info
        self.traj.delete_author_info('Author A', 'test')

    def test_get_author_info(self):
        self.assertTrue(
            isinstance(self.traj.get_author_info('Author A'), dict),
            f'''Wrong type returned. Got
            {type(self.traj.get_author_info('Author A'))} instead of dict.'''
        )
        self.assertIn(
            'email',
            self.traj.get_author_info('Author A'),
            f'''Missing key "email". Stored keys are
            {self.traj.get_author_info('Author A').keys()}.'''
        )
        self.assertIn(
            'affiliation',
            self.traj.get_author_info('Author A'),
            f'''Missing key "affiliation". Stored keys are
            {self.traj.get_author_info('Author A').keys()}.'''
        )

    def test_delete_author_info(self):
        # delete author information
        self.traj.delete_author_info('Author B', 'email')
        # check if it has been deleted correctly
        self.assertNotIn(
            'email',
            self.traj.get_author_info('Author B').keys(),
            '''Author info "email" has not been deleted.'''
        )
        # add the information again
        self.traj.add_author_info('Author B', 'email', 'B@beispiel.de')

    def test_authors(self):
        self.assertTrue(
            isinstance(self.traj.authors, list),
            f'''Invalid type. Got {type(self.traj.authors)} instead of list'''
        )
        self.assertEqual(
            len(self.traj.authors),
            2,
            f'''Incorrect len. Got {len(self.traj.authors)} instead of 2.
            Returned authors are {self.traj.authors}.'''
        )
        self.assertIn(
            'Author A',
            self.traj.authors,
            f'''Author "Author A" not in {self.traj.authors}.'''
        )
        self.assertIn(
            'Author B',
            self.traj.authors,
            f'''Author "Author B" not in {self.traj.authors}.'''
        )

    def test_add_software_info(self):
        # add info
        self.traj.add_software_info('test', 'v102')
        # check if info has been added correctly
        self.assertIn(
            'test',
            self.traj.software,
            f'''Software information has not been added correctly. "test" is
            not in {self.traj.software}.'''
        )
        self.assertEqual(
            self.traj.software['test'],
            'v102',
            f'''Software information has not been added correctly. Got
            {self.traj.software['test']} instead of "v102".'''
        )
        # delete software info
        self.traj.delete_software_info('test')

    def test_delete_software_info(self):
        # delete software information
        self.traj.delete_software_info('version')
        # check if it has been deleted correctly
        self.assertNotIn(
            'version',
            self.traj.software,
            f'''Information "version" has not been deleted. Got
            {self.traj.software}.'''
        )
        # add deleted information again
        self.traj.add_software_info('version', '4May2023')

    def test_software(self):
        self.assertTrue(
            isinstance(self.traj.software, dict),
            f'''Invalid type. Got {type(self.traj.software)} instead of dict'''
        )
        self.assertIn(
            'name',
            self.traj.software,
            f'''"name" is not in the software info dictionary. Got
            {self.traj.software}.'''
        )
        self.assertEqual(
            self.traj.software['name'],
            'LAMMPS',
            f'''Got wrong software name {self.traj.software['name']} instead of
            "LAMMPS".'''
        )

    def test_info(self):
        self.assertTrue(
            isinstance(self.traj.info, dict),
            f'''Invalid type. Got {type(self.traj.info)} instead of dict.'''
        )
        self.assertIn(
            'software',
            self.traj.info,
            f'''The key "software" is missing. Got {self.traj.info}.'''
        )
        self.assertIn(
            'authors',
            self.traj.info,
            f'''The key "authors" is missing. Got {self.traj.info}.'''
        )
        self.assertIn(
            'params',
            self.traj.info,
            f'''The key "params" is missing. Got {self.traj.info}.'''
        )

    def test_params(self):
        self.assertTrue(
            isinstance(self.traj.params, dict),
            f'''Invalid type. Got {type(self.traj.params)} instead of dict.'''
        )
        required_params = ['d', 'dt', 'nth', 'start', 'stop']
        for p in required_params:
            self.assertIn(
                p,
                self.traj.params,
                f'''Missing parameter "{p}". Got {self.traj.params}.'''
            )

    def test_add_param(self):
        # add parameter
        self.traj.add_param('A', 1)
        # check if parameter has been added correctly
        self.assertIn(
            'A',
            self.traj.params,
            f'''Parameter "A" has not been added. Got {self.traj.params}.'''
        )
        self.assertEqual(
            self.traj.params['A'],
            1,
            f'''Incorrect parameter value. Got {self.traj.params['A']} instead
            of 1.'''
        )
        # delete parameter
        self.traj.delete_param('A')

    def test_delete_param(self):
        # delete parameter
        self.traj.delete_param('dt')
        # check if parameter has been deleted correctly
        self.assertNotIn(
            'dt',
            self.traj.params,
            f'''Parameter "dt" has not been deleted. Got {self.traj.params}.'''
        )
        # add pararmeter again
        self.traj.add_param('dt', 1.0)

    def test_add_script(self):
        # add script
        self.traj.add_script(os.path.join(FIELDDIR, "solver.py"))
        # check if script has been added correctly
        self.assertIn(
            'solver.py',
            self.traj.scripts,
            f'''The script "solver.py" has not been added to the
            trajectory. Included files are {self.traj.scripts}.'''
        )
        # delete script again
        self.traj.delete_script('solver.py')

    def test_get_script(self):
        # add script
        self.traj.add_script(os.path.join(FIELDDIR, "solver.py"))
        # clean data directory
        if os.path.exists(os.path.join(DATADIR, "solver.py")):
            os.remove(os.path.join(DATADIR, "solver.py"))
        # test get_script method
        self.assertTrue(
            isinstance(self.traj.get_script('solver.py'), list),
            f'''Invalid type. Got {type(self.traj.get_script('solver.py'))}
            instead of list.'''
        )
        self.assertEqual(
            len(self.traj.get_script('solver.py')), 
            199,
            f'''Invalid length. Got
            {len(self.traj.get_script('solver.py'))} instead of 5.'''
        )
        # write script to file
        _ = self.traj.get_script(
            'solver.py',
            store = True, 
            directory = DATADIR
        )
        self.assertTrue(
            os.path.exists(os.path.join(DATADIR, "solver.py")),
            '''Script "solver.py" has not been stored.'''
        )
        # delete script again
        self.traj.delete_script('solver.py')
        # clean data directory again
        if os.path.exists(os.path.join(DATADIR, "solver.py")):
            os.remove(os.path.join(DATADIR, "solver.py"))

    def test_delete_script(self):
        # add script
        self.traj.add_script(os.path.join(FIELDDIR, "solver.py"))
        # delete script again
        self.traj.delete_script('solver.py')
        # check if script has been deleted correctly
        self.assertNotIn(
            'solver.py',
            self.traj.scripts,
            f'''Script "solver.py" has not been deleted. Available scripts
            are {self.traj.scripts}.'''
        )

    def test_scripts(self):
        # add script
        self.traj.add_script(os.path.join(FIELDDIR, "solver.py"))
        self.assertEqual(
            len(self.traj.scripts),
            1,
            f'''Invalid length. Got {len(self.traj.scripts)} instead of 1.'''
        )
        self.assertIn(
            'solver.py',
            self.traj.scripts,
            f'''Missing file "solver.py". Included files are
            {self.traj.scripts}.'''
        )

    def test_properties(self):
        self.assertEqual(
            self.traj.version, amep.__version__,
            f'''Invalid version. Got {self.traj.version} instead of
            {amep.__version__}.'''
        )
        self.assertEqual(
            self.traj.type,
            'field',
            f'''Invalid . Got {self.traj.type} instead of fields.'''
        )
        self.assertTrue(
            isinstance(self.traj.reader, amep.reader.ContinuumReader),
            f'''Invalid type. Got {type(self.traj.reader)} instead of 
            amep.reader.ContinuumReader.'''
        )
        self.assertEqual(
            self.traj.start,
            0.0,
            f'''Invalid start value. Got {self.traj.start} instead of 0.0.'''
        )
        self.assertEqual(
            self.traj.stop,
            1.0,
            f'''Invalid stop value. Got {self.traj.stop} instead of 1.0.'''
        )
        self.assertEqual(
            self.traj.nth,
            1,
            f'''Invalid nth value. Got {self.traj.nth} instead of 1.'''
        )
        self.assertEqual(
            self.traj.nframes,
            len(TIMESTEPS),
            f'''Invalid number of frames. Got {self.traj.nframes} instead of 
            {len(TIMESTEPS)}.'''
        )
        # check steps
        invalid_steps = False
        for step in self.traj.steps:
            if step not in TIMESTEPS:
                invalid_steps = True
        self.assertFalse(
            invalid_steps,
            f'''Invalid step detected. Got {self.traj.steps} instead of
            {TIMESTEPS}.'''
        )
        self.assertEqual(
            len(self.traj.steps),
            len(TIMESTEPS),
            f'''Invalid length of steps. Got {len(self.traj.steps)} instead of 
            {len(TIMESTEPS)}.'''
        )
        # check times
        invalid_times = False
        for time in self.traj.times:
            if time not in np.array(TIMESTEPS)*DT:
                invalid_times = True
        self.assertFalse(
            invalid_times,
            f'''Invalid time detected. Got {self.traj.times} instead of 
            {np.array(TIMESTEPS)*DT}.'''
        )
        self.assertEqual(
            len(self.traj.times),
            len(TIMESTEPS),
            f'''Invalid length of times. Got {len(self.traj.times)} instead of
            {len(TIMESTEPS)}.'''
        )
        self.assertEqual(
            self.traj.dt,
            DT,
            f'''Invalid time step. Got {self.traj.dt} instead of {DT}.'''
        )
        self.assertEqual(
            self.traj.dim,
            2,
            f'''Invalid dimension. Got {self.traj.dim} instead of 3.'''
        )
        self.assertEqual(
            self.traj.savedir,
            os.path.abspath(FIELDDIR),
            f'''Invalid savedir. Got {self.traj.savedir} instead of
            {FIELDDIR}.'''
        )


# =============================================================================
# BASEFUNCTION TESTS
# =============================================================================
class TestBaseFunction(unittest.TestCase):
    """Test generation and base utilities of the Function object."""

    @classmethod
    def setUpClass(cls):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        # create test data
        cls.x = np.linspace(0, 10, 100)
        cls.y = 10*cls.x + 5

    def test_init(self):
        # initialize function object
        function = amep.base.BaseFunction(2)
        # check number of parameters
        self.assertEqual(
            function.nparams,
            2,
            f"""Invalid number of parameters. Got {function.nparams}
            instead of 2."""
        )
        # check length of key and value lists
        self.assertEqual(
            len(function.keys),
            2,
            f"""Invalid number of keys. Got {len(function.keys)}
            instead of 2."""
        )
        self.assertEqual(
            len(function.errors),
            2,
            f"""Invalid length of fit error list. Got
            {len(function.errors)} instead of 2."""
        )
        self.assertEqual(
            len(function.params),
            2,
            f"""Invalid length of fit results list. Got
            {len(function.params)} instead of 2."""
        )

    def test_fit(self):
        # initialize function object
        function = amep.base.BaseFunction(2)
        # fit the test data
        function.fit(self.x, self.y)
        # check fit parameters
        self.assertTrue(
            np.isclose(function.params[0], 10.0),
            f"""Invalid parameter {function.keys[0]}. Got {function.params[0]}
            instead of 10.0."""
        )
        self.assertTrue(
            np.isclose(function.params[1], 5.0),
            f"""Invalid parameter {function.keys[1]}. Got {function.params[1]}
            instead of 5.0."""
        )

    def test_generate(self):
        # initialize function object
        function = amep.base.BaseFunction(2)
        # fit the test data
        function.fit(self.x, self.y)
        # generate y data
        y = function.generate(self.x)
        self.assertTrue(
            np.all(np.isclose(y, self.y)),
            """The generate method returned wrong data."""
        )

    def test_properties(self):
        # initialize function object
        function = amep.base.BaseFunction(2)
        # fit the test data
        function.fit(self.x, self.y)
        # check data type
        self.assertTrue(
            isinstance(function.keys, list),
            f"""Invalid type for property keys. Got {type(function.keys)}
            instead of list."""
        )
        self.assertTrue(
            isinstance(function.errors, np.ndarray),
            f"""Invalid type for property errors. Got {type(function.errors)}
            instead of np.ndarray."""
        )
        self.assertTrue(
            isinstance(function.params, np.ndarray),
            f"""Invalid type for property params. Got {type(function.params)}
            instead of np.ndarray."""
        )
        self.assertTrue(
            isinstance(function.results, dict),
            f"""Invalid type for property results. Got {type(function.results)}
            instead of dict."""
        )


# if __name__ == '__main__':
#     unittest.main()
