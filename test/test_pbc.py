# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) 2023-2024 Lukas Hecht and the AMEP development team.
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
Test units for the amep.pbc module.
"""
# =============================================================================
# IMPORT MODULES
# =============================================================================
import unittest
import numpy as np
import amep


# =============================================================================
# MAIN PBC TEST
# =============================================================================
class TestPbc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Basic setup. Generating test data.

        Returns
        -------
        None.

        """
        cls.coords = np.array([
            [2.0, 2.0, 0.0],
            [-1.0, 1.0, 0.0],
            [2.5, 3.0, 0.0],
            [-6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.7, 0.7, 0.0]
        ])
        cls.box = np.array([[-5, 5], [-5, 5], [-0.5, 0.5]])

    def test_fold(self):
        """
        Tests the `fold` method of the AMEP module `amep.pbc`.

        Returns
        -------
        None.

        """
        # generate test data
        unwrapped_coords = np.array([
            [2.0, 2.0, 0.75],
            [-1.0, 17.0, 0.0],
            [2.5, 3.0, 0.0],
            [-6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [7.0, 7.0, 0.0]
        ])
        box = np.array([[-5, 5], [-5, 5], [-0.5, 0.5]])
        # expected folded coords
        coords = np.array([
            [2.0, 2.0, -0.25],
            [-1.0, -3.0, 0.0],
            [2.5, 3.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
            [-3.0, -3.0, 0.0]
        ])
        # fold unwrapped coords back into the box
        folded = amep.pbc.fold(unwrapped_coords, box)
        # comparison
        compare = folded.round(3) == coords.round(3)
        # check result
        self.assertTrue(
            compare.all(),
            f'''folded coords do not match expected coords:\n
                unwrapped={unwrapped_coords},\n
                folded={folded},\n
                expected={coords},\n
                box={box}.'''
        )

    def test_pbc_points(self):
        # generate test data
        coords2d = np.array([
            [-2.5, 3.5, 0.0]
        ])
        coords2d2 = np.array([
            [-2.5, 3.5, 0.01],
            [-3.5, 1.5, 0.01]
        ])
        coords3d = np.array([
            [-2.5, 3.5, 0.25]
        ])
        box = np.array([[-5, 5], [-5, 5], [-0.5, 0.5]])
        # expected periodic image coordinates
        pbc_coords2d = np.array([
            [-2.5, 3.5, 0.0],
            [-12.5, -6.5, 0.0],
            [-12.5, 3.5, 0.0],
            [-12.5, 13.5, 0.0],
            [-2.5, -6.5, 0.0],
            [-2.5, 13.5, 0.0],
            [7.5, -6.5, 0.0],
            [7.5, 3.5, 0.0],
            [7.5, 13.5, 0.0]
        ])
        pbc_coords3d = np.array([
            [-2.5, 3.5, 0.25],
            [-12.5, -6.5, -0.75],
            [-12.5, -6.5, 0.25],
            [-12.5, -6.5, 1.25],
            [-12.5, 3.5, -0.75],
            [-12.5, 3.5, 0.25],
            [-12.5, 3.5, 1.25],
            [-12.5, 13.5, -0.75],
            [-12.5, 13.5, 0.25],
            [-12.5, 13.5, 1.25],
            [-2.5, -6.5, -0.75],
            [-2.5, -6.5, 0.25],
            [-2.5, -6.5, 1.25],
            [-2.5, 3.5, -0.75],
            [-2.5, 3.5, 1.25],
            [-2.5, 13.5, -0.75],
            [-2.5, 13.5, 0.25],
            [-2.5, 13.5, 1.25],
            [7.5, -6.5, -0.75],
            [7.5, -6.5, 0.25],
            [7.5, -6.5, 1.25],
            [7.5, 3.5, -0.75],
            [7.5, 3.5, 0.25],
            [7.5, 3.5, 1.25],
            [7.5, 13.5, -0.75],
            [7.5, 13.5, 0.25],
            [7.5, 13.5, 1.25]
        ])
        # comparison
        compare2d = amep.pbc.pbc_points(coords2d, box, enforce_nd=2) == pbc_coords2d
        compare3d = amep.pbc.pbc_points(coords3d, box, enforce_nd=3) == pbc_coords3d
        # check 2d data indentification
        shape = amep.pbc.pbc_points(coords2d, box, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (9, 3),
            f'''2d detection failed: Got shape {shape} instead of (9,3).
                2d data is not identified as such.'''
        )
        shape = amep.pbc.pbc_points(coords2d2, box).shape
        self.assertEqual(
            shape,
            (18, 3),
            f'''2d detection failed: Got shape {shape} instead of (9,3).
                2d data is not identified as such.'''
        )
        shape = amep.pbc.pbc_points(coords2d, box).shape
        self.assertEqual(
            shape,
            (27, 3),
            f'''2d detection failed: Got shape {shape} instead of (27,3).
                3d data is not identified as such.'''
        )
        shape = amep.pbc.pbc_points(coords3d, box).shape
        self.assertEqual(
            shape,
            (27, 3),
            f'''2d detection failed: Got shape {shape} instead of (27,3).
                3d data is not identified as such.'''
        )
        shape = amep.pbc.pbc_points(coords3d, box, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (9, 3),
            f'''2d detection failed: Got shape {shape} instead of (9,3).
                3d data is not identified as such.'''
        )
        # check enforce_nd keyword
        shape = amep.pbc.pbc_points(coords3d, box, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (9, 3),
            f'enforce_twod=True failure. Expected shape (9,3). Got {shape}.'
        )
        # check results
        self.assertTrue(
            compare2d.all(),
            f'''incorrect pbc coords for 2d data:\n
                coords={coords2d},\n
                expected={pbc_coords2d},\n
                result={amep.pbc.pbc_points(coords2d, box)},\n
                comparison={compare2d},\n
                box={box}.'''
        )
        self.assertTrue(
            compare3d.all(),
            f'''incorrect pbc coords for 3d data:\n
                coords={coords3d},\n
                expected={pbc_coords3d},\n
                result={amep.pbc.pbc_points(coords3d, box)},\n
                comparison={compare3d},\n
                box={box}.'''
        )
        # check index keyword
        returntype = type(amep.pbc.pbc_points(coords2d, box, index=True))
        self.assertIs(
            returntype,
            tuple,
            f'index=True failure. Expected a tuple to be returned. Got {returntype}.'
        )
        # check inclusive keyword
        shape = amep.pbc.pbc_points(coords2d, box, inclusive=False, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (8, 3),
            f'inclusive=False failure. Expected shape (8,3). Got {shape}.'
        )
        # check width keyword
        shape = amep.pbc.pbc_points(coords2d, box, width=0.5).shape
        self.assertEqual(
            shape,
            (4, 3),
            f'width=0.5 failure. Expected shape (4,3). Got {shape}.'
        )
        # check thickness keyword
        shape = amep.pbc.pbc_points(coords2d, box, thickness=5.0, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (4, 3),
            f'thickness=5.0 failure. Expected shape (4,3). Got {shape}.'
        )
        # check if width preferred before thickness
        shape = amep.pbc.pbc_points(
            coords2d, box, thickness=5.0, width=0.1
        ).shape
        self.assertEqual(
            shape,
            (1, 3),
            f'''thickness=5.0, width=0.1 failure. width should be preferred
                before thickness. Expected shape (1,3). Got {shape}.'''
        )

    def test_mirror_points(self):
        # generate test data
        coords2d = np.array([
            [-2.0, 3.0, 0.0]
        ])
        coords3d = np.array([
            [-2.0, 3.0, 0.25]
        ])
        box = np.array([[-5, 5], [-5, 5], [-0.5, 0.5]])
        # expected mirror image coordinates
        mirror_coords2d = np.array([
            [-2.0, 3.0, 0.0],
            [-2.0, 7.0, 0.0],
            [-8.0, 3.0, 0.0],
            [-8.0, 7.0, 0.0],
            [12.0, 3.0, 0.0],
            [12.0, 7.0, 0.0],
            [12.0, -13.0, 0.0],
            [-2.0, -13.0, 0.0],
            [-8.0, -13.0, 0.0]
        ])
        mirror_coords3d = np.array([
            [-2.0, 3.0, 0.25],
            [-2.0, 7.0, 0.25],
            [-8.0, 3.0, 0.25],
            [-8.0, 7.0, 0.25],
            [12.0, 3.0, 0.25],
            [12.0, 7.0, 0.25],
            [12.0, -13.0, 0.25],
            [-2.0, -13.0, 0.25],
            [-8.0, -13.0, 0.25],
            [-2.0, 3.0, 0.75],
            [-2.0, 7.0, 0.75],
            [-8.0, 3.0, 0.75],
            [-8.0, 7.0, 0.75],
            [12.0, 3.0, 0.75],
            [12.0, 7.0, 0.75],
            [12.0, -13.0, 0.75],
            [-2.0, -13.0, 0.75],
            [-8.0, -13.0, 0.75],
            [-2.0, 3.0, -1.25],
            [-2.0, 7.0, -1.25],
            [-8.0, 3.0, -1.25],
            [-8.0, 7.0, -1.25],
            [12.0, 3.0, -1.25],
            [12.0, 7.0, -1.25],
            [12.0, -13.0, -1.25],
            [-2.0, -13.0, -1.25],
            [-8.0, -13.0, -1.25]
        ])
        # comparison
        failed2d = [mc for mc in amep.pbc.mirror_points(coords2d,
                                                        box,
                                                        enforce_nd=2)
                    if not any((mc.round(3) == c.round(3)).all()
                               for c in mirror_coords2d)
                    ]
        failed3d = [mc for mc in amep.pbc.mirror_points(coords3d, box)
                    if not any((mc.round(3) == c.round(3)).all()
                               for c in mirror_coords3d)
                    ]
        # check 2d data indentification
        shape = amep.pbc.mirror_points(coords2d, box, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (9, 3),
            f'''2d detection failed: Got shape {shape} instead of (9,3).
                2d data is not identified as such.'''
        )
        shape = amep.pbc.mirror_points(coords2d, box).shape
        self.assertEqual(
            shape,
            (27, 3),
            f'''2d detection failed: Got shape {shape} instead of (27,3).
                3d data is not identified as such.'''
        )
        shape = amep.pbc.mirror_points(coords3d, box).shape
        self.assertEqual(
            shape,
            (27, 3),
            f'''2d detection failed: Got shape {shape} instead of (27,3).
                3d data is not identified as such.'''
        )
        # check enforce_twod keyword
        shape = amep.pbc.mirror_points(coords3d, box, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (9, 3),
            f'enforce_twod=True failure. Expected shape (9,3). Got {shape}.'
        )
        # check results
        self.assertTrue(
            failed2d==[],
            f'''incorrect mirror coords for 2d data.\n
                failed={failed2d}'''
        )
        self.assertTrue(
            failed3d==[],
            f'''incorrect mirror coords for 3d data.\n
                failed={failed3d}'''
        )
        # check index keyword
        returntype = type(amep.pbc.mirror_points(coords2d, box, index=True))
        self.assertIs(
            returntype,
            tuple,
            f'index=True failure. Expected a tuple to be returned. Got {returntype}.'
        )
        # check inclusive keyword
        shape = amep.pbc.mirror_points(coords2d, box, inclusive=False, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (8, 3),
            f'inclusive=False failure. Expected shape (8,3). Got {shape}.'
        )
        # check width keyword
        shape = amep.pbc.mirror_points(coords2d, box, width=0.5, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (4, 3),
            f'width=0.5 failure. Expected shape (4,3). Got {shape}.'
        )
        # check thickness keyword
        shape = amep.pbc.mirror_points(coords2d, box, thickness=5.0, enforce_nd=2).shape
        self.assertEqual(
            shape,
            (4, 3),
            f'thickness=5.0 failure. Expected shape (4,3). Got {shape}.'
        )
        # check if width preferred before thickness
        shape = amep.pbc.mirror_points(
            coords2d, box, thickness=5.0, width=0.1
        ).shape
        self.assertEqual(
            shape,
            (1, 3),
            f'''thickness=5.0, width=0.1 failure. width should be preferred
                before thickness. Expected shape (1,3). Got {shape}.'''
        )

    def test_pbc_diff(self):
        # generate test data
        coords1 = np.array([
            [4.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
            [0.0, 0.0, 0.45]
        ])
        coords2 = np.array([
            [-4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, -0.45]
        ])
        box = np.array([[-5, 5], [-5, 5], [-0.5, 0.5]])
        # expected result without pbc
        diff = np.array([
            [8.0, 0.0, 0.0],
            [0.0, -8.0, 0.0],
            [0.0, 0.0, 0.9]
        ])
        # expected result with pbc
        pbcdiff = np.array([
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -0.1]
        ])
        # check results
        diff_to_test = amep.pbc.pbc_diff(coords1, coords2, box, pbc=False)
        self.assertTrue(
            np.isclose(diff_to_test, diff).all(),
            f'''incorrect difference vectors with pbc=False:\n
                coords1={coords1},\n
                coords2={coords2},\n
                expected={diff},\n
                result={diff_to_test},\n
                box={box}.'''
        )
        pbcdiff_to_test = amep.pbc.pbc_diff(coords1, coords2, box, pbc=True)
        self.assertTrue(
            np.isclose(pbcdiff_to_test, pbcdiff).all(),
            f'''incorrect difference vectors with pbc=True:\n
                coords1={coords1},\n
                coords2={coords2},\n
                expected={pbcdiff},\n
                result={pbcdiff_to_test},\n
                box={box}.'''
        )

    def test_pbc_diff_rect(self):
        # generate test data
        coords1 = np.array([
            [4.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
            [0.0, 0.0, 0.45]
        ])
        coords2 = np.array([
            [-4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, -0.45]
        ])
        box = np.array([[-5, 5], [-5, 5], [-0.5, 0.5]])
        # expected result with pbc
        pbcdiff = np.array([
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -0.1]
        ])
        # check result
        pbcdiff_to_test = amep.pbc.pbc_diff_rect(coords1, coords2, box)
        self.assertTrue(
            np.isclose(pbcdiff_to_test, pbcdiff).all(),
            f'''incorrect difference vectors:\n
                coords1={coords1},\n
                coords2={coords2},\n
                expected={pbcdiff},\n
                result={pbcdiff_to_test},\n
                box={box}.'''
        )

    def test_kdtree(self):
        pass

    def test_find_pairs(self):
        pass

    def test_distance_matrix(self):
        pass

    def test_distances(self):
        pass       
