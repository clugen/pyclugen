# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the helper module."""

import numpy as np
from numpy.testing import assert_equal

from clugen.helper import fix_empty, fix_num_points


def test_fix_empty():
    """Test the fix_empty() function."""
    # No empty clusters
    clusts = np.array([11, 21, 10])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_empty(clusts, False)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, no fix
    clusts = np.array([0, 11, 21, 10, 0, 0])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_empty(clusts, True)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, fix
    clusts = np.array([5, 0, 21, 10, 0, 0, 101])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_empty(clusts, False)
    assert clusts is clusts_fixed
    assert np.sum(clusts_copy) == np.sum(clusts_fixed)
    assert np.any(clusts_copy != clusts_fixed)
    assert np.all(np.array(clusts_fixed) > 0)

    # # Empty clusters, fix, several equal maximums
    clusts = np.array([101, 5, 0, 21, 101, 10, 0, 0, 101, 100, 99, 0, 0, 0, 100])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_empty(clusts, False)
    assert clusts is clusts_fixed
    assert np.sum(clusts_copy) == sum(clusts_fixed)
    assert np.any(clusts_copy != clusts_fixed)
    assert np.all(np.array(clusts_fixed) > 0)

    # Empty clusters, no fix (flag)
    clusts = np.array([0, 10])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_empty(clusts, True)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, no fix (not enough points)
    clusts = np.array([0, 1, 1, 0, 0, 2, 0, 0])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_empty(clusts, False)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Works with 1D
    clusts = np.array([100])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_empty(clusts, True)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)


def test_fix_num_points():
    """Test the fix_num_points() function."""
    # No change
    clusts = np.array([10, 100, 42, 0, 12])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_num_points(clusts, np.sum(clusts))
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # # Fix due to too many points
    clusts = np.array([55, 12])
    clusts_copy = np.copy(clusts)
    num_pts = sum(clusts) - 14
    clusts_fixed = fix_num_points(clusts, num_pts)
    assert clusts is clusts_fixed
    assert np.any(clusts_copy != clusts_fixed)
    assert np.sum(clusts_fixed) == num_pts

    # Fix due to too few points
    clusts = np.array([0, 1, 0, 0])
    clusts_copy = np.copy(clusts)
    num_pts = 15
    clusts_fixed = fix_num_points(clusts, num_pts)
    assert clusts is clusts_fixed
    assert np.any(clusts_copy != clusts_fixed)
    assert np.sum(clusts_fixed) == num_pts

    # 1D - No change
    clusts = np.array([10])
    clusts_copy = np.copy(clusts)
    clusts_fixed = fix_num_points(clusts, sum(clusts))
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # 1D - Fix due to too many points
    clusts = np.array([241])
    clusts_copy = np.copy(clusts)
    num_pts = sum(clusts) - 20
    clusts_fixed = fix_num_points(clusts, num_pts)
    assert clusts is clusts_fixed
    assert np.any(clusts_copy != clusts_fixed)
    assert np.sum(clusts_fixed) == num_pts

    # 1D - Fix due to too few points
    clusts = np.array([0])
    clusts_copy = np.copy(clusts)
    num_pts = 8
    clusts_fixed = fix_num_points(clusts, num_pts)
    assert clusts is clusts_fixed
    assert np.any(clusts_copy != clusts_fixed)
    assert np.sum(clusts_fixed) == num_pts
