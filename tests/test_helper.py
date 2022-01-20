# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the helper module."""

import numpy as np
from numpy.testing import assert_equal

from clugen.helper import fix_empty


def test_fix_empty():
    """Test the points_on_line() function."""
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
