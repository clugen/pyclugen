# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the helper functions."""

import pytest
from numpy import all, any, array, copy, dot, sum
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_equal

from clugen.core import points_on_line
from clugen.helper import clupoints_n_1_template, fix_empty, fix_num_points


def test_clupoints_n_1_template(
    ndims, num_points, llength_mu, lat_std, prng, vector, uvector
):
    """Test the clupoints_n_1_template() function."""
    # Distance from points to projections will be 10
    dist_pt = 10

    # Get a direction
    direc = uvector(ndims)

    # Very simple dist_fn, always puts points at a distance of dist_pt
    def dist_fn(clu_num_points, ldisp, rg):
        return rg.choice(array([-dist_pt, dist_pt]), (clu_num_points, 1))

    # Create some point projections
    proj_dist_fn2ctr = llength_mu * prng.random((num_points, 1)) - llength_mu / 2
    proj = points_on_line(vector(ndims), direc, proj_dist_fn2ctr)

    # The clupoints_n_1_template function should run without warnings
    with pytest.warns(None) as wrec:
        pts = clupoints_n_1_template(proj, lat_std, direc, dist_fn, rng=prng)

    # Check that the function runs without warnings
    assert len(wrec) == 0

    # Check that number of points is the same as the number of projections
    assert pts.shape == proj.shape

    # In case of 1D, stop test
    if ndims == 1:
        return

    # The point minus its projection should yield an approximately
    # orthogonal vector to the cluster line
    for u in pts - proj:

        # Vector should be approximately orthogonal to the cluster line
        assert_allclose(dot(direc, u), 0, atol=1e-7)

        # Vector should have a magnitude of approximately dist_pt
        assert_allclose(norm(u), dist_pt)


def test_fix_empty():
    """Test the fix_empty() function."""
    # No empty clusters
    clusts = array([11, 21, 10])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_empty(clusts, False)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, no fix
    clusts = array([0, 11, 21, 10, 0, 0])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_empty(clusts, True)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, fix
    clusts = array([5, 0, 21, 10, 0, 0, 101])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_empty(clusts, False)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert sum(clusts_copy) == sum(clusts_fixed)
    assert any(clusts_copy != clusts_fixed)
    assert all(array(clusts_fixed) > 0)

    # # Empty clusters, fix, several equal maximums
    clusts = array([101, 5, 0, 21, 101, 10, 0, 0, 101, 100, 99, 0, 0, 0, 100])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_empty(clusts, False)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert sum(clusts_copy) == sum(clusts_fixed)
    assert any(clusts_copy != clusts_fixed)
    assert all(array(clusts_fixed) > 0)

    # Empty clusters, no fix (flag)
    clusts = array([0, 10])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_empty(clusts, True)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, no fix (not enough points)
    clusts = array([0, 1, 1, 0, 0, 2, 0, 0])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_empty(clusts, False)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Works with 1D
    clusts = array([100])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_empty(clusts, True)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)


def test_fix_num_points():
    """Test the fix_num_points() function."""
    # No change
    clusts = array([10, 100, 42, 0, 12])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_num_points(clusts, sum(clusts))
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # # Fix due to too many points
    clusts = array([55, 12])
    clusts_copy = copy(clusts)
    num_pts = sum(clusts) - 14
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_num_points(clusts, num_pts)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert any(clusts_copy != clusts_fixed)
    assert sum(clusts_fixed) == num_pts

    # Fix due to too few points
    clusts = array([0, 1, 0, 0])
    clusts_copy = copy(clusts)
    num_pts = 15
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_num_points(clusts, num_pts)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert any(clusts_copy != clusts_fixed)
    assert sum(clusts_fixed) == num_pts

    # 1D - No change
    clusts = array([10])
    clusts_copy = copy(clusts)
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_num_points(clusts, sum(clusts))
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # 1D - Fix due to too many points
    clusts = array([241])
    clusts_copy = copy(clusts)
    num_pts = sum(clusts) - 20
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_num_points(clusts, num_pts)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert any(clusts_copy != clusts_fixed)
    assert sum(clusts_fixed) == num_pts

    # 1D - Fix due to too few points
    clusts = array([0])
    clusts_copy = copy(clusts)
    num_pts = 8
    with pytest.warns(None) as wrec:
        clusts_fixed = fix_num_points(clusts, num_pts)
    assert len(wrec) == 0
    assert clusts is clusts_fixed
    assert any(clusts_copy != clusts_fixed)
    assert sum(clusts_fixed) == num_pts
