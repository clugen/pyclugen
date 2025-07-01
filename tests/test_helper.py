# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the helper functions."""

import warnings

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_equal

from pyclugen.core import points_on_line
from pyclugen.helper import angle_btw, clupoints_n_1_template, fix_empty, fix_num_points


def test_angle_btw():
    """Test the angle_btw() function."""
    # No blank line allowed here

    # Commonly used function for determining the angle between two vectors
    def common_angle_btw(v1, v2):
        return np.arccos(np.dot(v1, v2) / (norm(v1) * norm(v2)))

    # 2D
    u = np.array([1.5, 0])
    v = np.array([0.1, -0.4])
    assert_allclose(angle_btw(u, v), common_angle_btw(u, v))

    # 3D
    u = np.array([-1.5, 10, 0])
    v = np.array([0.99, 4.4, -1.1])
    assert_allclose(angle_btw(u, v), common_angle_btw(u, v))

    # 8D
    u = np.array([1.5, 0, 0, 0, 0, 0, 0, -0.5])
    v = np.array([7.5, -0.4, 0, 0, 0, -16.4, 0.1, -0.01])
    assert_allclose(angle_btw(u, v), common_angle_btw(u, v))


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
        return rg.choice(np.array([-dist_pt, dist_pt]), (clu_num_points, 1))

    # Create some point projections
    proj_dist_fn2ctr = llength_mu * prng.random((num_points, 1)) - llength_mu / 2
    proj = points_on_line(vector(ndims), direc, proj_dist_fn2ctr)

    # The clupoints_n_1_template function should run without warnings
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        pts = clupoints_n_1_template(proj, lat_std, direc, dist_fn, rng=prng)

    # Check that number of points is the same as the number of projections
    assert pts.shape == proj.shape

    # In case of 1D, stop test
    if ndims == 1:
        return

    # The point minus its projection should yield an approximately
    # orthogonal vector to the cluster line
    for u in pts - proj:
        # Vector should be approximately orthogonal to the cluster line
        assert_allclose(np.dot(direc, u), 0, atol=1e-7)

        # Vector should have a magnitude of approximately dist_pt
        assert_allclose(norm(u), dist_pt)


def test_fix_empty():
    """Test the fix_empty() function."""
    # No empty clusters
    clusts = np.array([11, 21, 10])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_empty(clusts, False)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, no fix
    clusts = np.array([0, 11, 21, 10, 0, 0])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_empty(clusts, True)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, fix
    clusts = np.array([5, 0, 21, 10, 0, 0, 101])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_empty(clusts, False)
    assert clusts is clusts_fixed
    assert np.sum(clusts_copy) == np.sum(clusts_fixed)
    assert np.any(clusts_copy != clusts_fixed)
    assert np.all(np.array(clusts_fixed) > 0)

    # # Empty clusters, fix, several equal maximums
    clusts = np.array([101, 5, 0, 21, 101, 10, 0, 0, 101, 100, 99, 0, 0, 0, 100])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_empty(clusts, False)
    assert clusts is clusts_fixed
    assert np.sum(clusts_copy) == np.sum(clusts_fixed)
    assert np.any(clusts_copy != clusts_fixed)
    assert np.all(np.array(clusts_fixed) > 0)

    # Empty clusters, no fix (flag)
    clusts = np.array([0, 10])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_empty(clusts, True)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Empty clusters, no fix (not enough points)
    clusts = np.array([0, 1, 1, 0, 0, 2, 0, 0])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_empty(clusts, False)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # Works with 1D
    clusts = np.array([100])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_empty(clusts, True)
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)


def test_fix_num_points():
    """Test the fix_num_points() function."""
    # No change
    clusts = np.array([10, 100, 42, 0, 12])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_num_points(clusts, np.sum(clusts))
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # # Fix due to too many points
    clusts = np.array([55, 12])
    clusts_copy = np.copy(clusts)
    num_pts = np.sum(clusts) - 14
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_num_points(clusts, num_pts)
    assert clusts is clusts_fixed
    assert np.any(clusts_copy != clusts_fixed)
    assert np.sum(clusts_fixed) == num_pts

    # Fix due to too few points
    clusts = np.array([0, 1, 0, 0])
    clusts_copy = np.copy(clusts)
    num_pts = 15
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_num_points(clusts, num_pts)
    assert clusts is clusts_fixed
    assert np.any(clusts_copy != clusts_fixed)
    assert np.sum(clusts_fixed) == num_pts

    # 1D - No change
    clusts = np.array([10])
    clusts_copy = np.copy(clusts)
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_num_points(clusts, np.sum(clusts))
    assert clusts is clusts_fixed
    assert_equal(clusts_copy, clusts_fixed)

    # 1D - Fix due to too many points
    clusts = np.array([241])
    clusts_copy = np.copy(clusts)
    num_pts = np.sum(clusts) - 20
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_num_points(clusts, num_pts)
    assert clusts is clusts_fixed
    assert np.any(clusts_copy != clusts_fixed)
    assert np.sum(clusts_fixed) == num_pts

    # 1D - Fix due to too few points
    clusts = np.array([0])
    clusts_copy = np.copy(clusts)
    num_pts = 8
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clusts_fixed = fix_num_points(clusts, num_pts)
    assert clusts is clusts_fixed
    assert np.any(clusts_copy != clusts_fixed)
    assert np.sum(clusts_fixed) == num_pts
