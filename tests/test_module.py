# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the algorithm module functions."""

import warnings

import numpy as np
from numpy.testing import assert_allclose

from pyclugen.core import points_on_line
from pyclugen.module import angle_deltas, clucenters, clupoints_n_1, clusizes, llengths


def test_angle_deltas(prng, num_clusters, angle_std):
    """Test the angle_deltas() function."""
    # The Get angle deltas
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        angles = angle_deltas(num_clusters, angle_std, rng=prng)

    # Check that return value has the correct dimensions
    assert angles.shape == (num_clusters,)

    # Check that all angles are between -π/2 and π/2
    assert np.all(np.abs(angles) <= np.pi / 2)


def test_clucenters(ndims, prng, num_clusters, cluoff_fn, clusep_fn):
    """Test the clucenters() function."""
    # Get cluster centers with the clucenters() function
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clu_ctrs = clucenters(
            num_clusters, clusep_fn(ndims), cluoff_fn(ndims), rng=prng
        )

    # Check that return value has the correct dimensions
    assert clu_ctrs.shape == (num_clusters, ndims)


def test_clupoints_n_1(ndims, num_points, prng, lat_std, llength_mu, uvector, vector):
    """Test the clupoints_n_1() function."""
    # Get center and direction
    ctr = vector(ndims)
    direc = uvector(ndims)

    # Create some point projections
    proj_dist_fn2ctr = llength_mu * prng.random(num_points) - llength_mu / 2
    proj = points_on_line(ctr, direc, proj_dist_fn2ctr)

    # Invoke the function being tested and get the points
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        pts = clupoints_n_1(proj, lat_std, llength_mu, direc, ctr, rng=prng)

    # Check that number of points is the same as the number of projections
    assert pts.shape == proj.shape

    # In case of 1D, stop test
    if ctr.size == 1:
        return

    # The point minus its projection should yield an approximately
    # orthogonal vector to the cluster line
    for u in pts - proj:
        assert_allclose(np.dot(direc, u), 0, atol=1e-7)


def test_clupoints_n(ndims, num_points, prng, lat_std, llength_mu, uvector, vector):
    """Test the clupoints_n() function."""
    # Get center and direction
    ctr = vector(ndims)
    direc = uvector(ndims)

    # Create some point projections
    proj_dist_fn2ctr = llength_mu * prng.random(num_points) - llength_mu / 2
    proj = points_on_line(ctr, direc, proj_dist_fn2ctr)

    # Invoke the function being tested and get the points
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        pts = clupoints_n_1(proj, lat_std, llength_mu, direc, ctr, rng=prng)

    # Check that number of points is the same as the number of projections
    assert pts.shape == proj.shape


def test_clusizes(prng, num_clusters, num_points, allow_empty):
    """Test the clusizes() function."""
    # Don't test if number of points is less than number of clusters and we
    # don't allow empty clusters
    if not allow_empty and num_points < num_clusters:
        return

    # Obtain the cluster sizes
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clu_sizes = clusizes(num_clusters, num_points, allow_empty, rng=prng)

    # Check that the output has the correct number of clusters
    assert clu_sizes.shape == (num_clusters,)

    # Check that the total number of points is correct
    assert np.sum(clu_sizes) == num_points

    # If empty clusters are not allowed, check that all of them have points
    if not allow_empty:
        assert np.min(clu_sizes) > 0


def test_llengths(prng, num_clusters, llength_mu, llength_sigma):
    """Test the llengths() function."""
    # Obtain the line lengths
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        lens = llengths(num_clusters, llength_mu, llength_sigma, rng=prng)

    # Check that return value has the correct dimensions
    assert lens.shape == (num_clusters,)

    # Check that all lengths are >= 0
    assert np.all(lens >= 0)
