# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the algorithm module functions."""

from numpy import abs, all, dot, pi
from numpy.testing import assert_allclose

from clugen.core import points_on_line
from clugen.module import angle_deltas, clucenters, clupoints_n_1


def test_angle_deltas(prng, num_clusters, angle_std):
    """Test the angle_deltas() function."""
    # Get angle deltas
    angles = angle_deltas(num_clusters, angle_std, rng=prng)

    # Check that return value has the correct dimensions
    assert angles.shape == (num_clusters,)

    # Check that all angles are between -π/2 and π/2
    assert all(abs(angles) <= pi / 2)


def test_clucenters(ndims, prng, num_clusters, clu_offset, clu_sep):
    """Test the clucenters() function."""
    # Get cluster centers with the clucenters() function
    clu_ctrs = clucenters(num_clusters, clu_sep, clu_offset, rng=prng)

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

    # Get the points
    pts = clupoints_n_1(proj, lat_std, llength_mu, direc, ctr, rng=prng)

    # Check that number of points is the same as the number of projections
    assert pts.shape == proj.shape

    # In case of 1D, stop test
    if ctr.size == 1:
        return

    # The point minus its projection should yield an approximately
    # orthogonal vector to the cluster line
    for u in pts - proj:
        assert_allclose(dot(direc, u), 0, atol=1e-7)
