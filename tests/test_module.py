# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the algorithm module functions."""

from numpy import abs, all, pi

from clugen.module import angle_deltas, clucenters


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
