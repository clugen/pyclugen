# Copyright (c) 2020, 2021 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the core module."""

import pytest
from numpy import vdot
from numpy.linalg import norm
from numpy.testing import assert_allclose

import clugen as cg
from clugen.core import rand_ortho_vector


def test_points_on_line(ndims, num_points, prng, llength_mu, uvector, vector):
    """Test the points_on_line() function."""
    # Number of directions to test
    ndirs = 3

    # Number of line centers to test
    ncts = 3

    # Avoid too many points, otherwise testing will be very slow
    if num_points >= 1000:
        return

    # Create some random distances from center
    dist2ctr = llength_mu * prng.random((num_points, 1)) - llength_mu / 2

    # Test for different directions
    for _i in range(ndirs):

        # Get a direction
        direc = uvector(ndims)

        # Test for different number of line centers
        for _j in range(ncts):

            # Get a line center
            ctr = vector(ndims)

            # Invoke the points_on_line function
            with pytest.warns(None) as wrec:
                pts = cg.points_on_line(ctr, direc, dist2ctr)

            # Check that the points_on_line function runs without warnings
            assert len(wrec) == 0

            # Check that the dimensions agree
            assert pts.shape == (num_points, ndims)

            # Check that distance of points to the line is approximately zero
            for pt in pts:
                # Get distance from current point to line
                pt = pt.reshape((ndims, 1))
                d = norm((pt - ctr) - vdot((pt - ctr), direc) * direc)
                # Check that it is approximately zero
                assert_allclose(d, 0, atol=1e-14)


def test_rand_ortho_vector(ndims, prng, uvector):
    """Test the rand_ortho_vector() function."""
    # Get a base vector
    u = uvector(ndims)

    # Invoke the rand_ortho_vector function on the base vector
    r = rand_ortho_vector(u, rng=prng)

    # Check that returned vector has the correct dimensions
    assert r.shape == (ndims, 1)

    # Check that returned vector has norm == 1
    assert_allclose(norm(r), 1, atol=1e-14)

    # Check that vectors u and r are orthogonal (only for nd > 1)
    if ndims > 1:
        # The dot product of orthogonal vectors must be (approximately) zero
        assert_allclose(vdot(u, r), 0, atol=1e-12)


def test_rand_unit_vector(ndims, prng):
    """Test the rand_unit_vector() function."""
    # Get a random unit vector
    r = cg.rand_unit_vector(ndims, rng=prng)

    # Check that returned vector has the correct dimensions
    assert r.shape == (ndims, 1)

    # Check that returned vector has norm == 1
    assert_allclose(norm(r), 1, atol=1e-14)
