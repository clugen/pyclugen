# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the core functions."""

import pytest
from numpy import abs, dot, isclose, pi
from numpy.linalg import norm
from numpy.testing import assert_allclose

from clugen.core import (
    points_on_line,
    rand_ortho_vector,
    rand_unit_vector,
    rand_vector_at_angle,
)

from .helpers import angle_btw


@pytest.fixture(params=[1, 10, 500])
def num_points(request):
    """Provides a number of points."""
    return request.param


def test_points_on_line(ndims, num_points, prng, llength_mu, uvector, vector):
    """Test the points_on_line() function."""
    # Create some random distances from center
    dist2ctr = llength_mu * prng.random((num_points, 1)) - llength_mu / 2

    # Get a direction
    direc = uvector(ndims)

    # Get a line center
    ctr = vector(ndims)

    # Invoke the points_on_line function
    with pytest.warns(None) as wrec:
        pts = points_on_line(ctr, direc, dist2ctr)

    # Check that the points_on_line function runs without warnings
    assert len(wrec) == 0

    # Check that the dimensions agree
    assert pts.shape == (num_points, ndims)

    # Check that distance of points to the line is approximately zero
    for pt in pts:
        # Get distance from current point to line
        d = norm((pt - ctr) - dot((pt - ctr), direc) * direc)
        # Check that it is approximately zero
        assert_allclose(d, 0, atol=1e-14)


def test_rand_ortho_vector(ndims, prng, uvector):
    """Test the rand_ortho_vector() function."""
    # Get a base unit vector
    u = uvector(ndims)

    # Invoke the rand_ortho_vector function on the base vector
    with pytest.warns(None) as wrec:
        r = rand_ortho_vector(u, rng=prng)

    # Check that the function runs without warnings
    assert len(wrec) == 0

    # Check that returned vector has the correct dimensions
    assert r.shape == (ndims,)

    # Check that returned vector has norm == 1
    assert_allclose(norm(r), 1, atol=1e-14)

    # Check that vectors u and r are orthogonal (only for nd > 1)
    if ndims > 1:
        # The dot product of orthogonal vectors must be (approximately) zero
        assert_allclose(dot(u, r), 0, atol=1e-12)


def test_rand_unit_vector(ndims, prng):
    """Test the rand_unit_vector() function."""
    # Get a random unit vector
    with pytest.warns(None) as wrec:
        r = rand_unit_vector(ndims, rng=prng)

    # Check that the function runs without warnings
    assert len(wrec) == 0

    # Check that returned vector has the correct dimensions
    assert r.shape == (ndims,)

    # Check that returned vector has norm == 1
    assert_allclose(norm(r), 1, atol=1e-14)


def test_rand_vector_at_angle(ndims, prng, uvector, angle_std):
    """Test the rand_vector_at_angle() function."""
    # Get a base unit vector
    u = uvector(ndims)

    # Invoke the rand_vector_at_angle function on the base vector
    with pytest.warns(None) as wrec:
        r = rand_vector_at_angle(u, angle_std, rng=prng)

    # Check that the function runs without warnings
    assert len(wrec) == 0

    # Check that returned vector has the correct dimensions
    assert r.shape == (ndims,)

    # Check that returned vector has norm == 1
    assert_allclose(norm(r), 1, atol=1e-14)

    # Check that vectors u and r have an angle of angle_std between them
    if ndims > 1 and abs(angle_std) < pi / 2:
        isclose(angle_btw(u, r), abs(angle_std), atol=1e-12)
