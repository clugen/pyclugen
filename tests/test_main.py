# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the main clugen() function."""

import pytest
from numpy import abs, all, arange, pi, sum, unique
from numpy.random import PCG64, Generator
from numpy.testing import assert_allclose

from clugen.main import clugen

from .helpers import angle_btw


@pytest.fixture(params=[0, 98765])
def prng(request):
    """Provides random number generators."""
    return Generator(PCG64(request.param))


@pytest.fixture(params=[1, 2, 3, 7])
def ndims(request):
    """Provides a number of dimensions."""
    return request.param


@pytest.fixture(params=[1, 10, 800])
def num_points(request):
    """Provides a number of points."""
    return request.param


@pytest.fixture(params=[1, 4, 25])
def num_clusters(request):
    """Provides a number of clusters."""
    return request.param


@pytest.fixture(params=[0.0, 20.0])
def lat_std(request):
    """Provides values for lat_std."""
    return request.param


@pytest.fixture(params=[0, pi / 32, pi / 2, pi, 2 * pi])
def angle_std(request):
    """Provides an angle standard deviation."""
    return request.param


def test_clugen_mandatory(
    prng,
    ndims,
    num_clusters,
    num_points,
    vector,
    angle_std,
    clu_sep,
    llength_mu,
    llength_sigma,
    lat_std,
):
    """Test the mandatory parameters of the clugen() function."""
    direc = vector(ndims)

    # By default, allow_empty is false, so clugen() must be given more points
    # than clusters...
    if num_points >= num_clusters:
        # ...in which case it runs without problem
        result = clugen(
            ndims,
            num_clusters,
            num_points,
            direc,
            angle_std,
            clu_sep,
            llength_mu,
            llength_sigma,
            lat_std,
            rng=prng,
        )
    else:
        # ...otherwise an ArgumentError will be thrown
        with pytest.raises(
            ValueError,
            match=f"A total of {num_points} points is not enough for "
            + f"{num_clusters} non-empty clusters",
        ):
            clugen(
                ndims,
                num_clusters,
                num_points,
                direc,
                angle_std,
                clu_sep,
                llength_mu,
                llength_sigma,
                lat_std,
                rng=prng,
            )
        return  # In this case, no need for more tests with this parameter set

    # Check dimensions of result variables
    assert result.points.shape == (num_points, ndims)
    assert result.point_clusters.shape == (num_points,)
    assert result.point_projections.shape == (num_points, ndims)
    assert result.cluster_sizes.shape == (num_clusters,)
    assert result.cluster_centers.shape == (num_clusters, ndims)
    assert result.cluster_directions.shape == (num_clusters, ndims)
    assert result.cluster_angles.shape == (num_clusters,)
    assert result.cluster_lengths.shape == (num_clusters,)

    # Check point cluster indexes
    assert all(unique(result.point_clusters) == arange(num_clusters))

    # Check total points
    assert sum(result.cluster_sizes) == num_points

    # Check that cluster directions have the correct angles with the main direction
    if ndims > 1:
        for i in range(num_clusters):
            assert_allclose(
                angle_btw(direc, result.cluster_directions[i, :]),
                abs(result.cluster_angles[i]),
                atol=1e-11,
            )
