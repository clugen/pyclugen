# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the main clugen() function."""

import pytest
from numpy import abs, all, arange, diag, linspace, ones, pi, sum, unique, zeros
from numpy.random import PCG64, Generator
from numpy.testing import assert_allclose

from clugen.main import clugen
from clugen.module import angle_deltas, clucenters, clusizes, llengths

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


@pytest.fixture(params=[0, pi / 8, pi])
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
        with pytest.warns(None) as wrec:
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
        # Check that the function runs without warnings
        assert len(wrec) == 0
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


@pytest.fixture(params=["norm", "unif", lambda l, n: linspace(-l / 2, l / 2, n)])
def ptdist_fn(request):
    """Provides a point distribution function."""
    return request.param


@pytest.fixture(
    params=["n-1", "n", lambda prj, ls, l, cd, cc, r: prj + ones(prj.shape)]
)
def ptoff_fn(request):
    """Provides a point offset function."""
    return request.param


def csz_equi_size(nclu, tpts, ae, rng):
    """Alternative cluster sizing function for testing purposes."""
    cs = zeros(nclu, dtype=int)
    for i in range(tpts):
        cs[i % nclu] += 1
    return cs


@pytest.fixture(params=[clusizes, csz_equi_size])
def csz_fn(request):
    """Provides a cluster sizes function."""
    return request.param


@pytest.fixture(
    params=[
        clucenters,
        lambda nc, cs, co, r: diag(arange(1, nc + 1)) @ ones((nc, cs.size)),
    ]
)
def cctr_fn(request):
    """Provides a cluster centers function."""
    return request.param


@pytest.fixture(params=[llengths, lambda nc, l, ls, r: 10 + 10 * r.random(nc)])
def llen_fn(request):
    """Provides a a line lengths function."""
    return request.param


@pytest.fixture(params=[angle_deltas, lambda nc, asd, r: zeros(nc)])
def lang_fn(request):
    """Provides a line angles function."""
    return request.param


def test_clugen_optional(
    prng,
    ndims,
    vector,
    clu_sep,
    clu_offset,
    allow_empty,
    ptdist_fn,
    ptoff_fn,
    csz_fn,
    cctr_fn,
    llen_fn,
    lang_fn,
):
    """Test the optional parameters of the clugen() function."""
    # Valid arguments
    nclu = 7
    tpts = 500
    astd = pi / 256
    len_mu = 9
    len_std = 1.2
    lat_std = 2

    # Get direction
    direc = vector(ndims)

    with pytest.warns(None) as wrec:
        result = clugen(
            ndims,
            nclu,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=allow_empty,
            cluster_offset=clu_offset,
            proj_dist_fn=ptdist_fn,
            point_dist_fn=ptoff_fn,
            clusizes_fn=csz_fn,
            clucenters_fn=cctr_fn,
            llengths_fn=llen_fn,
            angle_deltas_fn=lang_fn,
            rng=prng,
        )

    # Check that the function runs without warnings
    assert len(wrec) == 0

    # Check dimensions of result variables
    assert result.points.shape == (tpts, ndims)
    assert result.point_clusters.shape == (tpts,)
    assert result.point_projections.shape == (tpts, ndims)
    assert result.cluster_sizes.shape == (nclu,)
    assert result.cluster_centers.shape == (nclu, ndims)
    assert result.cluster_directions.shape == (nclu, ndims)
    assert result.cluster_angles.shape == (nclu,)
    assert result.cluster_lengths.shape == (nclu,)

    # Check point cluster indexes
    if not allow_empty:
        assert all(unique(result.point_clusters) == arange(nclu))
    else:
        assert all(result.point_clusters < nclu)

    # Check total points
    assert sum(result.cluster_sizes) == tpts
    # This might not be the case if the specified clusize_fn does not obey
    # the total number of points

    # Check that cluster directions have the correct angles with the main direction
    if ndims > 1:
        for i in range(nclu):
            assert_allclose(
                angle_btw(direc, result.cluster_directions[i, :]),
                abs(result.cluster_angles[i]),
                atol=1e-11,
            )
