# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the main clugen() function."""

import re
import warnings

import pytest
from numpy import abs, all, arange, array, diag, linspace, ones, pi, sum, unique, zeros
from numpy.random import PCG64, Generator
from numpy.testing import assert_allclose

from clugen.helper import angle_btw
from clugen.main import clugen
from clugen.module import angle_deltas, clucenters, clusizes, llengths


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
        with warnings.catch_warnings():
            # Check that the function runs without warnings
            warnings.simplefilter("error")
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


@pytest.fixture(params=["norm", "unif", lambda l, n, r: linspace(-l / 2, l / 2, n)])
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

    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
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


def test_clugen_exceptions(prng):
    """Test that clugen() raises the expected exceptions."""
    # Valid arguments
    nd = 3
    nclu = 5
    tpts = 1000
    direc = [1, 0, 0]
    astd = pi / 64
    clu_sep = [10, 10, 5]
    len_mu = 5
    len_std = 0.5
    lat_std = 0.3
    ae = True
    clu_off = [-1.5, 0, 2]
    pt_dist = "unif"
    pt_off = "n-1"
    csizes_fn = clusizes
    ccenters_fn = clucenters
    llengths_fn = llengths
    langles_fn = angle_deltas

    # Test passes with valid arguments
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clugen(
            nd,
            nclu,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Test passes with zero points since allow_empty is set to true
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        clugen(
            nd,
            nclu,
            0,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Invalid number of dimensions
    with pytest.raises(
        ValueError,
        match=re.escape("Number of dimensions, `num_dims`, must be > 0"),
    ):
        clugen(
            0,
            nclu,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Invalid number of clusters
    with pytest.raises(
        ValueError,
        match=re.escape("Number of clusters, `num_clust`, must be > 0"),
    ):
        clugen(
            nd,
            0,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Direction needs to have magnitude > 0
    with pytest.raises(
        ValueError,
        match="`direction` must have magnitude > 0",
    ):
        clugen(
            nd,
            nclu,
            tpts,
            [0, 0, 0],
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Direction needs to have nd dims
    bad_dir = array([1, 1])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of `direction` must be equal to `num_dims` "
            + f"({bad_dir.size} != {nd})"
        ),
    ):
        clugen(
            nd,
            nclu,
            tpts,
            bad_dir,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # cluster_sep needs to have nd dims
    bad_clusep = array([10, 0, 5, 1.4])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of `cluster_sep` must be equal to `num_dims` "
            + f"({bad_clusep.size} != {nd})"
        ),
    ):
        clugen(
            nd,
            nclu,
            tpts,
            direc,
            astd,
            bad_clusep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # cluster_offset needs to have nd dims
    bad_cluoff = array([0, 1])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of `cluster_offset` must be equal to `num_dims` "
            + f"({bad_cluoff.size} != {nd}"
        ),
    ):
        clugen(
            nd,
            nclu,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=bad_cluoff,
            proj_dist_fn=pt_dist,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Unknown proj_dist_fn given as string
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`proj_dist_fn` has to be either 'norm', 'unif' or user-defined function"
        ),
    ):
        clugen(
            nd,
            nclu,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn="bad_proj_dist_fn",
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Invalid proj_dist_fn given as function
    with pytest.raises(
        TypeError,
        match="argument",
    ):
        clugen(
            nd,
            nclu,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=lambda x: 0,
            point_dist_fn=pt_off,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Unknown point_dist_fn given as string
    with pytest.raises(
        ValueError,
        match=re.escape(
            "point_dist_fn has to be either 'n-1', 'n' or a user-defined function"
        ),
    ):
        clugen(
            nd,
            nclu,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn="bad_pt_off",
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )

    # Invalid point_dist_fn given as function
    with pytest.raises(
        TypeError,
        match="argument",
    ):
        clugen(
            nd,
            nclu,
            tpts,
            direc,
            astd,
            clu_sep,
            len_mu,
            len_std,
            lat_std,
            allow_empty=ae,
            cluster_offset=clu_off,
            proj_dist_fn=pt_dist,
            point_dist_fn=lambda x: 0,
            clusizes_fn=csizes_fn,
            clucenters_fn=ccenters_fn,
            llengths_fn=llengths_fn,
            angle_deltas_fn=langles_fn,
            rng=prng,
        )
