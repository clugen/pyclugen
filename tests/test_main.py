# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the main clugen() function."""

import re
import warnings

import pytest
from numpy import abs, all, arange, array, pi, repeat, sum, unique
from numpy.random import Generator, Philox
from numpy.testing import assert_allclose, assert_array_equal

from pyclugen.helper import angle_btw
from pyclugen.main import clugen
from pyclugen.module import angle_deltas, clucenters, clusizes, llengths


def test_clugen_mandatory(
    prng,
    ndims,
    num_clusters,
    num_points,
    vec_or_mat,
    angle_std,
    clusep_fn,
    llength_mu,
    llength_sigma,
    lat_std,
):
    """Test the mandatory parameters of the clugen() function."""
    direc = vec_or_mat(ndims, num_clusters)

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
                clusep_fn(ndims),
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
                clusep_fn(ndims),
                llength_mu,
                llength_sigma,
                lat_std,
                rng=prng,
            )
        return  # In this case, no need for more tests with this parameter set

    # Check dimensions of result variables
    assert result.points.shape == (num_points, ndims)
    assert result.clusters.shape == (num_points,)
    assert result.projections.shape == (num_points, ndims)
    assert result.sizes.shape == (num_clusters,)
    assert result.centers.shape == (num_clusters, ndims)
    assert result.directions.shape == (num_clusters, ndims)
    assert result.angles.shape == (num_clusters,)
    assert result.lengths.shape == (num_clusters,)

    # Check point cluster indexes
    assert all(unique(result.clusters) == arange(num_clusters))

    # Check total points
    assert sum(result.sizes) == num_points

    # Check that cluster directions have the correct angles with the main direction
    if ndims > 1:
        if direc.ndim == 1:
            direc = repeat(direc.reshape((1, -1)), num_clusters, axis=0)
        for i in range(num_clusters):
            assert_allclose(
                angle_btw(direc[i, :], result.directions[i, :]),
                abs(result.angles[i]),
                atol=1e-6,
            )


def test_clugen_optional(
    prng,
    ndims,
    vec_or_mat,
    clusep_fn,
    cluoff_fn,
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
    direc = vec_or_mat(ndims, nclu)

    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        result = clugen(
            ndims,
            nclu,
            tpts,
            direc,
            astd,
            clusep_fn(ndims),
            len_mu,
            len_std,
            lat_std,
            allow_empty=allow_empty,
            cluster_offset=cluoff_fn(ndims),
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
    assert result.clusters.shape == (tpts,)
    assert result.projections.shape == (tpts, ndims)
    assert result.sizes.shape == (nclu,)
    assert result.centers.shape == (nclu, ndims)
    assert result.directions.shape == (nclu, ndims)
    assert result.angles.shape == (nclu,)
    assert result.lengths.shape == (nclu,)

    # Check point cluster indexes
    if not allow_empty:
        assert all(unique(result.clusters) == arange(nclu))
    else:
        assert all(result.clusters < nclu)

    # Check total points
    assert sum(result.sizes) == tpts
    # This might not be the case if the specified clusize_fn does not obey
    # the total number of points

    # Check that cluster directions have the correct angles with the main direction
    if ndims > 1:
        if direc.ndim == 1:
            direc = repeat(direc.reshape((1, -1)), nclu, axis=0)
        for i in range(nclu):
            assert_allclose(
                angle_btw(direc[i, :], result.directions[i, :]),
                abs(result.angles[i]),
                atol=1e-11,
            )


def test_clugen_reproducibility(seed, ndims):
    """Test that clugen() provides reproducible results."""
    # This line can't be blank

    def run_clugen(seed, ndims):
        # Initialize a pseudo-random generator with the specified seed
        prng = Generator(Philox(seed))

        # Run clugen and return results
        return clugen(
            ndims,
            prng.integers(1, 20),  # Number of clusters
            prng.integers(1, 500),  # Number of points
            prng.random(ndims),  # Direction
            prng.random(),  # Angle dispersion
            prng.random(ndims),  # Cluster separation
            prng.random(),  # Line length average
            prng.random(),  # Line length dispersion
            prng.random(),  # Lateral dispersion
            rng=prng,
        )

    # Run clugen with specified seed and get results
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        r1 = run_clugen(seed, ndims)

    # Run clugen again with the same seed and get results
    with warnings.catch_warnings():
        # Check that the function runs without warnings
        warnings.simplefilter("error")
        r2 = run_clugen(seed, ndims)

    # Check that results are the same
    assert_array_equal(r1.points, r2.points)
    assert_array_equal(r1.clusters, r2.clusters)


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

    # Direction needs to have nd size (or nd columns)
    bad_dir = array([1, 1])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of directions in `direction` must be equal to `num_dims` "
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

    # Direction needs to have 1 or nclu rows
    bad_dir = repeat([[1, 1]], nclu + 1, axis=0)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of rows in `direction` must be the same as the number of "
            + f"clusters ({bad_dir.shape[0]} != {nclu})"
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

    # Direction needs to be a 1D array (vector) or 2D array (matrix)
    bad_dir = prng.random((nclu, nd, 2))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`direction` must be a vector (1D array) or a matrix (2D array), "
            + f"but is {bad_dir.ndim}D"
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
