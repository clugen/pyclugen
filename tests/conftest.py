# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Fixtures to be used by test functions."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import pytest
from numpy import arange, diag, linspace, ones, pi, zeros
from numpy.linalg import norm
from numpy.random import PCG64, Generator
from numpy.typing import NDArray

from pyclugen.module import angle_deltas, clucenters, clusizes, llengths


def _ptdist_evenly(ln: float, n: int, r: Generator) -> NDArray:
    """Alternative projection distribution function for testing purposes."""
    return linspace(-ln / 2, ln / 2, n)


def _ptoff_ones(
    prj: NDArray, ls: float, ln: float, cd: NDArray, cc: NDArray, r: Generator
) -> NDArray:
    """Alternative point offset function for testing purposes."""
    return prj + ones(prj.shape)


def _csz_equi_size(nclu: int, tpts: int, ae: bool, rng: Generator) -> NDArray:
    """Alternative cluster sizing function for testing purposes."""
    cs = zeros(nclu, dtype=int)
    for i in range(tpts):
        cs[i % nclu] += 1
    return cs


def _cctr_diag(nc: int, cs: NDArray, co: NDArray, r: Generator) -> NDArray:
    """Alternative cluster centers function for testing purposes."""
    return diag(arange(1, nc + 1)) @ ones((nc, cs.size))


def _llen_0_20(nc: int, ln: float, ls: float, r: Generator) -> NDArray:
    """Alternative line lengths function for testing purposes."""
    return 10 + 10 * r.random(nc)


def _lang_zeros(nc: int, asd: float, r: Generator) -> NDArray:
    """Alternative line angles function for testing purposes."""
    return zeros(nc)


def zeros_ones_and_randoms_factory(std, seeds):
    """Generates a list of function for generating vectors with a given size.

    The first function generates a vector with zeros.
    The second function generates a vector with ones.
    The remaining functions generate a random vector with elements drawn from the
    normal distribution using the bit generator given in `bitgen` with standard
    deviation given by `std`.
    """
    return [lambda sz: zeros(sz), lambda sz: ones(sz)] + [
        lambda sz, isd=osd: std * Generator(PCG64(isd)).normal(size=sz) for osd in seeds
    ]


def pytest_addoption(parser):
    """Add '--test-level' option to pytest."""
    parser.addoption(
        "--test-level",
        action="store",
        default="normal",
        help="test-level: fast, ci, normal or full",
        choices=("fast", "ci", "normal", "full"),
    )


seeds: Sequence[int]
t_ndims: Sequence[int]
t_num_points: Sequence[int]
t_num_clusters: Sequence[int]
t_lat_std: Sequence[float]
t_angle_std: Sequence[float]
ptdist: Sequence[str | Callable[[float, int, Generator], NDArray]]
ptoff: Sequence[
    str | Callable[[NDArray, float, float, NDArray, NDArray, Generator], NDArray]
]
csz: Sequence[Callable[[int, int, bool, Generator], NDArray]]
cctr: Sequence[Callable[[int, NDArray, NDArray, Generator], NDArray]]
llen: Sequence[Callable[[int, float, float, Generator], NDArray]]
lang: Sequence[Callable[[int, float, Generator], NDArray]]
t_ds_cg_n: Sequence[int]
t_ds_ot_n: Sequence[int]
t_ds_od_n: Sequence[int]
t_no_clusters_field: Sequence[bool]
t_ds_cgs_n: Sequence[int]


def pytest_report_header(config):
    """Show pyclugen test level when running pytest."""
    return f"pyclugen test level: {config.getoption('--test-level')}"


def pytest_generate_tests(metafunc):
    """Generate test data depending on '--test-level' option."""
    test_level = metafunc.config.getoption("--test-level")
    if test_level == "fast":
        # Fast test level, quick check if everything is working
        seeds = [123]
        t_ndims = [2]
        t_num_points = [50]
        t_num_clusters = [4]
        t_clusep_fn = zeros_ones_and_randoms_factory(1000, seeds)
        t_lat_std = [2.0]
        t_angle_std = [pi / 128]
        t_llen_mu = [5]
        t_llen_sigma = [0.5]
        t_ae = [True]
        t_cluoff_fn = zeros_ones_and_randoms_factory(1000, seeds)
        ptdist = ["norm"]
        ptoff = ["n-1"]
        csz = [clusizes]
        cctr = [clucenters]
        llen = [llengths]
        lang = [angle_deltas]
        t_ds_cg_n = [0, 1]
        t_ds_ot_n = [0]
        t_ds_od_n = [0, 1]
        t_no_clusters_field = (False,)
        t_ds_cgs_n = [2]
    elif test_level == "ci":
        # CI test level
        seeds = [123]
        t_ndims = [1, 2, 3]
        t_num_points = [1, 10, 500]
        t_num_clusters = [1, 10]
        t_clusep_fn = zeros_ones_and_randoms_factory(1000, seeds)
        t_lat_std = [0.0, 10.0]
        t_angle_std = [0, pi / 128, pi / 2, pi, 2 * pi]
        t_llen_mu = [0, 5]
        t_llen_sigma = [0, 0.5]
        t_ae = [True, False]
        t_cluoff_fn = zeros_ones_and_randoms_factory(1000, seeds)
        ptdist = ["norm", "unif", _ptdist_evenly]
        ptoff = ["n-1", "n", _ptoff_ones]
        csz = [clusizes, _csz_equi_size]
        cctr = [clucenters, _cctr_diag]
        llen = [llengths, _llen_0_20]
        lang = [angle_deltas, _lang_zeros]
        t_ds_cg_n = [0, 2]
        t_ds_ot_n = [0, 1]
        t_ds_od_n = [0, 1]
        t_no_clusters_field = [False, True]
        t_ds_cgs_n = [2, 3]
    elif test_level == "normal":
        seeds = [0, 123, 6789]
        t_ndims = [1, 2, 3, 10]
        t_num_points = [1, 10, 500, 2500]
        t_num_clusters = [1, 10, 50]
        t_clusep_fn = zeros_ones_and_randoms_factory(1000, seeds)
        t_lat_std = [0.0, 10.0]
        t_angle_std = [0, pi / 128, pi / 8, pi / 2, pi]
        t_llen_mu = [0, 5, 50]
        t_llen_sigma = [0, 0.5, 20]
        t_ae = [True, False]
        t_cluoff_fn = zeros_ones_and_randoms_factory(1000, seeds)
        ptdist = ["norm", "unif", _ptdist_evenly]
        ptoff = ["n-1", "n", _ptoff_ones]
        csz = [clusizes, _csz_equi_size]
        cctr = [clucenters, _cctr_diag]
        llen = [llengths, _llen_0_20]
        lang = [angle_deltas, _lang_zeros]
        t_ds_cg_n = [0, 1, 2]
        t_ds_ot_n = [0, 1]
        t_ds_od_n = [0, 1, 2]
        t_no_clusters_field = [False, True]
        t_ds_cgs_n = [2, 3, 4]
    elif test_level == "full":
        seeds = [0, 123, 6789, 9876543]
        t_ndims = [1, 2, 3, 5, 10, 30]
        t_num_points = [1, 10, 500, 2500, 10000]
        t_num_clusters = [1, 2, 5, 10, 50, 100]
        t_clusep_fn = zeros_ones_and_randoms_factory(1000, seeds)
        t_lat_std = [0.0, 5.0, 10.0, 500.0]
        t_angle_std = [0, pi / 128, pi / 32, pi / 4, pi / 2, pi, 2 * 3 / pi, 2 * pi]
        t_llen_mu = [0, 5, 50, 200]
        t_llen_sigma = [0, 0.5, 20, 63]
        t_ae = [True, False]
        t_cluoff_fn = zeros_ones_and_randoms_factory(1000, seeds)
        ptdist = ["norm", "unif", _ptdist_evenly]
        ptoff = ["n-1", "n", _ptoff_ones]
        csz = [clusizes, _csz_equi_size]
        cctr = [clucenters, _cctr_diag]
        llen = [llengths, _llen_0_20]
        lang = [angle_deltas, _lang_zeros]
        t_ds_cg_n = [0, 1, 2, 3]
        t_ds_ot_n = [0, 1, 2]
        t_ds_od_n = [0, 1, 2]
        t_no_clusters_field = [False, True]
        t_ds_cgs_n = [2, 3, 4, 5]
    else:
        raise ValueError(f"Unknown test level {test_level!r}")

    def param_if(param: str, value: Sequence[Any]):
        if param in metafunc.fixturenames:
            metafunc.parametrize(param, value)

    param_if("seed", seeds)
    param_if("prng", [Generator(PCG64(s)) for s in seeds])
    param_if("ndims", t_ndims)
    param_if("num_points", t_num_points)
    param_if("num_clusters", t_num_clusters)
    param_if("clusep_fn", t_clusep_fn)
    param_if("lat_std", t_lat_std)
    param_if("angle_std", t_angle_std)
    param_if("llength_mu", t_llen_mu)
    param_if("llength_sigma", t_llen_sigma)
    param_if("allow_empty", t_ae)
    param_if("cluoff_fn", t_cluoff_fn)
    param_if("ptdist_fn", ptdist)
    param_if("ptoff_fn", ptoff)
    param_if("csz_fn", csz)
    param_if("cctr_fn", cctr)
    param_if("llen_fn", llen)
    param_if("lang_fn", lang)
    param_if("ds_cg_n", t_ds_cg_n)
    param_if("ds_ot_n", t_ds_ot_n)
    param_if("ds_od_n", t_ds_od_n)
    param_if("no_clusters_field", t_no_clusters_field)
    param_if("ds_cgs_n", t_ds_cgs_n)


@pytest.fixture()
def vector(prng):
    """Provides random vectors."""

    def _vector(numel):
        return prng.random(numel)

    return _vector


@pytest.fixture()
def uvector(prng):
    """Provides random unit vectors."""

    def _uvector(numel):
        v = prng.random(numel)
        return v / norm(v)

    return _uvector


@pytest.fixture(params=["vec", "mat"])
def vec_or_mat(prng, request):
    """Provides a vector or a matrix."""
    if request.param == "vec":

        def _vec(ndim, nclu):
            return prng.random(ndim)

        return _vec
    elif request.param == "mat":

        def _mat(ndim, nclu):
            return prng.random((nclu, ndim))

        return _mat
    else:
        raise ValueError(f"Unknown fixture parameter {request.param!r}")
