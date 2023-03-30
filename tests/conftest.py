# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Fixtures to be used by test functions."""

from __future__ import annotations

import os
from typing import Callable, Sequence

import pytest
from numpy import arange, diag, linspace, ones, pi, zeros
from numpy.linalg import norm
from numpy.random import PCG64, Generator, Philox
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


def _is_test_mode(tm: str):
    """Return true if the specified test mode is set."""
    return str(os.environ.get("PYCLUGEN_TEST_MODE")).lower() == tm.lower()


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

if _is_test_mode("fast"):
    # Fast test mode, quick check if everything is working
    tm = "fast"
    seeds = [123]
    t_ndims = [2]
    t_num_points = [50]
    t_num_clusters = [4]
    t_lat_std = [2.0]
    t_angle_std = [pi / 128]
    ptdist = ["norm"]
    ptoff = ["n-1"]
    csz = [clusizes]
    cctr = [clucenters]
    llen = [llengths]
    lang = [angle_deltas]

elif os.environ.get("CI") is not None or _is_test_mode("ci"):
    # CI test mode
    tm = "ci"
    seeds = [123]
    t_ndims = [1, 2, 3]
    t_num_points = [1, 10, 500]
    t_num_clusters = [1, 10]
    t_lat_std = [0.0, 10.0]
    t_angle_std = [0, pi / 128, pi / 2, pi, 2 * pi]
    ptdist = ["norm", "unif", _ptdist_evenly]
    ptoff = ["n-1", "n", _ptoff_ones]
    csz = [clusizes, _csz_equi_size]
    cctr = [clucenters, _cctr_diag]
    llen = [llengths, _llen_0_20]
    lang = [angle_deltas, _lang_zeros]

elif not _is_test_mode("full"):
    # If not full, assume "normal" test mode
    tm = "normal"
    seeds = [0, 123, 6789]
    t_ndims = [1, 2, 3, 10]
    t_num_points = [1, 10, 500, 2500]
    t_num_clusters = [1, 10, 50]
    t_lat_std = [0.0, 10.0]
    t_angle_std = [0, pi / 128, pi / 8, pi / 2, pi]
    ptdist = ["norm", "unif", _ptdist_evenly]
    ptoff = ["n-1", "n", _ptoff_ones]
    csz = [clusizes, _csz_equi_size]
    cctr = [clucenters, _cctr_diag]
    llen = [llengths, _llen_0_20]
    lang = [angle_deltas, _lang_zeros]

else:
    # Full testing mode, can take a long time
    tm = "full"
    seeds = [0, 123, 6789, 9876543]
    t_ndims = [1, 2, 3, 5, 10, 30]
    t_num_points = [1, 10, 500, 2500, 10000]
    t_num_clusters = [1, 2, 5, 10, 50, 100]
    t_lat_std = [0.0, 5.0, 10.0, 500.0]
    t_angle_std = [0, pi / 128, pi / 32, pi / 4, pi / 2, pi, 2 * 3 / pi, 2 * pi]
    ptdist = ["norm", "unif", _ptdist_evenly]
    ptoff = ["n-1", "n", _ptoff_ones]
    csz = [clusizes, _csz_equi_size]
    cctr = [clucenters, _cctr_diag]
    llen = [llengths, _llen_0_20]
    lang = [angle_deltas, _lang_zeros]


def pytest_report_header(config):
    """Show pyclugen test mode when running pytest."""
    return f"pyclugen test mode: {tm}"


@pytest.fixture(params=seeds)
def seed(request):
    """Provides PRNG seeds."""
    return request.param


@pytest.fixture()
def prng(seed):
    """Provides random number generators."""
    return Generator(PCG64(seed))


@pytest.fixture(params=t_ndims)
def ndims(request):
    """Provides a number of dimensions."""
    return request.param


@pytest.fixture(params=t_num_points)
def num_points(request):
    """Provides a number of points."""
    return request.param


@pytest.fixture(params=t_num_clusters)
def num_clusters(request):
    """Provides a number of clusters."""
    return request.param


@pytest.fixture(params=t_lat_std)
def lat_std(request):
    """Provides values for lat_std."""
    return request.param


@pytest.fixture(params=[0, 10])
def llength_mu(request):
    """Provides a line length average."""
    return request.param


@pytest.fixture(params=[0, 15])
def llength_sigma(request):
    """Provides a line length dispersion."""
    return request.param


@pytest.fixture(params=[True, False])
def allow_empty(request):
    """Provides the values for the allow_empty parameter."""
    return request.param


@pytest.fixture(params=t_angle_std)
def angle_std(request):
    """Provides an angle standard deviation."""
    return request.param


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


def zeros_ones_and_randoms_factory(std, bitgen):
    """Generates a list of function for generating vectors with a given size.

    The first function generates a vector with zeros.
    The second function generates a vector with ones.
    The remaining functions generate a random vector with elements drawn from the
    normal distribution using the bit generator given in `bitgen` with standard
    deviation given by `std`.
    """
    return [lambda s, isd=0: zeros(s), lambda s, isd=0: ones(s)] + [
        lambda s, isd=osd: std * Generator(bitgen(isd)).normal(size=s) for osd in seeds
    ]


@pytest.fixture(params=zeros_ones_and_randoms_factory(1000, PCG64))
def clu_offset(request, ndims):
    """Provides random cluster offsets."""
    return request.param(ndims)


@pytest.fixture(params=zeros_ones_and_randoms_factory(1000, Philox))
def clu_sep(request, ndims):
    """Provides random cluster separations."""
    return request.param(ndims)


@pytest.fixture(params=ptdist)
def ptdist_fn(request):
    """Provides a point distribution function."""
    return request.param


@pytest.fixture(params=ptoff)
def ptoff_fn(request):
    """Provides a point offset function."""
    return request.param


@pytest.fixture(params=csz)
def csz_fn(request):
    """Provides a cluster sizes function."""
    return request.param


@pytest.fixture(params=cctr)
def cctr_fn(request):
    """Provides a cluster centers function."""
    return request.param


@pytest.fixture(params=llen)
def llen_fn(request):
    """Provides a a line lengths function."""
    return request.param


@pytest.fixture(params=lang)
def lang_fn(request):
    """Provides a line angles function."""
    return request.param
