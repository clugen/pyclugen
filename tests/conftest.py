# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Fixtures to be used by test functions."""

import os

import pytest
from numpy import ones, pi, zeros
from numpy.linalg import norm
from numpy.random import PCG64, Generator, Philox

if os.environ.get("CI") is not None:
    seeds = [123]
    t_ndims = [1, 3, 5]
    t_num_points = [1, 10, 500]
    t_num_clusters = [1, 10, 50]
    t_lat_std = [0.0, 10.0]
    t_angle_std = [0, pi / 256, pi, 2 * pi]
else:
    seeds = [0, 123, 9999, 9876543]
    t_ndims = [1, 2, 3, 5, 30]
    t_num_points = [1, 10, 500, 10000]
    t_num_clusters = [1, 2, 5, 10, 100]
    t_lat_std = [0.0, 5.0, 500.0]
    t_angle_std = [0, pi / 256, pi / 32, pi / 4, pi / 2, pi, 2 * pi]


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
