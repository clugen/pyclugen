# Copyright (c) 2020, 2021 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Fixtures to be used by test functions."""

import numpy as np
import pytest
from numpy import pi


@pytest.fixture(params=[0, 123, 9999, 9876543])
def seed(request):
    """Provides PRNG seeds."""
    return request.param


@pytest.fixture()
def prng(seed):
    """Provides random number generators."""
    return np.random.default_rng(seed)


@pytest.fixture(params=[1, 2, 3, 4, 30])
def ndims(request):
    """Provides a number of dimensions."""
    return request.param


@pytest.fixture(params=[1, 10, 500, 10000])
def num_points(request):
    """Provides a number of points."""
    return request.param


@pytest.fixture(params=[0, 10])
def llength_mu(request):
    """Provides a line length average."""
    return request.param


@pytest.fixture()
def vector(prng):
    """Provides random vectors."""

    def _vector(numel):
        return prng.random((numel, 1))

    return _vector


@pytest.fixture()
def uvector(prng):
    """Provides random unit vectors."""

    def _uvector(numel):
        v = prng.random((numel, 1))
        return v / np.linalg.norm(v)

    return _uvector


@pytest.fixture(params=[0, pi / 256, pi / 32, pi / 4, pi / 2, pi, 2 * pi])
def angle_std(request):
    """Provides angles."""
    return request.param
