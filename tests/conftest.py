# Copyright (c) 2020, 2021 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Fixtures to be used by test functions."""

import pytest


@pytest.fixture(params=[1, 2, 3, 4, 30])
def ndims(request):
    """Provides the number of dimensions to test with."""
    return request.param
