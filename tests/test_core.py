# Copyright (c) 2020, 2021 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the core module."""

import numpy as np

import clugen as cg


def test_points_on_line(ndims):
    """Test points_on_line()."""
    ctr = np.ones((ndims, 1))
    direc = np.ones((ndims, 1)) * 2
    dc = np.array([[-1.5, 2, -1.6, 10, 0, 1.1]]).T
    pol = cg.points_on_line(ctr, direc, dc)
    assert pol.shape == (dc.size, ndims)
