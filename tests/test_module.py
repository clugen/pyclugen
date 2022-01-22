# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Tests for the algorithm module functions."""

from clugen.module import clucenters


def test_clucenters(ndims, prng, num_clusters, clu_offset, clu_sep):
    """Test the clucenters() function."""
    # Get cluster centers with the clucenters() function
    clu_ctrs = clucenters(num_clusters, clu_sep, clu_offset, rng=prng)

    # Check that return value has the correct dimensions
    assert clu_ctrs.shape == (num_clusters, ndims)
