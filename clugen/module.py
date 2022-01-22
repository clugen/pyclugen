# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Algorithm module functions."""

from numpy import diag
from numpy.random import Generator
from numpy.typing import NDArray

from .shared import _default_rng


def angle_deltas():
    """Placeholder."""
    pass


def clucenters(
    num_clusters: int,
    clu_sep: NDArray,
    clu_offset: NDArray,
    rng: Generator = _default_rng,
) -> NDArray:
    """Placeholder."""
    # Obtain a num_clusters x num_dims matrix of uniformly distributed values
    # between -0.5 and 0.5 representing the relative cluster centers
    ctr_rel = rng.random((num_clusters, clu_sep.size)) - 0.5

    return num_clusters * (ctr_rel @ diag(clu_sep)) + clu_offset


def clupoints_n_1():
    """Placeholder."""
    pass


def clupoints_n():
    """Placeholder."""
    pass


def clusizes():
    """Placeholder."""
    pass


def llengths():
    """Placeholder."""
    pass
