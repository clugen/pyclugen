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
    r"""Determine cluster centers using the uniform distribution.

    The number of clusters (`num_clusters`) and the average cluster separation
    (`clu_sep`) are taken into account.

    More specifically, let \(c=\)`num_clusters`, \(\mathbf{s}=\)`clu_sep.reshape(-1,1)`,
    \(\mathbf{o}=\)`clu_offset.reshape(-1,1)`, \(n=\)`clu_sep.size` (i.e., number of
    dimensions). Cluster centers are obtained according to the following equation:

    $$
    \mathbf{C}=c\mathbf{U} \cdot \operatorname{diag}(\mathbf{s}) +
        \mathbf{1}\,\mathbf{o}^T
    $$

    where \(\mathbf{C}\) is the \(c \times n\) matrix of cluster centers,
    \(\mathbf{U}\) is an \(c \times n\) matrix of random values drawn from the
    uniform distribution between -0.5 and 0.5, and \(\mathbf{1}\) is an \(c \times
    1\) vector with all entries equal to 1.

    ## Examples

    >>> from clugen import clucenters
    >>> from numpy import array
    >>> from numpy.random import Generator, PCG64
    >>> prng = Generator(PCG64(123))
    >>> clucenters(3, array([30,10]), array([-50,50]), rng=prng)
    array([[-33.58833231,  36.61463056],
           [-75.16761145,  40.53115432],
           [-79.1684689 ,  59.3628352 ]])

    Args:
      num_clusters: Number of clusters.
      clu_sep: Average cluster separation ( \(n \times 1\) vector).
      clu_offset: Cluster offsets ( \(n \times 1\) vector).
      rng: Optional pseudo-random number generator.

    Returns:
        A \(c \times n\) matrix containing the cluster centers.
    """
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
