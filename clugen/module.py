# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Algorithm module functions."""

from numpy import abs, arctan2, cos, diag, pi, sign, sin, where
from numpy.random import Generator
from numpy.typing import NDArray

from .helper import clupoints_n_1_template
from .shared import _default_rng


def angle_deltas(
    num_clusters: int, angle_disp: float, rng: Generator = _default_rng
) -> NDArray:
    r"""Get angles between average cluster direction and cluster-supporting lines.

    Determine the angles between the average cluster direction and the
    cluster-supporting lines. These angles are obtained from a wrapped normal
    distribution ( \(\mu=0\), \(\sigma=\)`angle_disp`) with support in the interval
    \(\left[-\pi/2,\pi/2\right]\). Note this is different from the standard
    wrapped normal distribution, the support of which is given by the interval
    \(\left[-\pi,\pi\right]\).

    ## Examples:

    >>> from clugen import angle_deltas
    >>> from numpy import degrees, pi
    >>> from numpy.random import Generator, PCG64
    >>> prng = Generator(PCG64(123))
    >>> a_rad = angle_deltas(4, pi/8, rng=prng) # Angle dispersion of 22.5 degrees
    >>> a_rad
    array([-0.38842705, -0.14442948,  0.50576707,  0.07617358])
    >>> degrees(a_rad) # Show angle deltas in degrees
    array([-22.25523038,  -8.27519966,  28.97831838,   4.36442443])

    Args:
      num_clusters: Number of clusters.
      angle_disp: Angle dispersion, in radians.
      rng: Optional pseudo-random number generator.

    Returns:
       Angles between the average cluster direction and the cluster-supporting
       lines, given in radians in the interval \(\left[-\pi/2,\pi/2\right]\).
    """
    # Get random angle differences using the normal distribution
    angles = angle_disp * rng.normal(size=num_clusters)

    # Reduce angle differences to the interval [-π, π]
    angles = arctan2(sin(angles), cos(angles))

    # Make sure angle differences are within interval [-π/2, π/2]
    return where(abs(angles) > pi / 2, angles - sign(angles) * pi / 2, angles)


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


def clupoints_n_1(
    projs: NDArray,
    lat_disp: float,
    line_len: float,
    clu_dir: NDArray,
    clu_ctr: NDArray,
    rng: Generator = _default_rng,
) -> NDArray:
    """Placeholder."""
    # Define function to get distances from points to their projections on the
    # line (i.e., using the normal distribution)
    def dist_fn(clu_num_points, ldisp):
        return ldisp * rng.normal(size=clu_num_points)

    # Use clupoints_n_1_template() to do the heavy lifting
    return clupoints_n_1_template(projs, lat_disp, clu_dir, dist_fn, rng=rng)


def clupoints_n():
    """Placeholder."""
    pass


def clusizes():
    """Placeholder."""
    pass


def llengths():
    """Placeholder."""
    pass
