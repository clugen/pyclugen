# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""This module contains the algorithm module functions."""

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .helper import clupoints_n_1_template, fix_empty, fix_num_points
from .shared import _default_rng


def angle_deltas(
    num_clusters: int, angle_disp: float, rng: Generator = _default_rng
) -> NDArray:
    r"""Get angles between average cluster direction and cluster-supporting lines.

    Determine the angles between the average cluster direction and the
    cluster-supporting lines. These angles are obtained from a wrapped normal
    distribution ( $\mu=0$, $\sigma=$`angle_disp`) with support in the interval
    $\left[-\pi/2,\pi/2\right]$. Note this is different from the standard
    wrapped normal distribution, the support of which is given by the interval
    $\left[-\pi,\pi\right]$.

    Examples:
        >>> from pyclugen import angle_deltas
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
        lines, given in radians in the interval $\left[-\pi/2,\pi/2\right]$.
    """
    # Get random angle differences using the normal distribution
    angles = angle_disp * rng.normal(size=num_clusters)

    # Reduce angle differences to the interval [-π, π]
    angles = np.arctan2(np.sin(angles), np.cos(angles))

    # Make sure angle differences are within interval [-π/2, π/2]
    return np.where(
        np.abs(angles) > np.pi / 2, angles - np.sign(angles) * np.pi / 2, angles
    )


def clucenters(
    num_clusters: int,
    clu_sep: NDArray,
    clu_offset: NDArray,
    rng: Generator = _default_rng,
) -> NDArray:
    r"""Determine cluster centers using the uniform distribution.

    The number of clusters (`num_clusters`) and the average cluster separation
    (`clu_sep`) are taken into account.

    More specifically, let $c=$`num_clusters`, $\mathbf{s}=$`clu_sep.reshape(-1,1)`,
    $\mathbf{o}=$`clu_offset.reshape(-1,1)`, $n=$`clu_sep.size` (i.e., number of
    dimensions). Cluster centers are obtained according to the following equation:

    $$
    \mathbf{C}=c\mathbf{U} \cdot \operatorname{diag}(\mathbf{s}) +
        \mathbf{1}\,\mathbf{o}^T
    $$

    where $\mathbf{C}$ is the $c \times n$ matrix of cluster centers,
    $\mathbf{U}$ is an $c \times n$ matrix of random values drawn from the
    uniform distribution between -0.5 and 0.5, and $\mathbf{1}$ is an $c \times
    1$ vector with all entries equal to 1.

    Examples:
        >>> from pyclugen import clucenters
        >>> from numpy import array
        >>> from numpy.random import Generator, PCG64
        >>> prng = Generator(PCG64(123))
        >>> clucenters(3, array([30,10]), array([-50,50]), rng=prng)
        array([[-33.58833231,  36.61463056],
               [-75.16761145,  40.53115432],
               [-79.1684689 ,  59.3628352 ]])

    Args:
      num_clusters: Number of clusters.
      clu_sep: Average cluster separation ( $n \times 1$ vector).
      clu_offset: Cluster offsets ( $n \times 1$ vector).
      rng: Optional pseudo-random number generator.

    Returns:
        A $c \times n$ matrix containing the cluster centers.
    """
    # Obtain a num_clusters x num_dims matrix of uniformly distributed values
    # between -0.5 and 0.5 representing the relative cluster centers
    ctr_rel = rng.random((num_clusters, clu_sep.size)) - 0.5

    return num_clusters * (ctr_rel @ np.diag(clu_sep)) + clu_offset


def clupoints_n_1(
    projs: NDArray,
    lat_disp: float,
    line_len: float,
    clu_dir: NDArray,
    clu_ctr: NDArray,
    rng: Generator = _default_rng,
) -> NDArray:
    r"""Generate points from their $n$-D projections on a cluster-supporting line.

    Each point is placed on a hyperplane orthogonal to that line and centered at
    the point's projection, using the normal distribution ( $\mu=0$,
    $σ=$`lat_disp`).

    This function's main intended use is by the [`clugen()`][pyclugen.main.clugen]
    function, generating the final points when the `point_dist_fn` parameter is
    set to `"n-1"`.

    Examples:
        >>> from pyclugen import clupoints_n_1, points_on_line
        >>> from numpy import array, linspace
        >>> from numpy.random import Generator, PCG64
        >>> prng = Generator(PCG64(123))
        >>> projs = points_on_line(array([5,5]),     # Get 5 point projections
        ...                        array([1,0]),     # on a 2D line
        ...                        linspace(-4,4,5))
        >>> projs
        array([[1., 5.],
               [3., 5.],
               [5., 5.],
               [7., 5.],
               [9., 5.]])
        >>> clupoints_n_1(projs, 0.5, 1.0, array([1,0]), array([0,0]), rng=prng)
        array([[1.        , 5.49456068],
               [3.        , 5.18389333],
               [5.        , 5.64396263],
               [7.        , 5.09698721],
               [9.        , 5.46011545]])

    Args:
      projs: Point projections on the cluster-supporting line ( $p \times n$ matrix).
      lat_disp: Standard deviation for the normal distribution, i.e., cluster
        lateral dispersion.
      line_len: Length of cluster-supporting line (ignored).
      clu_dir: Direction of the cluster-supporting line.
      clu_ctr: Center position of the cluster-supporting line (ignored).
      rng: Optional pseudo-random number generator.

    Returns:
      Generated points ( $p \times n$ matrix).
    """
    # No blank line allowed here

    # Define function to get distances from points to their projections on the
    # line (i.e., using the normal distribution)
    def dist_fn(clu_num_points, ldisp, rg):
        return ldisp * rg.normal(size=clu_num_points)

    # Use clupoints_n_1_template() to do the heavy lifting
    return clupoints_n_1_template(projs, lat_disp, clu_dir, dist_fn, rng=rng)


def clupoints_n(
    projs: NDArray,
    lat_disp: float,
    line_len: float,
    clu_dir: NDArray,
    clu_ctr: NDArray,
    rng: Generator = _default_rng,
) -> NDArray:
    r"""Generate points from their $n$-D projections on a cluster-supporting line.

    Each point is placed around its projection using the normal distribution
    ( $\mu=0$, $σ=$`lat_disp`).

    This function's main intended use is by the [`clugen()`][pyclugen.main.clugen]
    function, generating the final points when the `point_dist_fn` parameter is
    set to `"n"`.

    Examples:
        >>> from pyclugen import clupoints_n, points_on_line
        >>> from numpy import array, linspace
        >>> from numpy.random import Generator, PCG64
        >>> prng = Generator(PCG64(123))
        >>> projs = points_on_line(array([5,5]),     # Get 5 point projections
        ...                        array([1,0]),     # on a 2D line
        ...                        linspace(-4,4,5))
        >>> projs
        array([[1., 5.],
               [3., 5.],
               [5., 5.],
               [7., 5.],
               [9., 5.]])
        >>> clupoints_n(projs, 0.5, 1.0, array([1,0]), array([0,0]), rng=prng)
        array([[0.50543932, 4.81610667],
               [3.64396263, 5.09698721],
               [5.46011545, 5.2885519 ],
               [6.68176818, 5.27097611],
               [8.84170227, 4.83880544]])

    Args:
      projs: Point projections on the cluster-supporting line ( $p \times n$ matrix).
      lat_disp: Standard deviation for the normal distribution, i.e., cluster
        lateral dispersion.
      line_len: Length of cluster-supporting line (ignored).
      clu_dir: Direction of the cluster-supporting line.
      clu_ctr: Center position of the cluster-supporting line (ignored).
      rng: Optional pseudo-random number generator.

    Returns:
      Generated points ( $p \times n$ matrix).
    """
    # Number of dimensions
    num_dims = clu_dir.size

    # Number of points in this cluster
    clu_num_points = projs.shape[0]

    # Get random displacement vectors for each point projection
    displ = lat_disp * rng.normal(size=(clu_num_points, num_dims))

    # Add displacement vectors to each point projection
    points = projs + displ

    return points


def clusizes(
    num_clusters: int,
    num_points: int,
    allow_empty: bool,
    rng: Generator = _default_rng,
) -> NDArray:
    r"""Determine cluster sizes, i.e., the number of points in each cluster.

    Cluster sizes are determined using the normal distribution (
    $\mu=$`num_points` $/$`num_clusters`, $\sigma=\mu/3$), and then
    assuring that the final cluster sizes add up to `num_points` via the
    [`fix_num_points()`][pyclugen.helper.fix_num_points] function.

    Examples:
        >>> from numpy.random import Generator, PCG64
        >>> from pyclugen import clusizes
        >>> prng = Generator(PCG64(123))
        >>> sizes = clusizes(4, 1000, True, rng=prng)
        >>> sizes
        array([166, 217, 354, 263])
        >>> int(sum(sizes))
        1000

    Args:
      num_clusters: Number of clusters.
      num_points: Total number of points.
      allow_empty: Allow empty clusters?
      rng: Optional pseudo-random number generator.

    Returns:
      Number of points in each cluster (vector of size `num_clusters`).
    """
    # Determine number of points in each cluster using the normal distribution

    # Consider the mean an equal division of points between clusters
    mean = num_points / num_clusters
    # The standard deviation is such that the interval [0, 2 * mean] will contain
    # ≈99.7% of cluster sizes
    std = mean / 3

    # Determine points with the normal distribution
    clu_num_points = std * rng.normal(size=num_clusters) + mean

    # Set negative values to zero
    clu_num_points = np.where(clu_num_points > 0, clu_num_points, 0)

    # Fix imbalances, so that num_points is respected
    if np.sum(clu_num_points) > 0:  # Be careful not to divide by zero
        clu_num_points *= num_points / np.sum(clu_num_points)

    # Round the real values to integers since a cluster sizes is represented by
    # an integer
    clu_num_points = np.rint(clu_num_points).astype(int)

    # Make sure total points is respected, which may not be the case at this time due
    # to rounding
    fix_num_points(clu_num_points, num_points)

    # If empty clusters are not allowed, make sure there aren't any
    if not allow_empty:
        fix_empty(clu_num_points)

    return clu_num_points


def llengths(
    num_clusters: int,
    llength: float,
    llength_disp: float,
    rng: Generator = _default_rng,
) -> NDArray:
    r"""Determine length of cluster-supporting lines.

    Line lengths are determined using the folded normal distribution (
    $\mu=$`llength`, $\sigma=$`llength_disp`).

    Examples:
        >>> from numpy.random import Generator, MT19937
        >>> from pyclugen import llengths
        >>> prng = Generator(MT19937(123))
        >>> llengths(4, 20, 3.5, rng=prng)
        array([19.50968733, 19.92482858, 25.99013804, 18.58029672])

    Args:
      num_clusters: Number of clusters.
      llength: Average line length.
      llength_disp: Line length dispersion.
      rng: Optional pseudo-random number generator.

    Returns:
      Lengths of cluster-supporting lines (vector of size `num_clusters`).
    """
    return np.abs(llength + llength_disp * rng.normal(size=num_clusters))
