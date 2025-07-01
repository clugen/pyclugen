# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""This module contains the helper functions."""

from typing import Callable

import numpy as np
from numpy.linalg import norm
from numpy.random import Generator
from numpy.typing import NDArray

from .core import rand_ortho_vector
from .shared import _default_rng


def angle_btw(v1: NDArray, v2: NDArray) -> float:
    r"""Angle between two $n$-dimensional vectors.

    Typically, the angle between two vectors `v1` and `v2` can be obtained with:

    ```python
    arccos(dot(u, v) / (norm(u) * norm(v)))
    ```

    However, this approach is numerically unstable. The version provided here is
    numerically stable and based on the
    [AngleBetweenVectors](https://github.com/JeffreySarnoff/AngleBetweenVectors.jl)
    Julia package by Jeffrey Sarnoff (MIT license), implementing an algorithm
    provided by Prof. W. Kahan in
    [these notes](https://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf)
    (see page 15).

    Examples:
        >>> from numpy import array, degrees
        >>> from pyclugen import angle_btw
        >>> v1 = array([1.0, 1.0, 1.0, 1.0])
        >>> v2 = array([1.0, 0.0, 0.0, 0.0])
        >>> float(degrees(angle_btw(v1, v2)))
        60.00000000000001

    Args:
      v1: First vector.
      v2: Second vector.

    Returns:
      Angle between `v1` and `v2` in radians.
    """
    u1 = v1 / norm(v1)
    u2 = v2 / norm(v2)

    y = u1 - u2
    x = u1 + u2

    return 2 * np.arctan(norm(y) / norm(x))


def clupoints_n_1_template(
    projs: NDArray,
    lat_disp: float,
    clu_dir: NDArray,
    dist_fn: Callable[[int, float, Generator], NDArray],
    rng: Generator = _default_rng,
) -> NDArray:
    r"""Create $p$ points from their $n$-D projections on a cluster-supporting line.

    Each point is placed on a hyperplane orthogonal to that line and centered at
    the point's projection. The function specified in `dist_fn` is used to perform
    the actual placement.

    This function is used internally by
    [`clupoints_n_1()`][pyclugen.module.clupoints_n_1] and may be useful for
    constructing user-defined final point placement strategies for the `point_dist_fn`
    parameter of the main [`clugen()`][pyclugen.main.clugen] function.

    Examples:
        >>> from numpy import array, zeros
        >>> from numpy.random import Generator, PCG64
        >>> from pyclugen import clupoints_n_1_template, points_on_line
        >>> ctr = zeros(2)
        >>> dir = array([1, 0])
        >>> pdist = array([-0.5, -0.2, 0.1, 0.3])
        >>> rng = Generator(PCG64(123))
        >>> proj = points_on_line(ctr, dir, pdist)
        >>> clupoints_n_1_template(proj, 0, dir, lambda p, l, r: r.random(p), rng=rng)
        array([[-0.5       ,  0.68235186],
               [-0.2       , -0.05382102],
               [ 0.1       ,  0.22035987],
               [ 0.3       , -0.18437181]])

    Args:
      projs: Point projections on the cluster-supporting line ( $p \times n$ matrix).
      lat_disp: Dispersion of points from their projection.
      clu_dir: Direction of the cluster-supporting line (unit vector).
      dist_fn: Function to place points on a second line, orthogonal to the first.
        The functions accepts as parameters the number of points in the current
        cluster, the `lateral_disp` parameter (the same passed to the
        [`clugen()`][pyclugen.main.clugen] function), and a random number generator,
        returning a vector containing the distance of each point to its projection
        on the cluster-supporting line.
      rng: An optional pseudo-random number generator for reproducible executions.

    Returns:
      Generated points ( $p \times n$ matrix).
    """
    # Number of dimensions
    num_dims = clu_dir.size

    # Number of points in this cluster
    clu_num_points = projs.shape[0]

    # Get distances from points to their projections on the line
    points_dist = dist_fn(clu_num_points, lat_disp, rng)

    # Get normalized vectors, orthogonal to the current line, for each point
    orth_vecs = np.zeros((clu_num_points, num_dims))

    for j in range(clu_num_points):
        orth_vecs[j, :] = rand_ortho_vector(clu_dir, rng=rng).ravel()

    # Set vector magnitudes
    orth_vecs = np.abs(points_dist).reshape(-1, 1) * orth_vecs

    # Add perpendicular vectors to point projections on the line,
    # yielding final cluster points
    points = projs + orth_vecs

    return points


def fix_empty(clu_num_points: NDArray, allow_empty: bool = False) -> NDArray:
    r"""Certifies that, given enough points, no clusters are left empty.

    This is done by removing a point from the largest cluster and adding it to an
    empty cluster while there are empty clusters. If the total number of points is
    smaller than the number of clusters (or if the `allow_empty` parameter is set
    to `true`), this function does nothing.

    This function is used internally by [`clusizes()`][pyclugen.module.clusizes]
    and might be useful for custom cluster sizing implementations given as the
    `clusizes_fn` parameter of the main [`clugen()`][pyclugen.main.clugen] function.

    Note that the array is changed in-place.

    Examples:
        >>> from numpy import array
        >>> from pyclugen import fix_empty
        >>> clusters = array([3, 4, 5, 0, 0])
        >>> fix_empty(clusters)
        array([3, 3, 4, 1, 1])
        >>> clusters # Verify that the array was changed in-place
        array([3, 3, 4, 1, 1])

    Args:
      clu_num_points: Number of points in each cluster (vector of size $c$),
        where $c$ is the number of clusters.
      allow_empty: Allow empty clusters?

    Returns:
      Number of points in each cluster, after being fixed by this function (vector
        of size $c$, which is the same reference than `clu_num_points`).
    """
    # If the allow_empty parameter is set to true, don't do anything and return
    # immediately; this is useful for quick `clusizes_fn` one-liners
    if not allow_empty:
        # Find empty clusters
        empty_clusts = [idx for idx, val in enumerate(clu_num_points) if val == 0]

        # If there are empty clusters and enough points for all clusters...
        if len(empty_clusts) > 0 and np.sum(clu_num_points) >= clu_num_points.size:
            # Go through the empty clusters...
            for i0 in empty_clusts:
                # ...get a point from the largest cluster and assign it to the
                # current empty cluster
                imax = np.argmax(clu_num_points)
                clu_num_points[imax] -= 1
                clu_num_points[i0] += 1

    return clu_num_points


def fix_num_points(clu_num_points: NDArray, num_points: int) -> NDArray:
    r"""Certifies that the values in the `clu_num_points` array add up to `num_points`.

    If this is not the case, the `clu_num_points` array is modified in-place,
    incrementing the value corresponding to the smallest cluster while
    `sum(clu_num_points) < num_points`, or decrementing the value corresponding to
    the largest cluster while `sum(clu_num_points) > num_points`.

    This function is used internally by [`clusizes()`][pyclugen.module.clusizes]
    and might be useful for custom cluster sizing implementations given as the
    `clusizes_fn` parameter of the main [`clugen()`][pyclugen.main.clugen] function.

    Examples:
        >>> from numpy import array
        >>> from pyclugen import fix_num_points
        >>> clusters = array([1, 6, 3])  # 10 total points
        >>> fix_num_points(clusters, 12) # But we want 12 total points
        array([3, 6, 3])
        >>> clusters # Verify that the array was changed in-place
        array([3, 6, 3])

    Args:
      clu_num_points: Number of points in each cluster (vector of size $c$),
        where $c$ is the number of clusters.
      num_points: The expected total number of points.

    Returns:
      Number of points in each cluster, after being fixed by this function (vector
        of size $c$, which is the same reference than `clu_num_points`).
    """
    while np.sum(clu_num_points) < num_points:
        imin = np.argmin(clu_num_points)
        clu_num_points[imin] += 1
    while np.sum(clu_num_points) > num_points:
        imax = np.argmax(clu_num_points)
        clu_num_points[imax] -= 1

    return clu_num_points
