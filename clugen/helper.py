# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Helper functions."""

from typing import Callable

from numpy import abs, argmax, argmin, sum, zeros
from numpy.random import Generator
from numpy.typing import NDArray

from .core import rand_ortho_vector
from .shared import _default_rng


def clupoints_n_1_template(
    projs: NDArray,
    lat_disp: float,
    clu_dir: NDArray,
    dist_fn: Callable[[int, float], NDArray],
    rng: Generator = _default_rng,
) -> NDArray:
    """Placeholder."""
    # Number of dimensions
    num_dims = clu_dir.size

    # Number of points in this cluster
    clu_num_points = projs.shape[0]

    # Get distances from points to their projections on the line
    points_dist = dist_fn(clu_num_points, lat_disp)

    # Get normalized vectors, orthogonal to the current line, for each point
    orth_vecs = zeros((clu_num_points, num_dims))

    for j in range(clu_num_points):
        orth_vecs[j, :] = rand_ortho_vector(clu_dir, rng=rng).ravel()

    # Set vector magnitudes
    orth_vecs = abs(points_dist) * orth_vecs

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

    This function is used internally by `module.clusizes()` and might be useful
    for custom cluster sizing implementations given as the `clusizes_fn` parameter
    of the main `main.clugen()` function.

    Note that the array is changed in-place.

    ## Examples:

    >>> from numpy import array
    >>> from clugen import fix_empty
    >>> clusters = array([3, 4, 5, 0, 0])
    >>> fix_empty(clusters)
    array([3, 3, 4, 1, 1])
    >>> clusters # Verify that the array was changed in-place
    array([3, 3, 4, 1, 1])

    Args:
      clu_num_points: Number of points in each cluster (vector of size \(c\)),
        where \(c\) is the number of clusters.
      allow_empty: Allow empty clusters?

    Returns:
      Number of points in each cluster, after being fixed by this function (vector
      of size \(c\) which is the same reference than `clu_num_points`).
    """
    # If the allow_empty parameter is set to true, don't do anything and return
    # immediately; this is useful for quick `clusizes_fn` one-liners
    if not allow_empty:

        # Find empty clusters
        empty_clusts = [idx for idx, val in enumerate(clu_num_points) if val == 0]

        # If there are empty clusters and enough points for all clusters...
        if len(empty_clusts) > 0 and sum(clu_num_points) >= clu_num_points.size:

            # Go through the empty clusters...
            for i0 in empty_clusts:

                # ...get a point from the largest cluster and assign it to the
                # current empty cluster
                imax = argmax(clu_num_points)
                clu_num_points[imax] -= 1
                clu_num_points[i0] += 1

    return clu_num_points


def fix_num_points(clu_num_points: NDArray, num_points: int) -> NDArray:
    r"""Certifies that the values in the `clu_num_points` array add up to `num_points`.

    If this is not the case, the `clu_num_points` array is modified in-place,
    incrementing the value corresponding to the smallest cluster while
    `sum(clu_num_points) < num_points`, or decrementing the value corresponding to
    the largest cluster while `sum(clu_num_points) > num_points`.

    This function is used internally by `module.clusizes()` and might be useful for
    custom cluster sizing implementations given as the `clusizes_fn` parameter of
    the main `main.clugen()` function.

    ## Examples:

    >>> from numpy import array
    >>> from clugen import fix_num_points
    >>> clusters = array([1, 6, 3])  # 10 total points
    >>> fix_num_points(clusters, 12) # But we want 12 total points
    array([3, 6, 3])
    >>> clusters # Verify that the array was changed in-place
    array([3, 6, 3])

    Args:
      clu_num_points: Number of points in each cluster (vector of size \(c\)),
        where \(c\) is the number of clusters.
      num_points: The expected total number of points.

    Returns:
      Number of points in each cluster, after being fixed by this function (vector
      of size \(c\) which is the same reference than `clu_num_points`).
    """
    while sum(clu_num_points) < num_points:
        imin = argmin(clu_num_points)
        clu_num_points[imin] += 1
    while sum(clu_num_points) > num_points:
        imax = argmax(clu_num_points)
        clu_num_points[imax] -= 1

    return clu_num_points
