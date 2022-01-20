# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Helper functions."""

from numpy import argmax, sum
from numpy.typing import NDArray


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
