# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Helper functions."""

from numpy import argmax, sum
from numpy.typing import NDArray


def fix_empty(clu_num_points: NDArray, allow_empty: bool = False) -> NDArray:
    """Certifies that, given enough points, no clusters are left empty."""
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
