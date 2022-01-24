# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Main function."""

from typing import Callable, NamedTuple, Optional, Union

from numpy import array, asarray, isclose, zeros
from numpy.linalg import norm
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from .module import angle_deltas, clucenters, clusizes, llengths
from .shared import _default_rng


class Clusters(NamedTuple):
    """Placeholder."""

    points: NDArray
    """Points."""

    point_clusters: NDArray
    """Point clusters."""

    point_projections: NDArray
    """Point projections."""

    cluster_sizes: NDArray
    """Cluster sizes."""

    cluster_centers: NDArray
    """Cluster centers."""

    cluster_directions: NDArray
    """Cluster directions."""

    cluster_angles: NDArray
    """Cluster angles."""

    cluster_lengths: NDArray
    """Cluster lengths."""


def clugen(
    num_dims: int,
    num_clusters: int,
    num_points: int,
    direction: ArrayLike,
    angle_disp: float,
    cluster_sep: ArrayLike,
    llength: float,
    llength_disp: float,
    lateral_disp: float,
    allow_empty: bool = False,
    cluster_offset: Optional[ArrayLike] = None,
    proj_dist_fn: Union[str, Callable[[float, int], NDArray]] = "norm",
    point_dist_fn: Union[
        str, Callable[[NDArray, float, float, NDArray, NDArray, Generator], NDArray]
    ] = "n-1",
    clusizes_fn: Callable[[int, int, bool, Generator], NDArray] = clusizes,
    clucenters_fn: Callable[[int, NDArray, NDArray, Generator], NDArray] = clucenters,
    llengths_fn: Callable[[int, float, float, Generator], NDArray] = llengths,
    angle_deltas_fn: Callable[[int, float, Generator], NDArray] = angle_deltas,
    rng: Generator = _default_rng,
) -> Clusters:
    """Placeholder."""
    # ############### #
    # Validate inputs #
    # ############### #

    # Check that number of dimensions is > 0
    if num_dims < 1:
        raise ValueError("Number of dimensions, `num_dims`, must be > 0")

    # Check that number of clusters is > 0
    if num_clusters < 1:
        raise ValueError("Number of clusters, `num_clust`, must be > 0")

    # Check that direction vector has magnitude > 0
    if isclose(norm(direction), 0):
        raise ValueError("`direction` must have magnitude > 0")

    # Check that direction has num_dims dimensions
    direction = asarray(direction)
    if direction.size != num_dims:
        raise ValueError(
            "Length of `direction` must be equal to `num_dims` "
            + f"({direction.size} != {num_dims})"
        )

    # If allow_empty is false, make sure there are enough points to distribute
    # by the clusters
    if (not allow_empty) and num_points < num_clusters:
        raise ValueError(
            "A total of $num_points points is not enough for "
            + f"{num_clusters} non-empty clusters"
        )

    # Check that cluster_sep has num_dims dimensions
    cluster_sep = asarray(cluster_sep)
    if cluster_sep.size != num_dims:
        raise ValueError(
            "Length of `cluster_sep` must be equal to `num_dims` "
            + f"({cluster_sep.size} != {num_dims})"
        )

    # If given, cluster_offset must have the correct number of dimensions,
    # if not given then it will be a num_dims x 1 vector of zeros
    if cluster_offset is None:
        cluster_offset = zeros(num_dims)
    else:
        cluster_offset = asarray(cluster_offset)
        if cluster_offset.size != num_dims:
            raise ValueError(
                "Length of `cluster_offset` must be equal to `num_dims` "
                + f"({cluster_offset.size} != {num_dims}"
            )

    return Clusters(
        array([]),
        array([]),
        array([]),
        array([]),
        array([]),
        array([]),
        array([]),
        array([]),
    )
