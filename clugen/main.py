# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Main function."""

from typing import Callable, NamedTuple, Optional, Union

from numpy import array, isclose
from numpy.linalg import norm
from numpy.random import Generator
from numpy.typing import NDArray

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
    direction: NDArray,
    angle_disp: float,
    llength: float,
    llength_disp: float,
    lateral_disp: float,
    allow_empty: bool = False,
    cluster_offset: Optional[NDArray] = None,
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
