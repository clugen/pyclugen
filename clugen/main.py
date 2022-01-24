# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Main function."""

from typing import Callable, NamedTuple, Optional, Union

from numpy import asarray, concatenate, cumsum, isclose, sum, zeros
from numpy.linalg import norm
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from .core import points_on_line, rand_vector_at_angle
from .module import (
    angle_deltas,
    clucenters,
    clupoints_n,
    clupoints_n_1,
    clusizes,
    llengths,
)
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

    # Check that proj_dist_fn specifies a valid way for projecting points along
    # cluster-supporting lines i.e., either "norm" (default), "unif" or a
    # user-defined function
    pointproj_fn: Callable[[float, int], NDArray]

    if callable(proj_dist_fn):
        # Use user-defined distribution; assume function accepts length of line
        # and number of points, and returns a number of points x 1 vector
        pointproj_fn = proj_dist_fn

    elif proj_dist_fn == "unif":
        # Point projections will be uniformly placed along cluster-supporting lines
        def pointproj_fn(length, n):
            return length * rng.random(n) - len / 2

    elif proj_dist_fn == "norm":
        # Use normal distribution for placing point projections along cluster-supporting
        # lines, mean equal to line center, standard deviation equal to 1/6 of line
        # length such that the line length contains ≈99.73% of the points
        def pointproj_fn(length, n):
            return (1.0 / 6.0) * length * rng.normal(size=n)

    else:
        raise ValueError(
            "`proj_dist_fn` has to be either 'norm', 'unif' or user-defined function"
        )

    # Check that point_dist_fn specifies a valid way for generating points given
    # their projections along cluster-supporting lines, i.e., either "n-1"
    # (default), "n" or a user-defined function
    pt_from_proj_fn: Callable[
        [NDArray, float, float, NDArray, NDArray, Generator], NDArray
    ]

    if num_dims == 1:
        # If 1D was specified, point projections are the points themselves
        def pt_from_proj_fn(projs, lat_disp, length, clu_dir, clu_ctr, rng=rng):
            return projs

    elif callable(point_dist_fn):
        # Use user-defined distribution; assume function accepts point projections
        # on the line, lateral disp., cluster direction and cluster center, and
        # returns a num_points x num_dims matrix containing the final points
        # for the current cluster
        pt_from_proj_fn = point_dist_fn

    elif point_dist_fn == "n-1":
        # Points will be placed on a hyperplane orthogonal to the cluster-supporting
        # line using a normal distribution centered at their intersection
        pt_from_proj_fn = clupoints_n_1

    elif point_dist_fn == "n":
        # Points will be placed using a multivariate normal distribution
        # centered at the point projection
        pt_from_proj_fn = clupoints_n

    else:
        raise ValueError(
            "point_dist_fn has to be either 'n-1', 'n' or a user-defined function"
        )

    # ############################ #
    # Determine cluster properties #
    # ############################ #

    # Normalize main direction
    direction /= norm(direction)

    # Determine cluster sizes
    cluster_sizes = clusizes_fn(num_clusters, num_points, allow_empty, rng)

    # Custom clusizes_fn's are not required to obey num_points, so we update
    # it here just in case it's different from what the user specified
    num_points = sum(cluster_sizes)

    # Determine cluster centers
    cluster_centers = clucenters_fn(num_clusters, cluster_sep, cluster_offset, rng)

    # Determine length of lines supporting clusters
    cluster_lengths = llengths_fn(num_clusters, llength, llength_disp, rng)

    # Obtain angles between main direction and cluster-supporting lines
    cluster_angles = angle_deltas_fn(num_clusters, angle_disp, rng)

    # Determine normalized cluster directions
    cluster_directions = zeros((num_clusters, num_dims))
    for i in range(num_clusters):
        cluster_directions[i, :] = rand_vector_at_angle(
            direction, cluster_angles[i], rng=rng
        )

    # ################################# #
    # Determine points for each cluster #
    # ################################# #

    # Aux. vector with cumulative sum of number of points in each cluster
    cumsum_points = concatenate(([0], cumsum(cluster_sizes)))

    # Pre-allocate data structures for holding cluster info and points
    point_clusters = zeros(num_points, dtype=int)  # Cluster indices of each point
    point_projections = zeros((num_points, num_dims))  # Point projections on
    #                                                  # cluster-supporting lines
    points = zeros((num_points, num_dims))  # Final points to be generated

    # Loop through clusters and create points for each one
    for i in range(num_clusters):

        # Start and end indexes for points in current cluster
        idx_start = cumsum_points[i] + 1
        idx_end = cumsum_points[i + 1]

        # Update cluster indices of each point
        point_clusters[idx_start:idx_end] = i

        # Determine distance of point projections from the center of the line
        ptproj_dist_fn_center = pointproj_fn(cluster_lengths[i], cluster_sizes[i])

        # Determine coordinates of point projections on the line using the
        # parametric line equation (this works since cluster direction is normalized)
        point_projections[idx_start:idx_end, :] = points_on_line(
            cluster_centers[i, :], cluster_directions[i, :], ptproj_dist_fn_center
        )

        # Determine points from their projections on the line
        points[idx_start:idx_end, :] = pt_from_proj_fn(
            point_projections[idx_start:idx_end, :],
            lateral_disp,
            cluster_lengths[i],
            cluster_directions[i, :],
            cluster_centers[i, :],
            rng,
        )

    return Clusters(
        points,
        point_clusters,
        point_projections,
        cluster_sizes,
        cluster_centers,
        cluster_directions,
        cluster_angles,
        cluster_lengths,
    )
