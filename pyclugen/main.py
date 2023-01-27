# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""This module contains the main `clugen()` function."""

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
    r"""Read-only container for results returned by [`clugen()`][pyclugen.main.clugen].

    The symbols presented in the instances variable below have the following
    meanings:

    - $n$ : Number of dimensions.
    - $p$ : Number of points.
    - $c$ : Number of clusters.
    """

    points: NDArray
    r"""$p \times n$ matrix containing the generated points for all clusters."""

    clusters: NDArray
    r"""Vector of size $p$ indicating the cluster each point in `points`
    belongs to."""

    projections: NDArray
    r"""$p \times n$ matrix with the point projections on the cluster-supporting
    lines."""

    sizes: NDArray
    r"""Vector of size $c$ with the number of points in each cluster."""

    centers: NDArray
    r"""$c \times n$ matrix with the coordinates of the cluster centers."""

    directions: NDArray
    r"""$c \times n$ matrix with the direction of each cluster-supporting line."""

    angles: NDArray
    r"""Vector of size $c$ with the angles between the cluster-supporting lines and
    the main direction."""

    lengths: NDArray
    r"""Vector of size $c$ with the lengths of the cluster-supporting lines."""


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
    proj_dist_fn: Union[str, Callable[[float, int, Generator], NDArray]] = "norm",
    point_dist_fn: Union[
        str, Callable[[NDArray, float, float, NDArray, NDArray, Generator], NDArray]
    ] = "n-1",
    clusizes_fn: Callable[[int, int, bool, Generator], NDArray] = clusizes,
    clucenters_fn: Callable[[int, NDArray, NDArray, Generator], NDArray] = clucenters,
    llengths_fn: Callable[[int, float, float, Generator], NDArray] = llengths,
    angle_deltas_fn: Callable[[int, float, Generator], NDArray] = angle_deltas,
    rng: Generator = _default_rng,
) -> Clusters:
    """Generate multidimensional clusters.

    !!! tip
        This is the main function of the **pyclugen** package, and possibly the
        only function most users will need.

    ## Examples:

        >>> import pyclugen as cg
        >>> import matplotlib.pyplot as plt
        >>> from numpy import pi
        >>> from numpy.random import Generator, PCG64
        >>> rng = Generator(PCG64(321))
        >>> out = cg.clugen(2, 5, 10000, [1, 0.5], pi/16, [10, 40], 10, 1, 2, rng=rng)
        >>> out.centers # What are the cluster centers?
        array([[ 20.02876212,  36.59611434],
               [-15.60290734, -26.52169579],
               [ 23.09775166,  91.66309916],
               [ -5.76816015,  54.9775074 ],
               [ -4.64224681,  78.40990876]])
        >>> plt.scatter(out.points[:,0],
        ...             out.points[:,1],
        ...             c=out.clusters) # doctest: +SKIP
        >>> plt.show() # doctest: +SKIP

    ![clugen](https://user-images.githubusercontent.com/3018963/151056890-c83c9509-b40d-4ab2-a842-f2a4706344c6.png)

    !!! Note
        In the descriptions below, the terms "average" and "dispersion" refer to
        measures of central tendency and statistical dispersion, respectively.
        Their exact meaning depends on several optional arguments.

    Args:
      num_dims: Number of dimensions.
      num_clusters: Number of clusters to generate.
      num_points: Total number of points to generate.
      direction: Average direction of the cluster-supporting lines (vector of size
        `num_dims`).
      angle_disp: Angle dispersion of cluster-supporting lines (radians).
      cluster_sep: Average cluster separation in each dimension (vector of size
        `num_dims`).
      llength: Average length of cluster-supporting lines.
      llength_disp: Length dispersion of cluster-supporting lines.
      lateral_disp: Cluster lateral dispersion, i.e., dispersion of points from their
        projection on the cluster-supporting line.
      allow_empty: Allow empty clusters? `False` by default.
      cluster_offset: Offset to add to all cluster centers (vector of size `num_dims`).
        By default the offset will be equal to `numpy.zeros(num_dims)`.
      proj_dist_fn: Distribution of point projections along cluster-supporting lines,
        with three possible values:

        - `"norm"` (default): Distribute point projections along lines using a normal
          distribution (μ=_line center_, σ=`llength/6`).
        - `"unif"`: Distribute points uniformly along the line.
        - User-defined function, which accepts two parameters, line length (`float`),
          number of points (`int`) and an instance of
          [`Generator`](https://numpy.org/doc/stable/reference/random/generator.html?highlight=generator#numpy.random.Generator),
          and returns an array containing the distance of each point projection to
          the center of the line. For example, the `"norm"` option roughly corresponds
          to `lambda l, n, rg: l * rg.random((n, 1)) / 6`.

      point_dist_fn: Controls how the final points are created from their projections
        on the cluster-supporting lines, with three possible values:

        - `"n-1"` (default): Final points are placed on a hyperplane orthogonal to
          the cluster-supporting line, centered at each point's projection, using the
          normal distribution (μ=0, σ=`lateral_disp`). This is done by the
          [`clupoints_n_1()`][pyclugen.module.clupoints_n_1] function.
        - `"n"`: Final points are placed around their projection on the
          cluster-supporting line using the normal distribution (μ=0,
          σ=`lateral_disp`). This is done by the
          [`clupoints_n()`][pyclugen.module.clupoints_n] function.
        - User-defined function: The user can specify a custom point placement
          strategy by passing a function with the same signature as
          [`clupoints_n_1()`][pyclugen.module.clupoints_n_1] and
          [`clupoints_n()`][pyclugen.module.clupoints_n].

      clusizes_fn: Distribution of cluster sizes. By default, cluster sizes are
        determined by the [`clusizes()`][pyclugen.module.clusizes] function, which
        uses the normal distribution (μ=`num_points`/`num_clusters`, σ=μ/3), and
        assures that the final cluster sizes add up to `num_points`. This parameter
        allows the user to specify a custom function for this purpose, which must
        follow [`clusizes()`][pyclugen.module.clusizes] signature. Note that custom
        functions are not required to strictly obey the `num_points` parameter.
      clucenters_fn: Distribution of cluster centers. By default, cluster centers
        are determined by the [`clucenters()`][pyclugen.module.clucenters] function,
        which uses the uniform distribution, and takes into account the `num_clusters`
        and `cluster_sep` parameters for generating well-distributed cluster centers.
        This parameter allows the user to specify a custom function for this purpose,
        which must follow [`clucenters()`][pyclugen.module.clucenters] signature.
      llengths_fn: Distribution of line lengths. By default, the lengths of
        cluster-supporting lines are determined by the
        [`llengths()`][pyclugen.module.llengths] function, which uses the folded
        normal distribution (μ=`llength`, σ=`llength_disp`). This parameter allows
        the user to specify a custom function for this purpose, which must follow
        [`llengths()`][pyclugen.module.llengths] signature.
      angle_deltas_fn: Distribution of line angle differences with respect to
        `direction`. By default, the angles between `direction` and the direction of
        cluster-supporting lines are determined by the
        [`angle_deltas()`][pyclugen.module.angle_deltas] function, which uses the
        wrapped normal distribution (μ=0, σ=`angle_disp`) with support in the interval
        [-π/2, π/2]. This parameter allows the user to specify a custom function for
        this purpose, which must follow [`angle_deltas()`][pyclugen.module.angle_deltas]
        signature.
      rng: An optional instance of [`Generator`][numpy.random.Generator] for
        reproducible executions.

    Returns:
      The generated clusters and associated information in the form of a
        [`Clusters`][pyclugen.main.Clusters] object.
    """
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
    arrdir = asarray(direction)
    if arrdir.size != num_dims:
        raise ValueError(
            "Length of `direction` must be equal to `num_dims` "
            + f"({arrdir.size} != {num_dims})"
        )

    # If allow_empty is false, make sure there are enough points to distribute
    # by the clusters
    if (not allow_empty) and num_points < num_clusters:
        raise ValueError(
            f"A total of {num_points} points is not enough for "
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
                + f"({cluster_offset.size} != {num_dims})"
            )

    # Check that proj_dist_fn specifies a valid way for projecting points along
    # cluster-supporting lines i.e., either "norm" (default), "unif" or a
    # user-defined function
    pointproj_fn: Callable[[float, int, Generator], NDArray]

    if callable(proj_dist_fn):
        # Use user-defined distribution; assume function accepts length of line
        # and number of points, and returns a number of points x 1 vector
        pointproj_fn = proj_dist_fn

    elif proj_dist_fn == "unif":
        # Point projections will be uniformly placed along cluster-supporting lines
        def pointproj_fn(length, n, rg):
            return length * rg.random(n) - length / 2

    elif proj_dist_fn == "norm":
        # Use normal distribution for placing point projections along cluster-supporting
        # lines, mean equal to line center, standard deviation equal to 1/6 of line
        # length such that the line length contains ≈99.73% of the points
        def pointproj_fn(length, n, rg):
            return (1.0 / 6.0) * length * rg.normal(size=n)

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
    arrdir = arrdir / norm(arrdir)

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
            arrdir, cluster_angles[i], rng=rng
        )

    # ################################# #
    # Determine points for each cluster #
    # ################################# #

    # Aux. vector with cumulative sum of number of points in each cluster
    cumsum_points = concatenate((asarray([0]), cumsum(cluster_sizes)))

    # Pre-allocate data structures for holding cluster info and points
    point_clusters = zeros(num_points, dtype=int)  # Cluster indices of each point
    point_projections = zeros((num_points, num_dims))  # Point projections on
    #                                                  # cluster-supporting lines
    points = zeros((num_points, num_dims))  # Final points to be generated

    # Loop through clusters and create points for each one
    for i in range(num_clusters):

        # Start and end indexes for points in current cluster
        idx_start = cumsum_points[i]
        idx_end = cumsum_points[i + 1]

        # Update cluster indices of each point
        point_clusters[idx_start:idx_end] = i

        # Determine distance of point projections from the center of the line
        ptproj_dist_fn_center = pointproj_fn(cluster_lengths[i], cluster_sizes[i], rng)

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
