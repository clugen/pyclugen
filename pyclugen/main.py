# Copyright (c) 2020-2025 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""This module contains the main `clugen()` function."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, MutableSet
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, cast

import numpy as np
from numpy.linalg import norm
from numpy.random import PCG64, Generator
from numpy.typing import ArrayLike, DTypeLike, NDArray

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
    proj_dist_fn: str | Callable[[float, int, Generator], NDArray] = "norm",
    point_dist_fn: (
        str | Callable[[NDArray, float, float, NDArray, NDArray, Generator], NDArray]
    ) = "n-1",
    clusizes_fn: Callable[[int, int, bool, Generator], NDArray] | ArrayLike = clusizes,
    clucenters_fn: (
        Callable[[int, NDArray, NDArray, Generator], NDArray] | ArrayLike
    ) = clucenters,
    llengths_fn: (
        Callable[[int, float, float, Generator], NDArray] | ArrayLike
    ) = llengths,
    angle_deltas_fn: (
        Callable[[int, float, Generator], NDArray] | ArrayLike
    ) = angle_deltas,
    rng: int | np.integer | Generator = _default_rng,
) -> Clusters:
    """Generate multidimensional clusters.

    !!! tip
        This is the main function of the **pyclugen** package, and possibly the
        only function most users will need.

    ## Examples:

        >>> import matplotlib.pyplot as plt
        >>> from pyclugen import clugen
        >>> from numpy import pi
        >>> out = clugen(2, 5, 10000, [1, 0.5], pi/16, [10, 40], 10, 1, 2, rng=321)
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
      direction: Average direction of the cluster-supporting lines. Can be a
        vector of length `num_dims` (same direction for all clusters) or a
        matrix of size `num_clusters` x `num_dims` (one direction per cluster).
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
        - User-defined function, which accepts three parameters, line length (`float`),
          number of points (`int`), and an instance of
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
        Alternatively, the user can specify an array of cluster sizes directly.
      clucenters_fn: Distribution of cluster centers. By default, cluster centers
        are determined by the [`clucenters()`][pyclugen.module.clucenters] function,
        which uses the uniform distribution, and takes into account the `num_clusters`
        and `cluster_sep` parameters for generating well-distributed cluster centers.
        This parameter allows the user to specify a custom function for this purpose,
        which must follow [`clucenters()`][pyclugen.module.clucenters] signature.
        Alternatively, the user can specify a matrix of size `num_clusters` x
        `num_dims` with the exact cluster centers.
      llengths_fn: Distribution of line lengths. By default, the lengths of
        cluster-supporting lines are determined by the
        [`llengths()`][pyclugen.module.llengths] function, which uses the folded
        normal distribution (μ=`llength`, σ=`llength_disp`). This parameter allows
        the user to specify a custom function for this purpose, which must follow
        [`llengths()`][pyclugen.module.llengths] signature. Alternatively, the user
        can specify an array of line lengths directly.
      angle_deltas_fn: Distribution of line angle differences with respect to
        `direction`. By default, the angles between `direction` and the direction of
        cluster-supporting lines are determined by the
        [`angle_deltas()`][pyclugen.module.angle_deltas] function, which uses the
        wrapped normal distribution (μ=0, σ=`angle_disp`) with support in the interval
        [-π/2, π/2]. This parameter allows the user to specify a custom function for
        this purpose, which must follow [`angle_deltas()`][pyclugen.module.angle_deltas]
        signature. Alternatively, the user can specify an array of angle deltas
        directly.
      rng: The seed for the random number generator or an instance of
        [`Generator`][numpy.random.Generator] for reproducible executions.

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

    # Convert given direction into a NumPy array
    arrdir: NDArray = np.asarray(direction)

    # Get number of dimensions in `direction` array
    dir_ndims = arrdir.ndim

    # Is direction a vector or a matrix?
    if dir_ndims == 1:
        # It's a vector, let's convert it into a row matrix, since this will be
        # useful down the road
        arrdir = arrdir.reshape((1, -1))
    elif dir_ndims == 2:
        # If a matrix was given (i.e. a main direction is given for each cluster),
        # check if the number of directions is the same as the number of clusters
        dir_size_1 = arrdir.shape[0]
        if dir_size_1 != num_clusters:
            raise ValueError(
                "Number of rows in `direction` must be the same as the "
                + f"number of clusters ({dir_size_1} != {num_clusters})"
            )
    else:
        # The `directions` array must be a vector or a matrix, so if we get here
        # it means we have invalid arguments
        raise ValueError(
            "`direction` must be a vector (1D array) or a matrix (2D array), "
            + f"but is {dir_ndims}D"
        )

    # Check that direction has num_dims dimensions
    dir_size_2 = arrdir.shape[1]
    if dir_size_2 != num_dims:
        raise ValueError(
            "Length of directions in `direction` must be equal to "
            + f"`num_dims` ({dir_size_2} != {num_dims})"
        )

    # Check that directions have magnitude > 0
    dir_magnitudes = np.apply_along_axis(norm, 1, arrdir)
    if np.any(np.isclose(dir_magnitudes, 0)):
        raise ValueError("Directions in `direction` must have magnitude > 0")

    # If allow_empty is false, make sure there are enough points to distribute
    # by the clusters
    if (not allow_empty) and num_points < num_clusters:
        raise ValueError(
            f"A total of {num_points} points is not enough for "
            + f"{num_clusters} non-empty clusters"
        )

    # Check that cluster_sep has num_dims dimensions
    cluster_sep = np.asarray(cluster_sep)
    if cluster_sep.size != num_dims:
        raise ValueError(
            "Length of `cluster_sep` must be equal to `num_dims` "
            + f"({cluster_sep.size} != {num_dims})"
        )

    # If given, cluster_offset must have the correct number of dimensions,
    # if not given then it will be a num_dims x 1 vector of zeros
    if cluster_offset is None:
        cluster_offset = np.zeros(num_dims)
    else:
        cluster_offset = np.asarray(cluster_offset)
        if cluster_offset.size != num_dims:
            raise ValueError(
                "Length of `cluster_offset` must be equal to `num_dims` "
                + f"({cluster_offset.size} != {num_dims})"
            )

    # If the user specified rng as an int, create a proper rng object
    rng_sel: Generator
    if isinstance(rng, Generator):
        rng_sel = cast(Generator, rng)
    elif isinstance(rng, (int, np.integer)):
        rng_sel = Generator(PCG64(cast(int, rng)))
    else:
        raise ValueError(
            f"`rng` must be an instance of int or Generator, but is {type(rng)}"
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
        def pt_from_proj_fn(projs, lat_disp, length, clu_dir, clu_ctr, rng=rng_sel):
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

    # Normalize main direction(s)
    arrdir = np.apply_along_axis(lambda a: a / norm(a), 1, arrdir)

    # If only one main direction was given, expand it for all clusters
    if dir_ndims == 1:
        arrdir = np.repeat(arrdir, num_clusters, axis=0)

    # Determine cluster sizes
    if callable(clusizes_fn):
        cluster_sizes = clusizes_fn(num_clusters, num_points, allow_empty, rng_sel)
    elif len(np.asarray(clusizes_fn)) == num_clusters:
        cluster_sizes = np.asarray(clusizes_fn)
    else:
        raise ValueError(
            "clusizes_fn has to be either a function or a `num_clusters`-sized array"
        )

    # Custom clusizes_fn's are not required to obey num_points, so we update
    # it here just in case it's different from what the user specified
    num_points = np.sum(cluster_sizes)

    # Determine cluster centers
    if callable(clucenters_fn):
        cluster_centers = clucenters_fn(
            num_clusters, cluster_sep, cluster_offset, rng_sel
        )
    elif np.asarray(clucenters_fn).shape == (num_clusters, num_dims):
        cluster_centers = np.asarray(clucenters_fn)
    else:
        raise ValueError(
            "clucenters_fn has to be either a function or a matrix of size "
            + "`num_clusters` x `num_dims`"
        )

    # Determine length of lines supporting clusters
    if callable(llengths_fn):
        cluster_lengths = llengths_fn(num_clusters, llength, llength_disp, rng_sel)
    elif len(np.asarray(llengths_fn)) == num_clusters:
        cluster_lengths = np.asarray(llengths_fn)
    else:
        raise ValueError(
            "llengths_fn has to be either a function or a `num_clusters`-sized array"
        )

    # Obtain angles between main direction and cluster-supporting lines
    if callable(angle_deltas_fn):
        cluster_angles = angle_deltas_fn(num_clusters, angle_disp, rng_sel)
    elif len(np.asarray(angle_deltas_fn)) == num_clusters:
        cluster_angles = np.asarray(angle_deltas_fn)
    else:
        raise ValueError(
            "angle_deltas_fn has to be either a function or a "
            + "`num_clusters`-sized array"
        )

    # Determine normalized cluster directions by applying the obtained angles
    cluster_directions = np.apply_along_axis(
        lambda v, a: rand_vector_at_angle(v, next(a), rng_sel),
        1,
        arrdir,
        iter(cluster_angles),
    )

    # ################################# #
    # Determine points for each cluster #
    # ################################# #

    # Aux. vector with cumulative sum of number of points in each cluster
    cumsum_points = np.concatenate((np.asarray([0]), np.cumsum(cluster_sizes)))

    # Pre-allocate data structures for holding cluster info and points
    point_clusters: NDArray = np.empty(
        num_points, dtype=np.int32
    )  # Cluster indices of each point
    point_projections = np.empty((num_points, num_dims))  # Point projections on
    #                                                  # cluster-supporting lines
    points = np.empty((num_points, num_dims))  # Final points to be generated

    # Loop through clusters and create points for each one
    for i in range(num_clusters):
        # Start and end indexes for points in current cluster
        idx_start = cumsum_points[i]
        idx_end = cumsum_points[i + 1]

        # Update cluster indices of each point
        point_clusters[idx_start:idx_end] = i

        # Determine distance of point projections from the center of the line
        ptproj_dist_fn_center = pointproj_fn(
            cluster_lengths[i], cluster_sizes[i], rng_sel
        )

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
            rng_sel,
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


@dataclass
class _FieldInfo:
    """Field information for merging datasets."""

    dtype: DTypeLike
    """The field data type, may be promoted when merging."""

    ncol: int
    """Number of columns in the data."""


def _getcols(a: NDArray) -> int:
    """Get number of columns from a NumPy array, returns 1 if just a vector."""
    return a.shape[1] if len(a.shape) > 1 else 1


def clumerge(
    *data: NamedTuple | Mapping[str, ArrayLike],
    fields: tuple[str, ...] = ("points", "clusters"),
    clusters_field: str | None = "clusters",
) -> dict[str, NDArray]:
    r"""Merges the fields (specified in `fields`) of two or more `data` sets.

    Merges the fields (specified in `fields`) of two or more `data` sets (named
    tuples or dictionaries). The fields to be merged need to have the same
    number of columns. The corresponding merged field will contain the rows of
    the fields to be merged, and will have a common supertype.

    The `clusters_field` parameter specifies a field containing integers that
    identify the cluster to which the respective points belongs to. If
    `clusters_field` is specified (by default it's specified as `"clusters"`),
    cluster assignments in individual datasets will be updated in the merged
    dataset so that clusters are considered separate. This parameter can be set
    to `None`, in which case no field will be considered as a special cluster
    assignments field.

    This function can be used to merge data sets generated with the
    [`clugen()`][pyclugen.main.clugen] function, by default merging the
    `points` and `clusters` fields in those data sets. It also works with
    arbitrary data by specifying alternative fields in the `fields` parameter.
    It can be used, for example, to merge third-party data with
    [`clugen()`][pyclugen.main.clugen]-generated data.

    Examples:
        >>> from pyclugen import clugen, clumerge
        >>> data1 = clugen(2, 5, 1000, [1, 1], 0.01, [20, 20], 14, 1.2, 1.5);
        >>> data2 = clugen(2, 3, 450, [0.8, -0.3], 0, [25, 21], 6, 0.4, 3.5);
        >>> data3 = clugen(2, 2, 600, [0, -0.7], 0.2, [15, 10], 1, 0.1, 5.2);
        >>> data_merged = clumerge(data1, data2, data3)

    Args:
      *data: One or more cluster data sets whose `fields` are to be merged.
      fields: Fields to be merged, which must exist in the data set given in
        `*data`.
      clusters_field: Field containing the integer cluster labels. If specified,
        cluster assignments in individual datasets will be updated in the merged
        dataset so that clusters are considered separate.

    Returns:
      A dictionary, where keys correspond to field names, and values to the
        merged numerical arrays.
    """
    # Number of elements in each array the merged dataset
    numel: int = 0

    # Number of columns of values in each field
    fields_info: dict[str, _FieldInfo] = {}

    # Merged dataset to output, initially empty
    output: dict[str, NDArray] = {}

    # Create a fields set
    fields_set: MutableSet[str] = set(fields)

    # If a clusters field is given, add it
    if clusters_field is not None:
        fields_set.add(str(clusters_field))

    # Data in dictionary format with NDArray views on data
    ddata: MutableSequence[Mapping[str, NDArray]] = []
    for dt in data:
        # If dt is a named tuple, convert it into a dictionary
        ddt: Mapping[str, ArrayLike]
        if isinstance(dt, dict):
            ddt = cast(dict, dt)
        else:
            ntdt = cast(NamedTuple, dt)
            ddt = ntdt._asdict()

        # Convert dictionary values to NDArrays
        ddtnp: Mapping[str, NDArray] = {k: np.asarray(v) for k, v in ddt.items()}

        # Add converted dictionary to our sequence of dictionaries
        ddata.append(ddtnp)

    # Cycle through data items
    for dt in ddata:
        # Number of elements in the current item
        numel_i: int = -1

        # Cycle through fields for the current item
        for field in fields_set:
            if field not in dt:
                raise ValueError(f"Data item does not contain required field `{field}`")
            elif field == clusters_field and not np.can_cast(
                dt[clusters_field].dtype, np.int64
            ):
                raise ValueError(f"`{clusters_field}` must contain integer types")

            # Get the field value
            value: NDArray = dt[field]

            # Number of elements in field value
            numel_tmp = len(value)

            # Check the number of elements in the field value
            if numel_i == -1:
                # First field: get number of elements in value (must be the same
                # for the remaining field values)
                numel_i = numel_tmp

            elif numel_tmp != numel_i:
                # Fields values after the first must have the same number of
                # elements
                raise ValueError(
                    "Data item contains fields with different sizes "
                    + f"({numel_tmp} != {numel_i})"
                )

            # Get/check info about the field value type
            if field not in fields_info:
                # If it's the first time this field appears, just get the info
                fields_info[field] = _FieldInfo(value.dtype, _getcols(value))

            else:
                # If this field already appeared in previous data items, get the
                # info and check/determine its compatibility with respect to
                # previous data items
                if _getcols(value) != fields_info[field].ncol:
                    # Number of columns must be the same
                    raise ValueError(f"Dimension mismatch in field `{field}`")

                # Get the common supertype
                fields_info[field].dtype = np.promote_types(
                    fields_info[field].dtype, value.dtype
                )

        # Update total number of elements
        numel += numel_i

    # Initialize output dictionary fields with room for all items
    for field in fields_info:
        if fields_info[field].ncol == 1:
            output[field] = np.empty((numel,), dtype=fields_info[field].dtype)
        else:
            output[field] = np.empty(
                (numel, fields_info[field].ncol), dtype=fields_info[field].dtype
            )

    # Copy items from input data to output dictionary, field-wise
    copied: int = 0
    last_cluster: int = 0

    # Create merged output
    for dt in ddata:
        # How many elements to copy for the current data item?
        tocopy: int = len(dt[fields[0]])

        # Cycle through each field and its information
        for field in fields_info:
            # Copy elements
            if field == clusters_field:
                # If this is a clusters field, update the cluster IDs
                old_clusters = np.unique(dt[clusters_field])
                new_clusters = list(
                    range(last_cluster + 1, last_cluster + len(old_clusters) + 1)
                )
                old2new = zip(old_clusters, new_clusters)
                mapping = dict(old2new)
                last_cluster = new_clusters[-1]

                output[field][copied : (copied + tocopy)] = [
                    mapping[val] for val in dt[clusters_field]
                ]

            else:
                # Otherwise just copy the elements
                ncol: int = fields_info[field].ncol
                output[field].flat[copied * ncol : (copied + tocopy) * ncol] = dt[field]

        # Update how many were copied so far
        copied += tocopy

    # Return result
    return output
