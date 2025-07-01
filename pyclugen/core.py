# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""This module contains the core functions."""

from math import tan

import numpy as np
from numpy.linalg import norm
from numpy.random import Generator
from numpy.typing import NDArray

from .shared import _default_rng


def points_on_line(
    center: NDArray, direction: NDArray, dist_center: NDArray
) -> NDArray:
    r"""Determine coordinates of points on a line.

    Determine coordinates of points on a line with `center` and `direction`,
    based on the distances from the center given in `dist_center`.

    This works by using the vector formulation of the line equation assuming
    `direction` is a $n$-dimensional unit vector. In other words, considering
    $\mathbf{d}=$`direction.reshape(-1,1)` ( $n \times 1$ vector),
    $\mathbf{c}=$`center.reshape(-1,1)` ( $n \times 1$ vector), and
    $\mathbf{w}=$ `dist_center.reshape(-1,1)` ( $p \times 1$ vector),
    the coordinates of points on the line are given by:

    $$
    \mathbf{P}=\mathbf{1}\,\mathbf{c}^T + \mathbf{w}\mathbf{d}^T
    $$

    where $\mathbf{P}$ is the $p \times n$ matrix of point coordinates on the
    line, and $\mathbf{1}$ is a $p \times 1$ vector with all entries equal to 1.

    Examples:
        >>> from pyclugen import points_on_line
        >>> from numpy import array, linspace
        >>> points_on_line(array([5., 5.]),
        ...                array([1., 0.]),
        ...                linspace(-4, 4, 5)) # 2D, 5 points
        array([[1., 5.],
               [3., 5.],
               [5., 5.],
               [7., 5.],
               [9., 5.]])
        >>> points_on_line(array([-2, 0, 0., 2]),
        ...                array([0., 0, -1, 0]),
        ...                array([10, -10])) # 4D, 2 points
        array([[ -2.,   0., -10.,   2.],
               [ -2.,   0.,  10.,   2.]])

    Args:
      center: Center of the line ( $n$-component vector).
      direction: Line direction ( $n$-component unit vector).
      dist_center: Distance of each point to the center of the line
        ( $p$-component vector, where $p$ is the number of points).

    Returns:
      Coordinates of points on the specified line ( $p \times n$ matrix).
    """
    return center.reshape(1, -1) + dist_center.reshape(-1, 1) @ direction.reshape(
        (1, -1)
    )


def rand_ortho_vector(u: NDArray, rng: Generator = _default_rng) -> NDArray:
    r"""Get a random unit vector orthogonal to `u`.

    Note that `u` is expected to be a unit vector itself.

    Examples:
        >>> from pyclugen import rand_ortho_vector
        >>> from numpy import isclose, dot
        >>> from numpy.linalg import norm
        >>> from numpy.random import Generator, PCG64
        >>> rng = Generator(PCG64(123))
        >>> r = rng.random(3) # Get a random vector with 3 components (3D)
        >>> r = r / norm(r) # Normalize it
        >>> r_ort = rand_ortho_vector(r, rng=rng) # Get random unit vector orth. to r
        >>> r_ort
        array([-0.1982903 , -0.61401512,  0.76398062])
        >>> bool(isclose(dot(r, r_ort), 0)) # Check that vectors are orthogonal
        True

    Args:
      u: Unit vector with $n$ components.
      rng: Optional pseudo-random number generator.

    Returns:
      A random unit vector with $n$ components orthogonal to `u`.
    """
    # If 1D, just return a random unit vector
    if u.size == 1:
        return rand_unit_vector(1, rng=rng)

    # Find a random, non-parallel vector to u
    while True:
        # Find normalized random vector
        r = rand_unit_vector(u.size, rng=rng)

        # If not parallel to u we can keep it and break the loop
        if not np.isclose(np.abs(np.dot(u, r)), 1):
            break

    # Get vector orthogonal to u using 1st iteration of Gram-Schmidt process
    v = r - np.dot(u, r) / np.dot(u, u) * u

    # Normalize it
    v = v / norm(v)

    # And return it
    return v


def rand_unit_vector(num_dims: int, rng: Generator = _default_rng) -> NDArray:
    r"""Get a random unit vector with `num_dims` components.

    Examples:
        >>> from pyclugen import rand_unit_vector
        >>> rand_unit_vector(4) # doctest: +SKIP
        array([ 0.48653889,  0.50753862,  0.05711487, -0.70881757])

        >>> from pyclugen import rand_unit_vector
        >>> from numpy.random import Generator, PCG64
        >>> rng = Generator(PCG64(123))
        >>> rand_unit_vector(2, rng=rng) # Reproducible
        array([ 0.3783202 , -0.92567479])

    Args:
      num_dims: Number of components in vector (i.e. vector size).
      rng: Optional pseudo-random number generator.

    Returns:
      A random unit vector with `num_dims` components.
    """
    r = rng.random(num_dims) - 0.5
    r = r / norm(r)
    return r


def rand_vector_at_angle(
    u: NDArray, angle: float, rng: Generator = _default_rng
) -> NDArray:
    r"""Get a random unit vector which is at `angle` radians of vector `u`.

    Note that `u` is expected to be a unit vector itself.

    Examples:
        >>> from pyclugen import rand_vector_at_angle
        >>> from numpy import arccos, array, degrees, pi, dot
        >>> from numpy.linalg import norm
        >>> from numpy.random import Generator, PCG64
        >>> rng = Generator(PCG64(123))
        >>> u = array([ 1.0, 0, 0.5, -0.5 ]) # Define a 4D vector
        >>> u = u / norm(u) # Normalize the vector
        >>> v = rand_vector_at_angle(u, pi/4, rng=rng) # Get a vector at 45 degrees
        >>> v
        array([ 0.633066  , -0.50953554, -0.10693823, -0.57285705])
        >>> float(degrees(arccos(dot(u, v) / norm(u) * norm(v)))) # u-v angle
        45.0

    Args:
      u: Unit vector with $n$ components.
      angle: Angle in radians.
      rng: Optional pseudo-random number generator.

    Returns:
      Random unit vector with $n$ components which is at `angle` radians
        with vector `u`.
    """
    if np.isclose(abs(angle), np.pi / 2) and u.size > 1:
        return rand_ortho_vector(u, rng=rng)
    elif -np.pi / 2 < angle < np.pi / 2 and u.size > 1:
        v = u + rand_ortho_vector(u, rng=rng) * tan(angle)
        return v / norm(v)
    else:
        # For |θ| > π/2 or the 1D case, simply return a random vector
        return rand_unit_vector(u.size, rng=rng)
