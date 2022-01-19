# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Core functions."""

from math import tan

from numpy import abs, isclose, pi, vdot
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
    `direction` is a \(n\)-dimensional unit vector. In other words, considering
    \(\mathbf{d}=\)`direction` ( \(n \times 1\) ), \(\mathbf{c}=\)`center`
    ( \(n \times 1\) ), and \(\mathbf{w}=\) `dist_center` ( \(p \times 1\) ),
    the coordinates of points on the line are given by:

    $$
    \mathbf{P}=\mathbf{1}\,\mathbf{c}^T + \mathbf{w}\mathbf{d}^T
    $$

    where \(\mathbf{P}\) is the \(p \times n\) matrix of point coordinates on the
    line, and \(\mathbf{1}\) is a \(p \times 1\) vector with all entries equal to 1.

    ## Examples:

    >>> import clugen as cg
    >>> import numpy as np
    >>> cg.points_on_line(np.array([[5.0, 5.0]]).T,
    ...                   np.array([[1.0, 0.0]]).T,
    ...                   np.array([np.linspace(-4, 4, 5)]).T) # 2D, 5 points
    array([[1., 5.],
           [3., 5.],
           [5., 5.],
           [7., 5.],
           [9., 5.]])

    >>> cg.points_on_line(np.array([[-2,0,0,2.]]).T,
    ...                   np.array([[0,0,-1.,0]]).T,
    ...                   np.array([[10, -10]]).T) # 4D, 2 points
    array([[ -2.,   0., -10.,   2.],
           [ -2.,   0.,  10.,   2.]])

    Args:
      center: Center of the line ( \(n \times 1\) vector).
      direction: Line direction ( \(n \times 1\) unit vector).
      dist_center: Distance of each point to the center of the line ( \(p \times
        1\) vector, where \(p\) is the number of points).

    Returns:
      Coordinates of points on the specified line ( \(p \times n\) matrix).
    """
    return center.T + dist_center @ direction.T


def rand_ortho_vector(u: NDArray, rng: Generator = _default_rng) -> NDArray:
    r"""Get a random unit vector orthogonal to `u`.

    Note that `u` is expected to be a unit vector itself.

    ## Examples

    >>> from clugen import rand_ortho_vector
    >>> from numpy.linalg import norm
    >>> from numpy.random import Generator, PCG64
    >>> rng = Generator(PCG64(123))
    >>> r = rng.random((3, 1)) # Get a random 3D vector
    >>> r = r / norm(r) # Normalize it
    >>> r_ort = rand_ortho_vector(r, rng=rng) # Get random unit vector orthogonal to r
    >>> r_ort
    array([[-0.1982903 ],
           [-0.61401512],
           [ 0.76398062]])

    >>> from numpy import isclose, vdot
    >>> isclose(vdot(r, r_ort), 0) # Check that vectors are indeed orthogonal
    True

    Args:
      u: \(n \times 1\) unit vector.
      rng: Optional pseudo-random number generator.

    Returns:
      A \(n \times 1\) random unit vector orthogonal to `u`.
    """
    # If 1D, just return a random unit vector
    if u.size == 1:
        return rand_unit_vector(1, rng=rng)

    # Find a random, non-parallel vector to u
    while True:

        # Find normalized random vector
        r = rand_unit_vector(u.size, rng=rng)

        # If not parallel to u we can keep it and break the loop
        if not isclose(abs(vdot(u, r)), 1):
            break

    # Get vector orthogonal to u using 1st iteration of Gram-Schmidt process
    v = r - vdot(u, r) / vdot(u, u) * u

    # Normalize it
    v = v / norm(v)

    # And return it
    return v


def rand_unit_vector(num_dims: int, rng: Generator = _default_rng) -> NDArray:
    r"""Get a `num_dims` \(\times 1\) random unit vector.

    ## Examples:

    >>> import clugen as cg
    >>> cg.rand_unit_vector(4) # doctest: +SKIP
    array([[-0.48915817],
           [-0.1507109 ],
           [ 0.8540957 ],
           [ 0.09236367]])

    >>> import numpy.random as nprand
    >>> rng = nprand.Generator(nprand.PCG64(123))
    >>> cg.rand_unit_vector(2, rng=rng)
    array([[ 0.3783202 ],
           [-0.92567479]])

    Args:
      num_dims: Number of dimensions.
      rng: Optional pseudo-random number generator.

    Returns:
      A `num_dims` \(\times 1\) random unit vector.
    """
    r = rng.random((num_dims, 1)) - 0.5
    r = r / norm(r)
    return r


def rand_vector_at_angle(
    u: NDArray, angle: float, rng: Generator = _default_rng
) -> NDArray:
    r"""Get a random unit vector which is at `angle` radians of vector `u`.

    Note that `u` is expected to be a unit vector itself.

    ## Examples:

    >>> from clugen import rand_vector_at_angle
    >>> from numpy import arccos, array, degrees, pi, vdot
    >>> from numpy.linalg import norm
    >>> from numpy.random import Generator, PCG64
    >>> rng = Generator(PCG64(123))
    >>> u = array([ 1.0, 0, 0.5, -0.5 ]).reshape((4, 1)) # Define a 4D vector
    >>> u = u / norm(u) # Normalize the vector
    >>> v = rand_vector_at_angle(u, pi/4, rng=rng) # Get a vector at 45 degrees
    >>> v
    array([[ 0.633066  ],
           [-0.50953554],
           [-0.10693823],
           [-0.57285705]])
    >>> degrees(arccos(vdot(u, v) / norm(u) * norm(v))) # Angle between u and v
    45.0

    Args:
      u: \(n \times 1\) unit vector.
      angle: Angle in radians.
      rng: Optional pseudo-random number generator.

    Returns:
      A `num_dims` \(\times 1\) random unit vector which is at `angle` radians
      with vector `u`.
    """
    if isclose(abs(angle), pi / 2) and u.size > 1:
        return rand_ortho_vector(u, rng=rng)
    elif -pi / 2 < angle < pi / 2 and u.size > 1:
        v = u + rand_ortho_vector(u, rng=rng) * tan(angle)
        return v / norm(v)
    else:
        # For |θ| > π/2 or the 1D case, simply return a random vector
        return rand_unit_vector(u.size, rng=rng)
