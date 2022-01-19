# Copyright (c) 2020, 2021 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Core functions."""

from numpy import abs, isclose, vdot
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
    """Get a random unit vector orthogonal to `u`.

    Note that `u` is expected to be a unit vector itself.

    Args:
      u: A unit vector.
      rng: Optional pseudo-random number generator.

    Returns:
      A random unit vector orthogonal to `u`.
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
    """Get a random unit vector with `num_dims` dimensions.

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
      A random unit vector with `num_dims` dimensions.
    """
    r = rng.random((num_dims, 1)) - 0.5
    r = r / norm(r)
    return r
