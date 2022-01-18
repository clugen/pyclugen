# Copyright (c) 2020, 2021 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Core functions."""

import numpy as np
import numpy.typing as npt

from .shared import _default_rng


def points_on_line(
    center: npt.NDArray, direction: npt.NDArray, dist_center: npt.NDArray
) -> npt.NDArray:
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


def rand_unit_vector(
    num_dims: int, rng: np.random.Generator = _default_rng
) -> npt.NDArray:
    """Get a random unit vector with `num_dims` dimensions.

    ## Examples:

    >>> import clugen as cg
    >>> cg.rand_unit_vector(4) # doctest: +SKIP
    array([-0.77876028,  0.33941554, -0.42085082,  0.31814165])

    >>> import numpy.random as nprand
    >>> rng = nprand.Generator(nprand.PCG64(123))
    >>> cg.rand_unit_vector(2, rng=rng)
    array([ 0.3783202 , -0.92567479])

    Args:
      num_dims: Number of dimensions.
      rng: Optional pseudo-random number generator.

    Returns:
      A random unit vector with `num_dims` dimensions.
    """
    r = rng.random(num_dims) - 0.5
    r = r / np.linalg.norm(r)
    return r
