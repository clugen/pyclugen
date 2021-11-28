"""Core functions."""

import numpy.typing as npt


def points_on_line(
    center: npt.NDArray, direction: npt.NDArray, dist_center: npt.NDArray
) -> npt.NDArray:
    """Determine coordinates of points on a line.

    Determine coordinates of points on a line with `center` and `direction`,
    based on the distances from the center given in `dist_center`.

    Example:

    >>> import clugen as cg
    >>> import numpy as np
    >>> center = np.array([[5.0, 5.0]]).T
    >>> dir = np.array([[1.0, 0.0]]).T
    >>> dc = np.array([np.linspace(-4, 4, 5)]).T
    >>> cg.points_on_line(center, dir, dc)
    array([[1., 5.],
           [3., 5.],
           [5., 5.],
           [7., 5.],
           [9., 5.]])

    Args:
      center: Todo.
      direction: Todo.
      dist_center: Todo.

    Returns:
      Todo.
    """
    return center.T + dist_center @ direction.T
