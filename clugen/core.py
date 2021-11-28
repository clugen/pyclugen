"""Core functions."""

import numpy.typing as npt


def points_on_line(
    center: npt.NDArray, direction: npt.NDArray, dist_center: npt.NDArray
) -> npt.NDArray:
    """Determine coordinates of points on a line.

    Determine coordinates of points on a line with `center` and `direction`,
    based on the distances from the center given in `dist_center`.

    Args:
      center: Todo.
      direction: Todo.
      dist_center: Todo.

    Returns:
      Todo.
    """
    return center.T + dist_center @ direction.T
