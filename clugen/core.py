"""
   Core functions.
"""

import numpy as np
import numpy.typing as npt

def points_on_line(
    center: npt.NDArray,
    direction: npt.NDArray,
    dist_center: npt.NDArray) -> npt.NDArray:

    return center.T + dist_center @ direction.T