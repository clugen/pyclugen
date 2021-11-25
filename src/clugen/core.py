"""
   Core functions.
"""

import numpy as np

def points_on_line(center, direction, dist_center):

    return center.T + dist_center @ direction.T