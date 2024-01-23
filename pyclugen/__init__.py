# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Various functions for multidimensional cluster generation in Python.

Note that:

1. [`clugen()`][pyclugen.main.clugen] is the main function of the **pyclugen**
   package, and possibly the only function most users will need.
2. Functions which accept `rng` as the last parameter are stochastic. Thus, in
   order to obtain the same result on separate invocations of these functions,
   pass them an instance of same pseudo-random number
   [`Generator`][numpy.random.Generator] initialized with the same seed.
"""


__all__ = [
    "Clusters",
    "clugen",
    "clumerge",
    "points_on_line",
    "rand_ortho_vector",
    "rand_unit_vector",
    "rand_vector_at_angle",
    "angle_deltas",
    "angle_btw",
    "clupoints_n_1_template",
    "fix_empty",
    "fix_num_points",
    "clucenters",
    "clupoints_n_1",
    "clupoints_n",
    "clusizes",
    "llengths",
]

from pyclugen.core import (
    points_on_line,
    rand_ortho_vector,
    rand_unit_vector,
    rand_vector_at_angle,
)
from pyclugen.helper import angle_btw, clupoints_n_1_template, fix_empty, fix_num_points
from pyclugen.main import Clusters, clugen, clumerge
from pyclugen.module import (
    angle_deltas,
    clucenters,
    clupoints_n,
    clupoints_n_1,
    clusizes,
    llengths,
)
