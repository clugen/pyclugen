# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Multidimensional cluster generation in Python."""


__all__ = [
    "Clusters",
    "clugen",
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

from clugen.core import (
    points_on_line,
    rand_ortho_vector,
    rand_unit_vector,
    rand_vector_at_angle,
)
from clugen.helper import angle_btw, clupoints_n_1_template, fix_empty, fix_num_points
from clugen.main import Clusters, clugen
from clugen.module import (
    angle_deltas,
    clucenters,
    clupoints_n,
    clupoints_n_1,
    clusizes,
    llengths,
)
