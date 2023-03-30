# Copyright (c) 2020-2023 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Shared data."""

import numpy as np

# Default pseudo-random number generator in case users don't specify one.
_default_rng = np.random.default_rng()
