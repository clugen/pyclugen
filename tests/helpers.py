# Copyright (c) 2020-2022 Nuno Fachada and contributors
# Distributed under the MIT License (See accompanying file LICENSE.txt or copy
# at http://opensource.org/licenses/MIT)

"""Helper functions for tests."""

from numpy import arctan, pi, signbit
from numpy.linalg import norm


def angle_btw(v1, v2):
    """Get angle between two vectors, useful for checking correctness of results.

    Common version is unstable: arccos(dot(u, v) / (norm(u) * norm(v)))
    This version is based on AngleBetweenVectors.jl by Jeffrey Sarnoff (MIT license),
    (https://github.com/JeffreySarnoff/AngleBetweenVectors.jl/blob/master/src/AngleBetweenVectors.jl)
    in turn based on these notes by Prof. W. Kahan, see page 15:
    https://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf
    """
    u1 = v1 / norm(v1)
    u2 = v2 / norm(v2)

    y = u1 - u2
    x = u1 + u2

    a0 = 2 * arctan(norm(y) / norm(x))

    if (not signbit(a0)) or signbit(pi - a0):
        return a0
    elif signbit(a0):
        return 0.0
    else:
        return pi
