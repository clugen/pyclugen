"""# Examples in 1D

This section contains several examples on how to generate 1D data with
**pyclugen**. To run the examples we first need to import the
[`clugen()`][clugen.main.clugen] function:"""

from clugen import clugen

#%%
# To make the examples exactly reproducible we'll import a random number
# generator from NumPy and pass it as a parameter to
# [`clugen()`][clugen.main.clugen]. We'll also create a small helper function
# for providing us a brand new seeded generator:

import numpy as np
from numpy.random import PCG64, Generator

def rng(seed):
    return Generator(PCG64(seed))

#%%
# To plot these examples we use the [`plot_examples_1d`](plot_functions.md#plot_examples_1d)
# function:

from plot_functions import plot_examples_1d

#%%
# ## Basic 1D example with density plot

#%%

seed = 23456

#%%

# Custom proj_dist_fn: point projections placed using the Weibull distribution
def proj_weibull(len, n, rng):
    return len / 2 * rng.weibull(1.5, size=n)

#%%

e79 = clugen(1, 3, 1000, [1], 0, [10], 6, 1.5, 0, rng=rng(seed))
e80 = clugen(1, 3, 1000, [1], 0, [10], 6, 1.5, 0, rng=rng(seed), proj_dist_fn="unif")
e81 = clugen(1, 3, 1000, [1], 0, [10], 6, 1.5, 0, rng=rng(seed), proj_dist_fn=proj_weibull)

#%%

plot_examples_1d(
    e79, "e79: proj_dist_fn = 'norm' (default)",
    e80, "e80: proj_dist_fn = 'unif'",
    e81, "e81: custom proj_dist_fn (Weibull)")
