"""# Examples in nD

This section contains several examples on how to generate nD (n > 3) data with
**pyclugen**. To run the examples we first need to import the
[`clugen()`][pyclugen.main.clugen] function:"""

from pyclugen import clugen

#%%
# To make the examples exactly reproducible we'll import a random number
# generator from NumPy and pass it as a parameter to
# [`clugen()`][pyclugen.main.clugen]. We'll also create a small helper function
# for providing us a brand new seeded generator:

import numpy as np
from numpy.random import PCG64, Generator

def rng(seed):
    return Generator(PCG64(seed))

#%%
# To plot these examples we use the [`plot_examples_nd`](plot_functions.md#plot_examples_nd)
# function:

from plot_functions import plot_examples_nd

#%%

#%%%
# ## 5D example with default optional arguments

seed = 123

#%%

# Number of dimensions
nd = 5

#%%

e085 = clugen(nd, 6, 1500, [1, 1, 0.5, 0, 0], np.pi / 16, 30 * np.ones(nd), 30, 4, 3, rng=rng(seed))

#%%

plot_examples_nd(e085, "e085: 5D with optional parameters set to defaults")

#%%
# ## 5D example with `proj_dist_fn = "unif"` and `point_dist_fn = "n"`

seed = 579

#%%

# Number of dimensions
nd = 5

#%%

e086 = clugen(nd, 6, 1500, [0.1, 0.3, 0.5, 0.3, 0.1], np.pi / 12, 30 * np.ones(nd), 35, 5, 3.5,
    proj_dist_fn="unif", point_dist_fn="n", rng=rng(seed))

#%%

plot_examples_nd(e086, "e086: 5D with proj_dist_fn=\"unif\" and point_dist_fn=\"n\"")

#%%
# ## 4D example with custom projection placement using the Beta distribution

seed = 963

#%%

# Number of dimensions
nd = 4

#%%

# Custom proj_dist_fn: point projections placed using the Beta distribution
def proj_beta(len, n, rng):
    return len * rng.beta(0.1, 0.1, size=n) - len / 2

#%%

e087 = clugen(nd, 5, 1500, np.ones(nd), np.pi / 6, 30 * np.ones(nd), 60, 15, 6, rng=rng(seed),
    proj_dist_fn=proj_beta)

#%%

plot_examples_nd(e087, "e087: 4D with custom proj_dist_fn (Beta)")
