"""# Examples in 2D

This section contains several examples on how to generate 2D data with
**pyclugen**. To run the examples we first need to import the
[`clugen()`][clugen.main.clugen] function:"""

from clugen import clugen

#%%
# To make the examples exactly reproducible we'll import a random number
# generator from NumPy and pass it as a parameter to
# [`clugen()`][clugen.main.clugen]. We'll also create a small helper function
# for providing us a brand new seeded generator:

import numpy as np
from numpy.random import PCG64, Generator, default_rng

def rng(seed):
    return Generator(PCG64(seed))

#%%
# To plot these examples we use the [`plot_examples_2d`](plot_functions.md#plot_examples_2d)
# function:

from plot_functions import plot_examples_2d


#%%
# ## Manipulating the direction of cluster-supporting lines
#
# ### Using the `direction` parameter

seed = 123

#%%

e01 = clugen(2, 4, 2000, [1, 0], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed))
e02 = clugen(2, 4, 200, [1, 1], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed))
e03 = clugen(2, 4, 200, [0, 1], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed))

#%%

plot_examples_2d(
    e01, "e01: direction = [1, 0]",
    e02, "e02: direction = [1, 1]",
    e03, "e03: direction = [0, 1]")

#%%
# ### Changing the `angle_disp` parameter and using a custom `angle_deltas_fn` function

seed = 321

#%%

# Custom angle_deltas function: arbitrarily rotate some clusters by 90 degrees
def angdel_90_fn(nclu, astd, rng):
    return rng.choice([0, np.pi / 2], size=nclu)

#%%

e04 = clugen(2, 6, 500, [1, 0], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed))
e05 = clugen(2, 6, 500, [1, 0], np.pi / 8, [10, 10], 10, 1.5, 0.5, rng=rng(seed))
e06 = clugen(2, 6, 500, [1, 0], 0, [10, 10], 10, 1.5, 0.5,
    angle_deltas_fn=angdel_90_fn, rng=rng(seed))

#%%

plot_examples_2d(
    e04, "e04: angle_disp = 0",
    e05, "e05: angle_disp = π/8",
    e06, "e06: custom angle_deltas function")

#%%
# ## Manipulating the length of cluster-supporting lines
#
# ### Using the `llength` parameter

seed = 567

#%%

e07 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10],  0, 0, 0.5,
    point_dist_fn="n", rng=rng(seed))
e08 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 10, 0, 0.5,
    point_dist_fn="n", rng=rng(seed))
e09 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 30, 0, 0.5,
    point_dist_fn="n", rng=rng(seed))

#%%

plot_examples_2d(
    e07, "e07: llength = 0",
    e08, "e08: llength = 10",
    e09, "e09: llength = 30")

#%%
# ### Changing the `llength_disp` parameter and using a custom `llengths_fn` function

seed = 567

#%%

# Custom llengths function: line lengths grow for each new cluster
def llen_grow_fn(nclu, llen, llenstd, rng):
    return llen * np.arange(nclu) + rng.normal(scale=llenstd, size=nclu)

#%%

e10 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 15,  0.0, 0.5,
    point_dist_fn="n", rng=rng(seed))
e11 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 15, 10.0, 0.5,
    point_dist_fn="n", rng=rng(seed))
e12 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 10,  0.1, 0.5,
    llengths_fn=llen_grow_fn, point_dist_fn="n", rng=rng(seed))

#%%

plot_examples_2d(
    e10, "e10: llength_disp = 0.0",
    e11, "e11: llength_disp = 5.0",
    e12, "e12: custom llengths function")

#%%
# ## Manipulating relative cluster positions
#
# ### Using the `cluster_sep` parameter

#%%

seed = 21

#%%

e13 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [10, 10], 10, 2, 2.5, rng=rng(seed))
e14 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [30, 10], 10, 2, 2.5, rng=rng(seed))
e15 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [10, 30], 10, 2, 2.5, rng=rng(seed))

#%%

plt = plot_examples_2d(
    e13, "e13: cluster_sep = [10, 10]",
    e14, "e14: cluster_sep = [30, 10]",
    e15, "e15: cluster_sep = [10, 30]")

#%%
# ### Changing the `cluster_offset` parameter and using a custom `clucenters_fn` function

seed = 21

#%%

# Custom clucenters function: places clusters in a diagonal
def centers_diag_fn(nclu, csep, coff, rng):
    return np.ones((nclu, len(csep))) * np.arange(1, nclu + 1)[:, None] * np.max(csep) + coff

#%%

e16 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [10, 10], 10, 2, 2.5,
    rng=rng(seed))
e17 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [10, 10], 10, 2, 2.5,
    cluster_offset=[20, -20], rng=rng(seed))
e18 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [10, 10], 10, 2, 2.5,
    cluster_offset=[-50, -50], clucenters_fn=centers_diag_fn, rng=rng(seed))

#%%

plt = plot_examples_2d(
    e16, "e16: default",
    e17, "e17: cluster_offset = [20, -20]",
    e18, "e18: custom clucenters function")
