"""# Examples in 3D

This section contains several examples on how to generate 3D data with
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
# To plot these examples we use the [`plot_examples_3d`](plot_functions.md#plot_examples_3d)
# function:

from plot_functions import plot_examples_3d

#%%

#%%
# ## Manipulating the direction of cluster-supporting lines
#
# ### Using the `direction` parameter

#%%

seed = 321

#%%

e40 = clugen(3, 4, 500, [1, 0, 0], 0, [10, 10, 10], 15, 1.5, 0.5, rng=rng(seed))
e41 = clugen(3, 4, 500, [1, 1, 1], 0, [10, 10, 10], 15, 1.5, 0.5, rng=rng(seed))
e42 = clugen(3, 4, 500, [0, 0, 1], 0, [10, 10, 10], 15, 1.5, 0.5, rng=rng(seed))

#%%

plt = plot_examples_3d(
    e40, "e40: direction = [1, 0, 0]",
    e41, "e41: direction = [1, 1, 1]",
    e42, "e42: direction = [0, 0, 1]")

#%%
# ### Changing the `angle_disp` parameter and using a custom `angle_deltas_fn` function

seed = 321

# Custom angle_deltas function: arbitrarily rotate some clusters by 90 degrees
def angdel_90_fn(nclu, astd, rng):
    return rng.choice([0, np.pi / 2], size=nclu)

#%%

e43 = clugen(3, 6, 1000, [1, 0, 0], 0, [10, 10, 10], 15, 1.5, 0.5, rng=rng(seed))
e44 = clugen(3, 6, 1000, [1, 0, 0], np.pi / 8, [10, 10, 10], 15, 1.5, 0.5, rng=rng(seed))
e45 = clugen(3, 6, 1000, [1, 0, 0], 0, [10, 10, 10], 15, 1.5, 0.5, rng=rng(seed),
    angle_deltas_fn = angdel_90_fn)

#%%

plt = plot_examples_3d(
    e43, "e43: angle_disp = 0",
    e44, "e44: angle_disp = π / 8",
    e45, "e45: custom angle_deltas function")


#%%
# ## Manipulating the length of cluster-supporting lines
#
# ### Using the `llength` parameter

seed = 789

#%%

e46 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 0, 0, 0.5, rng=rng(seed),
    point_dist_fn = "n")
e47 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 10, 0, 0.5, rng=rng(seed),
    point_dist_fn = "n")
e48 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 30, 0, 0.5, rng=rng(seed),
    point_dist_fn = "n")

#%%

plt = plot_examples_3d(
    e46, "e46: llength = 0",
    e47, "e47: llength = 10",
    e48, "e48: llength = 30")

#%%
# ### Changing the `llength_disp` parameter and using a custom `llengths_fn` function

seed = 765

#%%

# Custom llengths function: line lengths tend to grow for each new cluster
def llen_grow_fn(nclu, llen, llenstd, rng):
    return llen * np.arange(nclu) + rng.normal(scale=llenstd, size=nclu)

e49 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 15,  0.0, 0.5, rng=rng(seed),
    point_dist_fn = "n")
e50 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 15, 10.0, 0.5, rng=rng(seed),
    point_dist_fn = "n")
e51 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 10,  0.1, 0.5, rng=rng(seed),
    llengths_fn = llen_grow_fn, point_dist_fn = "n")

#%%

plt = plot_examples_3d(
    e49, "e49: llength_disp = 0.0",
    e50, "e50: llength_disp = 10.0",
    e51, "e51: custom llengths function")

#%%
# ## Manipulating relative cluster positions
#
# ### Using the `cluster_sep` parameter

seed = 765

#%%

e52 = clugen(3, 8, 1000, [1, 1, 1], np.pi / 4, [30, 10, 10], 25, 4, 3, rng=rng(seed))
e53 = clugen(3, 8, 1000, [1, 1, 1], np.pi / 4, [10, 30, 10], 25, 4, 3, rng=rng(seed))
e54 = clugen(3, 8, 1000, [1, 1, 1], np.pi / 4, [10, 10, 30], 25, 4, 3, rng=rng(seed))

#%%

plt = plot_examples_3d(
    e52, "e52: cluster_sep = [30, 10, 10]",
    e53, "e53: cluster_sep = [10, 30, 10]",
    e54, "e54: cluster_sep = [10, 10, 30]")

#%%
# ### Changing the `cluster_offset` parameter and using a custom `clucenters_fn` function

# Custom clucenters function: places clusters in a diagonal
def centers_diag_fn(nclu, csep, coff, rng):
    return np.ones((nclu, len(csep))) * np.arange(1, nclu + 1)[:, None] * np.max(csep) + coff

e55 = clugen(3, 8, 1000, [1, 1, 1], np.pi / 4, [10, 10, 10], 12, 3, 2.5, rng=rng(seed))
e56 = clugen(3, 8, 1000, [1, 1, 1], np.pi / 4, [10, 10, 10], 12, 3, 2.5, rng=rng(seed),
    cluster_offset = [30, -30, 30])
e57 = clugen(3, 8, 1000, [1, 1, 1], np.pi / 4, [10, 10, 10], 12, 3, 2.5, rng=rng(seed),
    cluster_offset = [-40, -40, -40], clucenters_fn = centers_diag_fn)

#%%

plt = plot_examples_3d(
    e55, "e55: default",
    e56, "e56: cluster_offset = [30, -30, 30]",
    e57, "e57: custom clucenters function")
