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
    angle_deltas_fn=angdel_90_fn)

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
    point_dist_fn="n")
e47 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 10, 0, 0.5, rng=rng(seed),
    point_dist_fn="n")
e48 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 30, 0, 0.5, rng=rng(seed),
    point_dist_fn="n")

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
    point_dist_fn="n")
e50 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 15, 10.0, 0.5, rng=rng(seed),
    point_dist_fn="n")
e51 = clugen(3, 5, 800, [1, 0, 0], np.pi / 10, [10, 10, 10], 10,  0.1, 0.5, rng=rng(seed),
    point_dist_fn="n", llengths_fn=llen_grow_fn)

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
    cluster_offset=[30, -30, 30])
e57 = clugen(3, 8, 1000, [1, 1, 1], np.pi / 4, [10, 10, 10], 12, 3, 2.5, rng=rng(seed),
    cluster_offset=[-40, -40, -40], clucenters_fn=centers_diag_fn)

#%%

plt = plot_examples_3d(
    e55, "e55: default",
    e56, "e56: cluster_offset=[30, -30, 30]",
    e57, "e57: custom clucenters function")

#%%
# ## Lateral dispersion and placement of point projections on the line
#
# ### Normal projection placement (default): `proj_dist_fn="norm"`

seed = 246

#%%

e58 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 0.0, rng=rng(seed))
e59 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 1.0, rng=rng(seed))
e60 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 3.0, rng=rng(seed))

#%%

plt = plot_examples_3d(
    e58, "e58: lateral_disp = 0",
    e59, "e59: lateral_disp = 1",
    e60, "e60: lateral_disp = 3")

#%%
# ### Uniform projection placement: `proj_dist_fn="unif"`

seed = 246

#%%

e61 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 0.0, rng=rng(seed),
    proj_dist_fn="unif")
e62 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 1.0, rng=rng(seed),
    proj_dist_fn="unif")
e63 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 3.0, rng=rng(seed),
    proj_dist_fn="unif")

#%%

plt = plot_examples_3d(
    e61, "e61: lateral_disp = 0",
    e62, "e62: lateral_disp = 1",
    e63, "e63: lateral_disp = 3")

#%%
# ### Custom projection placement using the Laplace distribution

seed = 246

#%%

# Custom proj_dist_fn: point projections placed using the Laplace distribution
def proj_laplace(len, n, rng):
    return rng.laplace(scale=len / 6, size=n)

#%%

e64 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 0.0, rng=rng(seed),
    proj_dist_fn=proj_laplace)
e65 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 1.0, rng=rng(seed),
    proj_dist_fn=proj_laplace)
e66 = clugen(3, 4, 1000, [1, 0, 0], np.pi / 2, [20, 20, 20], 13, 2, 3.0, rng=rng(seed),
    proj_dist_fn=proj_laplace)

#%%

plt = plot_examples_3d(
    e64, "e64: lateral_disp = 0",
    e65, "e65: lateral_disp = 1",
    e66, "e66: lateral_disp = 3")

#%%
# ## Controlling final point positions from their projections on the cluster-supporting line
#
# ### Points on hyperplane orthogonal to cluster-supporting line (default): `point_dist_fn="n-1"`

seed = 840

#%%

# Custom proj_dist_fn: point projections placed using the Laplace distribution
def proj_laplace(len, n, rng):
    return rng.laplace(scale=len / 6, size=n)

#%%

e67 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed))
e68 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed),
    proj_dist_fn="unif")
e69 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed),
    proj_dist_fn=proj_laplace)

#%%

plt = plot_examples_3d(
    e67, "e67: proj_dist_fn=\"norm\" (default)",
    e68, "e68: proj_dist_fn=\"unif\"",
    e69, "e69: custom proj_dist_fn (Laplace)")

#%%
# ### Points around projection on cluster-supporting line: `point_dist_fn="n"`

seed = 840

#%%

# Custom proj_dist_fn: point projections placed using the Laplace distribution
def proj_laplace(len, n, rng):
    return rng.laplace(scale=len / 6, size=n)

e70 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed),
    point_dist_fn="n")
e71 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed),
    point_dist_fn="n", proj_dist_fn="unif")
e72 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed),
    point_dist_fn="n", proj_dist_fn=proj_laplace)

#%%

plt = plot_examples_3d(
    e70, "e70: proj_dist_fn=\"norm\" (default)",
    e71, "e71: proj_dist_fn=\"unif\"",
    e72, "e72: custom proj_dist_fn (Laplace)")

#%%
# ### Custom point placement using the exponential distribution
#
# For this example we require the
# [`clupoints_n_1_template()`][clugen.helper.clupoints_n_1_template]
# helper function:

from clugen import clupoints_n_1_template

#%%

seed = 840

#%%

# Custom point_dist_fn: final points placed using the Exponential distribution
def clupoints_n_1_exp(projs, lat_std, len, clu_dir, clu_ctr, rng):
    def dist_exp(npts, lstd, rg):
        return lstd * rg.exponential(scale=2 / lstd, size=npts)
    return clupoints_n_1_template(projs, lat_std, clu_dir, dist_exp, rng=rng)

#%%

# Custom proj_dist_fn: point projections placed using the Laplace distribution
def proj_laplace(len, n, rng):
    return rng.laplace(scale=len / 6, size=n)

#%%

e73 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed),
    point_dist_fn=clupoints_n_1_exp)
e74 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed),
    point_dist_fn=clupoints_n_1_exp, proj_dist_fn="unif")
e75 = clugen(3, 5, 1500, [1, 0, 0], np.pi / 3, [20, 20, 20], 22, 3, 2, rng=rng(seed),
    point_dist_fn=clupoints_n_1_exp, proj_dist_fn=proj_laplace)

#%%

plt = plot_examples_3d(
    e73, "e73: proj_dist_fn=\"norm\" (default)",
    e74, "e74: proj_dist_fn=\"unif\"",
    e75, "e75: custom proj_dist_fn (Laplace)")

#%%
# ## Manipulating cluster sizes

seed = 555

#%%

# Custom clusizes_fn (e77): cluster sizes determined via the uniform distribution,
# no correction for total points
def clusizes_unif(nclu, npts, ae, rng):
    return rng.integers(low=1, high=2 * npts / nclu + 1, size=nclu)

#%%

# Custom clusizes_fn (e78): clusters all have the same size, no correction for total points
def clusizes_equal(nclu, npts, ae, rng):
    return (npts // nclu) * np.ones(nclu, dtype=int)

#%%

# Custom clucenters_fn (all): yields fixed positions for the clusters
def centers_fixed(nclu, csep, coff, rng):
    return np.array([
        [-csep[0], -csep[1], -csep[2]],
        [csep[0], -csep[1], -csep[2]],
        [-csep[0], csep[1], csep[2]],
        [csep[0], csep[1], csep[2]]])

#%%

e76 = clugen(3, 4, 1500, [1, 1, 1], np.pi, [20, 20, 20], 0, 0, 5, rng=rng(seed),
    clucenters_fn=centers_fixed, point_dist_fn="n")
e77 = clugen(3, 4, 1500, [1, 1, 1], np.pi, [20, 20, 20], 0, 0, 5, rng=rng(seed),
    clucenters_fn=centers_fixed, clusizes_fn=clusizes_unif, point_dist_fn="n")
e78 = clugen(3, 4, 1500, [1, 1, 1], np.pi, [20, 20, 20], 0, 0, 5, rng=rng(seed),
    clucenters_fn=centers_fixed, clusizes_fn=clusizes_equal, point_dist_fn="n")

#%%

plt = plot_examples_3d(
    e76, "e76: normal dist. (default)",
    e77, "e77: unif. dist. (custom)",
    e78, "e78: equal size (custom)")
