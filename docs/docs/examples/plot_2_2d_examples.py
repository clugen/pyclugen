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
e06 = clugen(2, 6, 500, [1, 0], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed),
    angle_deltas_fn=angdel_90_fn)

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

e07 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10],  0, 0, 0.5, rng=rng(seed),
    point_dist_fn="n")
e08 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 10, 0, 0.5, rng=rng(seed),
    point_dist_fn="n")
e09 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 30, 0, 0.5, rng=rng(seed),
    point_dist_fn="n")

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

e10 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 15,  0.0, 0.5, rng=rng(seed),
    point_dist_fn="n")
e11 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 15, 10.0, 0.5, rng=rng(seed),
    point_dist_fn="n")
e12 = clugen(2, 5, 800, [1, 0], np.pi / 10, [10, 10], 10,  0.1, 0.5, rng=rng(seed),
    llengths_fn=llen_grow_fn, point_dist_fn="n")

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

e16 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [10, 10], 10, 2, 2.5, rng=rng(seed))
e17 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [10, 10], 10, 2, 2.5, rng=rng(seed),
    cluster_offset=[20, -20])
e18 = clugen(2, 8, 1000, [1, 1], np.pi / 4, [10, 10], 10, 2, 2.5, rng=rng(seed),
    cluster_offset=[-50, -50], clucenters_fn=centers_diag_fn)

#%%

plt = plot_examples_2d(
    e16, "e16: default",
    e17, "e17: cluster_offset = [20, -20]",
    e18, "e18: custom clucenters function")

#%%
# ## Lateral dispersion and placement of point projections on the line
#
# ### Normal projection placement (default): `proj_dist_fn = "norm"`

seed = 654

#%%

e19 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 0.0, rng=rng(seed))
e20 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 1.0, rng=rng(seed))
e21 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 3.0, rng=rng(seed))

#%%

plt = plot_examples_2d(
    e19, "e19: lateral_disp = 0",
    e20, "e20: lateral_disp = 1",
    e21, "e21: lateral_disp = 3")

#%%
# ### Uniform projection placement: `proj_dist_fn = "unif"`

seed = 654

#%%

e22 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 0.0, rng=rng(seed),
    proj_dist_fn="unif")
e23 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 1.0, rng=rng(seed),
    proj_dist_fn="unif")
e24 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 3.0, rng=rng(seed),
    proj_dist_fn="unif")

#%%

plt = plot_examples_2d(
    e22, "e22: lateral_disp = 0",
    e23, "e23: lateral_disp = 1",
    e24, "e24: lateral_disp = 3")

#%%
# ### Custom projection placement using the Laplace distribution


# Custom proj_dist_fn: point projections placed using the Laplace distribution
def proj_laplace(len, n, rng):
    return rng.laplace(scale=len / 6, size=n)

#%%

e25 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 0.0, rng=rng(seed),
    proj_dist_fn=proj_laplace)
e26 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 1.0, rng=rng(seed),
    proj_dist_fn=proj_laplace)
e27 = clugen(2, 4, 1000, [1, 0], np.pi / 2, [20, 20], 13, 2, 3.0, rng=rng(seed),
    proj_dist_fn=proj_laplace)

#%%

plt = plot_examples_2d(
    e25, "e25: lateral_disp = 0",
    e26, "e26: lateral_disp = 1",
    e27, "e27: lateral_disp = 3")

#%%
# ## Controlling final point positions from their projections on the cluster-supporting line
#
# ### Points on hyperplane orthogonal to cluster-supporting line (default): `point_dist_fn = "n-1"`

seed = 1357

#%%

# Custom proj_dist_fn: point projections placed using the Laplace distribution
def proj_laplace(len, n, rng):
    return rng.laplace(scale=len / 6, size=n)

#%%

e28 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed))
e29 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed),
    proj_dist_fn="unif")
e30 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed),
    proj_dist_fn=proj_laplace)

#%%

plt = plot_examples_2d(
    e28, "e28: proj_dist_fn=\"norm\" (default)",
    e29, "e29: proj_dist_fn=\"unif\"",
    e30, "e30: custom proj_dist_fn (Laplace)")

#%%
# ### Points around projection on cluster-supporting line: `point_dist_fn = "n"`

seed = 1357

#%%

# Custom proj_dist_fn: point projections placed using the Laplace distribution
def proj_laplace(len, n, rng):
    return rng.laplace(scale=len / 6, size=n)

#%%

e31 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed),
    point_dist_fn="n")
e32 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed),
    point_dist_fn="n", proj_dist_fn="unif")
e33 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed),
    point_dist_fn="n", proj_dist_fn=proj_laplace)

#%%

plt = plot_examples_2d(
    e31, "e31: proj_dist_fn=\"norm\" (default)",
    e32, "e32: proj_dist_fn=\"unif\"",
    e33, "e33: custom proj_dist_fn (Laplace)")

#%%
# ### Custom point placement using the exponential distribution
#
# For this example we require the
# [`clupoints_n_1_template()`][clugen.helper.clupoints_n_1_template]
# helper function:

from clugen import clupoints_n_1_template

#%%

seed = 1357

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

e34 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed),
    point_dist_fn=clupoints_n_1_exp)
e35 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed),
    point_dist_fn=clupoints_n_1_exp, proj_dist_fn="unif")
e36 = clugen(2, 5, 1500, [1, 0], np.pi / 3, [20, 20], 12, 3, 1.0, rng=rng(seed),
    point_dist_fn=clupoints_n_1_exp, proj_dist_fn=proj_laplace)

#%%

plt = plot_examples_2d(
    e34, "e34: proj_dist_fn=\"norm\" (default)",
    e35, "e35: proj_dist_fn=\"unif\"",
    e36, "e36: custom proj_dist_fn (Laplace)")

#%%
# ## Manipulating cluster sizes

seed = 963

#%%

# Custom clusizes_fn (e38): cluster sizes determined via the uniform distribution,
# no correction for total points
def clusizes_unif(nclu, npts, ae, rng):
    return rng.integers(low=1, high=2 * npts / nclu + 1, size=nclu)

#%%

# Custom clusizes_fn (e39): clusters all have the same size, no correction for total points
def clusizes_equal(nclu, npts, ae, rng):
    return (npts // nclu) * np.ones(nclu, dtype=int)

#%%

# Custom clucenters_fn (all): yields fixed positions for the clusters
def centers_fixed(nclu, csep, coff, rng):
    return np.array([[-csep[0], -csep[1]], [csep[0], -csep[1]], [-csep[0], csep[1]], [csep[0], csep[1]]])

#%%

e37 = clugen(2, 4, 1500, [1, 1], np.pi, [20, 20], 0, 0, 5, rng=rng(seed),
    point_dist_fn="n", clucenters_fn=centers_fixed)
e38 = clugen(2, 4, 1500, [1, 1], np.pi, [20, 20], 0, 0, 5, rng=rng(seed),
    point_dist_fn="n", clucenters_fn=centers_fixed, clusizes_fn=clusizes_unif)
e39 = clugen(2, 4, 1500, [1, 1], np.pi, [20, 20], 0, 0, 5, rng=rng(seed),
    point_dist_fn="n", clucenters_fn=centers_fixed, clusizes_fn=clusizes_equal)

#%%

plt = plot_examples_2d(
    e37, "e37: normal dist. (default)",
    e38, "e38: unif. dist. (custom)",
    e39, "e39: equal size (custom)")
