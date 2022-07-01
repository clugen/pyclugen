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

from numpy import pi
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
    return rng.choice([0, pi / 2], size=nclu)

#%%

e04 = clugen(2, 6, 500, [1, 0], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed))
e05 = clugen(2, 6, 500, [1, 0], pi / 8, [10, 10], 10, 1.5, 0.5, rng=rng(seed))
e06 = clugen(2, 6, 500, [1, 0], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed), angle_deltas_fn=angdel_90_fn)

#%%

plot_examples_2d(
    e04, "e04: angle_disp = 0",
    e05, "e05: angle_disp = π/8",
    e06, "e06: custom angle_deltas function")
