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
