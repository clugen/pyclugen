"""# Examples in 2D

These are the 2D examples.
"""

import matplotlib.pyplot as plt
from numpy.random import PCG64, Generator

from clugen import clugen


def rng(seed):
    return Generator(PCG64(seed))


#%%

seed = 123

#%%

e01 = clugen(2, 4, 200, [1, 0], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed))
e02 = clugen(2, 4, 200, [1, 1], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed))
e03 = clugen(2, 4, 200, [0, 1], 0, [10, 10], 10, 1.5, 0.5, rng=rng(seed))

#%%

plt.scatter(e01.points[:, 0], e01.points[:, 1], c=e01.clusters)
plt.show()

#%%

plt.scatter(e02.points[:, 0], e02.points[:, 1], c=e02.clusters)
plt.show()

#%%

plt.scatter(e03.points[:, 0], e03.points[:, 1], c=e03.clusters)
plt.show()
