# pyclugen

**pyclugen** is Python package for generating multidimensional clusters. Each
cluster is supported by a line segment, the position, orientation and length of
which guide where the respective points are placed. The
[`clugen()`][clugen.main.clugen] function is provided for this purpose, as well
as a number of auxiliary functions, used internally and modularly by
[`clugen()`][clugen.main.clugen]. Users can swap these auxiliary functions by
their own customized versions, fine-tuning their cluster generation strategies,
or even use them as the basis for their own generation algorithms.

## How to install

From PyPI:

```text
pip install pyclugen
```

From source/GitHub:

```text
pip install git+https://github.com/clugen/pyclugen.git#egg=pyclugen
```

## Quick start

```python
import clugen as cg
import numpy as np
import matplotlib.pyplot as plt
```

```python
out2 = cg.clugen(2, 4, 400, [1, 0], np.pi / 8, [50, 10], 20, 1, 2)
plt.scatter(out2.points[:,0], out2.points[:,1], c=out2.clusters)
plt.show()
```

![2D example.](https://github.com/clugen/.github/blob/main/images/example2d_python.png?raw=true)

```python
out3 = cg.clugen(3, 5, 10000, [0.5, 0.5, 0.5], np.pi / 16, [10, 10, 10], 10, 1, 2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(out3.points[:,0], out3.points[:,1], out3.points[:,2], c=out3.clusters)
plt.show()
```

![3D example.](https://github.com/clugen/.github/blob/main/images/example3d_python.png?raw=true)

## Further reading

* [Theory: the _clugen_ algorithm in detail](theory)
* [Detailed usage examples](examples)
* [API](api)
* [Developing this package](dev)
