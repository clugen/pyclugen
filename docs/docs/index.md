# pyclugen

**pyclugen** is Python package for generating multidimensional clusters. Each
cluster is supported by a line segment, the position, orientation and length of
which guide where the respective points are placed. The
[`clugen()`][pyclugen.main.clugen] function is provided for this purpose, as well
as a number of auxiliary functions, used internally and modularly by
[`clugen()`][pyclugen.main.clugen]. Users can swap these auxiliary functions by
their own customized versions, fine-tuning their cluster generation strategies,
or even use them as the basis for their own generation algorithms.

## Installation

Install from PyPI:

```sh
pip install --upgrade pip
pip install pyclugen
```

Or directly from GitHub:

```text
pip install --upgrade pip
pip install git+https://github.com/clugen/pyclugen.git#egg=pyclugen
```

## Quick start

```python
from pyclugen import clugen
import matplotlib.pyplot as plt
```

```python
out2 = clugen(2, 4, 400, [1, 0], 0.4, [50, 10], 20, 1, 2)
plt.scatter(out2.points[:, 0], out2.points[:, 1], c=out2.clusters)
plt.show()
```

![2D example.](https://github.com/clugen/.github/blob/main/images/example2d_python.png?raw=true)

```python
out3 = clugen(3, 5, 10000, [0.5, 0.5, 0.5], 0.2, [10, 10, 10], 10, 1, 2)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(out3.points[:, 0], out3.points[:, 1], out3.points[:, 2], c=out3.clusters)
plt.show()
```

![3D example.](https://github.com/clugen/.github/blob/main/images/example3d_python.png?raw=true)

## Further reading

The *clugen* algorithm and its several implementations are detailed in the
following reference (please cite it if you use this software):

* Fachada, N. & de Andrade, D. (2023). Generating multidimensional clusters
  with support lines. *Knowledge-Based Systems*, 277, 110836.
  <https://doi.org/10.1016/j.knosys.2023.110836>
  ([arXiv preprint](https://doi.org/10.48550/arXiv.2301.10327))

## Also in this documentation

* [Theory: the clugen algorithm in detail](theory.md)
* [Detailed usage examples](generated/gallery/index.md)
* [Reference](reference.md)
* [Developing this package](dev.md)
