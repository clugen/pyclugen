[![Tests](https://github.com/clugen/pyclugen/actions/workflows/tests.yml/badge.svg)](https://github.com/clugen/pyclugen/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/clugen/pyclugen/branch/main/graph/badge.svg?token=3K5ZN35AJ5)](https://codecov.io/gh/clugen/pyclugen)
[![docs](https://img.shields.io/badge/docs-click_here-blue.svg)](https://clugen.github.io/pyclugen/)
[![PyPI](https://img.shields.io/pypi/v/pyclugen)](https://pypi.org/project/pyclugen/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyclugen?color=blueviolet)
[![MIT](https://img.shields.io/badge/license-MIT-yellowgreen.svg)](https://tldrlegal.com/license/mit-license)

# pyclugen

**pyclugen** is a Python implementation of the *clugen* algorithm for
generating multidimensional clusters with arbitrary distributions. Each cluster
is supported by a line segment, the position, orientation and length of which
guide where the respective points are placed.

See the [documentation](https://clugen.github.io/pyclugen/) and
[examples](https://clugen.github.io/pyclugen/generated/gallery/) for more
details.

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

## See also

* [CluGen.jl](https://github.com/clugen/CluGen.jl/), a Julia implementation of
  the *clugen* algorithm.
* [clugenr](https://github.com/clugen/clugenr/), an R implementation
  of the *clugen* algorithm.
* [MOCluGen](https://github.com/clugen/MOCluGen/), a MATLAB/Octave
  implementation of the *clugen* algorithm.

## Reference

If you use this software, please cite the following reference:

* Fachada, N. & de Andrade, D. (2023). Generating multidimensional clusters
  with support lines. *Knowledge-Based Systems*, 277, 110836.
  <https://doi.org/10.1016/j.knosys.2023.110836>
  ([arXiv preprint](https://doi.org/10.48550/arXiv.2301.10327))

## License

[MIT License](LICENSE.txt)
