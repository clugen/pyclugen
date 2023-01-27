[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://clugen.github.io/pyclugen/)
[![MIT](https://img.shields.io/badge/license-MIT-yellowgreen.svg)](https://tldrlegal.com/license/mit-license)

# pyclugen

**pyclugen** is a Python implementation of the *clugen* algorithm for
generating multidimensional clusters. Each cluster is supported by a line
segment, the position, orientation and length of which guide where the
respective points are placed.

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
import pyclugen as cg
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

## See also

* [CluGen.jl](https://github.com/clugen/CluGen.jl/), a Julia implementation of
  the *clugen* algorithm.
* [clugenr](https://github.com/clugen/clugenr/), an R implementation
  of the *clugen* algorithm.
* [MOCluGen](https://github.com/clugen/MOCluGen/), a MATLAB/Octave
  implementation of the *clugen* algorithm.

## Reference

If you use this software, please cite the following reference:

* Fachada, N. & de Andrade, D. (2023). Generating Multidimensional Clusters With
  Support Lines. <https://doi.org/10.48550/arXiv.2301.10327>.

## License

[MIT License](LICENSE.txt)
