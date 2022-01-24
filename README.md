# PyCluGen

A Python implementation of the CluGen algorithm.

## How to use

```python
import clugen as cg
import numpy as np
import matplotlib.pyplot as plt

out2 = cg.clugen(2, 4, 400, [1, 0], np.pi/8, [50, 10], 20, 1, 0.2)
plt.scatter(out2.points[:,0], out2.points[:,1], c=out2.point_clusters)
plt.show()

out3 = cg.clugen(3, 5, 10000, [0.5, 0.5, 0.5], np.pi/16, [10, 10, 10], 10, 1, 2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(out3.points[:,0], out3.points[:,1], out3.points[:,2], c=out3.point_clusters)
plt.show()
```

## How to install

### From PyPI

```sh
pip install pyclugen
```

### From source/GitHub

Directly using pip:

```text
pip install git+https://github.com/clugen/pyclugen.git#egg=pyclugen
```

Or each step at a time:

```text
git clone https://github.com/clugen/pyclugen.git
cd pyclugen
pip install .
```

### Installing for development and/or improving the package

```text
git clone https://github.com/clugen/pyclugen.git
cd pyclugen
python -m venv env
source env/bin/activate
pip install -e .[dev]
pre-commit install
```

On Windows replace `source env/bin/activate` with `. env\Scripts\activate`.

To build the documentation run:

```text
pdoc3 --force --html --output-dir docs/build/ --template-dir docs/templates clugen/
```

## License

[MIT License](LICENSE.txt)
