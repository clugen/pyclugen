# pyclugen

**pyclugen** is a Python package for generating multidimensional clusters using
the _clugen_ algorithm. It provides the [`clugen()`][clugen.main.clugen]
function for this purpose, as well as a number of auxiliary functions, used
internally and modularly by [`clugen()`][clugen.main.clugen]. Users can swap
these auxiliary functions by their own customized versions, fine-tuning their
cluster generation strategies, or even use them as the basis for their own
generation algorithms.

## Installation

### From PyPI

```text
pip install pyclugen
```

### From source/GitHub

```text
pip install git+https://github.com/clugen/pyclugen.git#egg=pyclugen
```

## Quick examples

TO DO: 2D example

TO DO: 3D example

## Further reading

* [Theory: the _clugen_ algorithm in detail](theory)
* [Examples](examples)
* [API](api)
* [Developing this package](dev)
