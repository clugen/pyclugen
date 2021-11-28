# PyCluGen

A Python implementation of the CluGen algorithm.

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

## License

[MIT License](LICENSE.txt)
