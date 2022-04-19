# Development

## Installing for development and/or improving the package

```text
$ git clone https://github.com/clugen/pyclugen.git
$ cd pyclugen
$ python -m venv env
$ source env/bin/activate
$ pip install -e .[dev]
$ pre-commit install
```

On Windows replace `source env/bin/activate` with `. env\Scripts\activate`.

## Run tests

```text
$ pytest
```

## Build docs

Considering we're in the `pyclugen` folder, run the following commands:

```text
$ cd docs
$ mkdocs build
```

The generated documentation will be placed in `docs/site`. Alternatively, the
documentation can be generated and served locally with:

```
$ mkdocs serve
```

## Code style

Code style is enforced with [flake8] (and a number of plugins), [black], and
[isort]. Some highlights include, but are not limited to:

* Encoding: UTF-8
* Indentation: 4 spaces (no tabs)
* Line size limit: 88 chars
* Newlines: Unix style, i.e. LF or \n

[black]: https://black.readthedocs.io/en/stable/
[flake8]: https://flake8.pycqa.org/en/latest/
[isort]: https://pycqa.github.io/isort/
