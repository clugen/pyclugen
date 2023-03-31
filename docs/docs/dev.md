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

Tests can be executed with the following command:

```text
$ pytest
```

The previous command runs the tests at `normal` level by default. This test
level can also be specified explicitly:

```text
$ pytest --test-level=normal
```

There are four test levels, from fastest to slowest (i.e., from less thorough to
more exhaustive): `fast`, `ci`, `normal` and `full`. The `fast` level tests all
functions using typical parameters, just to check if everything is working. The
`ci` level performs the minimal amount of testing that yields complete test
coverage. Beyond complete coverage, the `normal` and `full` levels also test
increasing combinations of parameters and PRNG seeds, which may be important to
root out rare corner cases. Note that the `full` level can be extremely slow.

To generate a test coverage report, run pytest as follows:

```text
$ pytest --cov=pyclugen --cov-report=html --test-level=ci
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
