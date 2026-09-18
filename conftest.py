"""Root conftest for the test suite.

The file has no fixtures. It exists for its location: pytest inserts the
directory containing the top-most conftest into ``sys.path`` before collecting
tests, and that is what lets ``tests/`` do ``from data_loader import ...`` and
``from smoking_model import ...``.

Why this is needed: with the default ``prepend`` import mode, pytest adds the
*first directory without an ``__init__.py``* above each test file. There is no
``tests/__init__.py``, so only ``tests/`` itself was added -- not the repository
root where the two modules live. Running ``python -m pytest`` masks the problem,
because that form puts the current directory on ``sys.path`` itself. It therefore
passed locally and failed in CI, where the console script ``pytest tests/ -v``
is used and exits with code 2 during collection.

Keep this file at the repository root; moving it into ``tests/`` reintroduces
the failure.
"""
