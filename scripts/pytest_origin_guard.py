"""Pytest plugin that aborts unless gragra imports from the expected source tree.

Load with ``-p pytest_origin_guard`` and set ``GRAGRA_EXPECTED_SRC`` to the
``src`` directory of the tree under test. This stops an editable install that
points at another worktree from silently running the tests against the wrong
code.
"""

import os

import gragra


def pytest_sessionstart():
    expected = os.path.realpath(os.environ["GRAGRA_EXPECTED_SRC"])
    origin = os.path.realpath(gragra.__file__)
    if not origin.startswith(expected + os.sep):
        raise RuntimeError(f"gragra resolves to {origin}, expected under {expected}")
