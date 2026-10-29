"""Tests for gwexpy adapter behavior when gwexpy is not installed."""

import sys

import numpy as np
import pytest

import gragra.adapters.gwexpy_adapter as gwexpy_adapter
from gragra.adapters import (
    to_gwexpy_scalar_field,
    to_gwexpy_tensor_field,
    to_gwexpy_vector_field,
)


def test_gwexpy_adapter_importable_when_missing():
    """Importing the adapter itself should not raise ImportError."""
    assert gwexpy_adapter is not None


def test_scalar_raises_importerror_when_missing(monkeypatch):
    """to_gwexpy_scalar_field raises ImportError if gwexpy is missing."""
    monkeypatch.setitem(sys.modules, "gwexpy", None)
    with pytest.raises(ImportError, match="runtime dependencies") as error:
        to_gwexpy_scalar_field(np.zeros(5))
    assert isinstance(error.value.__cause__, ImportError)


def test_vector_raises_importerror_when_missing(monkeypatch):
    """to_gwexpy_vector_field raises ImportError if gwexpy is missing."""
    monkeypatch.setitem(sys.modules, "gwexpy", None)
    with pytest.raises(ImportError, match="runtime dependencies") as error:
        to_gwexpy_vector_field(np.zeros((5, 3)))
    assert isinstance(error.value.__cause__, ImportError)


def test_tensor_raises_importerror_when_missing(monkeypatch):
    """to_gwexpy_tensor_field raises ImportError if gwexpy is missing."""
    monkeypatch.setitem(sys.modules, "gwexpy", None)
    with pytest.raises(ImportError, match="runtime dependencies") as error:
        to_gwexpy_tensor_field(np.zeros((5, 3, 3)))
    assert isinstance(error.value.__cause__, ImportError)
