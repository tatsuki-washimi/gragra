from __future__ import annotations

import builtins

import numpy as np
import pytest

import gragra.adapters.gwexpy_observable as adapter


def test_module_import_is_lazy_and_conversion_reports_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def missing_gwexpy(name, *args, **kwargs):
        if name == "gwexpy" or name.startswith("gwexpy."):
            raise ImportError("blocked for missing-dependency test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_gwexpy)
    with pytest.raises(ImportError, match="runtime dependencies") as error:
        adapter.to_gwexpy_observable(
            np.array([1.0]),
            np.array([[0.0, 0.0, 0.0]]),
            observable="potential",
            layout="static",
            axis_values=np.array([0.0]),
            metadata={
                "coordinate_frame": "lab",
                "coordinate_origin_m": np.zeros(3),
                "unit": "m²/s²",
            },
        )
    assert isinstance(error.value.__cause__, ImportError)
    assert str(error.value.__cause__) == "blocked for missing-dependency test"
