from __future__ import annotations

import builtins

import numpy as np
import pytest

import gragra.adapters.gwexpy_source as adapter
from gragra.gravity_sources.volume import VolumeElementSource


def test_module_import_is_lazy_and_forward_reports_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def missing_gwexpy(name, *args, **kwargs):
        if name == "gwexpy" or name.startswith("gwexpy.") or name == "astropy":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_gwexpy)
    source = VolumeElementSource(np.zeros((1, 3)), np.ones(1), np.ones(1))
    with pytest.raises(ImportError, match="runtime dependencies") as error:
        adapter.to_gwexpy_density_snapshot(
            source,
            time_s=0.0,
            density_kind="rho",
            coordinate_frame="lab",
            coordinate_origin_m=np.zeros(3),
        )
    assert isinstance(error.value.__cause__, ImportError)
    assert str(error.value.__cause__) == "blocked"
