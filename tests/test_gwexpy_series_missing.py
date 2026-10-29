"""Simulated missing GWexpy behavior without module-level imports."""

import sys

import numpy as np
import pytest

from gragra.adapters.gwexpy_density_series import to_gwexpy_density_time_series
from gragra.fields.density_series import DensityTimeSeries


def test_public_density_bridge_reports_missing_dependency(monkeypatch):
    series = DensityTimeSeries(
        np.zeros((1, 3)),
        np.ones(1),
        np.ones((1, 1)),
        np.array([0.0]),
        "rho",
        "local",
        np.zeros(3),
        ["one"],
    )
    monkeypatch.setitem(sys.modules, "gwexpy", None)
    with pytest.raises(ImportError, match="runtime dependencies") as error:
        to_gwexpy_density_time_series(series)
    assert isinstance(error.value.__cause__, ImportError)
