"""Third-wave subpackage exports remain importable without optional extras."""

from __future__ import annotations

import gragra
from gragra.adapters import (
    DensitySnapshotArrayBundle,
    GwexpyDensitySnapshotBundle,
    from_gwexpy_density_snapshot,
    to_gwexpy_density_snapshot,
)
from gragra.fields import (
    TetraP1DensityLinearizedSources,
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)


def test_third_wave_exports_are_subpackage_only() -> None:
    assert len(gragra.__all__) == 14
    assert "compare" not in gragra.__all__
    assert "GravityResult" not in gragra.__all__
    assert "Frame" not in gragra.__all__
    assert GwexpyDensitySnapshotBundle.__name__ == "GwexpyDensitySnapshotBundle"
    assert DensitySnapshotArrayBundle.__name__ == "DensitySnapshotArrayBundle"
    assert callable(to_gwexpy_density_snapshot)
    assert callable(from_gwexpy_density_snapshot)
    assert TetraP1DensityLinearizedSources.__name__ == "TetraP1DensityLinearizedSources"
    assert callable(piecewise_tetra_p1_density_sources)
    assert callable(compare_tetra_p1_density_representations)
