"""HDF5 exchange checks for the Wave 5 moving-density study."""

from __future__ import annotations

import pytest

pytest.importorskip("h5py")

from scripts.research_wave5_crosspaths import (  # noqa: E402
    _hdf5_series_metadata_equal,
    build_small_crosspath_inputs,
    hdf5_density_exchange,
    read_fixed_mesh_series_hdf5,
    write_fixed_mesh_series_hdf5,
)


def test_hdf5_density_exchange_preserves_values_metadata_and_observables(tmp_path):
    density, _displacement, _surface, targets = build_small_crosspath_inputs()
    result = hdf5_density_exchange(density, targets, tmp_path / "density.h5")

    assert result["metadata_equal"] == {
        "axis": True,
        "axis_kind": True,
        "coordinate_frame": True,
        "coordinate_origin": True,
        "density_kind": True,
        "geometry": True,
        "ids": True,
        "units": True,
        "values": True,
        "volumes": True,
    }
    for observable in ("potential", "acceleration", "gradient"):
        assert (
            result["comparison"][observable]["summary"]["maximum_normalized"] <= 1e-14
        )


@pytest.mark.parametrize(
    ("series_index", "quantity", "specific"),
    [
        (1, "displacement", {"background_mass"}),
        (2, "surface_density", {"areas", "normals"}),
    ],
)
def test_hdf5_small_non_density_metadata_is_observed(
    tmp_path, series_index, quantity, specific
):
    series = build_small_crosspath_inputs()[series_index]
    path = tmp_path / f"{quantity}.h5"
    write_fixed_mesh_series_hdf5(path, series)
    provider = read_fixed_mesh_series_hdf5(path)

    metadata = _hdf5_series_metadata_equal(series, provider, path, quantity)

    assert set(metadata) == {
        "positions",
        "ids",
        "axis",
        "axis_kind",
        "coordinate_frame",
        "coordinate_origin",
        "quantity",
        "phase",
        "units",
        "values",
        *specific,
    }
    assert all(metadata.values())
