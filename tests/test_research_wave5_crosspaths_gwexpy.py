"""Native GWexpy exchange checks for the Wave 5 moving-density study."""

from __future__ import annotations

import pytest

from scripts.research_wave5_crosspaths import (  # noqa: E402
    _gwexpy_series_metadata_equal,
    build_small_crosspath_inputs,
    from_gwexpy_displacement_series,
    from_gwexpy_surface_density_series,
    gwexpy_density_exchange,
    to_gwexpy_displacement_series,
    to_gwexpy_surface_density_series,
)


def test_gwexpy_density_exchange_uses_current_values_and_preserves_metadata():
    density, _displacement, _surface, targets = build_small_crosspath_inputs()
    result = gwexpy_density_exchange(
        density,
        targets,
        current_value_probe=(1, 1, 0.75),
    )

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
    probe = result["native_current_value_probe"]
    assert probe["restored_density_matches_mutated_native_value"]
    assert all(probe["observable_changed"].values())


@pytest.mark.parametrize(
    ("series_index", "quantity", "to_native", "from_native", "specific"),
    [
        (
            1,
            "displacement",
            to_gwexpy_displacement_series,
            from_gwexpy_displacement_series,
            {"background_mass"},
        ),
        (
            2,
            "surface_density",
            to_gwexpy_surface_density_series,
            from_gwexpy_surface_density_series,
            {"areas", "normals"},
        ),
    ],
)
def test_gwexpy_small_non_density_metadata_and_native_values_are_observed(
    series_index, quantity, to_native, from_native, specific
):
    series = build_small_crosspath_inputs()[series_index]
    bundle = to_native(series)
    restored = from_native(bundle)

    metadata = _gwexpy_series_metadata_equal(series, bundle, restored, quantity)

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
