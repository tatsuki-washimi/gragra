from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

import gragra.adapters.gwexpy_source as adapter
from gragra.gravity_sources.volume import VolumeElementSource


def _source(density: object = np.array([2.0, 3.5])) -> VolumeElementSource:
    return VolumeElementSource(
        positions_m=np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]]),
        volumes_m3=np.array([0.25, 2.0]),
        density_kg_m3=density,
    )


def _bundle(*, density: object = np.array([2.0, 3.5]), kind: str = "rho"):
    return adapter.to_gwexpy_density_snapshot(
        _source(density),
        time_s=12.5,
        density_kind=kind,
        coordinate_frame="laboratory",
        coordinate_origin_m=np.array([0.1, 0.2, 0.3]),
        source_ids=["left", "right"],
        source_metadata={"run": "fixture", "nested": {"tag": 3}},
    )


def test_round_trip_preserves_native_schema_and_source_values():
    bundle = _bundle()
    field = bundle.field
    restored = adapter.from_gwexpy_density_snapshot(bundle)
    assert field.shape == (1, 2, 1, 1)
    assert np.array_equal(field.value[:, :, 0, 0], [[2.0, 3.5]])
    assert tuple(field.axis_names) == ("time", "point", "dummy_y", "dummy_z")
    assert restored.time_s == 12.5
    assert restored.source.density_kg_m3.dtype == np.float64
    assert np.array_equal(restored.source.density_kg_m3, [2.0, 3.5])
    assert np.array_equal(restored.source.positions_m, _source().positions_m)
    assert np.array_equal(restored.source.volumes_m3, _source().volumes_m3)
    assert np.array_equal(restored.source.as_weighted_source().weights_kg, [0.5, 7.0])
    assert restored.companion["unit"] == "kg/m^3"
    assert set(restored.companion) == {
        "density_kind",
        "positions_m",
        "volumes_m3",
        "time_s",
        "unit",
        "coordinate_frame",
        "coordinate_origin_m",
        "source_ids",
        "source_metadata",
    }


def test_scalar_density_broadcasts_and_inverse_reads_current_native_value():
    bundle = _bundle(density=4.0)
    assert np.array_equal(bundle.field.value[:, :, 0, 0], [[4.0, 4.0]])
    bundle.field.value[0, 1, 0, 0] = 8.0
    restored = adapter.from_gwexpy_density_snapshot(bundle)
    assert np.array_equal(restored.source.density_kg_m3, [4.0, 8.0])
    assert np.array_equal(restored.source.as_weighted_source().weights_kg, [1.0, 16.0])


def test_inverse_accepts_a_caller_constructed_native_field_with_sidecar():
    bundle = _bundle(kind="delta_rho")
    from astropy import units
    from gwexpy import fields

    caller_field = fields.ScalarField(
        np.array([[[[-2.0]], [[6.0]]]]),
        unit="kg/m^3",
        axis0=np.array([12.5]) * units.s,
        axis1=np.arange(2.0),
        axis2=np.array([0.0]),
        axis3=np.array([0.0]),
        axis_names=["time", "point", "dummy_y", "dummy_z"],
        axis0_domain="time",
        space_domain={"point": "real", "dummy_y": "real", "dummy_z": "real"},
    )
    restored = adapter.from_gwexpy_density_snapshot(
        adapter.GwexpyDensitySnapshotBundle(caller_field, bundle.companion)
    )
    assert np.array_equal(restored.source.density_kg_m3, [-2.0, 6.0])


def test_companion_is_defensive_recursively_immutable_and_array_bundle_copies():
    source = _source()
    origin = np.array([0.1, 0.2, 0.3])
    metadata = {"array": np.array([4.0]), "nested": {"id": "a"}}
    bundle = adapter.to_gwexpy_density_snapshot(
        source,
        time_s=1.0,
        density_kind="rho",
        coordinate_frame="lab",
        coordinate_origin_m=origin,
        source_metadata=metadata,
    )
    origin[0] = 9.0
    metadata["array"][0] = 9.0
    restored = adapter.from_gwexpy_density_snapshot(bundle)
    assert isinstance(bundle.companion, MappingProxyType)
    assert bundle.companion["coordinate_origin_m"][0] == 0.1
    assert bundle.companion["source_metadata"]["array"][0] == 4.0
    assert not bundle.companion["positions_m"].flags.writeable
    assert not restored.source.positions_m.flags.writeable
    with pytest.raises(FrozenInstanceError):
        bundle.field = None


@pytest.mark.parametrize(
    ("density", "kind", "error"),
    [
        (np.array([1 + 2j, 2 + 3j]), "rho", TypeError),
        (np.array([-1.0, 2.0]), "rho", ValueError),
        (np.array([1.0, np.inf]), "delta_rho", ValueError),
        (np.array([True, False]), "delta_rho", TypeError),
        (np.array([1.0]), "rho", ValueError),
        (np.array([1.0, 2.0]), "wrong", ValueError),
    ],
)
def test_forward_rejects_invalid_density(density, kind, error):
    with pytest.raises(error):
        _bundle(density=density, kind=kind)


def test_forward_rejects_invalid_sidecar_arguments_and_metadata():
    source = _source()
    args = {
        "time_s": 1.0,
        "density_kind": "rho",
        "coordinate_frame": "lab",
        "coordinate_origin_m": np.zeros(3),
    }
    with pytest.raises(ValueError):
        adapter.to_gwexpy_density_snapshot(source, **{**args, "coordinate_frame": ""})
    with pytest.raises(ValueError):
        adapter.to_gwexpy_density_snapshot(source, **{**args, "source_ids": ["x", "x"]})
    with pytest.raises(ValueError):
        adapter.to_gwexpy_density_snapshot(source, **{**args, "source_ids": ["x"]})
    with pytest.raises(TypeError):
        adapter.to_gwexpy_density_snapshot(source, **{**args, "source_ids": ["x", 2]})
    with pytest.raises(TypeError):
        adapter.to_gwexpy_density_snapshot(
            source, **{**args, "source_metadata": {"x": object()}}
        )
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="cyclic"):
        adapter.to_gwexpy_density_snapshot(
            source, **{**args, "source_metadata": cyclic}
        )


def test_inverse_rejects_bare_field_and_schema_tampering():
    bundle = _bundle()
    with pytest.raises(TypeError, match="GwexpyDensitySnapshotBundle"):
        adapter.from_gwexpy_density_snapshot(bundle.field)
    bad = adapter.GwexpyDensitySnapshotBundle(bundle.field, {"time_s": 12.5})
    with pytest.raises(ValueError, match="companion"):
        adapter.from_gwexpy_density_snapshot(bad)
    bundle.field._unit = "g/cm3"
    with pytest.raises(ValueError, match="unit"):
        adapter.from_gwexpy_density_snapshot(bundle)


def test_inverse_rejects_axes_shape_domain_and_rho_tampering():
    from astropy import units

    bundle = _bundle()
    bundle.field._axis0_index[0] += 1e-4 * units.s
    with pytest.raises(ValueError, match="axis0"):
        adapter.from_gwexpy_density_snapshot(bundle)
    fresh = _bundle()
    fresh.field.value[0, 0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="rho"):
        adapter.from_gwexpy_density_snapshot(fresh)
    fresh = _bundle()
    fresh.field._axis1_index = fresh.field._axis1_index.value * units.m
    with pytest.raises(ValueError, match="spatial axis"):
        adapter.from_gwexpy_density_snapshot(fresh)


def test_inverse_rejects_scaled_axis_units():
    from astropy import units

    bundle = _bundle()
    bundle.field._axis0_index = bundle.field._axis0_index.value * units.ms
    with pytest.raises(ValueError, match="SI seconds"):
        adapter.from_gwexpy_density_snapshot(bundle)
    bundle = _bundle()
    bundle.field._axis1_index = bundle.field._axis1_index.value * units.percent
    with pytest.raises(ValueError, match="unscaled dimensionless"):
        adapter.from_gwexpy_density_snapshot(bundle)


@pytest.mark.parametrize("axis_name", ["_axis0_index", "_axis1_index"])
def test_inverse_rejects_complex_native_axes_even_with_zero_imaginary_part(axis_name):
    from astropy import units

    bundle = _bundle()
    native_axis = getattr(bundle.field, axis_name)
    unit = units.s if axis_name == "_axis0_index" else units.dimensionless_unscaled
    setattr(bundle.field, axis_name, native_axis.value.astype(complex) * unit)
    with pytest.raises(ValueError, match="real finite"):
        adapter.from_gwexpy_density_snapshot(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: setattr(f, "_axis1_name", "x"), "names"),
        (lambda f: setattr(f, "_axis0_domain", "frequency"), "domains"),
        (lambda f: setattr(f, "_space_domains", {"point": "frequency"}), "domains"),
        (lambda f: setattr(f, "_axis1_index", np.array([4.0, 5.0])), "spatial axis"),
        (lambda f: setattr(f, "_axis2_index", np.array([1.0])), "spatial axis"),
    ],
)
def test_inverse_rejects_native_axis_names_domains_and_values(mutate, message):
    bundle = _bundle()
    mutate(bundle.field)
    with pytest.raises(ValueError, match=message):
        adapter.from_gwexpy_density_snapshot(bundle)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_inverse_rejects_nonfinite_current_native_value(bad_value):
    bundle = _bundle(kind="delta_rho")
    bundle.field.value[0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        adapter.from_gwexpy_density_snapshot(bundle)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("time_s"),
        lambda c: c.__setitem__("unexpected", 1),
        lambda c: c.__setitem__("positions_m", np.zeros((1, 3))),
        lambda c: c.__setitem__("volumes_m3", np.array([0.0, 1.0])),
        lambda c: c.__setitem__("coordinate_origin_m", np.zeros(2)),
        lambda c: c.__setitem__("time_s", np.inf),
        lambda c: c.__setitem__("source_ids", ("only-one",)),
    ],
)
def test_inverse_revalidates_tampered_companion(mutate):
    bundle = _bundle()
    companion = dict(bundle.companion)
    mutate(companion)
    with pytest.raises((TypeError, ValueError)):
        caller_bundle = adapter.GwexpyDensitySnapshotBundle(bundle.field, companion)
        adapter.from_gwexpy_density_snapshot(caller_bundle)


def test_forward_rejects_empty_source_and_metadata_object_array():
    empty = VolumeElementSource(np.empty((0, 3)), np.empty(0), np.empty(0))
    with pytest.raises(ValueError, match="at least one"):
        adapter.to_gwexpy_density_snapshot(
            empty,
            time_s=0.0,
            density_kind="rho",
            coordinate_frame="lab",
            coordinate_origin_m=np.zeros(3),
        )
    with pytest.raises(TypeError, match="object"):
        adapter.to_gwexpy_density_snapshot(
            _source(),
            time_s=0.0,
            density_kind="rho",
            coordinate_frame="lab",
            coordinate_origin_m=np.zeros(3),
            source_metadata={"bad": np.array([object()], dtype=object)},
        )
