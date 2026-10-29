from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

import gragra.adapters.gwexpy_observable as adapter

UNITS = {
    "potential": "m²/s²",
    "acceleration": "m/s²",
    "gradient": "1/s²",
}
COMPONENT_SHAPES = {
    "potential": (),
    "acceleration": (3,),
    "gradient": (3, 3),
}


def _case(observable: str, layout: str, *, complex_values: bool = False):
    m, samples = 2, 3
    suffix = COMPONENT_SHAPES[observable]
    if layout == "static":
        shape = (m, *suffix)
        axis = np.array([12.5])
    elif layout == "time_first":
        shape = (samples, m, *suffix)
        axis = np.array([-0.2, 0.1, 0.7])
    else:
        shape = (m, samples, *suffix)
        axis = np.array([-2.0, 0.0, 4.0])
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 0.25
    if complex_values:
        values = values + 1j * (values + 0.5)
    positions = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    metadata = {
        "coordinate_frame": "laboratory",
        "coordinate_origin_m": np.array([0.1, 0.2, 0.3]),
        "target_ids": ["A", "B"],
        "t_ref_s": 10.0,
        "phase_origin_m": None,
        "fourier_convention": "exp(-iwt)",
        "name": "fixture",
        "channel": "X1",
        "source_metadata": {"revision": "abc", "indices": [1, 2]},
        "quadrature_metadata": {"weights": np.array([0.25, 0.75])},
        "material_metadata": {"density": 2300.0},
        "unit": UNITS[observable],
    }
    return values, positions, axis, metadata


@pytest.mark.parametrize("observable", list(UNITS))
@pytest.mark.parametrize("layout", ["static", "time_first", "frequency_target_first"])
def test_round_trip_all_observables_and_layouts(observable, layout):
    values, positions, axis, metadata = _case(
        observable, layout, complex_values=layout == "frequency_target_first"
    )
    bundle = adapter.to_gwexpy_observable(
        values,
        positions,
        observable=observable,
        layout=layout,
        axis_values=axis,
        metadata=metadata,
    )
    restored = adapter.from_gwexpy_observable(bundle)

    assert isinstance(bundle, adapter.GwexpyObservableBundle)
    assert isinstance(restored, adapter.ObservableArrayBundle)
    assert restored.values.dtype == values.dtype
    assert np.array_equal(restored.values, values)
    assert not restored.values.flags.writeable
    assert np.array_equal(restored.companion["positions_m"], positions)
    assert np.array_equal(restored.companion["axis_values"], axis)
    assert restored.companion["unit"] == UNITS[observable]
    assert "unit" not in restored.companion["metadata"]

    field = bundle.field
    components = [field] if observable == "potential" else list(field.values())
    expected_native_shape = (1 if layout == "static" else 3, 2, 1, 1)
    assert all(component.shape == expected_native_shape for component in components)
    if observable == "acceleration":
        assert list(field.keys()) == ["x", "y", "z"]
    if observable == "gradient":
        assert list(field.keys()) == [(i, j) for i in range(3) for j in range(3)]


def test_inverse_reads_current_native_component_values():
    values, positions, axis, metadata = _case("acceleration", "time_first")
    bundle = adapter.to_gwexpy_observable(
        values,
        positions,
        observable="acceleration",
        layout="time_first",
        axis_values=axis,
        metadata=metadata,
    )
    bundle.field["y"].value[2, 1, 0, 0] = 987.5
    restored = adapter.from_gwexpy_observable(bundle)
    assert restored.values[2, 1, 1] == 987.5
    assert values[2, 1, 1] != 987.5


def test_inverse_uses_explicit_tensor_keys_not_dictionary_order():
    values, positions, axis, metadata = _case("gradient", "time_first")
    bundle = adapter.to_gwexpy_observable(
        values,
        positions,
        observable="gradient",
        layout="time_first",
        axis_values=axis,
        metadata=metadata,
    )
    reversed_field = type(bundle.field)(
        dict(reversed(tuple(bundle.field.items()))), rank=2
    )
    reordered = adapter.GwexpyObservableBundle(reversed_field, bundle.companion)
    assert np.array_equal(adapter.from_gwexpy_observable(reordered).values, values)


def test_omitted_unit_uses_the_observable_canonical_si_unit():
    values, positions, axis, metadata = _case("potential", "static")
    metadata.pop("unit")
    bundle = adapter.to_gwexpy_observable(
        values,
        positions,
        observable="potential",
        layout="static",
        axis_values=axis,
        metadata=metadata,
    )
    assert bundle.companion["unit"] == "m²/s²"
    assert adapter.from_gwexpy_observable(bundle).companion["unit"] == "m²/s²"


def test_float64_normalization_rejects_overflow():
    values, positions, axis, metadata = _case("potential", "static")
    values = np.array(
        [np.longdouble(np.finfo(np.float64).max) * np.longdouble(2.0)] * 2
    )
    with pytest.raises(ValueError, match="float64"):
        adapter.to_gwexpy_observable(
            values,
            positions,
            observable="potential",
            layout="static",
            axis_values=axis,
            metadata=metadata,
        )


def test_inputs_and_recursive_metadata_are_defensively_frozen():
    values, positions, axis, metadata = _case("potential", "static")
    origin = metadata["coordinate_origin_m"]
    weights = metadata["quadrature_metadata"]["weights"]
    bundle = adapter.to_gwexpy_observable(
        values,
        positions,
        observable="potential",
        layout="static",
        axis_values=axis,
        metadata=metadata,
    )
    positions[0, 0] = 999
    axis[0] = 999
    origin[0] = 999
    weights[0] = 999
    metadata["source_metadata"]["revision"] = "changed"

    companion = bundle.companion
    assert isinstance(companion, MappingProxyType)
    assert companion["positions_m"][0, 0] == 1.0
    assert companion["axis_values"][0] == 12.5
    assert companion["metadata"]["coordinate_origin_m"][0] == 0.1
    assert companion["metadata"]["quadrature_metadata"]["weights"][0] == 0.25
    assert companion["metadata"]["source_metadata"]["revision"] == "abc"
    assert isinstance(companion["metadata"]["target_ids"], tuple)
    assert not companion["positions_m"].flags.writeable
    assert not companion["metadata"]["quadrature_metadata"]["weights"].flags.writeable
    with pytest.raises(FrozenInstanceError):
        bundle.field = None


def test_blockwise_round_trip_can_be_combined_without_transpose():
    values, positions, axis, metadata = _case(
        "gradient", "frequency_target_first", complex_values=True
    )
    blocks = []
    for target_slice in (slice(0, 1), slice(1, 2)):
        block_metadata = dict(metadata)
        block_metadata["target_ids"] = metadata["target_ids"][target_slice]
        bundle = adapter.to_gwexpy_observable(
            values[target_slice],
            positions[target_slice],
            observable="gradient",
            layout="frequency_target_first",
            axis_values=axis,
            metadata=block_metadata,
        )
        blocks.append(adapter.from_gwexpy_observable(bundle).values)
    assert np.array_equal(np.concatenate(blocks, axis=0), values)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda v, p, a, m: (v.astype(bool), p, a, m), TypeError),
        (lambda v, p, a, m: (v.astype(object), p, a, m), TypeError),
        (lambda v, p, a, m: (np.full_like(v, np.nan), p, a, m), ValueError),
        (lambda v, p, a, m: (v, p.astype(complex), a, m), TypeError),
        (lambda v, p, _a, m: (v, p, np.array([0.0, 0.0, 1.0]), m), ValueError),
        (lambda v, p, a, m: (v, p, a, {**m, "unknown": 1}), TypeError),
        (lambda v, p, a, m: (v, p, a, {**m, "target_ids": ["A", "A"]}), ValueError),
        (lambda v, p, a, m: (v, p, a, {**m, "unit": "km/s²"}), ValueError),
    ],
)
def test_invalid_inputs_are_rejected(mutate, error):
    values, positions, axis, metadata = _case("acceleration", "time_first")
    values, positions, axis, metadata = mutate(values, positions, axis, metadata)
    with pytest.raises(error):
        adapter.to_gwexpy_observable(
            values,
            positions,
            observable="acceleration",
            layout="time_first",
            axis_values=axis,
            metadata=metadata,
        )


def test_complex_values_require_explicit_phase_metadata():
    values, positions, axis, metadata = _case(
        "potential", "static", complex_values=True
    )
    for key in ("t_ref_s", "phase_origin_m", "fourier_convention"):
        invalid = dict(metadata)
        invalid.pop(key)
        with pytest.raises(ValueError, match=key):
            adapter.to_gwexpy_observable(
                values,
                positions,
                observable="potential",
                layout="static",
                axis_values=axis,
                metadata=invalid,
            )


def test_recursive_metadata_rejects_cycles_and_object_arrays():
    values, positions, axis, metadata = _case("potential", "static")
    cyclic = {}
    cyclic["self"] = cyclic
    metadata["source_metadata"] = cyclic
    with pytest.raises(ValueError, match="cyclic"):
        adapter.to_gwexpy_observable(
            values,
            positions,
            observable="potential",
            layout="static",
            axis_values=axis,
            metadata=metadata,
        )
    metadata["source_metadata"] = {"bad": np.array([object()], dtype=object)}
    with pytest.raises(TypeError, match="object"):
        adapter.to_gwexpy_observable(
            values,
            positions,
            observable="potential",
            layout="static",
            axis_values=axis,
            metadata=metadata,
        )


def test_inverse_rejects_bare_field_and_tampering():
    values, positions, axis, metadata = _case("acceleration", "time_first")
    bundle = adapter.to_gwexpy_observable(
        values,
        positions,
        observable="acceleration",
        layout="time_first",
        axis_values=axis,
        metadata=metadata,
    )
    with pytest.raises(TypeError, match="GwexpyObservableBundle"):
        adapter.from_gwexpy_observable(bundle.field)

    missing = adapter.GwexpyObservableBundle(
        type(bundle.field)({"x": bundle.field["x"], "y": bundle.field["y"]}),
        bundle.companion,
    )
    with pytest.raises(ValueError, match="components"):
        adapter.from_gwexpy_observable(missing)

    bundle.field["x"]._axis0_index[0] += 1e-4 * bundle.field["x"]._axis0_index.unit
    with pytest.raises(ValueError, match="axis0"):
        adapter.from_gwexpy_observable(bundle)

    fresh = adapter.to_gwexpy_observable(
        values,
        positions,
        observable="acceleration",
        layout="time_first",
        axis_values=axis,
        metadata=metadata,
    )
    fresh.field["z"]._unit = "kg"
    with pytest.raises(ValueError, match="unit"):
        adapter.from_gwexpy_observable(fresh)


@pytest.mark.parametrize("axis_name", ["_axis1_index", "_axis2_index", "_axis3_index"])
def test_inverse_rejects_spatial_axis_value_or_unit_tampering(axis_name):
    from astropy import units

    values, positions, axis, metadata = _case("potential", "static")
    bundle = adapter.to_gwexpy_observable(
        values,
        positions,
        observable="potential",
        layout="static",
        axis_values=axis,
        metadata=metadata,
    )
    native_axis = getattr(bundle.field, axis_name)
    setattr(bundle.field, axis_name, native_axis.value * units.m)
    with pytest.raises(ValueError, match="spatial axis"):
        adapter.from_gwexpy_observable(bundle)


def test_direct_array_bundle_copies_and_freezes_values():
    values = np.array([1, 2, 3], dtype=np.int64)
    bundle = adapter.ObservableArrayBundle(values, {"nested": {"value": 1}})
    values[0] = 99
    assert np.array_equal(bundle.values, [1, 2, 3])
    assert not bundle.values.flags.writeable
    assert isinstance(bundle.companion, MappingProxyType)
