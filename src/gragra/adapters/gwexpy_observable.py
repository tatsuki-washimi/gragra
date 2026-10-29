"""Round-trip multi-target observable results through GWexpy fields."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

_UNITS = {
    "potential": "m²/s²",
    "acceleration": "m/s²",
    "gradient": "1/s²",
}
_COMPONENT_SHAPES = {
    "potential": (),
    "acceleration": (3,),
    "gradient": (3, 3),
}
_LAYOUTS = {"static", "time_first", "frequency_target_first"}
_VECTOR_KEYS = ("x", "y", "z")
_TENSOR_KEYS = tuple((i, j) for i in range(3) for j in range(3))
_METADATA_KEYS = {
    "coordinate_frame",
    "coordinate_origin_m",
    "target_ids",
    "t_ref_s",
    "phase_origin_m",
    "fourier_convention",
    "name",
    "channel",
    "source_metadata",
    "quadrature_metadata",
    "material_metadata",
    "unit",
}
_PROVENANCE_KEYS = (
    "source_metadata",
    "quadrature_metadata",
    "material_metadata",
)


def _freeze(value: Any, active: set[int] | None = None) -> Any:
    """Return a defensive, recursively immutable copy of supported metadata."""
    if active is None:
        active = set()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("metadata object arrays are not supported")
        if not (
            np.issubdtype(value.dtype, np.number)
            or np.issubdtype(value.dtype, np.bool_)
            or np.issubdtype(value.dtype, np.str_)
        ):
            raise TypeError(f"unsupported metadata array dtype {value.dtype}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("metadata arrays must contain only finite values")
        copied = np.array(value, copy=True)
        copied.setflags(write=False)
        return copied
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError("cyclic metadata is not supported")
        active.add(marker)
        try:
            copied = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("metadata mapping keys must be strings")
                copied[key] = _freeze(item, active)
            return MappingProxyType(copied)
        finally:
            active.remove(marker)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        marker = id(value)
        if marker in active:
            raise ValueError("cyclic metadata is not supported")
        active.add(marker)
        try:
            return tuple(_freeze(item, active) for item in value)
        finally:
            active.remove(marker)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Real):
        if not np.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value.item() if isinstance(value, np.generic) else value
    raise TypeError(f"unsupported metadata value type {type(value).__name__}")


@dataclass(frozen=True)
class GwexpyObservableBundle:
    """A native GWexpy field plus immutable gragra sidecar metadata."""

    field: object
    companion: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "companion", _freeze(self.companion))


@dataclass(frozen=True)
class ObservableArrayBundle:
    """A read-only NumPy observable array plus immutable sidecar metadata."""

    values: np.ndarray
    companion: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a numpy ndarray")
        values = np.array(self.values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "companion", _freeze(self.companion))


def _import_gwexpy():
    try:
        import gwexpy
        from astropy import units
    except ImportError as exc:
        raise ImportError(
            "GWexpy observable import failed; reinstall gragra with its runtime "
            "dependencies."
        ) from exc
    return gwexpy, units


def _numeric_array(name: str, value: object, *, real: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray")
    if np.issubdtype(value.dtype, np.bool_) or value.dtype.hasobject:
        raise TypeError(f"{name} must be a numeric non-boolean array")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if real and np.iscomplexobj(value):
        raise TypeError(f"{name} must be real-valued")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    dtype = np.float64 if real or not np.iscomplexobj(value) else np.complex128
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.asarray(value, dtype=dtype)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} cannot be represented as finite {dtype.__name__}")
    return normalized


def _layout_dimensions(
    values: np.ndarray, observable: str, layout: str
) -> tuple[int, int]:
    if observable not in _UNITS:
        raise ValueError(
            f"observable must be one of {tuple(_UNITS)}, got {observable!r}"
        )
    if layout not in _LAYOUTS:
        raise ValueError(
            f"layout must be one of {tuple(sorted(_LAYOUTS))}, got {layout!r}"
        )
    suffix = _COMPONENT_SHAPES[observable]
    if layout == "static":
        expected_ndim = 1 + len(suffix)
        if values.ndim != expected_ndim or values.shape[1:] != suffix:
            raise ValueError(
                f"{observable} static values must have shape (M, {suffix!r})"
            )
        m, samples = values.shape[0], 1
    elif layout == "time_first":
        expected_ndim = 2 + len(suffix)
        if values.ndim != expected_ndim or values.shape[2:] != suffix:
            raise ValueError(
                f"{observable} time_first values must have shape (T, M, {suffix!r})"
            )
        samples, m = values.shape[:2]
    else:
        expected_ndim = 2 + len(suffix)
        if values.ndim != expected_ndim or values.shape[2:] != suffix:
            raise ValueError(
                f"{observable} frequency_target_first values must have "
                f"shape (M, F, {suffix!r})"
            )
        m, samples = values.shape[:2]
    if m < 1 or samples < 1:
        raise ValueError("target and sample dimensions must be non-empty")
    return m, samples


def _finite_real(name: str, value: object, *, allow_none: bool = False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _real_vector3(name: str, value: object, *, allow_none: bool = False):
    if value is None and allow_none:
        return None
    array = _numeric_array(name, value, real=True)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    copied = np.array(array, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


def _normalize_metadata(
    metadata: Mapping[str, object],
    *,
    observable: str,
    m: int,
    needs_phase: bool,
) -> tuple[str, Mapping[str, object]]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    unknown = set(metadata) - _METADATA_KEYS
    if unknown:
        raise TypeError(f"unknown metadata keys: {sorted(unknown)}")
    for required in ("coordinate_frame", "coordinate_origin_m"):
        if required not in metadata:
            raise ValueError(f"metadata.{required} is required")
    frame = metadata["coordinate_frame"]
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError("metadata.coordinate_frame must be a non-empty string")
    unit = metadata.get("unit", _UNITS[observable])
    if not isinstance(unit, str) or unit != _UNITS[observable]:
        raise ValueError(
            f"metadata.unit must be the canonical SI label {_UNITS[observable]!r}"
        )
    target_ids = metadata.get("target_ids", tuple(str(i) for i in range(m)))
    if (
        not isinstance(target_ids, Sequence)
        or isinstance(target_ids, (str, bytes))
        or len(target_ids) != m
        or any(not isinstance(item, str) for item in target_ids)
    ):
        raise TypeError(f"metadata.target_ids must be a sequence of {m} strings")
    if len(set(target_ids)) != m:
        raise ValueError("metadata.target_ids must be unique")
    if needs_phase:
        for required in ("t_ref_s", "phase_origin_m", "fourier_convention"):
            if required not in metadata:
                raise ValueError(f"metadata.{required} is required")
    t_ref = _finite_real("metadata.t_ref_s", metadata.get("t_ref_s"), allow_none=True)
    phase_origin = _real_vector3(
        "metadata.phase_origin_m",
        metadata.get("phase_origin_m"),
        allow_none=True,
    )
    convention = metadata.get("fourier_convention")
    if convention not in (None, "exp(-iwt)", "exp(+iwt)"):
        raise ValueError(
            "metadata.fourier_convention must be 'exp(-iwt)', 'exp(+iwt)', or None"
        )
    if needs_phase and (t_ref is None or convention is None):
        missing = "t_ref_s" if t_ref is None else "fourier_convention"
        raise ValueError(f"metadata.{missing} is required")
    normalized: dict[str, object] = {
        "coordinate_frame": frame,
        "coordinate_origin_m": _real_vector3(
            "metadata.coordinate_origin_m", metadata["coordinate_origin_m"]
        ),
        "target_ids": tuple(target_ids),
        "t_ref_s": t_ref,
        "phase_origin_m": phase_origin,
        "fourier_convention": convention,
    }
    for key in ("name", "channel"):
        value = metadata.get(key)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"metadata.{key} must be a string or None")
        normalized[key] = value
    for key in _PROVENANCE_KEYS:
        value = metadata.get(key, {})
        if not isinstance(value, Mapping):
            raise TypeError(f"metadata.{key} must be a mapping")
        normalized[key] = value
    return unit, _freeze(normalized)


def _normalize_axis(axis_values: object, layout: str, samples: int) -> np.ndarray:
    axis = _numeric_array("axis_values", axis_values, real=True)
    if axis.ndim != 1 or len(axis) != samples:
        raise ValueError(f"axis_values must have shape ({samples},)")
    if layout != "static" and np.any(np.diff(axis) <= 0):
        raise ValueError("axis_values must be strictly increasing")
    copied = np.array(axis, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


def _native_values(values: np.ndarray, layout: str) -> np.ndarray:
    if layout == "static":
        return values[np.newaxis, ...]
    if layout == "time_first":
        return values
    return np.swapaxes(values, 0, 1)


def _make_scalar(gwexpy, units, data: np.ndarray, unit: str, axis, layout: str):
    axis_unit = units.Hz if layout == "frequency_target_first" else units.s
    domain = "frequency" if layout == "frequency_target_first" else "time"
    axis_name = "frequency" if domain == "frequency" else "time"
    return gwexpy.fields.ScalarField(
        data[:, :, np.newaxis, np.newaxis],
        unit=unit,
        axis0=axis * axis_unit,
        axis1=np.arange(data.shape[1], dtype=np.float64),
        axis2=np.array([0.0]),
        axis3=np.array([0.0]),
        axis_names=[axis_name, "point", "dummy_y", "dummy_z"],
        axis0_domain=domain,
        space_domain={"point": "real", "dummy_y": "real", "dummy_z": "real"},
    )


def to_gwexpy_observable(
    values: np.ndarray,
    positions_m: np.ndarray,
    *,
    observable: str,
    layout: str,
    axis_values: np.ndarray,
    metadata: Mapping[str, object],
) -> GwexpyObservableBundle:
    """Convert a gragra multi-target result into a native GWexpy field bundle."""
    normalized_values = _numeric_array("values", values)
    m, samples = _layout_dimensions(normalized_values, observable, layout)
    positions = _numeric_array("positions_m", positions_m, real=True)
    if positions.shape != (m, 3):
        raise ValueError(f"positions_m must have shape ({m}, 3)")
    positions = np.array(positions, dtype=np.float64, copy=True)
    positions.setflags(write=False)
    axis = _normalize_axis(axis_values, layout, samples)
    unit, normalized_metadata = _normalize_metadata(
        metadata,
        observable=observable,
        m=m,
        needs_phase=np.iscomplexobj(normalized_values)
        or layout == "frequency_target_first",
    )
    gwexpy, units = _import_gwexpy()
    native = _native_values(normalized_values, layout)
    suffix = _COMPONENT_SHAPES[observable]
    if observable == "potential":
        field = _make_scalar(gwexpy, units, native, unit, axis, layout)
    elif observable == "acceleration":
        components = {
            key: _make_scalar(gwexpy, units, native[..., i], unit, axis, layout)
            for i, key in enumerate(_VECTOR_KEYS)
        }
        field = gwexpy.fields.VectorField(components)
    else:
        components = {
            key: _make_scalar(
                gwexpy, units, native[..., key[0], key[1]], unit, axis, layout
            )
            for key in _TENSOR_KEYS
        }
        field = gwexpy.fields.TensorField(components, rank=len(suffix))
    companion = {
        "observable": observable,
        "layout": layout,
        "positions_m": positions,
        "axis_values": axis,
        "unit": unit,
        "metadata": normalized_metadata,
    }
    return GwexpyObservableBundle(field, companion)


def _field_components(gwexpy, field, observable: str):
    if observable == "potential":
        if not isinstance(field, gwexpy.fields.ScalarField):
            raise TypeError("potential bundle field must be a ScalarField")
        return (field,)
    if observable == "acceleration":
        if not isinstance(field, gwexpy.fields.VectorField):
            raise TypeError("acceleration bundle field must be a VectorField")
        if set(field.keys()) != set(_VECTOR_KEYS):
            raise ValueError("VectorField components must be exactly x, y, z")
        return tuple(field[key] for key in _VECTOR_KEYS)
    if not isinstance(field, gwexpy.fields.TensorField):
        raise TypeError("gradient bundle field must be a TensorField")
    if set(field.keys()) != set(_TENSOR_KEYS):
        raise ValueError(
            "TensorField components must be the complete 3x3 tuple-key set"
        )
    return tuple(field[key] for key in _TENSOR_KEYS)


def _validate_native_component(
    component,
    *,
    gwexpy,
    units,
    unit: str,
    axis: np.ndarray,
    layout: str,
    m: int,
) -> np.ndarray:
    if not isinstance(component, gwexpy.fields.ScalarField):
        raise TypeError("all native components must be ScalarField instances")
    expected_shape = (len(axis), m, 1, 1)
    if component.shape != expected_shape:
        raise ValueError(
            f"native component shape {component.shape} does not match {expected_shape}"
        )
    expected_unit = units.Unit(unit)
    if component.unit != expected_unit:
        raise ValueError(
            f"native component unit {component.unit!r} does not match {unit!r}"
        )
    axis_unit = units.Hz if layout == "frequency_target_first" else units.s
    try:
        native_axis = np.asarray(component._axis0_index.to_value(axis_unit))
    except Exception as exc:
        raise ValueError("native axis0 has the wrong physical unit") from exc
    if not np.allclose(native_axis, axis, rtol=0.0, atol=1e-12):
        raise ValueError("native axis0 does not match companion axis_values")
    spatial_axes = (
        ("_axis1_index", np.arange(m, dtype=np.float64)),
        ("_axis2_index", np.array([0.0])),
        ("_axis3_index", np.array([0.0])),
    )
    for axis_name, expected in spatial_axes:
        try:
            native_spatial = np.asarray(
                getattr(component, axis_name).to_value(units.dimensionless_unscaled)
            )
        except Exception as exc:
            raise ValueError(
                f"native spatial axis {axis_name} must be dimensionless"
            ) from exc
        if not np.array_equal(native_spatial, expected):
            raise ValueError(
                f"native spatial axis {axis_name} does not match the adapter layout"
            )
    expected_axis_names = (
        "frequency" if layout == "frequency_target_first" else "time",
        "point",
        "dummy_y",
        "dummy_z",
    )
    if tuple(component.axis_names) != expected_axis_names:
        raise ValueError("native spatial axis names do not match the adapter layout")
    if component.space_domains != {
        "point": "real",
        "dummy_y": "real",
        "dummy_z": "real",
    }:
        raise ValueError("native spatial axis domains do not match the adapter layout")
    expected_domain = "frequency" if layout == "frequency_target_first" else "time"
    if component.axis0_domain != expected_domain:
        raise ValueError("native axis0 domain does not match companion layout")
    values = np.asarray(component.value)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("native component values must be finite numeric values")
    return values[:, :, 0, 0]


def from_gwexpy_observable(
    bundle: GwexpyObservableBundle,
) -> ObservableArrayBundle:
    """Recover current native field values using the immutable companion."""
    if not isinstance(bundle, GwexpyObservableBundle):
        raise TypeError("bundle must be a GwexpyObservableBundle, not a bare Field")
    companion = bundle.companion
    expected_keys = {
        "observable",
        "layout",
        "positions_m",
        "axis_values",
        "unit",
        "metadata",
    }
    if set(companion) != expected_keys:
        raise ValueError("companion keys do not match the observable bundle contract")
    if "unit" in companion["metadata"]:
        raise ValueError("companion metadata must not duplicate the top-level unit")
    observable = companion["observable"]
    layout = companion["layout"]
    if observable not in _UNITS or layout not in _LAYOUTS:
        raise ValueError("companion observable or layout is invalid")
    unit = companion["unit"]
    if unit != _UNITS[observable]:
        raise ValueError("companion unit does not match observable")
    positions = _numeric_array(
        "companion.positions_m", companion["positions_m"], real=True
    )
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 1:
        raise ValueError("companion.positions_m must have shape (M, 3), M >= 1")
    m = len(positions)
    axis = _numeric_array("companion.axis_values", companion["axis_values"], real=True)
    if axis.ndim != 1 or len(axis) < 1:
        raise ValueError("companion.axis_values must be a non-empty 1D array")
    if layout != "static" and np.any(np.diff(axis) <= 0):
        raise ValueError("companion.axis_values must be strictly increasing")
    if layout == "static" and len(axis) != 1:
        raise ValueError("static companion axis_values must have length 1")
    metadata_with_unit = dict(companion["metadata"])
    metadata_with_unit["unit"] = unit
    gwexpy, units = _import_gwexpy()
    components = _field_components(gwexpy, bundle.field, observable)
    native_parts = [
        _validate_native_component(
            component,
            gwexpy=gwexpy,
            units=units,
            unit=unit,
            axis=axis,
            layout=layout,
            m=m,
        )
        for component in components
    ]
    if observable == "potential":
        native = native_parts[0]
    elif observable == "acceleration":
        native = np.stack(native_parts, axis=-1)
    else:
        native = np.stack(native_parts, axis=-1).reshape(len(axis), m, 3, 3)
    _normalize_metadata(
        metadata_with_unit,
        observable=observable,
        m=m,
        needs_phase=np.iscomplexobj(native) or layout == "frequency_target_first",
    )
    if layout == "static":
        values = native[0]
    elif layout == "time_first":
        values = native
    else:
        values = np.swapaxes(native, 0, 1)
    dtype = np.complex128 if np.iscomplexobj(values) else np.float64
    return ObservableArrayBundle(np.asarray(values, dtype=dtype), companion)
