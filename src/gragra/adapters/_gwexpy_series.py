"""Shared implementation for fixed-mesh GWexpy series bridges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gragra.fields._series import _freeze


def _native():
    try:
        import gwexpy
        from astropy import units
    except ImportError as exc:
        raise ImportError(
            "GWexpy series import failed; reinstall gragra with its runtime "
            "dependencies."
        ) from exc
    return gwexpy, units


@dataclass(frozen=True)
class GwexpySeriesBundle:
    """Native field and immutable sidecar; the sidecar deliberately has no values."""

    field: object
    companion: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "companion", _freeze(self.companion))


def _companion(
    series, quantity: str, geometry: dict[str, object]
) -> Mapping[str, object]:
    result = {
        "quantity": quantity,
        "positions_m": series.positions_m,
        "axis_kind": series.axis_kind,
        "axis_values": series.axis_values,
        "coordinate_frame": series.coordinate_frame,
        "coordinate_origin_m": series.coordinate_origin_m,
        "source_ids": series.source_ids,
        "source_metadata": series.source_metadata,
        **geometry,
    }
    if series.axis_kind == "frequency":
        result.update(
            t_ref_s=series.t_ref_s,
            phase_origin_m=series.phase_origin_m,
            fourier_convention=series.fourier_convention,
        )
    return _freeze(result)


def _scalar(values, series, unit):
    gwexpy, units = _native()
    axis_unit = units.s if series.axis_kind == "time" else units.Hz
    return gwexpy.fields.ScalarField(
        values[:, :, None, None],
        unit=unit,
        axis0=series.axis_values * axis_unit,
        axis1=np.arange(series.source_count),
        axis2=np.array([0.0]),
        axis3=np.array([0.0]),
        axis_names=[
            "time" if series.axis_kind == "time" else "frequency",
            "point",
            "dummy_y",
            "dummy_z",
        ],
        axis0_domain=series.axis_kind,
        space_domain={"point": "real", "dummy_y": "real", "dummy_z": "real"},
    )


def _vector(values, series):
    gwexpy, units = _native()
    axis_unit = units.s if series.axis_kind == "time" else units.Hz
    args = {
        "axis0": series.axis_values * axis_unit,
        "axis1": np.arange(series.source_count),
        "axis2": np.array([0.0]),
        "axis3": np.array([0.0]),
        "axis_names": [
            "time" if series.axis_kind == "time" else "frequency",
            "point",
            "dummy_y",
            "dummy_z",
        ],
        "axis0_domain": series.axis_kind,
        "space_domain": {"point": "real", "dummy_y": "real", "dummy_z": "real"},
    }
    return gwexpy.fields.VectorField(
        {
            k: gwexpy.fields.ScalarField(values[:, :, i, None, None], unit="m", **args)
            for i, k in enumerate(("x", "y", "z"))
        }
    )


def _current_scalar(bundle: GwexpySeriesBundle, unit: str, quantity: str) -> np.ndarray:
    gwexpy, units = _native()
    field = bundle.field
    c = bundle.companion
    if not isinstance(field, gwexpy.fields.ScalarField):
        raise TypeError("bundle field must be a ScalarField")
    s, n = len(c["axis_values"]), len(c["source_ids"])
    if field.shape != (s, n, 1, 1) or field.unit != units.Unit(unit):
        raise ValueError("native field shape or unit does not match sidecar")
    _validate_native_layout(field, c, units, quantity)
    result = np.asarray(field.value)[:, :, 0, 0]
    if not np.isfinite(result).all():
        raise ValueError("native field values must be finite")
    return result


def _current_vector(bundle: GwexpySeriesBundle) -> np.ndarray:
    gwexpy, units = _native()
    field = bundle.field
    c = bundle.companion
    if not isinstance(field, gwexpy.fields.VectorField):
        raise TypeError("bundle field must be a VectorField")
    s, n = len(c["axis_values"]), len(c["source_ids"])
    keys = tuple(field.keys())
    if set(keys) != {"x", "y", "z"} or len(keys) != 3:
        raise ValueError("native VectorField must contain exactly x, y, and z")
    components = []
    for key in ("x", "y", "z"):
        component = field[key]
        if component.shape != (s, n, 1, 1) or component.unit != units.m:
            raise ValueError("native vector component layout or unit does not match")
        _validate_native_layout(component, c, units, "displacement")
        values = np.asarray(component.value)
        if not np.isfinite(values).all():
            raise ValueError("native displacement values must be finite")
        components.append(values[:, :, 0, 0])
    return np.stack(components, axis=-1)


def _validate_native_layout(
    field: object, companion: Mapping[str, object], units: object, quantity: str
) -> None:
    """Reject native changes to the axis layout rather than trusting sidecar axes."""
    if companion.get("quantity") != quantity:
        raise ValueError("companion quantity does not match bridge")
    ids = companion.get("source_ids")
    if (
        not isinstance(ids, tuple)
        or any(not isinstance(item, str) for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("companion source_ids are invalid")
    axis_kind = companion["axis_kind"]
    if axis_kind not in {"time", "frequency"}:
        raise ValueError("companion axis_kind is invalid")
    if axis_kind == "frequency":
        if companion.get("fourier_convention") != "Re[q_hat exp(-2pi i f(t-t_ref_s))]":
            raise ValueError("companion Fourier convention is invalid")
        if "t_ref_s" not in companion or "phase_origin_m" not in companion:
            raise ValueError("frequency companion lacks phase metadata")
        t_ref_s = companion["t_ref_s"]
        phase_origin_m = np.asarray(companion["phase_origin_m"])
        if (
            isinstance(t_ref_s, bool)
            or not np.isscalar(t_ref_s)
            or not np.isfinite(t_ref_s)
            or phase_origin_m.shape != (3,)
            or not np.isfinite(phase_origin_m).all()
        ):
            raise ValueError("frequency companion phase metadata is invalid")
    elif any(
        key in companion for key in ("t_ref_s", "phase_origin_m", "fourier_convention")
    ):
        raise ValueError("time companion must not contain frequency phase metadata")
    names = (
        "time" if axis_kind == "time" else "frequency",
        "point",
        "dummy_y",
        "dummy_z",
    )
    if tuple(field.axis_names) != names or field.axis0_domain != axis_kind:
        raise ValueError("native axis names or domain do not match sidecar")
    expected_values = (
        np.asarray(companion["axis_values"]),
        np.arange(len(companion["source_ids"])),
        np.array([0.0]),
        np.array([0.0]),
    )
    expected_units = (
        units.s if axis_kind == "time" else units.Hz,
        units.dimensionless_unscaled,
        units.dimensionless_unscaled,
        units.dimensionless_unscaled,
    )
    for index, expected, expected_unit in zip(
        range(4), expected_values, expected_units, strict=True
    ):
        native = getattr(field, f"_axis{index}_index")
        if native.unit != expected_unit:
            raise ValueError("native axis unit does not match series layout")
        actual = np.asarray(native.to_value(expected_unit))
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError("native axis values or ordering do not match sidecar")
    domains = {"point": "real", "dummy_y": "real", "dummy_z": "real"}
    if field.space_domains != domains:
        raise ValueError("native spatial domains do not match series layout")
