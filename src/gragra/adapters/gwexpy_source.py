"""Bounded GWexpy bridge for one real volume-density snapshot."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from gragra.adapters.gwexpy_observable import _freeze
from gragra.gravity_sources.volume import VolumeElementSource

_KEYS = {
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
_AXIS_NAMES = ("time", "point", "dummy_y", "dummy_z")
_SPACE_DOMAINS = {"point": "real", "dummy_y": "real", "dummy_z": "real"}


def _import_gwexpy():
    try:
        import gwexpy
        from astropy import units
    except ImportError as exc:
        raise ImportError(
            "GWexpy density snapshot import failed; reinstall gragra with its "
            "runtime dependencies."
        ) from exc
    return gwexpy, units


def _real_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray")
    if value.dtype.hasobject or np.issubdtype(value.dtype, np.bool_):
        raise TypeError(f"{name} must be a numeric non-boolean array")
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be real-valued")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.asarray(value, dtype=np.float64)
    if not np.isfinite(copied).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(copied, dtype=np.float64, copy=True)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _density(source: VolumeElementSource, kind: str) -> np.ndarray:
    if not isinstance(source, VolumeElementSource):
        raise TypeError("source must be a VolumeElementSource")
    if kind not in ("rho", "delta_rho"):
        raise ValueError("density_kind must be 'rho' or 'delta_rho'")
    n = len(source.positions_m)
    if n < 1:
        raise ValueError("source must contain at least one volume element")
    raw = np.asarray(source.density_kg_m3)
    if (
        raw.dtype.hasobject
        or np.issubdtype(raw.dtype, np.bool_)
        or np.iscomplexobj(raw)
    ):
        raise TypeError("source density must be real numeric values")
    if raw.ndim == 0:
        values = np.full(n, raw.item(), dtype=np.float64)
    elif raw.ndim == 1 and raw.shape == (n,):
        values = np.asarray(raw, dtype=np.float64)
    else:
        raise ValueError(f"source density must have shape ({n},)")
    if not np.isfinite(values).all():
        raise ValueError("source density must contain only finite values")
    if kind == "rho" and (values < 0).any():
        raise ValueError("rho density must be non-negative")
    return np.array(values, dtype=np.float64, copy=True)


def _companion(
    source: VolumeElementSource,
    *,
    time_s: object,
    density_kind: object,
    coordinate_frame: object,
    coordinate_origin_m: object,
    source_ids: Sequence[str] | None,
    source_metadata: Mapping[str, object] | None,
) -> tuple[np.ndarray, Mapping[str, object]]:
    if not isinstance(density_kind, str):
        raise ValueError("density_kind must be 'rho' or 'delta_rho'")
    density = _density(source, density_kind)
    n = len(density)
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("coordinate_frame must be a non-empty string")
    origin = _real_array("coordinate_origin_m", coordinate_origin_m, (3,))
    if source_ids is None:
        ids = tuple(str(i) for i in range(n))
    else:
        if isinstance(source_ids, (str, bytes)) or not isinstance(source_ids, Sequence):
            raise TypeError(f"source_ids must be a sequence of {n} strings")
        if len(source_ids) != n:
            raise ValueError(f"source_ids must have length {n}")
        if any(not isinstance(item, str) for item in source_ids):
            raise TypeError(f"source_ids must be a sequence of {n} strings")
        ids = tuple(source_ids)
    if len(set(ids)) != n:
        raise ValueError("source_ids must be unique")
    if source_metadata is None:
        source_metadata = {}
    if not isinstance(source_metadata, Mapping):
        raise TypeError("source_metadata must be a mapping")
    positions = _real_array("source.positions_m", source.positions_m, (n, 3))
    volumes = _real_array("source.volumes_m3", source.volumes_m3, (n,))
    if (volumes <= 0.0).any():
        raise ValueError("source.volumes_m3 must contain only positive values")
    result = {
        "density_kind": density_kind,
        "positions_m": positions,
        "volumes_m3": volumes,
        "time_s": _finite_real("time_s", time_s),
        "unit": "kg/m^3",
        "coordinate_frame": coordinate_frame,
        "coordinate_origin_m": origin,
        "source_ids": ids,
        "source_metadata": source_metadata,
    }
    return density, _freeze(result)


@dataclass(frozen=True)
class GwexpyDensitySnapshotBundle:
    """A native ``ScalarField`` plus its immutable density sidecar."""

    field: object
    companion: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "companion", _freeze(self.companion))


@dataclass(frozen=True)
class DensitySnapshotArrayBundle:
    """Restored volume source plus its immutable density sidecar."""

    source: VolumeElementSource
    companion: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.source, VolumeElementSource):
            raise TypeError("source must be a VolumeElementSource")
        object.__setattr__(self, "companion", _freeze(self.companion))

    @property
    def time_s(self) -> float:
        return float(self.companion["time_s"])


def to_gwexpy_density_snapshot(
    source: VolumeElementSource,
    *,
    time_s: float,
    density_kind: Literal["rho", "delta_rho"],
    coordinate_frame: str,
    coordinate_origin_m: np.ndarray,
    source_ids: Sequence[str] | None = None,
    source_metadata: Mapping[str, object] | None = None,
) -> GwexpyDensitySnapshotBundle:
    """Convert one current real density snapshot to a native GWexpy field."""
    density, companion = _companion(
        source,
        time_s=time_s,
        density_kind=density_kind,
        coordinate_frame=coordinate_frame,
        coordinate_origin_m=coordinate_origin_m,
        source_ids=source_ids,
        source_metadata=source_metadata,
    )
    gwexpy, units = _import_gwexpy()
    field = gwexpy.fields.ScalarField(
        density[np.newaxis, :, np.newaxis, np.newaxis],
        unit="kg/m^3",
        axis0=np.array([companion["time_s"]]) * units.s,
        axis1=np.arange(len(density), dtype=np.float64),
        axis2=np.array([0.0]),
        axis3=np.array([0.0]),
        axis_names=list(_AXIS_NAMES),
        axis0_domain="time",
        space_domain=_SPACE_DOMAINS,
    )
    return GwexpyDensitySnapshotBundle(field, companion)


def _validated_companion(
    companion: Mapping[str, object],
) -> tuple[str, np.ndarray, np.ndarray, float]:
    if not isinstance(companion, Mapping) or set(companion) != _KEYS:
        raise ValueError("companion keys do not match the density snapshot contract")
    kind = companion["density_kind"]
    if kind not in ("rho", "delta_rho"):
        raise ValueError("companion density_kind is invalid")
    if companion["unit"] != "kg/m^3":
        raise ValueError("companion unit must be 'kg/m^3'")
    if (
        not isinstance(companion["coordinate_frame"], str)
        or not companion["coordinate_frame"].strip()
    ):
        raise ValueError("companion coordinate_frame is invalid")
    time_s = _finite_real("companion.time_s", companion["time_s"])
    positions_raw = companion["positions_m"]
    if (
        not isinstance(positions_raw, np.ndarray)
        or positions_raw.ndim != 2
        or positions_raw.shape[1:] != (3,)
        or len(positions_raw) < 1
    ):
        raise ValueError("companion positions_m must have shape (N, 3), N >= 1")
    n = len(positions_raw)
    positions = _real_array("companion.positions_m", positions_raw, (n, 3))
    volumes = _real_array("companion.volumes_m3", companion["volumes_m3"], (n,))
    if (volumes <= 0).any():
        raise ValueError("companion volumes_m3 must be positive")
    _real_array("companion.coordinate_origin_m", companion["coordinate_origin_m"], (3,))
    ids = companion["source_ids"]
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise TypeError("companion source_ids must be N strings")
    if len(ids) != n:
        raise ValueError("companion source_ids must have length N")
    if any(not isinstance(item, str) for item in ids):
        raise TypeError("companion source_ids must be N strings")
    if len(set(ids)) != n:
        raise ValueError("companion source_ids must be unique")
    if not isinstance(companion["source_metadata"], Mapping):
        raise TypeError("companion source_metadata must be a mapping")
    _freeze(companion["source_metadata"])
    return kind, positions, volumes, time_s


def _native_axis_values(native_axis: object, unit: object) -> np.ndarray:
    """Require a native coordinate axis to be finite, real, and unscaled."""
    if not hasattr(native_axis, "unit") or native_axis.unit != unit:
        raise ValueError("native axis has the wrong physical unit")
    raw = np.asarray(native_axis.value)
    if (
        raw.dtype.hasobject
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("native axis values must be real finite numeric values")
    try:
        return np.asarray(native_axis.to_value(unit))
    except Exception as exc:
        raise ValueError("native axis has the wrong physical unit") from exc


def from_gwexpy_density_snapshot(
    bundle: GwexpyDensitySnapshotBundle,
) -> DensitySnapshotArrayBundle:
    """Restore a source from the field's current values and explicit sidecar."""
    if not isinstance(bundle, GwexpyDensitySnapshotBundle):
        raise TypeError(
            "bundle must be a GwexpyDensitySnapshotBundle, not a bare Field"
        )
    kind, positions, volumes, time_s = _validated_companion(bundle.companion)
    gwexpy, units = _import_gwexpy()
    field = bundle.field
    if not isinstance(field, gwexpy.fields.ScalarField):
        raise TypeError("bundle field must be a ScalarField")
    n = len(positions)
    if field.shape != (1, n, 1, 1):
        raise ValueError("native field shape does not match density snapshot layout")
    if field.unit != units.Unit("kg/m^3"):
        raise ValueError("native field unit does not match 'kg/m^3'")
    native_axis0 = field._axis0_index
    if not hasattr(native_axis0, "unit") or native_axis0.unit != units.s:
        raise ValueError("native axis0 must use SI seconds")
    axis0 = _native_axis_values(native_axis0, units.s)
    if axis0.shape != (1,) or not np.allclose(axis0, [time_s], rtol=0.0, atol=1e-12):
        raise ValueError("native axis0 does not match companion time_s")
    for name, expected in (
        ("_axis1_index", np.arange(n, dtype=np.float64)),
        ("_axis2_index", np.array([0.0])),
        ("_axis3_index", np.array([0.0])),
    ):
        native_axis = getattr(field, name)
        if (
            not hasattr(native_axis, "unit")
            or native_axis.unit != units.dimensionless_unscaled
        ):
            raise ValueError(
                "native spatial axis must use the unscaled dimensionless unit"
            )
        actual = _native_axis_values(native_axis, units.dimensionless_unscaled)
        if not np.array_equal(actual, expected):
            raise ValueError(
                "native spatial axis does not match density snapshot layout"
            )
    if (
        tuple(field.axis_names) != _AXIS_NAMES
        or field.axis0_domain != "time"
        or field.space_domains != _SPACE_DOMAINS
    ):
        raise ValueError(
            "native axis names or domains do not match density snapshot layout"
        )
    values = np.asarray(field.value)
    if (
        values.dtype.hasobject
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
    ):
        raise TypeError("native density values must be real numeric values")
    if not np.isfinite(values).all():
        raise ValueError("native density values must be finite")
    density = np.asarray(values[0, :, 0, 0], dtype=np.float64)
    if kind == "rho" and (density < 0).any():
        raise ValueError("rho density must be non-negative")
    source = VolumeElementSource(positions, volumes, density)
    return DensitySnapshotArrayBundle(source, bundle.companion)
