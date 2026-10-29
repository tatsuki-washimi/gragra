"""Immutable fixed-geometry series contracts used by streaming evaluators."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _freeze(value: object, active: set[int] | None = None) -> object:
    """Make a defensive recursive immutable copy without optional adapters."""
    if active is None:
        active = set()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("metadata object arrays are not supported")
        copied = np.array(value, copy=True)
        copied.setflags(write=False)
        return copied
    if isinstance(value, Mapping):
        if id(value) in active:
            raise ValueError("cyclic metadata is not supported")
        active.add(id(value))
        try:
            copied = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("metadata mapping keys must be strings")
                copied[key] = _freeze(item, active)
            return MappingProxyType(copied)
        finally:
            active.remove(id(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if id(value) in active:
            raise ValueError("cyclic metadata is not supported")
        active.add(id(value))
        try:
            return tuple(_freeze(item, active) for item in value)
        finally:
            active.remove(id(value))
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Real) and np.isfinite(value):
        return value.item() if isinstance(value, np.generic) else value
    raise TypeError(f"unsupported metadata value type {type(value).__name__}")


def _array(name, value, shape, *, complex_ok=False):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray")
    if (
        value.dtype.hasobject
        or np.issubdtype(value.dtype, np.bool_)
        or not np.issubdtype(value.dtype, np.number)
    ):
        raise TypeError(f"{name} must be a numeric non-boolean array")
    if not complex_ok and np.iscomplexobj(value):
        raise TypeError(f"{name} must be real-valued")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    out = np.array(value, dtype=np.complex128 if complex_ok else np.float64, copy=True)
    out.setflags(write=False)
    return out


def _axis(values, kind):
    if kind not in ("time", "frequency"):
        raise ValueError("axis_kind must be 'time' or 'frequency'")
    a = _array("axis_values", values, (len(values),))
    if len(a) < 1:
        raise ValueError("axis_values must not be empty")
    if kind == "time":
        if not np.all(np.diff(a) > 0):
            raise ValueError("time axis must be strictly increasing")
    elif not (np.all(a > 0) and np.all(np.diff(a) > 0)):
        raise ValueError("frequency axis must be positive and strictly increasing")
    return a


def _common(
    obj,
    positions_m,
    axis_values,
    axis_kind,
    coordinate_frame,
    coordinate_origin_m,
    source_ids,
    source_metadata,
):
    n = len(positions_m)
    positions = _array("positions_m", positions_m, (n, 3))
    axis = _axis(axis_values, axis_kind)
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("coordinate_frame must be a non-empty string")
    origin = _array("coordinate_origin_m", coordinate_origin_m, (3,))
    if source_ids is None:
        raise TypeError("source_ids must be explicitly provided as N strings")
    valid_container = isinstance(source_ids, (Sequence, np.ndarray))
    if (
        isinstance(source_ids, (str, bytes))
        or not valid_container
        or np.asarray(source_ids).ndim != 1
        or len(source_ids) != n
        or any(not isinstance(x, str) for x in source_ids)
    ):
        raise TypeError("source_ids must be N strings")
    ids = tuple(source_ids)
    if any(not source_id for source_id in ids):
        raise ValueError("source_ids must be non-empty strings")
    if len(set(ids)) != n:
        raise ValueError("source_ids must be unique")
    if not isinstance(source_metadata if source_metadata is not None else {}, Mapping):
        raise TypeError("source_metadata must be a mapping")
    for k, v in {
        "positions_m": positions,
        "axis_values": axis,
        "axis_kind": axis_kind,
        "coordinate_frame": coordinate_frame,
        "coordinate_origin_m": origin,
        "source_ids": ids,
        "source_metadata": _freeze(source_metadata or {}),
    }.items():
        object.__setattr__(obj, k, v)


def _finite_real(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class _Series:
    positions_m: np.ndarray
    axis_values: np.ndarray
    axis_kind: str
    coordinate_frame: str
    coordinate_origin_m: np.ndarray
    source_ids: tuple[str, ...]
    source_metadata: Mapping[str, object]

    @property
    def sample_count(self):
        return len(self.axis_values)

    @property
    def source_count(self):
        return len(self.positions_m)
