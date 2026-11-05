"""Build gridded fields from static source quantities or dynamic contributions.

The grid stores source-position contributions. Each source is assigned to one
physical cell by its coordinates; the selected target identifies which
source-to-target response is placed at those coordinates. Optional spatial
projections reduce one grid axis while retaining the other axes and all
vector/tensor components.

Bare grid coordinates and bare ``frequency_hz`` values are interpreted in SI
meters and hertz. Astropy quantities are accepted when they are convertible to
those units.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from astropy import units as u

from gragra._evaluation_setup import CoordinatePrecisionError, check_pair_precision
from gragra._plotting._diagnostics import (
    _check_finite_pairs,
    _check_precision_candidates,
    _resolve_source,
    _resolved_from,
    _select_target,
    _source_positions_in_frame,
)
from gragra._plotting._source_contracts import SourceContributionResult
from gragra._plotting._source_quantities import (
    source_quantity_data,
)
from gragra.kernels._source_terms import iter_source_kernel_terms
from gragra.observables._source_terms import iter_point_observable_source_terms

_GG_COMPONENT_INDICES = {
    "xx": (0, 0),
    "xy": (0, 1),
    "xz": (0, 2),
    "yx": (1, 0),
    "yy": (1, 1),
    "yz": (1, 2),
    "zx": (2, 0),
    "zy": (2, 1),
    "zz": (2, 2),
}
_GG_COMPONENT_ALIASES = {
    f"{prefix}_{name}": index
    for prefix in ("GG", "gg")
    for name, index in _GG_COMPONENT_INDICES.items()
}
_PHYSICAL_ALIASES = {
    "acceleration_kernel": "acceleration_kernel",
    "acceleration_contribution": "acceleration_contribution",
    "gravity_gradient_kernel": "gravity_gradient_kernel",
    "gravity_gradient_contribution": "gravity_gradient_contribution",
    "gravity_gradient": "gravity_gradient_contribution",
    "gradient_kernel": "gravity_gradient_kernel",
    "gradient_contribution": "gravity_gradient_contribution",
    "GG_kernel": "gravity_gradient_kernel",
    "gg_kernel": "gravity_gradient_kernel",
    "GG_contribution": "gravity_gradient_contribution",
    "gg_contribution": "gravity_gradient_contribution",
    "GG": "gravity_gradient_contribution",
    "gg": "gravity_gradient_contribution",
    "gradient": "gravity_gradient_contribution",
    **dict.fromkeys(_GG_COMPONENT_ALIASES, "gravity_gradient_contribution"),
}
_DYNAMIC_QUANTITIES = {"complex_contribution", "phase", "interference_fraction"}
_PHYSICAL_UNITS = {
    "acceleration_kernel": u.m**-2,
    "gravity_gradient_kernel": u.m**-3,
    "acceleration_contribution": u.m / u.s**2,
    "gravity_gradient_contribution": u.s**-2,
}
_BLOCK_SIZE = 256


@dataclass(frozen=True, eq=False)
class SourceFieldResult:
    """A native GWexpy Field plus an explicit validity mask and provenance."""

    field: Any
    valid: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid)
        if valid.dtype.kind != "b":
            raise TypeError("valid must be a boolean array")
        if hasattr(self.field, "to_array"):
            field_shape = np.asarray(self.field.to_array()).shape
        else:
            field_shape = np.asarray(self.field).shape
        if valid.shape != field_shape:
            raise ValueError(
                "valid must have the same shape as the field's to_array() values"
            )
        copied = np.array(valid, dtype=bool, copy=True, order="C")
        frozen = np.frombuffer(copied.tobytes(order="C"), dtype=bool).reshape(
            copied.shape
        )
        metadata = _freeze_metadata(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "valid", frozen)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, eq=False)
class _Grid:
    edges_m: tuple[np.ndarray, np.ndarray, np.ndarray]
    centers_m: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int, int]


@dataclass(frozen=True, eq=False)
class _SourceValues:
    positions_m: np.ndarray
    source_index: np.ndarray
    values: np.ndarray | None
    unit: u.UnitBase
    quantity: str
    aggregation: str
    frame: Any
    coordinate_origin_m: np.ndarray
    source_metadata: Any
    target_id: str | None = None
    target_xyz_m: np.ndarray | None = None
    target_trace: Any | None = None
    volumes_m3: np.ndarray | None = None
    global_metadata: Any = None
    physical_observable: str | None = None
    softening_m: float = 0.0
    dynamic_result: SourceContributionResult | None = None
    frequency_indices: np.ndarray | None = None
    frequency_hz: float | None = None
    frequencies_hz: np.ndarray | None = None
    physical_weights: np.ndarray | None = None
    source_type: str | None = None


def source_field(
    data: object,
    quantity: str,
    *,
    target_id: str | None = None,
    grid: object,
    aggregation: str,
    frequency_hz: object | None = None,
    norm: str | None = None,
    projection: object | None = None,
    spatial_projection: str | None = None,
    softening_m: float | None = None,
):
    """Return a ScalarField, VectorField, or TensorField for source positions.

    ``grid`` is an explicit tuple ``(x_edges, y_edges, z_edges)``. Each edge
    vector is finite and strictly increasing and defines closed coverage at
    both outer boundaries. Internal boundaries go to the cell on their right;
    the final edge is included in the last cell. Every source is assigned once.
    Static extensive values and physical contributions use ``sum``; static
    intensive values use a verified volume-weighted ``mean``. Dynamic source
    responses are additive and use ``sum``.

    ``GG_xx``/``GG_xy`` and the other named GG component aliases select one
    signed component from the static gravity-gradient contribution. ``GG``
    retains the full tensor, and ``GG_kernel`` selects the G-free tensor kernel.

    Static fields receive a structural singleton time axis at zero seconds.
    Dynamic fields keep every stored frequency when ``frequency_hz`` is
    omitted; an exact stored frequency can be selected explicitly. Set
    ``spatial_projection`` to ``"xy"``, ``"xz"`` or ``"yz"`` to sum along the
    omitted spatial axis, or to compute a volume-weighted mean for intensive
    static values. Use :func:`source_field_result` for phase and
    interference-fraction maps so their validity mask is retained.
    """
    if not isinstance(quantity, str):
        raise TypeError("quantity must be a string")
    if quantity in {"phase", "interference_fraction"}:
        raise ValueError(
            "phase and interference_fraction require source_field_result so "
            "their validity mask is retained"
        )
    return _source_field_result(
        data,
        quantity,
        target_id=target_id,
        grid=grid,
        aggregation=aggregation,
        frequency_hz=frequency_hz,
        norm=norm,
        projection=projection,
        spatial_projection=spatial_projection,
        softening_m=softening_m,
    ).field


def source_field_result(
    data: object,
    quantity: str,
    *,
    target_id: str | None = None,
    grid: object,
    aggregation: str,
    frequency_hz: object | None = None,
    norm: str | None = None,
    projection: object | None = None,
    spatial_projection: str | None = None,
    softening_m: float | None = None,
) -> SourceFieldResult:
    """Return a source-position field, validity mask, and immutable metadata."""
    return _source_field_result(
        data,
        quantity,
        target_id=target_id,
        grid=grid,
        aggregation=aggregation,
        frequency_hz=frequency_hz,
        norm=norm,
        projection=projection,
        spatial_projection=spatial_projection,
        softening_m=softening_m,
    )


def _source_field_result(
    data: object,
    quantity: str,
    *,
    target_id: str | None,
    grid: object,
    aggregation: str,
    frequency_hz: object | None,
    norm: str | None,
    projection: object | None,
    spatial_projection: str | None,
    softening_m: float | None,
) -> SourceFieldResult:
    if not isinstance(quantity, str):
        raise TypeError("quantity must be a string")
    if norm is not None and not isinstance(norm, str):
        raise TypeError("norm must be a string or None")
    if norm is not None and projection is not None:
        raise ValueError("norm and projection are mutually exclusive")
    if spatial_projection is not None:
        if not isinstance(spatial_projection, str):
            raise TypeError("spatial_projection must be 'xy', 'xz', 'yz', or None")
        spatial_projection = spatial_projection.lower()
        if spatial_projection not in {"xy", "xz", "yz"}:
            raise ValueError("spatial_projection must be 'xy', 'xz', 'yz', or None")
    if quantity in {"phase", "interference_fraction"} and (
        norm is not None or projection is not None
    ):
        raise ValueError(
            "phase and interference_fraction cannot be combined with norm or projection"
        )
    component_index = _GG_COMPONENT_ALIASES.get(quantity)
    if component_index is not None:
        if norm is not None or projection is not None:
            raise ValueError(
                "a GG component alias cannot be combined with norm or projection"
            )
        left = np.zeros(3, dtype=np.float64)
        right = np.zeros(3, dtype=np.float64)
        left[component_index[0]] = 1.0
        right[component_index[1]] = 1.0
        projection = (left, right)
    normalized_grid = _normalize_grid(grid)
    if not isinstance(aggregation, str):
        raise TypeError("aggregation must be 'sum' or 'mean'")
    if aggregation not in {"sum", "mean"}:
        raise ValueError("aggregation must be 'sum' or 'mean'")

    if isinstance(data, SourceContributionResult):
        source_data = _dynamic_source_values(
            data,
            quantity,
            target_id=target_id,
            aggregation=aggregation,
            frequency_hz=frequency_hz,
            softening_m=softening_m,
        )
    else:
        if frequency_hz is not None:
            raise ValueError("frequency_hz applies only to SourceContributionResult")
        source_data = _static_source_values(
            data,
            quantity,
            target_id=target_id,
            aggregation=aggregation,
            softening_m=softening_m,
        )

    if source_data.aggregation != aggregation:
        raise ValueError("aggregation does not match the quantity contract")
    if source_data.physical_observable is not None and aggregation != "sum":
        raise ValueError("physical source contributions require aggregation='sum'")
    if source_data.dynamic_result is not None and aggregation != "sum":
        raise ValueError("dynamic source contributions require aggregation='sum'")

    dynamic = source_data.dynamic_result is not None
    frequency_count = len(source_data.frequency_indices) if dynamic else 1
    flat_cells = _source_cells(
        source_data.positions_m, source_data.source_index, normalized_grid
    )
    component_shape = _component_shape(source_data)
    dtype = np.complex128 if dynamic else np.float64
    n_cells = int(np.prod(normalized_grid.shape))
    accumulator_shape = (
        (frequency_count, n_cells, *component_shape)
        if dynamic
        else (n_cells, *component_shape)
    )
    accumulated = np.zeros(accumulator_shape, dtype=dtype)
    weights_by_cell = np.zeros(n_cells, dtype=np.float64)
    if aggregation == "mean":
        if source_data.volumes_m3 is None:
            raise ValueError("volume-weighted mean requires verified source volumes")
        if source_data.volumes_m3.shape != (len(source_data.positions_m),):
            raise ValueError("verified source volumes must align with source positions")

    if dynamic:
        _accumulate_dynamic(
            accumulated,
            flat_cells,
            source_data,
            weights_by_cell,
        )
    elif source_data.physical_observable is not None:
        _accumulate_static_physical(
            accumulated,
            flat_cells,
            source_data,
            weights_by_cell,
        )
    else:
        assert source_data.values is not None
        _accumulate_values(
            accumulated,
            flat_cells,
            source_data.values,
            source_data.volumes_m3 if aggregation == "mean" else None,
            weights_by_cell,
        )

    original_grid = normalized_grid
    if dynamic:
        binned = accumulated.reshape(
            (frequency_count, *normalized_grid.shape, *component_shape)
        )
        valid = np.ones((frequency_count, *normalized_grid.shape), dtype=bool)
    else:
        binned = accumulated.reshape((*normalized_grid.shape, *component_shape))
        if spatial_projection is not None:
            projection_axis = _spatial_projection_axis(spatial_projection)
            if aggregation == "mean":
                weights_grid = weights_by_cell.reshape(normalized_grid.shape)
                binned = np.sum(
                    binned,
                    axis=projection_axis,
                    keepdims=True,
                )
                projected_weights = np.sum(
                    weights_grid, axis=projection_axis, keepdims=True
                )
                binned, valid = _volume_weighted_average(
                    binned, projected_weights, component_shape
                )
            else:
                binned = np.sum(binned, axis=projection_axis, keepdims=True)
                projected_shape = list(normalized_grid.shape)
                projected_shape[projection_axis] = 1
                valid = np.ones(projected_shape, dtype=bool)
            normalized_grid = _project_grid(normalized_grid, spatial_projection)
        elif aggregation == "mean":
            binned, valid = _volume_weighted_average(
                binned, weights_by_cell.reshape(normalized_grid.shape), component_shape
            )
        else:
            valid = np.ones(normalized_grid.shape, dtype=bool)

    if spatial_projection is not None and dynamic:
        projection_axis = _spatial_projection_axis(spatial_projection) + 1
        binned = np.sum(binned, axis=projection_axis, keepdims=True)
        normalized_grid = _project_grid(normalized_grid, spatial_projection)
        valid = np.ones((frequency_count, *normalized_grid.shape), dtype=bool)

    total = None
    if dynamic and quantity == "interference_fraction":
        data_record = source_data.dynamic_result._data
        assert source_data.frequency_indices is not None
        assert source_data.target_id is not None
        target_index = data_record.target_ids.index(source_data.target_id)
        total = data_record.total_values[
            source_data.frequency_indices, target_index, ...
        ]

    if dynamic:
        derived = [
            _derive_field_values(
                binned[index],
                valid[index],
                unit=source_data.unit,
                component_shape=component_shape,
                quantity=quantity,
                dynamic=True,
                total=None if total is None else total[index],
                norm=norm,
                projection=projection,
            )
            for index in range(frequency_count)
        ]
        binned = np.stack([item[0] for item in derived])
        valid = np.stack([item[1] for item in derived])
        unit = derived[0][2]
        component_shape = derived[0][3]
    else:
        binned, valid, unit, component_shape = _derive_field_values(
            binned,
            valid,
            unit=source_data.unit,
            component_shape=component_shape,
            quantity=quantity,
            dynamic=False,
            total=None,
            norm=norm,
            projection=projection,
        )

    base_valid_rank = 4 if dynamic else 3
    if component_shape and valid.ndim == base_valid_rank:
        value_valid = np.broadcast_to(
            valid[(...,) + (None,) * len(component_shape)], binned.shape
        )
    else:
        value_valid = valid
    invalid_values = ~np.isfinite(binned)
    allowed_nan = (aggregation == "mean") & np.isnan(binned) & ~value_valid
    if np.any(invalid_values & ~allowed_nan):
        raise ValueError("field values are not finite after aggregation")

    field = _make_field(
        binned,
        unit,
        normalized_grid,
        frequencies_hz=source_data.frequencies_hz,
    )
    if dynamic:
        valid_native = value_valid.reshape(
            (frequency_count, *normalized_grid.shape, *component_shape)
        )
    else:
        valid_native = value_valid.reshape((*normalized_grid.shape, *component_shape))[
            np.newaxis, ...
        ]

    metadata = dict(
        _field_metadata(source_data, normalized_grid, quantity, aggregation, unit)
    )
    if spatial_projection is not None:
        projection_axis = _spatial_projection_axis(spatial_projection)
        metadata.update(
            {
                "spatial_projection": spatial_projection,
                "projected_axis": ("x", "y", "z")[projection_axis],
                "projection_range_m": (
                    float(original_grid.edges_m[projection_axis][0]),
                    float(original_grid.edges_m[projection_axis][-1]),
                ),
                "unprojected_grid_edges_m": tuple(
                    tuple(float(item) for item in axis)
                    for axis in original_grid.edges_m
                ),
            }
        )
    return SourceFieldResult(field=field, valid=valid_native, metadata=metadata)


def _static_source_values(
    setup: object,
    quantity: str,
    *,
    target_id: str | None,
    aggregation: str,
    softening_m: float | None,
) -> _SourceValues:
    canonical_physical = _PHYSICAL_ALIASES.get(quantity)
    if canonical_physical is not None:
        eps = _normalize_static_softening(softening_m)
        resolved = _resolved_from(setup)
        view = _resolve_source(resolved)
        target_index = _select_target(resolved, target_id, required=True)
        assert target_index is not None
        target_key = resolved.targets.ids[target_index]
        trace = resolved.target_traces[target_index]
        positions, frame = _source_positions_in_frame(view, resolved)
        target_xyz = np.asarray(trace.point_m, dtype=np.float64)
        quantity_kind = (
            "acceleration"
            if canonical_physical.startswith("acceleration")
            else "gravity_gradient"
        )
        if canonical_physical.endswith("_kernel"):
            weights = np.ones(len(positions), dtype=np.float64)
        else:
            weights = np.asarray(view.weights_kg)
        if aggregation != "sum":
            raise ValueError("physical source contributions require aggregation='sum'")
        return _SourceValues(
            positions_m=positions,
            source_index=np.arange(len(positions), dtype=np.int64),
            values=None,
            unit=_PHYSICAL_UNITS[canonical_physical],
            quantity=canonical_physical,
            aggregation=aggregation,
            frame=frame,
            coordinate_origin_m=frame.origin_m,
            source_metadata=_static_source_metadata(view),
            target_id=target_key,
            target_xyz_m=target_xyz,
            target_trace=trace,
            physical_weights=np.array(weights, copy=True),
            source_type=view.source_type,
            physical_observable=(
                f"kernel:{quantity_kind}"
                if canonical_physical.endswith("_kernel")
                else quantity_kind
            ),
            softening_m=eps,
        )

    if softening_m is None:
        softening_m = 0.0
    try:
        record = source_quantity_data(
            setup,
            quantity,
            target_id=target_id,
            softening_m=softening_m,
        )
    except ValueError:
        raise
    if record.extensive and aggregation != "sum":
        raise ValueError("extensive source quantities require aggregation='sum'")
    if not record.extensive and aggregation != "mean":
        raise ValueError("intensive source quantities require aggregation='mean'")

    volumes = None
    if aggregation == "mean":
        resolved = _resolved_from(setup)
        volumes = _resolve_source(resolved).volumes()
        if volumes.shape != record.values.shape or not np.isfinite(volumes).all():
            raise ValueError(
                "verified source volumes must be finite and source-aligned"
            )
    return _SourceValues(
        positions_m=record.positions_m,
        source_index=record.source_index,
        values=record.values,
        unit=record.unit,
        quantity=record.quantity,
        aggregation=aggregation,
        frame=record.frame,
        coordinate_origin_m=record.coordinate_origin_m,
        source_metadata=record.source_metadata,
        target_id=record.target_id,
        target_xyz_m=record.target_xyz_m,
        target_trace=record.target_trace,
        volumes_m3=volumes,
        softening_m=0.0,
        source_type=record.source_type,
    )


def _dynamic_source_values(
    result: SourceContributionResult,
    quantity: str,
    *,
    target_id: str | None,
    aggregation: str,
    frequency_hz: object | None,
    softening_m: float | None,
) -> _SourceValues:
    if quantity not in _DYNAMIC_QUANTITIES:
        raise ValueError(
            "dynamic source fields support 'complex_contribution', 'phase', and "
            "'interference_fraction'"
        )
    if quantity in {"phase", "interference_fraction"}:
        if softening_m is not None and not _softening_matches(
            softening_m, result._data.softening_m
        ):
            raise ValueError("softening_m override must match the stored value")
    elif softening_m is not None and not _softening_matches(
        softening_m, result._data.softening_m
    ):
        raise ValueError("softening_m override must match the stored value")
    if aggregation != "sum":
        raise ValueError("dynamic source contributions require aggregation='sum'")
    data = result._data
    if target_id is None:
        if len(data.target_ids) != 1:
            raise ValueError(
                "target_id is required when the result has multiple targets"
            )
        target_id = data.target_ids[0]
    if not isinstance(target_id, str) or target_id not in data.target_ids:
        raise ValueError(f"unknown target_id {target_id!r}")
    if frequency_hz is None:
        frequency_indices = np.arange(len(data.frequencies_hz), dtype=np.intp)
        frequency = None
        selected_frequencies = np.asarray(data.frequencies_hz, dtype=np.float64)
    else:
        frequency = _frequency_value_hz(frequency_hz)
        matches = np.flatnonzero(data.frequencies_hz == frequency)
        if len(matches) != 1:
            raise ValueError(
                "frequency_hz must exactly match one stored frequency; "
                f"got {frequency:g} Hz"
            )
        frequency_index = int(matches[0])
        frequency_indices = np.asarray([frequency_index], dtype=np.intp)
        selected_frequencies = np.asarray([frequency], dtype=np.float64)
    unit = u.Unit(data.unit)
    if quantity == "phase":
        unit = u.rad
    elif quantity == "interference_fraction":
        unit = u.dimensionless_unscaled
    return _SourceValues(
        positions_m=data.positions_m,
        source_index=data.source_index,
        values=None,
        unit=unit,
        quantity=quantity,
        aggregation=aggregation,
        frame=data.frame,
        coordinate_origin_m=data.coordinate_origin_m,
        source_metadata=data.source_metadata,
        target_id=target_id,
        target_xyz_m=data.targets_m[data.target_ids.index(target_id)],
        softening_m=data.softening_m,
        global_metadata=getattr(data, "global_metadata", None),
        dynamic_result=result,
        frequency_indices=frequency_indices,
        frequency_hz=frequency,
        frequencies_hz=selected_frequencies,
    )


def _normalize_static_softening(value: float | None) -> float:
    if value is None:
        return 0.0
    return _validated_softening(value)


def _softening_matches(value: float, expected: float) -> bool:
    return _validated_softening(value) == float(expected)


def _validated_softening(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("softening_m must be a finite non-negative scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("softening_m must be a finite non-negative scalar") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("softening_m must be a finite non-negative scalar")
    return result


def _frequency_value_hz(value: object) -> float:
    if isinstance(value, u.Quantity):
        raw = np.asarray(value.to_value(u.Hz))
    else:
        raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("frequency_hz must be one finite scalar in Hz")
    frequency = float(raw)
    if not np.isfinite(frequency):
        raise ValueError("frequency_hz must be one finite scalar in Hz")
    return frequency


def _normalize_grid(grid: object) -> _Grid:
    if isinstance(grid, Mapping):
        if set(grid) != {"x", "y", "z"}:
            raise ValueError("grid mapping must contain exactly x, y, and z edges")
        raw_axes = (grid["x"], grid["y"], grid["z"])
    elif isinstance(grid, (tuple, list)) and len(grid) == 3:
        raw_axes = tuple(grid)
    else:
        raise TypeError("grid must be a 3-tuple (x_edges, y_edges, z_edges)")

    edges = []
    centers = []
    for label, raw in zip(("x", "y", "z"), raw_axes, strict=True):
        if isinstance(raw, u.Quantity):
            values = np.asarray(raw.to_value(u.m))
        else:
            values = np.asarray(raw)
        if values.dtype.kind == "b" or values.dtype.kind not in "iuf":
            raise TypeError(f"{label} edges must be a real numeric array")
        values = np.array(values, dtype=np.float64, copy=True)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError(
                f"{label} edges must be a 1D vector with at least two edges"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{label} edges must be finite")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError(f"{label} edges must be strictly increasing")
        edges.append(values)
        centers.append((values[:-1] + values[1:]) / 2.0)
    edge_tuple = tuple(edges)
    center_tuple = tuple(centers)
    for array in (*edge_tuple, *center_tuple):
        array.setflags(write=False)
    return _Grid(
        edges_m=edge_tuple,
        centers_m=center_tuple,
        shape=tuple(len(axis) - 1 for axis in edge_tuple),
    )


def _source_cells(
    positions_m: np.ndarray, source_index: np.ndarray, grid: _Grid
) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("source positions must have shape (N, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("source positions must be finite")
    if source_index.shape != (len(positions),):
        raise ValueError("source_index must align with source positions")
    coords = []
    for axis_index, edges in enumerate(grid.edges_m):
        values = positions[:, axis_index]
        outside = (values < edges[0]) | (values > edges[-1])
        if outside.any():
            row = int(np.flatnonzero(outside)[0])
            raise ValueError(
                "grid does not cover source index "
                f"{int(source_index[row])} on {'xyz'[axis_index]} axis"
            )
        indices = np.searchsorted(edges, values, side="right") - 1
        indices[values == edges[-1]] = len(edges) - 2
        coords.append(indices)
    return np.ravel_multi_index(tuple(coords), grid.shape)


def _component_shape(source_data: _SourceValues) -> tuple[int, ...]:
    if source_data.dynamic_result is not None:
        return source_data.dynamic_result._data.component_shape
    if source_data.physical_observable is not None:
        return (
            (3,) if source_data.physical_observable.endswith("acceleration") else (3, 3)
        )
    return ()


def _accumulate_values(
    output: np.ndarray,
    flat_cells: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray | None,
    weights_by_cell: np.ndarray,
) -> None:
    values = np.asarray(values)
    if values.shape[0] != len(flat_cells):
        raise ValueError("source values must align with grid assignments")
    if weights is not None:
        values = values * weights.reshape((len(weights),) + (1,) * (values.ndim - 1))
        np.add.at(weights_by_cell, flat_cells, weights)
    else:
        np.add.at(weights_by_cell, flat_cells, 1.0)
    component_count = int(np.prod(output.shape[1:])) if output.ndim > 1 else 1
    flat_values = np.asarray(values).reshape(len(values), component_count)
    flat_output = output.reshape(output.shape[0], component_count)
    for component in range(component_count):
        np.add.at(flat_output[:, component], flat_cells, flat_values[:, component])


def _accumulate_dynamic(
    output: np.ndarray,
    flat_cells: np.ndarray,
    source_data: _SourceValues,
    weights_by_cell: np.ndarray,
) -> None:
    assert source_data.dynamic_result is not None
    assert source_data.frequency_indices is not None
    record = source_data.dynamic_result._data
    target_index = record.target_ids.index(source_data.target_id)
    for start, stop, block in record.iter_source_blocks(_BLOCK_SIZE):
        selected = np.asarray(
            block[source_data.frequency_indices, target_index], dtype=np.complex128
        )
        if not np.isfinite(selected).all():
            raise ValueError("dynamic source blocks must contain finite values")
        for frequency_index in range(len(source_data.frequency_indices)):
            _accumulate_values(
                output[frequency_index],
                flat_cells[start:stop],
                selected[frequency_index],
                None,
                weights_by_cell,
            )


def _accumulate_static_physical(
    output: np.ndarray,
    flat_cells: np.ndarray,
    source_data: _SourceValues,
    weights_by_cell: np.ndarray,
) -> None:
    observable = source_data.physical_observable
    assert observable is not None
    kind, _, name = observable.partition(":")
    _check_static_physical_precision(source_data, name)
    if kind == "kernel":
        blocks = iter_source_kernel_terms(
            source_data.positions_m,
            np.ones(len(source_data.positions_m), dtype=np.float64),
            np.asarray(source_data.target_xyz_m).reshape(1, 3),
            observable=name,
            softening_m=source_data.softening_m,
            chunk_size=_BLOCK_SIZE,
        )
    else:
        assert source_data.physical_weights is not None
        blocks = iter_point_observable_source_terms(
            source_data.positions_m,
            source_data.physical_weights,
            np.asarray(source_data.target_xyz_m).reshape(1, 3),
            observable=observable,
            softening_m=source_data.softening_m,
            chunk_size=_BLOCK_SIZE,
        )
    for pair_terms in blocks:
        source_slice, values = pair_terms
        _accumulate_values(
            output,
            flat_cells[source_slice],
            np.asarray(values[0]),
            None,
            weights_by_cell,
        )


def _check_static_physical_precision(
    source_data: _SourceValues, observable: str
) -> None:
    """Apply diagnostics' pair gate while allowing softened exact collisions."""
    assert source_data.target_trace is not None
    assert source_data.target_id is not None
    assert source_data.target_xyz_m is not None
    with np.errstate(over="ignore", invalid="ignore"):
        difference = source_data.positions_m - source_data.target_xyz_m
    _check_finite_pairs(
        source_data.positions_m,
        difference,
        source_data.target_trace,
        source_data.target_id,
        observable,
    )
    try:
        _check_precision_candidates(
            source_data.positions_m,
            difference,
            source_data.target_trace,
            source_data.target_id,
            observable,
        )
    except CoordinatePrecisionError as error:
        if source_data.softening_m == 0.0 or error.reason != "collision":
            raise
        for index, position in enumerate(source_data.positions_m):
            try:
                check_pair_precision(
                    position,
                    source_data.target_trace,
                    target_id=source_data.target_id,
                    source_index=int(source_data.source_index[index]),
                    observables=(observable,),
                )
            except CoordinatePrecisionError as pair_error:
                if pair_error.reason == "collision":
                    continue
                raise


def _derive_field_values(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    unit: u.UnitBase,
    component_shape: tuple[int, ...],
    quantity: str,
    dynamic: bool,
    total: np.ndarray | None,
    norm: str | None,
    projection: object | None,
) -> tuple[np.ndarray, np.ndarray, u.UnitBase, tuple[int, ...]]:
    if not dynamic and quantity in {"phase", "interference_fraction"}:
        raise ValueError(f"{quantity} requires dynamic complex source data")
    if quantity in {"phase", "interference_fraction"} and (
        norm is not None or projection is not None
    ):
        raise ValueError(
            "phase and interference_fraction cannot be combined with norm or projection"
        )
    if norm is not None and projection is not None:
        raise ValueError("norm and projection are mutually exclusive")

    if quantity == "phase":
        zero = values == 0
        derived_valid = valid[(...,) + (None,) * len(component_shape)] & ~zero
        result = np.zeros(values.shape, dtype=np.float64)
        result[derived_valid] = np.angle(values[derived_valid])
        return result, derived_valid, u.rad, component_shape

    if quantity == "interference_fraction":
        if total is None:
            raise ValueError("interference_fraction requires a global complex total")
        total = np.asarray(total, dtype=np.complex128)
        result = np.zeros(values.shape, dtype=np.float64)
        derived_valid = np.broadcast_to(
            valid[(...,) + (None,) * len(component_shape)], values.shape
        ).copy()
        for index in np.ndindex(component_shape or ()):
            comp_total = total[index] if component_shape else total.item()
            if comp_total == 0:
                if component_shape:
                    derived_valid[(...,) + index] = False
                else:
                    derived_valid[...] = False
                continue
            comp_values = values[(...,) + index] if component_shape else values
            # Extended real intermediates preserve tiny aligned components
            # when a cell also contains a much larger orthogonal component.
            # On the supported environment, long double spans the full range
            # of products formed from finite complex128 components.
            value_real = comp_values.real.astype(np.longdouble, copy=False)
            value_imag = comp_values.imag.astype(np.longdouble, copy=False)
            total_real = np.longdouble(comp_total.real)
            total_imag = np.longdouble(comp_total.imag)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                denominator = total_real * total_real + total_imag * total_imag
                ratio = (
                    value_real * total_real + value_imag * total_imag
                ) / denominator
                if component_shape:
                    result[(...,) + index] = ratio
                else:
                    result[...] = ratio
        return result, derived_valid, u.dimensionless_unscaled, component_shape

    if norm is not None:
        if component_shape == (3,):
            if norm != "euclidean":
                raise ValueError("vector norm must be 'euclidean'")
            values = np.sqrt(np.sum(np.abs(values) ** 2, axis=-1))
            component_shape = ()
            valid = np.all(valid[(...,) + (None,)], axis=-1)
        elif component_shape == (3, 3):
            if norm == "frobenius":
                values = np.sqrt(np.sum(np.abs(values) ** 2, axis=(-2, -1)))
            elif norm == "spectral":
                values = np.linalg.svd(values, compute_uv=False)[..., 0]
            else:
                raise ValueError("tensor norm must be 'frobenius' or 'spectral'")
            component_shape = ()
        else:
            raise ValueError("norm applies only to vector or tensor source fields")

    if projection is not None:
        if component_shape == (3,):
            direction = _normalize_direction(projection, "projection")
            values = np.einsum("...i,i->...", values, direction)
            component_shape = ()
            valid = np.all(valid[(...,) + (None,)], axis=-1)
        elif component_shape == (3, 3):
            left, right = _tensor_projection_directions(projection)
            values = np.einsum("i,...ij,j->...", left, values, right)
            component_shape = ()
        else:
            raise ValueError(
                "projection applies only to vector or tensor source fields"
            )
    return values, valid, unit, component_shape


def _normalize_direction(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iuf" or raw.shape != (3,):
        raise ValueError(f"{name} must be a real finite nonzero 3-vector")
    direction = np.asarray(raw, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be a real finite nonzero 3-vector")
    return direction / norm


def _tensor_projection_directions(value: object) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("tensor projection requires (left, right) direction vectors")
    return _normalize_direction(value[0], "left projection"), _normalize_direction(
        value[1], "right projection"
    )


def _spatial_projection_axis(projection: str) -> int:
    """Return the xyz axis collapsed by an XY, XZ, or YZ projection."""
    return {"yz": 0, "xz": 1, "xy": 2}[projection]


def _project_grid(grid: _Grid, projection: str) -> _Grid:
    """Keep projected-axis coverage as one cell and preserve the other axes."""
    collapsed_axis = _spatial_projection_axis(projection)
    edges = []
    centers = []
    for axis, axis_edges in enumerate(grid.edges_m):
        if axis == collapsed_axis:
            projected_edges = np.array(
                [axis_edges[0], axis_edges[-1]], dtype=np.float64
            )
            projected_centers = np.array(
                [0.5 * axis_edges[0] + 0.5 * axis_edges[-1]], dtype=np.float64
            )
            projected_edges.setflags(write=False)
            projected_centers.setflags(write=False)
            edges.append(projected_edges)
            centers.append(projected_centers)
        else:
            edges.append(axis_edges)
            centers.append(grid.centers_m[axis])
    edge_tuple = tuple(edges)
    center_tuple = tuple(centers)
    return _Grid(
        edges_m=edge_tuple,
        centers_m=center_tuple,
        shape=tuple(len(axis) - 1 for axis in edge_tuple),
    )


def _volume_weighted_average(
    numerator: np.ndarray,
    weights_m3: np.ndarray,
    component_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Divide accumulated quantity-volume products by accumulated volume."""
    numerator = np.asarray(numerator)
    weights = np.asarray(weights_m3, dtype=np.float64)
    if numerator.shape[:3] != weights.shape:
        raise ValueError("volume weights must align with spatial field cells")
    if numerator.shape[3:] != component_shape:
        raise ValueError("field components do not align with volume weights")
    valid = weights > 0.0
    denominator = weights[(...,) + (None,) * len(component_shape)]
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result, valid


def _make_field(
    values: np.ndarray,
    unit: u.UnitBase,
    grid: _Grid,
    *,
    frequencies_hz: np.ndarray | None,
):
    from gwexpy import fields

    dynamic = frequencies_hz is not None
    axis0 = (
        np.array([0.0]) * u.s
        if not dynamic
        else np.asarray(frequencies_hz, dtype=np.float64) * u.Hz
    )
    domain = "frequency" if dynamic else "time"
    axis_name = "frequency" if dynamic else "time"
    axes = [axis0, *(center * u.m for center in grid.centers_m)]
    common = {
        "unit": unit,
        "axis0": axes[0],
        "axis1": axes[1],
        "axis2": axes[2],
        "axis3": axes[3],
        "axis_names": [axis_name, "x", "y", "z"],
        "axis0_domain": domain,
        "space_domain": {"x": "real", "y": "real", "z": "real"},
    }
    native = values if dynamic else values[np.newaxis, ...]
    spatial_rank = values.ndim - int(dynamic)
    if spatial_rank == 3:
        return fields.ScalarField(native, **common)
    if spatial_rank == 4 and values.shape[-1] == 3:
        return fields.VectorField(
            {
                key: fields.ScalarField(native[..., index], **common)
                for index, key in enumerate(("x", "y", "z"))
            }
        )
    if spatial_rank == 5 and values.shape[-2:] == (3, 3):
        return fields.TensorField(
            {
                (i, j): fields.ScalarField(native[..., i, j], **common)
                for i in range(3)
                for j in range(3)
            },
            rank=2,
        )
    raise ValueError(f"unsupported field value shape {values.shape}")


def _field_metadata(
    source_data: _SourceValues,
    grid: _Grid,
    quantity: str,
    aggregation: str,
    unit: u.UnitBase,
) -> Mapping[str, Any]:
    dynamic_data = (
        None if source_data.dynamic_result is None else source_data.dynamic_result._data
    )
    metadata = {
        "grid_edges_m": tuple(
            tuple(float(item) for item in axis) for axis in grid.edges_m
        ),
        "grid_centers_m": tuple(
            tuple(float(item) for item in axis) for axis in grid.centers_m
        ),
        "unit": unit.to_string(),
        "frame": source_data.frame,
        "coordinate_origin_m": tuple(
            float(item) for item in source_data.coordinate_origin_m
        ),
        "quantity": quantity,
        "canonical_quantity": source_data.quantity,
        "aggregation": aggregation,
        "target_id": source_data.target_id,
        "target_xyz_m": (
            None
            if source_data.target_xyz_m is None
            else tuple(float(item) for item in source_data.target_xyz_m)
        ),
        "frequency_hz": source_data.frequency_hz,
        "softening_m": source_data.softening_m,
        "source_index": tuple(int(item) for item in source_data.source_index),
        "source_positions_m": tuple(
            tuple(float(value) for value in row) for row in source_data.positions_m
        ),
        "source_ids": None if dynamic_data is None else dynamic_data.source_ids,
        "source_type": source_data.source_type,
        "source_metadata": source_data.source_metadata,
        "global_metadata": source_data.global_metadata,
        "t_ref_s": None if dynamic_data is None else dynamic_data.t_ref_s,
        "phase_origin_m": (
            None
            if dynamic_data is None or dynamic_data.phase_origin_m is None
            else tuple(float(item) for item in dynamic_data.phase_origin_m)
        ),
        "fourier_convention": (
            None if dynamic_data is None else dynamic_data.fourier_convention
        ),
        "peak_phasor": None if dynamic_data is None else dynamic_data.peak_phasor,
        "input_provenance": {
            "source_metadata": source_data.source_metadata,
            "global_metadata": source_data.global_metadata,
        },
        "frequencies_hz": (
            None
            if source_data.frequencies_hz is None
            else tuple(float(item) for item in source_data.frequencies_hz)
        ),
        "frequency_indices": (
            None
            if source_data.frequency_indices is None
            else tuple(int(item) for item in source_data.frequency_indices)
        ),
        "axis0_semantics": (
            "structural singleton; not physical time"
            if dynamic_data is None
            else (
                "all stored frequencies"
                if source_data.frequency_hz is None
                else "selected frequency"
            )
        ),
    }
    if dynamic_data is not None:
        metadata["observable"] = dynamic_data.observable
    return _freeze_metadata(metadata)


def _static_source_metadata(view: object) -> Mapping[str, Any] | None:
    bound = view._bound
    if hasattr(bound, "provenance"):
        return bound.provenance
    if hasattr(bound, "discretization"):
        return bound.discretization
    return None


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[Any, Any] = {}
        for key, item in value.items():
            frozen[_freeze_metadata(key)] = _freeze_metadata(item)
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("metadata object arrays are not supported")
        copied = np.array(value, copy=True, order="C")
        return np.frombuffer(copied.tobytes(order="C"), dtype=copied.dtype).reshape(
            copied.shape
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.generic):
        return _freeze_metadata(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("metadata floating-point values must be finite")
    return value
