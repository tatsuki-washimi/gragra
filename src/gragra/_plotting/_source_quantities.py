"""Static source quantity selectors and shared geometry helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from astropy import units as u

from gragra._arrays import _check_no_bool, as_xyz_table
from gragra._evaluation_setup import CoordinatePrecisionError, check_pair_precision
from gragra._plotting._diagnostics import (
    _extract_values,
    _resolve_source,
    _resolved_from,
    _select_target,
    _source_distance_squared,
    _source_positions_in_frame,
)
from gragra.geometry import Frame
from gragra.kernels._source_terms import iter_source_kernel_terms
from gragra.targets.points import TargetPoints as NumericTargetPoints

_GG_NORM_BLOCK_SIZE = 256


@dataclass(frozen=True)
class QuantitySpec:
    """Canonical static source quantity metadata."""

    canonical: str
    aliases: tuple[str, ...]
    unit: u.UnitBase
    extensive: bool
    target_dependent: bool
    requires_density: bool = False
    requires_kernel_norm: bool = False


_SPECS = (
    QuantitySpec("distance", ("r",), u.m, False, True),
    QuantitySpec("inverse_r", ("1/r", "inverse_distance"), u.m**-1, False, True),
    QuantitySpec(
        "inverse_square",
        ("1/r^2", "r^-2", "inverse_r2", "inverse_square_distance"),
        u.m**-2,
        False,
        True,
    ),
    QuantitySpec(
        "inverse_cube",
        ("1/r^3", "inverse_r3", "inverse_cube_distance"),
        u.m**-3,
        False,
        True,
    ),
    QuantitySpec(
        "gravity_gradient_kernel_norm",
        ("GG_kernel_norm",),
        u.m**-3,
        True,
        True,
        requires_kernel_norm=True,
    ),
    QuantitySpec("density", ("rho", "ρ"), u.kg / u.m**3, False, False, True),
    QuantitySpec(
        "density_over_r",
        ("rho_over_r", "rho/r", "ρ/r", "density/r"),
        u.kg / u.m**4,
        False,
        True,
        True,
    ),
    QuantitySpec(
        "density_over_r2",
        ("rho_over_r2", "rho/r^2", "ρ/r²", "density/r2"),
        u.kg / u.m**5,
        False,
        True,
        True,
    ),
    QuantitySpec(
        "density_over_r3",
        (
            "rho_over_r3",
            "rho/r^3",
            "ρ/r³",
            "density/r3",
            "gg_weight_density",
        ),
        u.kg / u.m**6,
        False,
        True,
        True,
    ),
    QuantitySpec("mass", (), u.kg, True, False),
    QuantitySpec("mass_over_r", ("mass/r",), u.kg / u.m, True, True),
    QuantitySpec(
        "mass_over_r2",
        ("mass/r2", "mass/r^2", "dm/r^2", "Δm/r²"),
        u.kg / u.m**2,
        True,
        True,
    ),
    QuantitySpec(
        "mass_over_r3",
        ("mass/r3", "mass/r^3", "dm/r^3", "Δm/r³", "gg_weight"),
        u.kg / u.m**3,
        True,
        True,
    ),
)

QUANTITY_SPECS: Mapping[str, QuantitySpec] = MappingProxyType(
    {spec.canonical: spec for spec in _SPECS}
)
QUANTITY_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        name: spec.canonical
        for spec in _SPECS
        for name in (spec.canonical, *spec.aliases)
    }
)


def _as_real_float64(values: object, name: str) -> np.ndarray:
    """Copy real numeric values to float64 without discarding imaginary parts."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must contain only real values")
    return np.array(raw, dtype=np.float64, copy=True)


def resolve_quantity(quantity: str) -> QuantitySpec:
    """Resolve one exact canonical quantity ID or registered alias.

    The selector is a fixed table lookup. Expressions and spelling variants
    that are not listed in :data:`QUANTITY_ALIASES` are rejected.
    """
    if not isinstance(quantity, str):
        raise TypeError("quantity must be a string")
    try:
        return QUANTITY_SPECS[QUANTITY_ALIASES[quantity]]
    except KeyError as error:
        choices = ", ".join(sorted(QUANTITY_ALIASES))
        raise ValueError(
            f"unknown source quantity {quantity!r}; expected {choices}"
        ) from error


@dataclass(frozen=True, eq=False)
class StaticSourceQuantityData:
    """Immutable per-source static values and their geometry context."""

    values: np.ndarray
    positions_m: np.ndarray
    source_index: np.ndarray
    quantity: str
    unit: u.UnitBase
    extensive: bool
    frame: Frame
    coordinate_origin_m: np.ndarray
    target_id: str | None = None
    target_xyz_m: np.ndarray | None = None
    target_trace: object | None = None
    source_type: str | None = None
    source_metadata: Mapping[str, Any] | None = None
    kernel_norm: str | None = None
    softening_m: float = 0.0
    radial_distance_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = _as_real_float64(self.values, "values")
        positions = np.array(self.positions_m, dtype=np.float64, copy=True)
        indices = np.array(self.source_index, dtype=np.int64, copy=True)
        origin = np.array(self.coordinate_origin_m, dtype=np.float64, copy=True)
        if values.ndim != 1 or positions.shape != (len(values), 3):
            raise ValueError("values and positions_m must align on source axis")
        if indices.shape != (len(values),) or not np.array_equal(
            indices, np.arange(len(values))
        ):
            raise ValueError("source_index must be the stable ordered index 0..N-1")
        if origin.shape != (3,):
            raise ValueError("coordinate_origin_m must have shape (3,)")
        if self.kernel_norm is not None and (
            not isinstance(self.kernel_norm, str)
            or self.kernel_norm not in {"frobenius", "spectral"}
        ):
            raise ValueError("kernel_norm must be 'frobenius', 'spectral', or None")
        softening = float(self.softening_m)
        if not np.isfinite(softening) or softening < 0.0:
            raise ValueError("softening_m must be finite and non-negative")
        if self.kernel_norm is None and softening != 0.0:
            raise ValueError("softening_m is only retained for a kernel norm")
        radial_distance = None
        if self.radial_distance_m is not None:
            radial_distance = _as_real_float64(
                self.radial_distance_m, "radial_distance_m"
            )
            if radial_distance.shape != values.shape:
                raise ValueError("radial_distance_m must align with values")
            if not np.isfinite(radial_distance).all() or np.any(radial_distance < 0.0):
                raise ValueError("radial_distance_m must be finite and non-negative")
        if not (
            np.isfinite(values).all()
            and np.isfinite(positions).all()
            and np.isfinite(origin).all()
        ):
            raise ValueError("source quantity data must be finite")
        target = None
        if self.target_xyz_m is not None:
            target = np.array(self.target_xyz_m, dtype=np.float64, copy=True)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("target_xyz_m must be a finite 3-vector")
            target.setflags(write=False)
        for array in (values, positions, indices, origin):
            array.setflags(write=False)
        if radial_distance is not None:
            radial_distance.setflags(write=False)
        metadata = self.source_metadata
        if metadata is not None:
            metadata = MappingProxyType(dict(metadata))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "source_index", indices)
        object.__setattr__(self, "coordinate_origin_m", origin)
        object.__setattr__(self, "target_xyz_m", target)
        object.__setattr__(self, "source_metadata", metadata)
        object.__setattr__(self, "softening_m", softening)
        object.__setattr__(self, "radial_distance_m", radial_distance)


def source_quantity_data(
    setup: object,
    quantity: str,
    *,
    target_id: str | None = None,
    frame: Frame | None = None,
    softening_m: float = 0.0,
    norm: str | None = None,
) -> StaticSourceQuantityData:
    """Extract one registered scalar quantity for every static source.

    Geometry ratios use the unsoftened Euclidean distance. The gravity-gradient
    kernel norm instead uses the existing Plummer point kernel and requires an
    explicit Frobenius or spectral norm. Density values are available only
    when existing source diagnostics can verify them against source mass
    weights and volumes.
    """
    spec = resolve_quantity(quantity)
    if spec.requires_kernel_norm:
        if not isinstance(norm, str) or norm not in {"frobenius", "spectral"}:
            raise ValueError("norm must be 'frobenius' or 'spectral'")
        kernel_norm = norm
    else:
        if norm is not None:
            raise ValueError("norm applies only to gravity_gradient_kernel_norm")
        kernel_norm = None
    if isinstance(softening_m, (bool, np.bool_)):
        raise ValueError("softening_m must be a finite non-negative scalar")
    try:
        softening = float(softening_m)
    except (TypeError, ValueError) as error:
        raise ValueError("softening_m must be a finite non-negative scalar") from error
    if not np.isfinite(softening) or softening < 0.0:
        raise ValueError("softening_m must be a finite non-negative scalar")
    if not spec.requires_kernel_norm and softening != 0.0:
        raise ValueError(
            "softening_m must be zero; static geometry ratios are unsoftened"
        )

    resolved = _resolved_from(setup)
    view = _resolve_source(resolved)
    target_index = _select_target(resolved, target_id, required=spec.target_dependent)
    positions, output_frame = _source_positions_in_frame(view, resolved, frame)

    target_id_value: str | None = None
    target_xyz: np.ndarray | None = None
    target_trace = None
    distance: np.ndarray | None = None
    radial_distance: np.ndarray | None = None
    if target_index is not None:
        target_id_value = resolved.targets.ids[target_index]
        target_trace = resolved.target_traces[target_index]
        target_xyz = np.asarray(target_trace.point_m, dtype=np.float64)
        if spec.requires_kernel_norm:
            radial_distance = _check_kernel_norm_precision(
                view.positions_m,
                target_trace,
                target_id=target_id_value,
                softening_m=softening,
            )
        else:
            squared = _source_distance_squared(
                view,
                resolved,
                target_index,
                target_id=target_id_value,
                quantity=spec.canonical,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                distance = np.sqrt(squared)
            radial_distance = distance
        if output_frame is not resolved.computation_frame:
            target_xyz = resolved.computation_frame.transform_points_m(
                target_xyz.reshape(1, 3), output_frame
            )[0]
    if spec.canonical == "distance":
        assert distance is not None
        values = distance
    elif spec.canonical == "inverse_r":
        assert distance is not None
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            values = 1.0 / distance
    elif spec.canonical == "inverse_square":
        assert target_index is not None
        values = _extract_values(
            view,
            "inverse_square",
            resolved=resolved,
            target_index=target_index,
            frame=frame,
        )
    elif spec.canonical == "inverse_cube":
        assert distance is not None
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            inverse_r = 1.0 / distance
            values = inverse_r * inverse_r * inverse_r
    elif spec.requires_kernel_norm:
        assert target_trace is not None
        values = _gravity_gradient_kernel_norm_values(
            view.positions_m,
            np.asarray(target_trace.point_m, dtype=np.float64),
            norm=kernel_norm,
            softening_m=softening,
        )
    elif spec.canonical == "density":
        values = view.density()
    elif spec.canonical == "mass":
        values = view.weights_kg
    elif spec.canonical.endswith("_over_r3"):
        assert distance is not None
        base = view.density() if spec.requires_density else view.weights_kg
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            inverse_r = 1.0 / distance
            values = base * inverse_r * inverse_r * inverse_r
    else:
        assert distance is not None
        base = view.density() if spec.requires_density else view.weights_kg
        divisor = distance if spec.canonical.endswith("_over_r") else distance**2
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            values = base / divisor
    values = _as_real_float64(values, "static source quantities")
    if not np.isfinite(values).all():
        bad = int(np.flatnonzero(~np.isfinite(values))[0])
        raise ValueError(f"{spec.canonical} is not finite at source index {bad}")

    bound = view._bound
    if hasattr(bound, "provenance"):
        source_metadata = bound.provenance
    elif hasattr(bound, "discretization"):
        source_metadata = bound.discretization
    else:
        source_metadata = None
    return StaticSourceQuantityData(
        values=values,
        positions_m=positions,
        source_index=np.arange(len(positions), dtype=np.int64),
        quantity=spec.canonical,
        unit=spec.unit,
        extensive=spec.extensive,
        frame=output_frame,
        coordinate_origin_m=output_frame.origin_m,
        target_id=target_id_value,
        target_xyz_m=target_xyz,
        target_trace=target_trace,
        source_type=view.source_type,
        source_metadata=source_metadata,
        kernel_norm=kernel_norm,
        softening_m=softening if spec.requires_kernel_norm else 0.0,
        radial_distance_m=radial_distance,
    )


def _gravity_gradient_kernel_norm_values(
    positions_m: np.ndarray,
    target_xyz_m: np.ndarray,
    *,
    norm: str | None,
    softening_m: float,
) -> np.ndarray:
    """Compute each G-free source tensor norm before any source aggregation."""
    if not isinstance(norm, str) or norm not in {"frobenius", "spectral"}:
        raise ValueError("norm must be 'frobenius' or 'spectral'")
    values = np.empty(len(positions_m), dtype=np.float64)
    weights = np.ones(len(positions_m), dtype=np.float64)
    for source_slice, tensors in iter_source_kernel_terms(
        positions_m,
        weights,
        target_xyz_m.reshape(1, 3),
        observable="gravity_gradient",
        softening_m=softening_m,
        chunk_size=_GG_NORM_BLOCK_SIZE,
    ):
        source_tensors = tensors[0]
        if norm == "frobenius":
            block_values = np.linalg.norm(source_tensors, ord="fro", axis=(-2, -1))
        else:
            block_values = np.linalg.svd(source_tensors, compute_uv=False)[..., 0]
        values[source_slice] = block_values
    return values


def _check_kernel_norm_precision(
    positions_m: np.ndarray,
    target_trace: object,
    *,
    target_id: str,
    softening_m: float,
) -> np.ndarray:
    """Return radii after checking precision, allowing softened collisions."""
    from gragra._plotting._diagnostics import (
        _check_finite_pairs,
        _check_precision_candidates,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        difference = positions_m - target_trace.point_m
    _check_finite_pairs(
        positions_m, difference, target_trace, target_id, "gravity_gradient"
    )
    try:
        _check_precision_candidates(
            positions_m, difference, target_trace, target_id, "gravity_gradient"
        )
    except CoordinatePrecisionError as error:
        if softening_m == 0.0 or error.reason != "collision":
            raise
        for index, position in enumerate(positions_m):
            try:
                check_pair_precision(
                    position,
                    target_trace,
                    target_id=target_id,
                    source_index=index,
                    observables=("gravity_gradient",),
                )
            except CoordinatePrecisionError as pair_error:
                if pair_error.reason == "collision":
                    continue
                raise
    radii = np.linalg.norm(difference, axis=1)
    if not np.isfinite(radii).all():
        bad = int(np.flatnonzero(~np.isfinite(radii))[0])
        raise ValueError(
            f"gravity_gradient_kernel_norm radius is not finite at source index {bad}"
        )
    return radii


def normalize_targets_m(targets_m: object) -> np.ndarray:
    """Normalize targets to a copied read-only SI ``(M,3)`` array.

    Accepts bare numeric coordinates or the numeric-layer
    ``gragra.targets.points.TargetPoints``. Numeric coordinates are
    interpreted in the caller's source frame and origin.
    """
    if isinstance(targets_m, NumericTargetPoints):
        values = targets_m.positions_m
    else:
        if hasattr(targets_m, "ids") and hasattr(targets_m, "frame"):
            raise TypeError(
                "targets_m accepts only numeric coordinates or "
                "gragra.targets.points.TargetPoints"
            )
        _check_no_bool(targets_m, "targets_m")
        raw = np.asarray(targets_m)
        if np.iscomplexobj(raw):
            raise TypeError("targets_m must contain only real coordinates")
        if raw.ndim == 1 and raw.shape == (3,):
            raw = raw[np.newaxis, :]
        try:
            values = as_xyz_table("targets_m", raw)
        except (ValueError, TypeError) as error:
            raise TypeError(
                "targets_m must be numeric coordinates or "
                f"gragra.targets.points.TargetPoints: {error}"
            ) from error
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def source_radial_membership(
    distances_m: object, edges_m: object
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Assign each target/source distance to underflow, a finite bin, or overflow.

    Finite bins are half-open ``[left,right)`` except that the final right edge
    belongs to the final bin. The returned membership is an integer array of
    shape ``(M,N)`` whose values index the returned ordered ``group_ids``.
    """
    raw_distances = np.asarray(distances_m)
    if raw_distances.dtype.kind == "b" or raw_distances.dtype.kind not in "iuf":
        raise ValueError("distances_m must be a finite real (M,N) array")
    distances = np.asarray(raw_distances, dtype=np.float64)
    if distances.ndim != 2 or not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distances_m must be a finite non-negative (M,N) array")
    raw_edges = np.asarray(edges_m)
    if raw_edges.dtype.kind == "b" or np.iscomplexobj(raw_edges):
        raise ValueError("edges_m must be a finite real vector")
    try:
        edges = np.asarray(edges_m, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "edges_m must be a finite strictly increasing vector"
        ) from error
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or (np.diff(edges) <= 0.0).any()
    ):
        raise ValueError("edges_m must be a finite strictly increasing vector")
    n_bins = len(edges) - 1
    indices = np.searchsorted(edges, distances, side="right") - 1
    indices[distances < edges[0]] = -1
    indices[distances > edges[-1]] = n_bins
    indices[distances == edges[-1]] = n_bins - 1
    membership = indices + 1
    membership = np.asarray(membership, dtype=np.int64)
    membership.setflags(write=False)
    group_ids = ("underflow", *(f"bin_{index}" for index in range(n_bins)), "overflow")
    return membership, tuple(group_ids)
