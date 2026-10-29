"""Source-element diagnostics shared by histogram and plotting code.

This module imports neither Matplotlib nor GWexpy. Density and volume are
returned only when they can be checked against the mass weights that the
binding actually holds; position-only diagnostics never trigger that check.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy import units as u

from gragra._evaluation_setup import (
    CoordinatePrecisionError,
    DiscreteSource,
    ResolvedEvaluation,
    _StaticPointSourceBinding,
    check_pair_precision,
)
from gragra.constants import G_SI
from gragra.geometry._frame import Frame
from gragra.gravity_sources.volume import VolumeElementSource

QUANTITIES = ("density", "height", "distance", "inverse_square")
QUANTITIES_MAP = (
    "kernel_norm",
    "delta_a_norm",
    "delta_a_x",
    "delta_a_y",
    "delta_a_z",
    "delta_a_projection",
)
TARGET_INDEPENDENT_QUANTITIES = frozenset({"density", "height"})
STANDARD_XUNIT = {
    "density": u.kg / u.m**3,
    "height": u.m,
    "distance": u.m,
    "inverse_square": u.m**-2,
}


@dataclass(frozen=True, eq=False)
class _MapData:
    """Preprocessed spatial map diagnostic data for one target."""

    positions_m: np.ndarray
    values: np.ndarray
    delta_a_m_s2: np.ndarray | None
    quantity: str
    unit_str: str
    target_id: str
    frame: Frame


# Same relative tolerance that the Discrete-source mass check uses in the
# evaluation setup, so both layers accept the same DiscreteSource weights.
_DISCRETE_WEIGHT_RTOL = 1e-12
_EPS64 = np.finfo(np.float64).eps
_TINY64 = np.finfo(np.float64).tiny


@dataclass(frozen=True, eq=False)
class _SourceView:
    """Source positions and mass weights, with density and volume on demand."""

    positions_m: np.ndarray
    weights_kg: np.ndarray
    source_type: str
    _bound: object
    _model_source: object

    def density(self) -> np.ndarray:
        """Per-element density in kg/m^3, verified against the mass weights."""
        return self._elements[0]

    def volumes(self) -> np.ndarray:
        """Per-element volume in m^3, verified against the mass weights."""
        return self._elements[1]

    @cached_property
    def _elements(self) -> tuple[np.ndarray, np.ndarray]:
        """Verified density and volume, checked once per view."""
        if isinstance(self._bound, DiscreteSource):
            return _verify_discrete(self._bound, self.weights_kg)
        if isinstance(self._model_source, VolumeElementSource):
            return _verify_volume_elements(
                self._model_source, self.positions_m, self.weights_kg
            )
        raise ValueError(
            f"{self.source_type} carries mass weights only; density and volume "
            "are not defined for it and are not inferred from weights_kg"
        )


def _cell_volume_m3(source: DiscreteSource) -> float:
    raw = source.discretization.get("cell_volume_m3")
    if raw is None:
        raise ValueError(
            "DiscreteSource discretization has no cell_volume_m3, so its mass "
            "weights cannot be checked against density or volume; density and "
            "volume diagnostics are unavailable for this source"
        )
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, numbers.Real):
        raise ValueError("cell_volume_m3 must be a real number")
    cell = float(raw)
    if not (np.isfinite(cell) and cell > 0.0):
        raise ValueError("cell_volume_m3 must be a positive finite number")
    return cell


def _verify_discrete(
    source: DiscreteSource, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cell = _cell_volume_m3(source)
    density = float(source.density_kg_m3)
    if not np.allclose(weights, density * cell, rtol=_DISCRETE_WEIGHT_RTOL, atol=0.0):
        raise ValueError(
            "DiscreteSource weights_kg disagree with density_kg_m3 * "
            f"cell_volume_m3 beyond relative tolerance {_DISCRETE_WEIGHT_RTOL:g}"
        )
    count = len(weights)
    return np.full(count, density, dtype=np.float64), np.full(
        count, cell, dtype=np.float64
    )


def _verify_volume_elements(
    model: VolumeElementSource, positions: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(model.density_kg_m3):
        raise ValueError("complex density cannot be verified against mass weights")
    expected = model.as_weighted_source()
    if not (
        np.array_equal(expected.positions_m, positions)
        and np.array_equal(expected.weights_kg, weights)
    ):
        raise ValueError(
            "bound source does not reproduce the model's VolumeElementSource "
            "positions and weights in element order"
        )
    count = len(positions)
    density = np.array(
        np.broadcast_to(np.asarray(model.density_kg_m3, dtype=np.float64), (count,)),
        dtype=np.float64,
    )
    volumes = np.array(model.volumes_m3, dtype=np.float64)
    return density, volumes


def _resolved_from(setup: object) -> ResolvedEvaluation:
    candidate = getattr(setup, "resolved", setup)
    if not isinstance(candidate, ResolvedEvaluation):
        raise TypeError("setup must be an EvaluationSetup or ResolvedEvaluation")
    return candidate


def _resolve_source(resolved: ResolvedEvaluation) -> _SourceView:
    bound = resolved.source
    if bound is None:
        raise ValueError(
            "source diagnostics need a discrete point source; analytic models "
            "have no source elements"
        )
    if isinstance(bound, DiscreteSource):
        source_type = "DiscreteSource"
    elif isinstance(bound, _StaticPointSourceBinding):
        source_type = bound.source_type
    else:
        raise TypeError("unsupported source binding")
    return _SourceView(
        positions_m=bound.positions_m,
        weights_kg=bound.weights_kg,
        source_type=source_type,
        _bound=bound,
        _model_source=getattr(resolved.model, "source", None),
    )


def _select_target(
    resolved: ResolvedEvaluation, target_id: str | None, *, required: bool
) -> int | None:
    """Return the index of the one target a quantity uses, or None."""
    if not required:
        if target_id is not None:
            raise ValueError("target_id applies only to distance and inverse_square")
        return None
    ids = tuple(resolved.targets.ids)
    if not ids:
        raise ValueError("distance and inverse_square need at least one target")
    if target_id is None:
        if len(ids) > 1:
            raise ValueError(
                f"target_id is required when the setup has {len(ids)} targets"
            )
        return 0
    if target_id not in ids:
        raise ValueError(f"unknown target_id {target_id!r}")
    return ids.index(target_id)


def _check_frame(resolved: ResolvedEvaluation, frame: Frame | None) -> None:
    """Reject a frame that is not connected to the computation frame."""
    if frame is None:
        return
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame or None")
    resolved.computation_frame._path_to(frame)


def _require_finite(values: np.ndarray, name: str) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.all():
        index = int(np.flatnonzero(~finite)[0])
        raise ValueError(f"{name} is not finite at element index {index}")
    return values


def _pair_message(text: str, target_id: str, index: int) -> str:
    return f"{text}: target ID={target_id!r}, source index={index}"


def _extract_values(
    view: _SourceView,
    quantity: str,
    *,
    resolved: ResolvedEvaluation,
    target_index: int | None,
    frame: Frame | None,
) -> np.ndarray:
    """Return one float64 value per source element, in SI units."""
    if quantity == "density":
        return np.asarray(view.density(), dtype=np.float64)
    if quantity == "height":
        return _heights(view, resolved, frame)
    if target_index is None:
        raise ValueError(f"{quantity} requires a target")
    target_id = resolved.targets.ids[target_index]
    squared = _separations_squared(
        view,
        resolved.target_traces[target_index],
        target_id=target_id,
        quantity=quantity,
    )
    if quantity == "distance":
        return np.sqrt(squared)
    return _inverse_squares(squared, target_id)


def _heights(
    view: _SourceView, resolved: ResolvedEvaluation, frame: Frame | None
) -> np.ndarray:
    if frame is None or frame is resolved.computation_frame:
        return np.array(view.positions_m[:, 2], dtype=np.float64)
    transformed = resolved.computation_frame.transform_points_m(view.positions_m, frame)
    return np.array(transformed[:, 2], dtype=np.float64)


def _difference_and_squared(
    view: _SourceView,
    trace: object,
    *,
    target_id: str,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative displacement and squared distances in computation frame.

    The checks mirror the discrete-pair validation of the evaluation setup,
    applied to the single selected target. Collisions and cross-frame near
    coincidences raise CoordinatePrecisionError. Additionally, a nonzero
    separation whose squared sum is zero or subnormal is rejected, because the
    distance and inverse-square values derived from it would not be usable.
    """
    positions = view.positions_m
    with np.errstate(over="ignore", invalid="ignore"):
        difference = positions - trace.point_m
    _check_finite_pairs(positions, difference, trace, target_id, quantity)
    squared = _squared_sum(difference, target_id)
    chebyshev = _check_precision_candidates(
        positions, difference, trace, target_id, quantity
    )
    underflow = (chebyshev > 0.0) & (squared < _TINY64)
    if underflow.any():
        index = int(np.flatnonzero(underflow)[0])
        raise ValueError(
            _pair_message(
                "squared separation is zero or subnormal for a nonzero separation; "
                "distance and inverse square cannot be represented",
                target_id,
                index,
            )
        )
    return difference, squared


def _separations_squared(
    view: _SourceView,
    trace: object,
    *,
    target_id: str,
    quantity: str,
) -> np.ndarray:
    return _difference_and_squared(view, trace, target_id=target_id, quantity=quantity)[
        1
    ]


def _check_finite_pairs(
    positions: np.ndarray,
    difference: np.ndarray,
    trace: object,
    target_id: str,
    quantity: str,
) -> None:
    finite = np.isfinite(difference).all(axis=1)
    if finite.all():
        return
    index = int(np.flatnonzero(~finite)[0])
    try:
        check_pair_precision(
            positions[index],
            trace,
            target_id=target_id,
            source_index=index,
            observables=(quantity,),
        )
    except CoordinatePrecisionError:
        raise
    except ValueError as cause:
        raise ValueError(
            _pair_message("pair distance is not finite", target_id, index)
        ) from cause
    raise ValueError(_pair_message("pair distance is not finite", target_id, index))


def _squared_sum(difference: np.ndarray, target_id: str) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise"):
            return np.sum(np.square(difference), axis=1)
    except FloatingPointError as cause:
        index = _first_overflow_index(difference)
        raise ValueError(
            _pair_message("squared separation overflows float64", target_id, index)
        ) from cause


def _first_overflow_index(difference: np.ndarray) -> int:
    # Only reached after a vectorized overflow, so the Python loop is rare.
    for index, offset in enumerate(difference):
        try:
            with np.errstate(over="raise", invalid="raise"):
                np.sum(np.square(offset))
        except FloatingPointError:
            return index
    return -1


def _check_precision_candidates(
    positions: np.ndarray,
    difference: np.ndarray,
    trace: object,
    target_id: str,
    quantity: str,
) -> np.ndarray:
    """Return the Chebyshev distances after the collision and proximity gate."""
    chebyshev = np.maximum(difference.max(axis=1), -difference.min(axis=1))
    candidates = chebyshev == 0.0
    if trace.edge_count:
        source_scales = np.max(np.abs(positions), axis=1)
        scales = np.maximum(trace.scale_m, source_scales)
        thresholds = 64 * trace.edge_count * _EPS64 * scales
        candidates |= chebyshev <= thresholds
    for index in np.flatnonzero(candidates):
        check_pair_precision(
            positions[index],
            trace,
            target_id=target_id,
            source_index=int(index),
            observables=(quantity,),
        )
    return chebyshev


def _inverse_squares(squared: np.ndarray, target_id: str) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse = 1.0 / squared
    bad = ~np.isfinite(inverse)
    if bad.any():
        index = int(np.flatnonzero(bad)[0])
        raise ValueError(
            _pair_message("inverse square is not a finite value", target_id, index)
        )
    return inverse


def _first_cube_overflow_index(squared: np.ndarray, r: np.ndarray) -> int:
    for index, (s, rad) in enumerate(zip(squared, r, strict=True)):
        try:
            with np.errstate(over="raise", invalid="raise"):
                prod = s * rad
        except FloatingPointError:
            return index
        if not np.isfinite(prod):
            return index
    # Fallback only: the caller invokes this helper after an overflow was
    # raised, so a hit is expected. Returning 0 is a guard for the unreachable
    # case where no element reproduces the overflow; it does not change behavior.
    return 0


def _extract_map_data(
    setup: object,
    quantity: str,
    *,
    target_id: str | None = None,
    frame: Frame | None = None,
    direction: object | None = None,
) -> _MapData:
    """Extract positions, field values, and acceleration contributions for plot_map."""
    if quantity not in QUANTITIES_MAP:
        raise ValueError(
            f"unknown quantity {quantity!r}; expected one of {QUANTITIES_MAP}"
        )
    resolved = _resolved_from(setup)
    view = _resolve_source(resolved)
    target_index = _select_target(resolved, target_id, required=True)
    assert target_index is not None
    target_id_str = resolved.targets.ids[target_index]
    _check_frame(resolved, frame)

    # Normalize direction if delta_a_projection
    unit_dir: np.ndarray | None = None
    if quantity == "delta_a_projection":
        if direction is None:
            raise ValueError("direction is required for delta_a_projection")
        d_arr = np.asarray(direction)
        if np.iscomplexobj(d_arr):
            raise ValueError("direction must contain only real numbers")
        d_arr = np.asarray(d_arr, dtype=np.float64)
        if d_arr.shape != (3,):
            raise ValueError(f"direction must have shape (3,), got {d_arr.shape}")
        if not np.isfinite(d_arr).all():
            raise ValueError("direction must contain only finite numbers")
        # Numerically stable normalization avoiding overflow/underflow in squared norm
        d_max = float(np.max(np.abs(d_arr)))
        if d_max == 0.0 or not np.isfinite(d_max):
            raise ValueError("direction vector cannot have zero norm")
        d_scaled = d_arr / d_max
        d_norm_scaled = float(np.linalg.norm(d_scaled))
        if d_norm_scaled == 0.0 or not np.isfinite(d_norm_scaled):
            raise ValueError("direction vector cannot have zero norm")
        unit_dir = d_scaled / d_norm_scaled
    else:
        if direction is not None:
            raise ValueError(
                f"direction applies only to delta_a_projection, not {quantity!r}"
            )

    trace = resolved.target_traces[target_index]
    difference, squared = _difference_and_squared(
        view, trace, target_id=target_id_str, quantity=quantity
    )

    # Destination Frame setup for positions
    target_frame = resolved.computation_frame if frame is None else frame
    if target_frame is resolved.computation_frame:
        positions_transformed = np.asarray(view.positions_m, dtype=np.float64)
    else:
        positions_transformed = resolved.computation_frame.transform_points_m(
            view.positions_m, target_frame
        )

    # FAST PATH: kernel_norm needs only 1/r^2, independent of r^3 or mass weights
    if quantity == "kernel_norm":
        values = _inverse_squares(squared, target_id_str)
        return _MapData(
            positions_m=positions_transformed,
            values=values,
            delta_a_m_s2=None,
            quantity=quantity,
            unit_str="m^-2",
            target_id=target_id_str,
            frame=target_frame,
        )

    # 1. r^3 calculation with overflow and underflow safety checks
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.sqrt(squared)
    with np.errstate(over="raise", invalid="raise"):
        try:
            r3 = squared * r
        except FloatingPointError as cause:
            index = _first_cube_overflow_index(squared, r)
            raise ValueError(
                _pair_message("cubed distance overflows float64", target_id_str, index)
            ) from cause
    if not np.isfinite(r3).all():
        bad = np.flatnonzero(~np.isfinite(r3))
        index = int(bad[0])
        raise ValueError(
            _pair_message("cubed distance overflows float64", target_id_str, index)
        )
    underflow_r3 = (squared > 0.0) & (r3 < _TINY64)
    if underflow_r3.any():
        index = int(np.flatnonzero(underflow_r3)[0])
        raise ValueError(
            _pair_message(
                "cubed distance is zero or subnormal; kernel cannot be computed",
                target_id_str,
                index,
            )
        )

    # 2. Gravitational kernel K_i = d_i / r_i^3
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        k_vecs = difference / r3[:, np.newaxis]
    if not np.isfinite(k_vecs).all():
        bad = np.flatnonzero(~np.isfinite(k_vecs).all(axis=1))
        index = int(bad[0])
        raise ValueError(
            _pair_message("gravitational kernel is not finite", target_id_str, index)
        )

    # 3. Acceleration delta_a_i = G_SI * w_i * K_i
    weights = view.weights_kg
    if np.iscomplexobj(weights):
        raise ValueError("complex weights are not supported for plot_map")
    weights = np.asarray(weights, dtype=np.float64)

    # Robust multiplication per source element:
    # 1) G_SI * (weights * k_vecs) avoids underflow when weights is tiny but k_vecs is
    #    large (e.g. w=1e-320 kg, r=1e-50 m).
    # 2) For any rows where weights * k_vecs overflows float64, fallback to
    #    (weights * G_SI) * k_vecs (e.g. w=1e210 kg, r=1e-50 m).
    # Row-by-row replacement ensures mixed populations (tiny + huge masses)
    # are both computed accurately without cross-contamination.
    with np.errstate(over="ignore", invalid="ignore"):
        delta_a = G_SI * (weights[:, np.newaxis] * k_vecs)

    non_finite = ~np.isfinite(delta_a).all(axis=1)
    if non_finite.any():
        w_sub = weights[non_finite]
        k_sub = k_vecs[non_finite]
        with np.errstate(over="ignore", invalid="ignore"):
            delta_a_fallback = (w_sub * G_SI)[:, np.newaxis] * k_sub
        delta_a[non_finite] = delta_a_fallback

    if not np.isfinite(delta_a).all():
        bad = np.flatnonzero(~np.isfinite(delta_a).all(axis=1))
        index = int(bad[0])
        raise ValueError(
            _pair_message(
                "acceleration contribution overflows float64", target_id_str, index
            )
        )

    nonzero_w = weights != 0.0
    # Numerically stable 3D Euclidean norm via np.hypot:
    # avoids underflow at e.g. (1e-190)^2 = 1e-380 in naive sum-of-squares
    with np.errstate(over="ignore"):
        da_norm_stable = np.hypot(
            np.hypot(delta_a[:, 0], delta_a[:, 1]),
            delta_a[:, 2],
        )
    if nonzero_w.any():
        underflow_da = nonzero_w & (da_norm_stable < _TINY64)
        if underflow_da.any():
            index = int(np.flatnonzero(underflow_da)[0])
            raise ValueError(
                _pair_message(
                    "acceleration contribution underflows float64", target_id_str, index
                )
            )

    # 4. Destination Frame transformation for vectors
    if target_frame is resolved.computation_frame:
        delta_a_transformed = delta_a
    else:
        delta_a_transformed = resolved.computation_frame.transform_vectors_m(
            delta_a, target_frame
        )

    # 5. Extract values
    if quantity == "delta_a_norm":
        if not np.isfinite(da_norm_stable).all():
            bad = np.flatnonzero(~np.isfinite(da_norm_stable))
            index = int(bad[0])
            raise ValueError(
                _pair_message(
                    "acceleration contribution overflows float64", target_id_str, index
                )
            )
        values = da_norm_stable
        unit_str = "m/s^2"
    elif quantity == "delta_a_x":
        values = delta_a_transformed[:, 0]
        unit_str = "m/s^2"
    elif quantity == "delta_a_y":
        values = delta_a_transformed[:, 1]
        unit_str = "m/s^2"
    elif quantity == "delta_a_z":
        values = delta_a_transformed[:, 2]
        unit_str = "m/s^2"
    elif quantity == "delta_a_projection":
        assert unit_dir is not None
        values = delta_a_transformed @ unit_dir
        unit_str = "m/s^2"
    else:
        raise ValueError(f"unhandled quantity {quantity!r}")

    return _MapData(
        positions_m=positions_transformed,
        values=values,
        delta_a_m_s2=delta_a_transformed,
        quantity=quantity,
        unit_str=unit_str,
        target_id=target_id_str,
        frame=target_frame,
    )
