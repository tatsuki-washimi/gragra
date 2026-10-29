"""Gen1 causal point-force gravity model in an infinite elastic medium."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable

from gragra._arrays import _check_no_bool
from gragra._evaluation_setup import check_pair_precision
from gragra._interactions import Newtonian
from gragra.geometry._frame import Frame, _lengths_m
from gragra.gravity_sources._force_history import ForceHistory
from gragra.results._context import EvaluationContext
from gragra.results._coords import gps_offset_seconds
from gragra.results._field_definition import GravityFieldDefinition
from gragra.results._gen1_definitions import gen1_field_definition
from gragra.results._record import OBSERVABLES
from gragra.results._representation import TimeSeriesRepresentation
from gragra.results._time import TimeSampling
from gragra.targets._high_points import TargetPoints

_OBSERVABLES = ("potential", "acceleration", "gradient")


def _direction(value: object) -> tuple[float, float, float]:
    _check_no_bool(value, "direction")
    raw = np.asarray(value)
    if raw.shape != (3,) or raw.dtype.kind not in "iuf":
        raise ValueError("direction must be a real three-vector")
    vector = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("direction must be finite")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError("direction must be nonzero")
    scaled = vector / scale
    unit = scaled / np.linalg.norm(scaled)
    return tuple(float(value) for value in unit)


def _vp_m_s(value: object) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError("vp must be an Astropy speed Quantity")
    _check_no_bool(value, "vp")
    raw = np.asarray(value.to_value(u.m / u.s))
    if raw.shape != () or np.iscomplexobj(raw):
        raise ValueError("vp must be a real scalar")
    speed = float(raw)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("vp must be positive and finite")
    return speed


@dataclass(frozen=True, eq=False, init=False)
class PointForceModelSpec:
    """Validated immutable source, material and time-axis inputs."""

    position_m: tuple[float, float, float]
    direction: tuple[float, float, float]
    history: ForceHistory
    frame: Frame
    vp_m_s: float
    sampling: TimeSampling
    representation: TimeSeriesRepresentation
    field_definition: GravityFieldDefinition
    interaction: Newtonian
    g_convention: str
    _parameters: QTable
    _formulas: pd.DataFrame
    _references: pd.DataFrame

    def __init__(
        self,
        *,
        position: u.Quantity,
        direction: object,
        history: ForceHistory,
        frame: Frame,
        vp: u.Quantity,
        sampling: TimeSampling,
    ) -> None:
        position_m = _lengths_m("position", position, (3,))
        if not isinstance(history, ForceHistory):
            raise TypeError("history must be a ForceHistory")
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if not isinstance(sampling, TimeSampling):
            raise TypeError("sampling must be a TimeSampling")
        object.__setattr__(
            self, "position_m", tuple(float(value) for value in position_m)
        )
        object.__setattr__(self, "direction", _direction(direction))
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "vp_m_s", _vp_m_s(vp))
        object.__setattr__(self, "sampling", sampling)
        object.__setattr__(self, "representation", TimeSeriesRepresentation())
        object.__setattr__(
            self, "field_definition", gen1_field_definition("point_force")
        )
        object.__setattr__(self, "interaction", Newtonian())
        object.__setattr__(self, "g_convention", "SI_G_applied_once")
        object.__setattr__(self, "_parameters", self._make_parameters())
        object.__setattr__(self, "_formulas", self._make_formulas())
        object.__setattr__(self, "_references", self._make_references())

    @property
    def parameters(self) -> QTable:
        return self._parameters

    def _make_parameters(self) -> QTable:
        parameters = QTable(
            {
                "position_x": [self.position_m[0] * u.m],
                "position_y": [self.position_m[1] * u.m],
                "position_z": [self.position_m[2] * u.m],
                "direction_x": [self.direction[0]],
                "direction_y": [self.direction[1]],
                "direction_z": [self.direction[2]],
                "vp": [self.vp_m_s * u.m / u.s],
                "force_onset_gps_seconds": [self.history.onset_gps[0]],
                "force_onset_gps_nanoseconds": [self.history.onset_gps[1]],
                "sampling_epoch_gps_seconds": [self.sampling.epoch_gps[0]],
                "sampling_epoch_gps_nanoseconds": [self.sampling.epoch_gps[1]],
                "dt": [self.sampling.dt_seconds * u.s],
                "sample_count": [self.sampling.count],
                "frame": [self.frame.name],
                "after": [self.history.after],
            }
        )
        parameters["force_history_knots"] = (
            self.history.knots.to_value(u.s)[None, :] * u.s
        )
        parameters["force_history_coefficients"] = self.history.coefficients[None, :, :]
        parameters[
            "force_history_knots"
        ].description = (
            "Elapsed interval knots measured from the force-history GPS onset."
        )
        parameters["force_history_coefficients"].description = (
            "Per-interval coefficients in ascending powers of normalized local u; "
            "interval rows follow the knots and trailing zeros pad lower degrees."
        )
        return parameters

    @property
    def formulas(self) -> pd.DataFrame:
        return self._formulas

    def _make_formulas(self) -> pd.DataFrame:
        expressions = {
            "potential": "Phi = -G mu J / r^2",
            "acceleration": "a = G[(J/r^3)e + (f/(vp^2 r)-3J/r^3)mu n]",
            "gradient": "Gamma = G[C(e outer n+n outer e+mu I)+mu D(n outer n)]",
        }
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expressions[name],
                    "convention": (
                        "causal C1 signed history; G-free native values; G applied once"
                    ),
                }
                for name in OBSERVABLES
            ]
        )

    @property
    def references(self) -> pd.DataFrame:
        return self._references

    def _make_references(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source": "docs/design/gen1/formula-ledger.md",
                    "detail": "Gen1 point-force Green function and causal delay moment",
                }
            ]
        )


@dataclass(frozen=True, eq=False)
class PointForceModel:
    """Causal point force whose outputs use the supplied uniform GPS sampling."""

    spec: PointForceModelSpec

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = OBSERVABLES,
        output_frame: Frame | None = None,
    ):
        """Fix target coordinates and return the shared analytic evaluator."""
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        if not len(targets):
            raise ValueError("targets must be nonempty")
        computation_frame = self.spec.frame
        if output_frame is None:
            output_frame = targets.frame
        if not isinstance(output_frame, Frame):
            raise TypeError("output_frame must be a Frame")
        targets.frame._path_to(computation_frame)
        computation_frame._path_to(output_frame)
        traces = tuple(
            targets.frame.trace_point_m(point, computation_frame)
            for point in targets.positions_m
        )
        positions = np.stack([trace.point_m for trace in traces])
        context = EvaluationContext(
            model_spec=self.spec,
            targets=targets,
            frame=output_frame,
            observables=observables,
            representation=self.spec.representation,
            time_sampling=self.spec.sampling,
            g_convention=self.spec.g_convention,
        )
        from gragra._analytic_evaluation import AnalyticEvaluation, NativeEvaluation

        def evaluate_native(points: np.ndarray, names: tuple[str, ...]):
            return NativeEvaluation(
                _evaluate_point_force_native(self.spec, points, names)
            )

        def validate_targets(points: np.ndarray) -> None:
            if points.shape != positions.shape:
                raise ValueError("target positions changed after construction")
            for index, trace in enumerate(traces):
                check_pair_precision(
                    self.spec.position_m,
                    trace,
                    target_id=targets.ids[index],
                    source_index=0,
                    observables=context.observables,
                )

        return AnalyticEvaluation(
            context=context,
            computation_frame=computation_frame,
            target_positions_m=positions,
            evaluate_native=evaluate_native,
            validate_targets=validate_targets,
        )


def _requested_observables(observables: object) -> tuple[str, ...]:
    if isinstance(observables, (str, bytes)):
        raise TypeError("observables must be an iterable of observable names")
    try:
        names = tuple(observables)
    except TypeError as error:
        raise TypeError(
            "observables must be an iterable of observable names"
        ) from error
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in _OBSERVABLES for name in names)
    ):
        raise ValueError("observables must be a nonempty unique known subset")
    return names


def _evaluate_point_force_native(
    spec: PointForceModelSpec,
    target_positions_m: object,
    observables: object,
) -> dict[str, np.ndarray]:
    """Evaluate G-free observables as target/component/sample arrays."""
    if not isinstance(spec, PointForceModelSpec):
        raise TypeError("spec must be a PointForceModelSpec")
    names = _requested_observables(observables)
    points = np.asarray(target_positions_m)
    if points.dtype.kind not in "iuf" or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("target_positions_m must be a real (N, 3) table")
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        raise ValueError("target_positions_m must be nonempty")
    if not np.isfinite(points).all():
        raise ValueError("target_positions_m must be finite")

    source = np.asarray(spec.position_m)
    direction = np.asarray(spec.direction)
    with np.errstate(over="ignore", invalid="ignore"):
        vectors = points - source
        distances = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.isfinite(distances).all():
        raise ValueError("target-source separation must be finite")
    if np.any(distances == 0.0):
        index = int(np.flatnonzero(distances == 0.0)[0])
        raise ValueError(f"point force source collision at target index {index}")
    normals = vectors / distances[:, None]
    mu = normals @ direction

    sampling = spec.sampling
    start_elapsed = gps_offset_seconds(sampling.epoch_gps, spec.history.onset_gps)
    sample_elapsed = (
        start_elapsed
        + np.arange(sampling.count, dtype=np.float64) * sampling.dt_seconds
    )
    phi = (
        np.empty((len(points), sampling.count), dtype=np.float64)
        if "potential" in names
        else None
    )
    acceleration = (
        np.empty((len(points), 3, sampling.count), dtype=np.float64)
        if "acceleration" in names
        else None
    )
    gradient = (
        np.empty((len(points), 3, 3, sampling.count), dtype=np.float64)
        if "gradient" in names
        else None
    )
    speed = spec.vp_m_s
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_speed = np.divide(1.0, speed)
        inverse_speed2 = inverse_speed * inverse_speed
        inverse_speed3 = inverse_speed2 * inverse_speed
    if not np.isfinite([inverse_speed, inverse_speed2, inverse_speed3]).all():
        raise ValueError("vp is too small for finite point-force coefficients")

    for target_index, (distance, normal, cosine) in enumerate(
        zip(distances, normals, mu, strict=True)
    ):
        delay = float(distance / speed)
        if not np.isfinite(delay) or delay <= 0.0:
            raise ValueError(
                f"point-force delay is non-finite at target index {target_index}"
            )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            inverse_r = np.divide(1.0, distance)
            inverse_r2 = inverse_r * inverse_r
            inverse_r3 = inverse_r2 * inverse_r
            inverse_r4 = inverse_r2 * inverse_r2
        if not np.isfinite([inverse_r, inverse_r2, inverse_r3, inverse_r4]).all():
            raise ValueError(
                f"target-source separation is too small at target index {target_index}"
            )
        for sample_index, elapsed in enumerate(sample_elapsed):
            force_time = float(elapsed - delay)
            moment = spec.history._delay_moment_at_elapsed_seconds(
                float(elapsed), delay
            )

            if phi is not None:
                phi[target_index, sample_index] = -cosine * moment * inverse_r2
            if acceleration is not None or gradient is not None:
                force = spec.history._value_at_elapsed_seconds(force_time)
                if acceleration is not None:
                    acceleration[target_index, :, sample_index] = (
                        moment * inverse_r3 * direction
                        + (
                            force * inverse_speed2 * inverse_r
                            - 3.0 * moment * inverse_r3
                        )
                        * cosine
                        * normal
                    )
            if gradient is not None:
                force_derivative = spec.history._derivative_at_elapsed_seconds(
                    force_time
                )
                c_value = (
                    force * inverse_speed2 * inverse_r2 - 3.0 * moment * inverse_r4
                )
                d_value = (
                    15.0 * moment * inverse_r4
                    - 6.0 * force * inverse_speed2 * inverse_r2
                    - force_derivative * inverse_speed3 * inverse_r
                )
                gradient[target_index, :, :, sample_index] = c_value * (
                    np.outer(direction, normal)
                    + np.outer(normal, direction)
                    + cosine * np.eye(3)
                ) + cosine * d_value * np.outer(normal, normal)

    result = {}
    if phi is not None:
        result["potential"] = phi[:, None, :]
    if acceleration is not None:
        result["acceleration"] = acceleration
    if gradient is not None:
        result["gradient"] = gradient.reshape(len(points), 9, sampling.count)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ArithmeticError("point-force evaluation produced non-finite values")
    return result


def point_force(
    position: u.Quantity,
    direction: object,
    history: ForceHistory,
    *,
    frame: Frame,
    vp: u.Quantity,
    sampling: TimeSampling,
) -> PointForceModel:
    """Create a causal point-force model in a homogeneous elastic medium."""
    return PointForceModel(
        PointForceModelSpec(
            position=position,
            direction=direction,
            history=history,
            frame=frame,
            vp=vp,
            sampling=sampling,
        )
    )


__all__ = ["point_force"]
