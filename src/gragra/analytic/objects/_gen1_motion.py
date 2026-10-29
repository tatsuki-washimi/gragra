"""Approved Gen1 point-mass motion models and native Newtonian fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable

from gragra._analytic_evaluation import AnalyticEvaluation, NativeEvaluation
from gragra._arrays import _check_no_bool
from gragra._interactions import Newtonian
from gragra.geometry._frame import Frame, _lengths_m
from gragra.gravity_sources.weighted_points import PointMass, PointMassCloud
from gragra.results._context import EvaluationContext
from gragra.results._coords import gps_offset_seconds, normalize_gps
from gragra.results._field_definition import GravityFieldDefinition
from gragra.results._gen1_definitions import gen1_field_definition
from gragra.results._record import OBSERVABLES
from gragra.results._representation import TimeSeriesRepresentation
from gragra.results._spectral import PhasorRepresentation
from gragra.results._time import TimeSampling
from gragra.targets._high_points import TargetPoints

MotionKind = Literal["constant_speed", "oscillation", "rotation"]
_OBSERVABLES = OBSERVABLES


def _quantity_array(
    name: str,
    value: object,
    unit: u.UnitBase,
    shape: tuple[int, ...],
    *,
    allow_complex: bool = False,
) -> np.ndarray:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    _check_no_bool(value, name)
    raw = np.asarray(value.to_value(unit))
    allowed_kinds = "iufc" if allow_complex else "iuf"
    if raw.shape != shape or raw.dtype.kind not in allowed_kinds:
        qualifier = "real or complex" if allow_complex else "real"
        raise ValueError(f"{name} must be a {qualifier} quantity with shape {shape}")
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    result = np.asarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _real_quantity_scalar(
    name: str,
    value: object,
    unit: u.UnitBase,
    *,
    positive: bool = False,
) -> float:
    raw = _quantity_array(name, value, unit, ())
    scalar = float(raw)
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _angle_radians(name: str, value: object) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an angular Quantity")
    if value.unit.physical_type != "angle":
        raise u.UnitConversionError(f"{name} must have angular units")
    return _real_quantity_scalar(name, value, u.rad)


def _axis_unit(value: object) -> tuple[float, float, float]:
    _check_no_bool(value, "axis")
    if isinstance(value, u.Quantity):
        if value.unit.physical_type != "dimensionless":
            raise u.UnitConversionError("axis must have dimensionless units")
        raw = np.asarray(value.to_value(u.dimensionless_unscaled))
    else:
        raw = np.asarray(value)
    if raw.shape != (3,) or raw.dtype.kind not in "iuf":
        raise ValueError("axis must be a dimensionless real three-vector")
    vector = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("axis must be finite")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError("axis must be nonzero")
    scaled = vector / scale
    norm = float(np.hypot(np.hypot(scaled[0], scaled[1]), scaled[2]))
    unit = scaled / norm
    return tuple(float(component) for component in unit)


def _snapshot_source(
    source: object, *, allow_point_mass: bool
) -> tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]]:
    if isinstance(source, PointMass):
        if not allow_point_mass:
            raise TypeError("cloud must be a PointMassCloud")
        masses = np.asarray([source.mass_kg], dtype=np.float64)
        positions = np.asarray(source.position_m, dtype=np.float64).reshape(1, 3)
    elif isinstance(source, PointMassCloud):
        masses = np.asarray(source.masses_kg, dtype=np.float64)
        positions = np.asarray(source.positions_m, dtype=np.float64)
    else:
        accepted = (
            "PointMass or PointMassCloud" if allow_point_mass else "PointMassCloud"
        )
        raise TypeError(f"source must be a {accepted}")
    if masses.ndim != 1 or positions.shape != (len(masses), 3):
        raise ValueError("source masses and positions have inconsistent shapes")
    if not np.isfinite(masses).all() or np.any(masses <= 0.0):
        raise ValueError("source masses must be positive and finite")
    if not np.isfinite(positions).all():
        raise ValueError("source positions must be finite")
    return (
        tuple(float(mass) for mass in masses),
        tuple(tuple(float(x) for x in position) for position in positions),
    )


def _observables(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("observables must be an iterable of observable names")
    try:
        names = tuple(value)
    except TypeError as exc:
        raise TypeError("observables must be an iterable of observable names") from exc
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in _OBSERVABLES for name in names)
    ):
        raise ValueError("observables must be a nonempty unique known subset")
    return names


@dataclass(frozen=True, eq=False)
class _MotionSpec:
    """Immutable SI snapshot shared by the three approved motion families."""

    kind: MotionKind
    masses_kg: tuple[float, ...]
    positions_m: tuple[tuple[float, float, float], ...]
    frame: Frame
    sampling: TimeSampling | None
    motion_epoch_gps: tuple[int, int] | None = None
    velocity_m_s: tuple[float, float, float] | None = None
    displacement_m: tuple[float | complex, float | complex, float | complex] | None = (
        None
    )
    frequency_hz: float | None = None
    reference_time_gps: tuple[int, int] | None = None
    axis: tuple[float, float, float] | None = None
    angular_velocity_rad_s: float | None = None
    origin_m: tuple[float, float, float] | None = None
    angle0_rad: float | None = None
    angle_epoch_gps: tuple[int, int] | None = None
    representation: TimeSeriesRepresentation | PhasorRepresentation = field(
        default_factory=TimeSeriesRepresentation
    )
    field_definition: GravityFieldDefinition = field(init=False)
    interaction: Newtonian = field(default_factory=Newtonian)
    g_convention: str = "SI_G_applied_once"
    _parameters: QTable = field(init=False, repr=False)
    _formulas: pd.DataFrame = field(init=False, repr=False)
    _references: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        definition_model = "oscillation" if self.kind == "oscillation" else self.kind
        object.__setattr__(
            self, "field_definition", gen1_field_definition(definition_model)
        )
        object.__setattr__(self, "_parameters", self._make_parameters())
        object.__setattr__(self, "_formulas", self._make_formulas())
        object.__setattr__(self, "_references", self._make_references())

    @property
    def parameters(self) -> QTable:
        return self._parameters

    @property
    def formulas(self) -> pd.DataFrame:
        return self._formulas

    @property
    def references(self) -> pd.DataFrame:
        return self._references

    def _make_parameters(self) -> QTable:
        masses = np.asarray(self.masses_kg, dtype=np.float64)
        positions = np.asarray(self.positions_m, dtype=np.float64).reshape(-1, 3)
        parameters = QTable()
        parameters["source_masses"] = masses[None, :] * u.kg
        parameters["source_positions"] = positions[None, :, :] * u.m
        parameters["frame"] = [self.frame.name]
        if self.kind == "constant_speed":
            assert self.velocity_m_s is not None and self.motion_epoch_gps is not None
            assert self.sampling is not None
            parameters["velocity"] = np.asarray(self.velocity_m_s)[None, :] * u.m / u.s
            _add_epoch(parameters, "motion_epoch", self.motion_epoch_gps)
            _add_sampling(parameters, self.sampling)
        elif self.kind == "oscillation":
            assert self.displacement_m is not None and self.frequency_hz is not None
            assert self.reference_time_gps is not None
            parameters["displacement"] = np.asarray(self.displacement_m)[None, :] * u.m
            parameters["frequency"] = [self.frequency_hz * u.Hz]
            _add_epoch(parameters, "reference_time", self.reference_time_gps)
        else:
            assert self.axis is not None and self.angular_velocity_rad_s is not None
            assert self.origin_m is not None and self.angle0_rad is not None
            assert self.angle_epoch_gps is not None and self.sampling is not None
            parameters["axis"] = np.asarray(self.axis)[None, :]
            parameters["angular_velocity"] = [self.angular_velocity_rad_s * u.rad / u.s]
            parameters["origin"] = np.asarray(self.origin_m)[None, :] * u.m
            parameters["angle0"] = [self.angle0_rad * u.rad]
            _add_epoch(parameters, "angle_epoch", self.angle_epoch_gps)
            _add_sampling(parameters, self.sampling)
        return parameters

    def _make_formulas(self) -> pd.DataFrame:
        if self.kind == "oscillation":
            expressions = {
                "potential": "delta_Phi = -G sum(m (r dot u) / r^3)",
                "acceleration": "delta_a = G sum(m (u/r^3 - 3(r dot u)r/r^5))",
                "gradient": (
                    "delta_Gamma = G sum(m (-3(u outer r+r outer u+(r dot u)I) "
                    "/r^5 + 15(r dot u)rr/r^7))"
                ),
            }
            convention = (
                "linearized peak phasor; exp(-2 pi i f (t-reference_time)); "
                "no factor 1/2"
            )
        else:
            expressions = {
                "potential": "Phi = -G sum(m/r)",
                "acceleration": "a = -G sum(m r/r^3)",
                "gradient": "Gamma = G sum(m (3 rr/r^5-I/r^3))",
            }
            convention = (
                "exact moving point masses; G-free native values; G applied once"
            )
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expressions[name],
                    "convention": convention,
                }
                for name in OBSERVABLES
            ]
        )

    def _make_references(self) -> pd.DataFrame:
        detail = (
            "Gen1 first-order point-mass displacement field"
            if self.kind == "oscillation"
            else "Gen1 exact constant-speed or fixed-axis rotating point-mass field"
        )
        return pd.DataFrame(
            [{"source": "docs/design/gen1/formula-ledger.md", "detail": detail}]
        )


def _add_epoch(parameters: QTable, prefix: str, epoch: tuple[int, int]) -> None:
    parameters[f"{prefix}_gps_seconds"] = [epoch[0]]
    parameters[f"{prefix}_gps_nanoseconds"] = [epoch[1]]


def _add_sampling(parameters: QTable, sampling: TimeSampling) -> None:
    _add_epoch(parameters, "sampling_epoch", sampling.epoch_gps)
    parameters["dt"] = [sampling.dt_seconds * u.s]
    parameters["sample_count"] = [sampling.count]


def _source_arrays(spec: _MotionSpec) -> tuple[np.ndarray, np.ndarray]:
    masses = np.asarray(spec.masses_kg, dtype=np.float64)
    positions = np.asarray(spec.positions_m, dtype=np.float64).reshape(-1, 3)
    return masses, positions


def _sample_elapsed(
    sampling: TimeSampling, epoch_gps: tuple[int, int], index: int
) -> float:
    elapsed = gps_offset_seconds(sampling.epoch_gps, epoch_gps)
    elapsed += float(index) * sampling.dt_seconds
    if not np.isfinite(elapsed):
        raise ValueError("motion epoch offset must be finite")
    return elapsed


def _positions_at_sample(spec: _MotionSpec, index: int) -> np.ndarray:
    _, reference_positions = _source_arrays(spec)
    sampling = spec.sampling
    if sampling is None:
        raise TypeError("moving point-mass fields require TimeSampling")
    if spec.kind == "constant_speed":
        assert spec.motion_epoch_gps is not None and spec.velocity_m_s is not None
        elapsed = _sample_elapsed(sampling, spec.motion_epoch_gps, index)
        with np.errstate(over="ignore", invalid="ignore"):
            positions = reference_positions + elapsed * np.asarray(spec.velocity_m_s)
    else:
        assert spec.kind == "rotation"
        assert spec.angle_epoch_gps is not None and spec.axis is not None
        assert spec.angular_velocity_rad_s is not None and spec.angle0_rad is not None
        assert spec.origin_m is not None
        elapsed = _sample_elapsed(sampling, spec.angle_epoch_gps, index)
        angle = spec.angle0_rad + spec.angular_velocity_rad_s * elapsed
        if not np.isfinite(angle):
            raise ValueError("rotation angle must remain finite at every sample")
        axis = np.asarray(spec.axis)
        origin = np.asarray(spec.origin_m)
        relative = reference_positions - origin
        cosine = np.cos(angle)
        sine = np.sin(angle)
        with np.errstate(over="ignore", invalid="ignore"):
            rotated = (
                relative * cosine
                + np.cross(axis, relative) * sine
                + np.outer(relative @ axis, axis) * (1.0 - cosine)
            )
            positions = origin + rotated
    if not np.isfinite(positions).all():
        raise ValueError("moved source positions must remain finite at every sample")
    return positions


def _separations(points: np.ndarray, positions: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        relative = points[:, None, :] - positions[None, :, :]
        distances = np.hypot(
            np.hypot(relative[:, :, 0], relative[:, :, 1]), relative[:, :, 2]
        )
    if not np.isfinite(relative).all() or not np.isfinite(distances).all():
        raise ValueError("target-source separation must be finite")
    return relative, distances


def _check_no_collisions(
    points: np.ndarray,
    positions: np.ndarray,
    *,
    sample_index: int | None,
    coordinate_scale_m: float | np.ndarray = 0.0,
    frame_transform_edges: int | np.ndarray = 0,
) -> None:
    _, distances = _separations(points, positions)
    target_scale = np.max(np.abs(points), axis=1)[:, None]
    if len(positions):
        source_scale = np.max(np.abs(positions), axis=1)[None, :]
        scale = np.maximum(target_scale, source_scale)
    else:
        scale = np.empty((len(points), 0), dtype=np.float64)
    extra_scale = np.asarray(coordinate_scale_m, dtype=np.float64)
    if extra_scale.shape == ():
        scale = np.maximum(scale, float(extra_scale))
    elif extra_scale.shape == (len(points),):
        scale = np.maximum(scale, extra_scale[:, None])
    else:
        raise ValueError("collision roundoff scale must be scalar or per-target")
    edge_counts = np.asarray(frame_transform_edges)
    if edge_counts.shape == ():
        edge_counts = np.full(len(points), edge_counts.item())
    if (
        edge_counts.shape != (len(points),)
        or not np.issubdtype(edge_counts.dtype, np.integer)
        or np.any(edge_counts < 0)
    ):
        raise ValueError(
            "frame transform edge counts must be nonnegative per-target integers"
        )
    # Keep the existing coordinate scale and baseline tolerance. Each traversed
    # Frame edge adds rounding opportunities to the target-source separation.
    # The count is per target so unrelated distant targets cannot widen another
    # target's collision envelope, and a same-frame pair keeps the old rule.
    tolerance = (
        4.0
        * np.finfo(np.float64).eps
        * scale
        * (1.0 + edge_counts.astype(np.float64)[:, None])
    )
    collisions = np.argwhere((distances == 0.0) | (distances <= tolerance))
    if len(collisions):
        target_index, source_index = (int(value) for value in collisions[0])
        sample_detail = "" if sample_index is None else f" at sample {sample_index}"
        raise ValueError(
            f"point-mass collision{sample_detail} at target index {target_index}, "
            f"source index {source_index}"
        )


def _sample_operation_roundoff_scale_m(spec: _MotionSpec, index: int) -> float:
    _, initial_positions = _source_arrays(spec)
    if spec.kind == "constant_speed":
        assert spec.motion_epoch_gps is not None and spec.velocity_m_s is not None
        assert spec.sampling is not None
        elapsed = _sample_elapsed(spec.sampling, spec.motion_epoch_gps, index)
        movement = elapsed * np.asarray(spec.velocity_m_s)
        return max(
            float(np.max(np.abs(initial_positions), initial=0.0)),
            float(np.max(np.abs(movement), initial=0.0)),
        )
    assert spec.kind == "rotation" and spec.origin_m is not None
    origin = np.asarray(spec.origin_m)
    return max(
        float(np.max(np.abs(origin), initial=0.0)),
        float(np.max(np.abs(initial_positions - origin), initial=0.0)),
    )


def _validate_motion_targets(
    spec: _MotionSpec,
    points: np.ndarray,
    frame_roundoff_scales_m: np.ndarray,
    frame_transform_edges: np.ndarray,
) -> None:
    frame_scales = np.asarray(frame_roundoff_scales_m, dtype=np.float64)
    if frame_scales.shape != (len(points),) or not np.isfinite(frame_scales).all():
        raise ValueError("frame transform roundoff scales must match targets")
    edge_counts = np.asarray(frame_transform_edges)
    if (
        edge_counts.shape != (len(points),)
        or not np.issubdtype(edge_counts.dtype, np.integer)
        or np.any(edge_counts < 0)
    ):
        raise ValueError("frame transform edge counts must match targets")
    if spec.kind == "oscillation":
        _, positions = _source_arrays(spec)
        _check_no_collisions(
            points,
            positions,
            sample_index=None,
            coordinate_scale_m=frame_scales,
            frame_transform_edges=edge_counts,
        )
        return
    if spec.sampling is None:
        raise TypeError("moving point-mass fields require TimeSampling")
    for index in range(spec.sampling.count):
        positions = _positions_at_sample(spec, index)
        operation_scale_m = _sample_operation_roundoff_scale_m(spec, index)
        if spec.kind == "rotation":
            assert spec.origin_m is not None
            operation_scale_m = max(
                operation_scale_m,
                float(np.max(np.abs(spec.origin_m), initial=0.0)),
            )
        coordinate_scale_m = np.maximum(frame_scales, operation_scale_m)
        _check_no_collisions(
            points,
            positions,
            sample_index=index,
            coordinate_scale_m=coordinate_scale_m,
            frame_transform_edges=edge_counts,
        )


def _requested_point_fields(
    points: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    names: tuple[str, ...],
    *,
    coordinate_scale_m: float = 0.0,
) -> dict[str, np.ndarray]:
    relative, distances = _separations(points, positions)
    _check_no_collisions(
        points,
        positions,
        sample_index=None,
        coordinate_scale_m=coordinate_scale_m,
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_r = 1.0 / distances
        inverse_r2 = inverse_r * inverse_r
        inverse_r3 = inverse_r2 * inverse_r
        normals = relative / distances[:, :, None]
        result: dict[str, np.ndarray] = {}
        if "potential" in names:
            result["potential"] = -np.sum(masses[None, :] * inverse_r, axis=1)
        if "acceleration" in names:
            result["acceleration"] = -np.sum(
                masses[None, :, None] * normals * inverse_r2[:, :, None],
                axis=1,
            )
        if "gradient" in names:
            dyad = normals[:, :, :, None] * normals[:, :, None, :]
            result["gradient"] = np.sum(
                masses[None, :, None, None]
                * (3.0 * dyad - np.eye(3)[None, None, :, :])
                * inverse_r3[:, :, None, None],
                axis=1,
            )
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ArithmeticError("point-mass evaluation produced non-finite values")
    return result


def _evaluate_exact_native(
    spec: _MotionSpec, points: np.ndarray, names: tuple[str, ...]
) -> NativeEvaluation:
    masses, _ = _source_arrays(spec)
    sampling = spec.sampling
    if sampling is None:
        raise TypeError("exact motion requires TimeSampling")
    component_counts = {"potential": 1, "acceleration": 3, "gradient": 9}
    values = {
        name: np.empty(
            (len(points), component_counts[name], sampling.count), dtype=np.float64
        )
        for name in names
    }
    for sample_index in range(sampling.count):
        positions = _positions_at_sample(spec, sample_index)
        coordinate_scale_m = 0.0
        if spec.kind == "rotation":
            assert spec.origin_m is not None
            coordinate_scale_m = float(np.max(np.abs(spec.origin_m)))
        sample_values = _requested_point_fields(
            points,
            positions,
            masses,
            names,
            coordinate_scale_m=coordinate_scale_m,
        )
        for name in names:
            array = sample_values[name]
            if name == "potential":
                values[name][:, :, sample_index] = array[:, None]
            elif name == "gradient":
                values[name][:, :, sample_index] = array.reshape(len(points), 9)
            else:
                values[name][:, :, sample_index] = array
    return NativeEvaluation({name: values[name] for name in names})


def _evaluate_oscillation_native(
    spec: _MotionSpec, points: np.ndarray, names: tuple[str, ...]
) -> NativeEvaluation:
    masses, positions = _source_arrays(spec)
    assert spec.displacement_m is not None
    displacement = np.asarray(spec.displacement_m, dtype=np.complex128)
    relative, distances = _separations(points, positions)
    _check_no_collisions(points, positions, sample_index=None)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_r = 1.0 / distances
        inverse_r2 = inverse_r * inverse_r
        inverse_r3 = inverse_r2 * inverse_r
        inverse_r4 = inverse_r2 * inverse_r2
        inverse_r5 = inverse_r4 * inverse_r
        normals = relative / distances[:, :, None]
        projection = np.einsum("nsi,i->ns", relative, displacement)
        result: dict[str, np.ndarray] = {}
        if "potential" in names:
            result["potential"] = -np.sum(
                masses[None, :] * projection * inverse_r3, axis=1
            )[:, None, None]
        if "acceleration" in names:
            components = (
                displacement[None, None, :] * inverse_r3[:, :, None]
                - 3.0 * projection[:, :, None] * normals * inverse_r4[:, :, None]
            )
            result["acceleration"] = np.sum(masses[None, :, None] * components, axis=1)[
                :, :, None
            ]
        if "gradient" in names:
            displacement_normal = (
                displacement[None, None, :, None] * normals[:, :, None, :]
                + normals[:, :, :, None] * displacement[None, None, None, :]
            )
            identity = np.eye(3)[None, None, :, :]
            normal_dyad = normals[:, :, :, None] * normals[:, :, None, :]
            components = -3.0 * (
                displacement_normal * inverse_r4[:, :, None, None]
                + projection[:, :, None, None] * identity * inverse_r5[:, :, None, None]
            ) + 15.0 * (
                projection[:, :, None, None]
                * normal_dyad
                * inverse_r5[:, :, None, None]
            )
            gradient = np.sum(masses[None, :, None, None] * components, axis=1)
            result["gradient"] = gradient.reshape(len(points), 9, 1)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ArithmeticError("oscillation evaluation produced non-finite values")
    return NativeEvaluation(result)


def _at(
    spec: _MotionSpec,
    targets: TargetPoints,
    *,
    observables: tuple[str, ...],
    output_frame: Frame | None,
) -> AnalyticEvaluation:
    if not isinstance(targets, TargetPoints):
        raise TypeError("targets must be TargetPoints")
    if not len(targets):
        raise ValueError("targets must be nonempty")
    if output_frame is None:
        output_frame = targets.frame
    if not isinstance(output_frame, Frame):
        raise TypeError("output_frame must be a Frame")
    computation_frame = spec.frame
    targets.frame._path_to(computation_frame)
    computation_frame._path_to(output_frame)
    traces = tuple(
        targets.frame.trace_point_m(point, computation_frame)
        for point in targets.positions_m
    )
    positions = np.stack([trace.point_m for trace in traces])
    frame_roundoff_scales_m = np.asarray(
        [trace.scale_m for trace in traces], dtype=np.float64
    )
    frame_transform_edges = np.asarray(
        [trace.edge_count for trace in traces], dtype=np.int64
    )
    names = _observables(observables)
    context_arguments: dict[str, object] = {
        "model_spec": spec,
        "targets": targets,
        "frame": output_frame,
        "observables": names,
        "representation": spec.representation,
        "g_convention": spec.g_convention,
    }
    if spec.kind == "oscillation":
        assert spec.frequency_hz is not None and spec.reference_time_gps is not None
        context_arguments["frequencies"] = np.array([spec.frequency_hz]) * u.Hz
        context_arguments["reference_time"] = spec.reference_time_gps
    else:
        assert spec.sampling is not None
        context_arguments["time_sampling"] = spec.sampling
    context = EvaluationContext(**context_arguments)

    def evaluate_native(points: np.ndarray, requested: tuple[str, ...]):
        if spec.kind == "oscillation":
            return _evaluate_oscillation_native(spec, points, requested)
        return _evaluate_exact_native(spec, points, requested)

    return AnalyticEvaluation(
        context=context,
        computation_frame=computation_frame,
        target_positions_m=positions,
        evaluate_native=evaluate_native,
        validate_targets=lambda points: _validate_motion_targets(
            spec, points, frame_roundoff_scales_m, frame_transform_edges
        ),
    )


@dataclass(frozen=True, eq=False)
class _MotionModel:
    spec: _MotionSpec

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = OBSERVABLES,
        output_frame: Frame | None = None,
    ) -> AnalyticEvaluation:
        return _at(
            self.spec,
            targets,
            observables=observables,
            output_frame=output_frame,
        )


def constant_velocity(
    point_mass: PointMass,
    velocity: u.Quantity,
    *,
    frame: Frame,
    motion_epoch: object,
    sampling: TimeSampling,
) -> _MotionModel:
    """Create an exact constant-velocity point-mass model."""
    if not isinstance(point_mass, PointMass):
        raise TypeError("point_mass must be a PointMass")
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if not isinstance(sampling, TimeSampling):
        raise TypeError("sampling must be a TimeSampling")
    masses, positions = _snapshot_source(point_mass, allow_point_mass=True)
    velocity_m_s = _quantity_array("velocity", velocity, u.m / u.s, (3,))
    return _MotionModel(
        _MotionSpec(
            kind="constant_speed",
            masses_kg=masses,
            positions_m=positions,
            frame=frame,
            sampling=sampling,
            motion_epoch_gps=normalize_gps(motion_epoch),
            velocity_m_s=tuple(float(value) for value in velocity_m_s),
        )
    )


def translational_oscillation(
    point_mass_or_cloud: PointMass | PointMassCloud,
    displacement: u.Quantity,
    *,
    frame: Frame,
    frequency: u.Quantity,
    reference_time: object,
) -> _MotionModel:
    """Create a first-order uniform-translation phasor model."""
    if not isinstance(point_mass_or_cloud, (PointMass, PointMassCloud)):
        raise TypeError("point_mass_or_cloud must be a PointMass or PointMassCloud")
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    masses, positions = _snapshot_source(point_mass_or_cloud, allow_point_mass=True)
    displacement_m = _quantity_array(
        "displacement", displacement, u.m, (3,), allow_complex=True
    )
    frequency_hz = _real_quantity_scalar("frequency", frequency, u.Hz, positive=True)
    reference_time_gps = normalize_gps(reference_time)
    return _MotionModel(
        _MotionSpec(
            kind="oscillation",
            masses_kg=masses,
            positions_m=positions,
            frame=frame,
            sampling=None,
            displacement_m=tuple(
                complex(value) if np.iscomplexobj(displacement_m) else float(value)
                for value in displacement_m
            ),
            frequency_hz=frequency_hz,
            reference_time_gps=reference_time_gps,
            representation=PhasorRepresentation(),
        )
    )


def fixed_axis_rotation(
    cloud: PointMassCloud,
    axis: object,
    angular_velocity: u.Quantity,
    *,
    frame: Frame,
    origin: u.Quantity,
    angle0: u.Quantity,
    angle_epoch: object,
    sampling: TimeSampling,
) -> _MotionModel:
    """Create an exact fixed-axis, constant-angular-velocity cloud model."""
    if not isinstance(cloud, PointMassCloud):
        raise TypeError("cloud must be a PointMassCloud")
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")
    if not isinstance(sampling, TimeSampling):
        raise TypeError("sampling must be a TimeSampling")
    masses, positions = _snapshot_source(cloud, allow_point_mass=False)
    axis_unit = _axis_unit(axis)
    angular_velocity_rad_s = _real_quantity_scalar(
        "angular_velocity", angular_velocity, u.rad / u.s
    )
    origin_m = _lengths_m("origin", origin, (3,))
    angle0_rad = _angle_radians("angle0", angle0)
    angle_epoch_gps = normalize_gps(angle_epoch)
    return _MotionModel(
        _MotionSpec(
            kind="rotation",
            masses_kg=masses,
            positions_m=positions,
            frame=frame,
            sampling=sampling,
            axis=axis_unit,
            angular_velocity_rad_s=angular_velocity_rad_s,
            origin_m=tuple(float(value) for value in origin_m),
            angle0_rad=angle0_rad,
            angle_epoch_gps=angle_epoch_gps,
        )
    )


__all__ = [
    "constant_velocity",
    "translational_oscillation",
    "fixed_axis_rotation",
]
