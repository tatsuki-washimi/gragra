"""Owned Rayleigh inputs and interpretation shared across high-level lanes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeries
from gwpy.time import LIGOTimeGPS

from gragra._evaluation_setup import OBSERVABLES, density_kg_m3
from gragra._interactions import Newtonian
from gragra.constants import G_SI
from gragra.geometry._frame import Frame, PointTrace, _lengths_m
from gragra.geometry._halfspace import HalfSpace
from gragra.results._context import EvaluationContext
from gragra.results._coords import _frequencies_hz, normalize_gps
from gragra.results._coords import gps_offset_seconds as gps_offset_seconds
from gragra.results._field_definition import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
    GravityFieldDefinition,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
)
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints
from gragra.waves.rayleigh import rayleigh_speed


def _scalar_si(name: str, value: u.Quantity, unit: u.UnitBase) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an astropy Quantity")
    raw = np.asarray(value.to_value(unit))
    if raw.shape != () or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _medium_inputs(
    geometry: HalfSpace, density: u.Quantity, vp: u.Quantity, vs: u.Quantity
) -> tuple[float, float, float]:
    if not isinstance(geometry, HalfSpace):
        raise TypeError("geometry must be a HalfSpace")
    rho = density_kg_m3(density)
    if rho < 0:
        raise ValueError("density must be nonnegative")
    vp_si = _scalar_si("vp", vp, u.m / u.s)
    vs_si = _scalar_si("vs", vs, u.m / u.s)
    if not (0 < vs_si < vp_si and vp_si**2 > (4 / 3) * vs_si**2):
        raise ValueError("speeds must satisfy 0 < vs < vp and vp² > 4vs²/3")
    return rho, vp_si, vs_si


def _phase_origin_m(value: u.Quantity) -> np.ndarray:
    origin = _lengths_m("phase_origin", value, (3,))
    if origin[2] != 0:
        raise ValueError("phase_origin must be on the local z=0 surface")
    return origin


def _direction_and_speed(
    direction: object, vp_si: float, vs_si: float, frequencies: np.ndarray
) -> tuple[np.ndarray, float]:
    raw_direction = np.asarray(direction)
    if np.iscomplexobj(raw_direction):
        raise ValueError("direction must be real")
    direction_array = np.asarray(raw_direction, dtype=np.float64)
    if direction_array.shape != (3,) or not np.isfinite(direction_array).all():
        raise ValueError("direction must be a finite three-vector")
    scale = float(np.max(np.abs(direction_array)))
    if scale == 0:
        raise ValueError("direction must be nonzero")
    scaled = direction_array / scale
    magnitude = float(np.linalg.norm(scaled))
    if abs(scaled[2]) > 1e-12 * magnitude:
        raise ValueError("direction must be horizontal within 1e-12 relative tolerance")
    direction_unit = scaled / magnitude
    direction_unit[2] = 0.0
    speed = rayleigh_speed(vp_si, vs_si)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        k = 2 * np.pi * frequencies / speed
    if not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError("derived Rayleigh wavenumbers must be finite and positive")
    return direction_unit, float(speed)


def _rayleigh_output_definition() -> GravityFieldDefinition:
    reference = ReferenceDefinition(
        ReferenceStatus.DEFINED,
        ReferenceTemporalKind.STATE,
        "zero_displacement_halfspace",
    )
    change = ChangeDefinition(
        ChangeForm.LINEARIZED_RESPONSE,
        reference,
        changed_input="displacement",
        order=1,
        fixed_conditions=("geometry", "density", "vp", "vs", "free_surface_boundary"),
    )
    return GravityFieldDefinition.field_change(
        source_kind="halfspace_perturbation",
        contribution=Contribution("total", ("bulk", "surface")),
        potential_reference=PotentialReference.VERTICAL_FAR_FIELD_DECAY_POSITIVE_KH,
        change=change,
    )


@dataclass(frozen=True, eq=False, init=False)
class RayleighModelSpec:
    geometry: HalfSpace
    density_kg_m3: float
    vp_m_s: float
    vs_m_s: float
    frequencies_hz: tuple[float, ...]
    _amplitude_bytes: bytes
    displacement_name: str | None
    displacement_channel: str | None
    displacement_epoch_gps: tuple[int, int] | None
    displacement_unit: u.UnitBase
    displacement_input_unit: u.UnitBase
    representation: PhasorRepresentation
    reference_time_gps: tuple[int, int]
    _phase_origin_tuple_m: tuple[float, float, float]
    _direction_tuple: tuple[float, float, float]
    rayleigh_speed_m_s: float
    field_definition: GravityFieldDefinition
    interaction: Newtonian
    g_convention: str
    law_name: str
    law_parameters: Mapping[str, object]
    gravitational_constant_si: float

    def __init__(
        self,
        geometry: HalfSpace,
        density: u.Quantity,
        vp: u.Quantity,
        vs: u.Quantity,
        vertical_displacement: FrequencySeries,
        representation: PhasorRepresentation,
        reference_time: object,
        phase_origin: u.Quantity,
        direction: object,
    ) -> None:
        rho, vp_si, vs_si = _medium_inputs(geometry, density, vp, vs)
        if not isinstance(vertical_displacement, FrequencySeries):
            raise TypeError("vertical_displacement must be a FrequencySeries")
        frequencies = _frequencies_hz(vertical_displacement.frequencies)
        amplitudes = np.asarray(
            vertical_displacement.to_value(u.m), dtype=np.complex128
        )
        if amplitudes.shape != frequencies.shape or not np.isfinite(amplitudes).all():
            raise ValueError(
                "vertical_displacement must have finite amplitudes matching frequencies"
            )
        if not isinstance(representation, PhasorRepresentation):
            raise TypeError("representation must be a PhasorRepresentation")
        gps = normalize_gps(reference_time)
        origin = _phase_origin_m(phase_origin)
        direction_unit, speed = _direction_and_speed(
            direction, vp_si, vs_si, frequencies
        )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "density_kg_m3", rho)
        object.__setattr__(self, "vp_m_s", vp_si)
        object.__setattr__(self, "vs_m_s", vs_si)
        object.__setattr__(self, "frequencies_hz", tuple(float(x) for x in frequencies))
        object.__setattr__(self, "_amplitude_bytes", amplitudes.copy().tobytes())
        object.__setattr__(self, "displacement_name", vertical_displacement.name)
        object.__setattr__(
            self,
            "displacement_channel",
            None
            if vertical_displacement.channel is None
            else str(vertical_displacement.channel),
        )
        object.__setattr__(
            self,
            "displacement_epoch_gps",
            None
            if vertical_displacement.epoch is None
            else normalize_gps(vertical_displacement.epoch),
        )
        object.__setattr__(self, "displacement_unit", u.m)
        object.__setattr__(self, "displacement_input_unit", vertical_displacement.unit)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "reference_time_gps", gps)
        object.__setattr__(
            self, "_phase_origin_tuple_m", tuple(float(x) for x in origin)
        )
        object.__setattr__(
            self, "_direction_tuple", tuple(float(x) for x in direction_unit)
        )
        object.__setattr__(self, "rayleigh_speed_m_s", float(speed))
        object.__setattr__(
            self,
            "field_definition",
            _rayleigh_output_definition(),
        )
        object.__setattr__(self, "g_convention", "SI_G_applied_once")
        object.__setattr__(self, "interaction", Newtonian())
        object.__setattr__(self, "law_name", "newton")
        object.__setattr__(self, "law_parameters", MappingProxyType({}))
        object.__setattr__(self, "gravitational_constant_si", G_SI)

    @property
    def density(self) -> u.Quantity:
        return self.density_kg_m3 * u.kg / u.m**3

    @property
    def vertical_displacement_m(self) -> np.ndarray:
        return np.frombuffer(self._amplitude_bytes, dtype=np.complex128)

    @property
    def vertical_displacement(self) -> FrequencySeries:
        epoch = (
            None
            if self.displacement_epoch_gps is None
            else LIGOTimeGPS(*self.displacement_epoch_gps)
        )
        return FrequencySeries(
            self.vertical_displacement_m.copy(),
            frequencies=np.asarray(self.frequencies_hz) * u.Hz,
            unit=self.displacement_unit,
            name=self.displacement_name,
            channel=self.displacement_channel,
            epoch=epoch,
        )

    @property
    def phase_origin_m(self) -> np.ndarray:
        return np.asarray(self._phase_origin_tuple_m)

    @property
    def direction(self) -> np.ndarray:
        return np.asarray(self._direction_tuple)

    @classmethod
    def _from_response(
        cls,
        response_spec: RayleighResponseSpec,
        vertical_displacement: FrequencySeries,
        representation: PhasorRepresentation,
        reference_time: object,
    ) -> RayleighModelSpec:
        """Bind actual phasor input to an already validated response specification."""
        if type(response_spec) is not RayleighResponseSpec:
            raise TypeError("response_spec must be exactly a RayleighResponseSpec")
        if not isinstance(vertical_displacement, FrequencySeries):
            raise TypeError("vertical_displacement must be a FrequencySeries")
        frequencies = _frequencies_hz(vertical_displacement.frequencies)
        if tuple(float(x) for x in frequencies) != response_spec.frequencies_hz:
            raise ValueError(
                "vertical_displacement frequencies must match response frequencies"
            )
        amplitudes = np.asarray(
            vertical_displacement.to_value(u.m), dtype=np.complex128
        )
        if amplitudes.shape != frequencies.shape or not np.isfinite(amplitudes).all():
            raise ValueError(
                "vertical_displacement must have finite amplitudes matching frequencies"
            )
        if not isinstance(representation, PhasorRepresentation):
            raise TypeError("representation must be a PhasorRepresentation")
        gps = normalize_gps(reference_time)
        spec = object.__new__(cls)
        for name in (
            "geometry",
            "density_kg_m3",
            "vp_m_s",
            "vs_m_s",
            "frequencies_hz",
            "_phase_origin_tuple_m",
            "_direction_tuple",
            "rayleigh_speed_m_s",
            "g_convention",
            "interaction",
            "law_name",
            "law_parameters",
            "gravitational_constant_si",
        ):
            object.__setattr__(spec, name, getattr(response_spec, name))
        object.__setattr__(spec, "_amplitude_bytes", amplitudes.copy().tobytes())
        object.__setattr__(spec, "displacement_name", vertical_displacement.name)
        object.__setattr__(
            spec,
            "displacement_channel",
            None
            if vertical_displacement.channel is None
            else str(vertical_displacement.channel),
        )
        object.__setattr__(
            spec,
            "displacement_epoch_gps",
            None
            if vertical_displacement.epoch is None
            else normalize_gps(vertical_displacement.epoch),
        )
        object.__setattr__(spec, "displacement_unit", u.m)
        object.__setattr__(spec, "displacement_input_unit", vertical_displacement.unit)
        object.__setattr__(spec, "representation", representation)
        object.__setattr__(spec, "reference_time_gps", gps)
        object.__setattr__(spec, "field_definition", response_spec.output_definition)
        return spec


@dataclass(frozen=True, eq=False, init=False)
class RayleighResponseSpec:
    """Validated Rayleigh medium and phase geometry without input amplitude or time."""

    geometry: HalfSpace
    density_kg_m3: float
    vp_m_s: float
    vs_m_s: float
    frequencies_hz: tuple[float, ...]
    _phase_origin_tuple_m: tuple[float, float, float]
    _direction_tuple: tuple[float, float, float]
    rayleigh_speed_m_s: float
    output_definition: GravityFieldDefinition
    interaction: Newtonian
    contract: object
    g_convention: str
    law_name: str
    law_parameters: Mapping[str, object]
    gravitational_constant_si: float

    def __init__(
        self,
        geometry: HalfSpace,
        density: u.Quantity,
        vp: u.Quantity,
        vs: u.Quantity,
        frequencies: u.Quantity,
        phase_origin: u.Quantity,
        direction: object,
    ) -> None:
        from gragra.results._response import ResponseContract

        rho, vp_si, vs_si = _medium_inputs(geometry, density, vp, vs)
        axis = _frequencies_hz(frequencies)
        origin = _phase_origin_m(phase_origin)
        direction_unit, speed = _direction_and_speed(direction, vp_si, vs_si, axis)
        contract = ResponseContract(
            input_name="vertical_displacement",
            input_quantity="surface_vertical_displacement",
            input_unit=u.m,
            input_channel_ids=("vertical_displacement",),
            input_components=("z",),
            input_frame=geometry.frame,
            input_location_m=origin,
            input_distribution="rayleigh_plane_wave",
            normalization="complex phasor amplitude at input_location",
            basis_amplitude=1 * u.m,
            mapping_kind="local_linearization",
            action="per_frequency",
            storage="precomputed",
            varied="source",
            target_position="fixed",
            law_name="newton",
            law_parameters={},
            gravitational_constant_si=G_SI,
            linearization_assessment="not_assessed",
        )
        for name, value in (
            ("geometry", geometry),
            ("density_kg_m3", rho),
            ("vp_m_s", vp_si),
            ("vs_m_s", vs_si),
            ("frequencies_hz", tuple(float(x) for x in axis)),
            ("_phase_origin_tuple_m", tuple(float(x) for x in origin)),
            ("_direction_tuple", tuple(float(x) for x in direction_unit)),
            ("rayleigh_speed_m_s", speed),
            ("output_definition", _rayleigh_output_definition()),
            ("contract", contract),
            ("g_convention", "SI_G_applied_once"),
            ("interaction", Newtonian()),
            ("law_name", contract.law_name),
            ("law_parameters", contract.law_parameters),
            ("gravitational_constant_si", contract.gravitational_constant_si),
        ):
            object.__setattr__(self, name, value)

    @property
    def density(self) -> u.Quantity:
        return self.density_kg_m3 * u.kg / u.m**3

    @property
    def phase_origin_m(self) -> np.ndarray:
        return np.asarray(self._phase_origin_tuple_m)

    @property
    def direction(self) -> np.ndarray:
        return np.asarray(self._direction_tuple)


class RayleighSurfacePrecisionError(ValueError):
    def __init__(
        self,
        *,
        target_id: str,
        signed_distance_m: float,
        threshold_m: float,
        edge_count: int,
        scale_m: float,
        reason: str,
    ) -> None:
        self.target_id = target_id
        self.signed_distance_m = signed_distance_m
        self.threshold_m = threshold_m
        self.edge_count = edge_count
        self.scale_m = scale_m
        self.reason = reason
        self.unit = "m"
        self.observables = OBSERVABLES
        super().__init__(
            f"{reason}: target ID={target_id!r}, "
            f"surface z={signed_distance_m:g} m, threshold={threshold_m:g} m"
        )


def _position_z_row(source_frame: Frame, destination_frame: Frame) -> np.ndarray:
    """Return the z row of the linear map used by ``trace_point_m``."""
    up, down = source_frame._path_to(destination_frame)
    jacobian = np.eye(3, dtype=np.float64)
    for frame in up:
        jacobian = frame.basis @ jacobian
    for frame in down:
        jacobian = np.linalg.solve(frame.basis, jacobian)
    return jacobian[2]


def check_surface_precision(
    trace: PointTrace, source_frame: Frame, surface_frame: Frame, *, target_id: str
) -> str:
    """Reject z=0 and cross-frame z whose sign cannot be resolved.

    L is the greatest absolute contribution to surface z from coordinates
    or origins on this target's shortest common-ancestor path. The threshold
    is 64 K eps L (K traversed edges); equality is rejected. Each contribution
    uses the position-map Jacobian along the same frame path as the point trace.
    """
    up, down = source_frame._path_to(surface_frame)
    nodes = [source_frame]
    nodes.extend(frame.parent for frame in up)
    nodes.extend(down)
    scale = 0.0
    for frame, coordinates in zip(nodes, trace.coordinates_m, strict=True):
        row = _position_z_row(frame, surface_frame)
        scale = max(scale, float(np.sum(np.abs(row * coordinates))))
    for index, origin in enumerate(trace.origins_m):
        origin_frame = nodes[index + 1] if index < len(up) else nodes[index]
        row = _position_z_row(origin_frame, surface_frame)
        scale = max(scale, float(np.sum(np.abs(row * origin))))
    distance = float(trace.point_m[2])
    threshold = 64 * trace.edge_count * np.finfo(np.float64).eps * scale
    reason = (
        "surface"
        if distance == 0
        else "coordinate_precision"
        if trace.edge_count and abs(distance) <= threshold
        else None
    )
    if reason is not None:
        raise RayleighSurfacePrecisionError(
            target_id=target_id,
            signed_distance_m=distance,
            threshold_m=threshold,
            edge_count=trace.edge_count,
            scale_m=scale,
            reason=reason,
        )
    return "above" if distance > 0 else "below"


def _resolve_rayleigh_targets(
    spec: RayleighModelSpec | RayleighResponseSpec,
    targets: TargetPoints,
    output_frame: Frame,
) -> tuple[Frame, tuple[PointTrace, ...], bytes]:
    if not isinstance(targets, TargetPoints):
        raise TypeError("targets must be TargetPoints")
    if not len(targets):
        raise ValueError("targets must be nonempty")
    if not isinstance(output_frame, Frame):
        raise TypeError("output_frame must be a Frame")
    frame = spec.geometry.frame
    frame._path_to(output_frame)
    traces = tuple(
        targets.frame.trace_point_m(point, frame) for point in targets.positions_m
    )
    for target_id, trace in zip(targets.ids, traces, strict=True):
        offset = trace.point_m[:2] - spec.phase_origin_m[:2]
        with np.errstate(over="ignore", invalid="ignore"):
            phases = (
                2 * np.pi * np.asarray(spec.frequencies_hz) / spec.rayleigh_speed_m_s
            ) * np.dot(spec.direction[:2], offset)
        if not np.isfinite(phases).all():
            raise ValueError(f"derived phase is not finite for target ID={target_id!r}")
    positions = np.stack([trace.point_m for trace in traces])
    return frame, traces, positions.tobytes()


@dataclass(frozen=True, eq=False, init=False)
class RayleighResolvedEvaluation:
    model: object
    model_spec: RayleighModelSpec
    targets: TargetPoints
    computation_frame: Frame
    output_frame: Frame
    observables: tuple[str, ...]
    target_traces: tuple[PointTrace, ...]
    _target_positions_bytes: bytes

    def __init__(
        self, model: object, targets: TargetPoints, output_frame: Frame
    ) -> None:
        spec = getattr(model, "spec", None)
        if type(spec) is not RayleighModelSpec:
            raise TypeError("model.spec must be exactly a RayleighModelSpec")
        frame, traces, positions_bytes = _resolve_rayleigh_targets(
            spec, targets, output_frame
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "model_spec", spec)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "computation_frame", frame)
        object.__setattr__(self, "output_frame", output_frame)
        object.__setattr__(self, "observables", OBSERVABLES)
        object.__setattr__(self, "target_traces", traces)
        object.__setattr__(self, "_target_positions_bytes", positions_bytes)

    @property
    def target_positions_m(self) -> np.ndarray:
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            len(self.targets), 3
        )

    @property
    def parameters(self) -> QTable:
        spec = self.model_spec
        return QTable(
            {
                "density": [spec.density],
                "vp": [spec.vp_m_s * u.m / u.s],
                "vs": [spec.vs_m_s * u.m / u.s],
                "rayleigh_speed": [spec.rayleigh_speed_m_s * u.m / u.s],
                "frequency_min": [min(spec.frequencies_hz) * u.Hz],
                "frequency_max": [max(spec.frequencies_hz) * u.Hz],
                "displacement_max": [
                    float(np.max(np.abs(spec.vertical_displacement_m))) * u.m
                ],
                "direction_x": [spec.direction[0]],
                "direction_y": [spec.direction[1]],
                "direction_z": [spec.direction[2]],
                "phase_origin_x": [spec.phase_origin_m[0] * u.m],
                "phase_origin_y": [spec.phase_origin_m[1] * u.m],
                "reference_gps_seconds": [spec.reference_time_gps[0]],
                "reference_gps_nanoseconds": [spec.reference_time_gps[1]],
                "frequency_count": [len(spec.frequencies_hz)],
                "target_count": [len(self.targets)],
                "geometry_frame": [spec.geometry.frame.name],
                "target_frame": [self.targets.frame.name],
                "output_frame": [self.output_frame.name],
            }
        )

    @property
    def formulas(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": (
                        "H19 Eq. 96-98; spatial derivatives for "
                        "acceleration and gradient"
                    ),
                    "convention": (
                        "exp(+i k_h·(r_h-r0_h)); exp(-2πif(t-t_ref)); "
                        "total bulk plus surface; G once"
                    ),
                }
                for name in OBSERVABLES
            ]
        )

    @property
    def references(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source": (
                        "gragra.observables.rayleigh_literature."
                        "rayleigh_halfspace_allspace_observables"
                    ),
                    "detail": "SI bulk plus surface Rayleigh half-space; H19 Eq. 96-98",
                }
            ]
        )


@dataclass(frozen=True, eq=False, init=False)
class RayleighResponseEvaluation:
    """One target and output-frame request for amplitude-free Rayleigh coefficients."""

    model: object
    model_spec: RayleighResponseSpec
    targets: TargetPoints
    computation_frame: Frame
    output_frame: Frame
    observables: tuple[str, ...]
    target_traces: tuple[PointTrace, ...]
    _target_positions_bytes: bytes

    def __init__(
        self, model: object, targets: TargetPoints, output_frame: Frame
    ) -> None:
        spec = getattr(model, "spec", None)
        if type(spec) is not RayleighResponseSpec:
            raise TypeError("model.spec must be exactly a RayleighResponseSpec")
        frame, traces, positions_bytes = _resolve_rayleigh_targets(
            spec, targets, output_frame
        )
        for name, value in (
            ("model", model),
            ("model_spec", spec),
            ("targets", targets),
            ("computation_frame", frame),
            ("output_frame", output_frame),
            ("observables", OBSERVABLES),
            ("target_traces", traces),
            ("_target_positions_bytes", positions_bytes),
        ):
            object.__setattr__(self, name, value)

    @property
    def frequencies_hz(self) -> tuple[float, ...]:
        return self.model_spec.frequencies_hz

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self.targets.ids

    @property
    def target_positions_m(self) -> np.ndarray:
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            len(self.targets), 3
        )

    @property
    def parameters(self) -> QTable:
        spec = self.model_spec
        return QTable(
            {
                "density": [spec.density],
                "vp": [spec.vp_m_s * u.m / u.s],
                "vs": [spec.vs_m_s * u.m / u.s],
                "rayleigh_speed": [spec.rayleigh_speed_m_s * u.m / u.s],
                "frequency_min": [min(spec.frequencies_hz) * u.Hz],
                "frequency_max": [max(spec.frequencies_hz) * u.Hz],
                "direction_x": [spec.direction[0]],
                "direction_y": [spec.direction[1]],
                "direction_z": [spec.direction[2]],
                "phase_origin_x": [spec.phase_origin_m[0] * u.m],
                "phase_origin_y": [spec.phase_origin_m[1] * u.m],
                "frequency_count": [len(spec.frequencies_hz)],
                "target_count": [len(self.targets)],
                "geometry_frame": [spec.geometry.frame.name],
                "target_frame": [self.targets.frame.name],
                "output_frame": [self.output_frame.name],
                "law_name": [spec.law_name],
            }
        )

    @property
    def formulas(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": (
                        "H19 Eq. 96-98; spatial derivatives for "
                        "acceleration and gradient"
                    ),
                    "convention": (
                        "exp(+i k_h·(r_h-r0_h)); exp(-2πif(t-t_ref)); "
                        "total bulk plus surface; G once"
                    ),
                }
                for name in OBSERVABLES
            ]
        )

    @property
    def references(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source": (
                        "gragra.observables.rayleigh_literature."
                        "rayleigh_halfspace_allspace_observables"
                    ),
                    "detail": "SI bulk plus surface Rayleigh half-space; H19 Eq. 96-98",
                }
            ]
        )

    def _constructor_kwargs(
        self, *, coefficients: Mapping[str, np.ndarray], record: object
    ) -> dict[str, object]:
        """Supply shared constructor arguments for coefficient producers."""
        return {
            "contract": self.model_spec.contract,
            "output_definition": self.model_spec.output_definition,
            "evaluation": self,
            "coefficients": coefficients,
            "record": record,
        }

    def _applied_context(
        self,
        value: FrequencySeries,
        *,
        representation: PhasorRepresentation,
        reference_time: object,
    ) -> _AppliedRayleighContext:
        spec = RayleighModelSpec._from_response(
            self.model_spec, value, representation, reference_time
        )
        from gragra.analytic.seismic._rayleigh import _RayleighModel

        resolved = RayleighResolvedEvaluation(
            _RayleighModel(spec), self.targets, self.output_frame
        )
        return _AppliedRayleighContext(resolved, spec.vertical_displacement_m.tobytes())


@dataclass(frozen=True, eq=False)
class _AppliedRayleighContext:
    """Actual input and result bindings used when applying a Rayleigh response."""

    evaluation: RayleighResolvedEvaluation
    _input_bytes: bytes
    backend: str = "gravity_response:rayleigh_literature"
    context: EvaluationContext = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from gragra.results._context import EvaluationContext

        spec = self.evaluation.model_spec
        context = EvaluationContext(
            model_spec=spec,
            targets=self.evaluation.targets,
            frame=self.evaluation.output_frame,
            observables=self.evaluation.observables,
            representation=spec.representation,
            frequencies=np.asarray(spec.frequencies_hz) * u.Hz,
            reference_time=spec.reference_time_gps,
            g_convention=spec.g_convention,
        )
        object.__setattr__(self, "context", context)

    @property
    def bindings(self) -> Mapping[str, object]:
        return MappingProxyType(
            {name: self.context.binding(name) for name in self.context.observables}
        )

    @property
    def parameters(self) -> QTable:
        return self.evaluation.parameters

    @property
    def formulas(self) -> pd.DataFrame:
        return self.evaluation.formulas

    @property
    def references(self) -> pd.DataFrame:
        return self.evaluation.references

    def validate(self):
        from gragra._rayleigh_evaluation_setup import RayleighEvaluationSetup

        return RayleighEvaluationSetup(self.evaluation).validate()

    def run(self):
        raise RuntimeError(
            "An applied Rayleigh response cannot be rerun from result.evaluation; "
            "call response.apply() with the input phasor to apply the stored "
            "coefficients."
        )

    @property
    def input_values(self) -> np.ndarray:
        return np.frombuffer(self._input_bytes, dtype=np.complex128)

    def pack(self, values: Mapping[str, np.ndarray]) -> object:
        from gragra._rayleigh_evaluation_setup import _native_matrices
        from gragra.results import GravityResult

        count = len(self.evaluation.targets)
        bins = len(self.evaluation.model_spec.frequencies_hz)
        arrays = {}
        for name in OBSERVABLES:
            components = {"potential": 1, "acceleration": 3, "gradient": 9}[name]
            raw = np.asarray(values[name], dtype=np.complex128)
            if raw.shape != (count, components, bins):
                raise ValueError(f"{name} applied values have an invalid shape")
            trailing = (
                ()
                if name == "potential"
                else (3,)
                if name == "acceleration"
                else (3, 3)
            )
            arrays[name] = raw.transpose(2, 0, 1).reshape((bins, count, *trailing))
        return GravityResult(
            **_native_matrices(
                arrays,
                self.evaluation.targets.ids,
                self.evaluation.model_spec.frequencies_hz,
            )
        )


def rotate_rayleigh_vector(
    vector: object, source: Frame, destination: Frame
) -> np.ndarray:
    array = np.asarray(vector, dtype=np.complex128)
    if array.shape[-1:] != (3,) or not np.isfinite(array).all():
        raise ValueError("vector must have finite trailing dimension 3")
    result = np.einsum("ij,...j->...i", source.rotation_to(destination), array)
    if not np.isfinite(result).all():
        raise ValueError("rotated vector is not finite")
    return result


def rotate_rayleigh_tensor(
    tensor: object, source: Frame, destination: Frame
) -> np.ndarray:
    array = np.asarray(tensor, dtype=np.complex128)
    if array.shape[-2:] != (3, 3) or not np.isfinite(array).all():
        raise ValueError("tensor must have finite trailing shape (3,3)")
    rotation = source.rotation_to(destination)
    result = np.einsum("ij,...jk,lk->...il", rotation, array, rotation)
    if not np.isfinite(result).all():
        raise ValueError("rotated tensor is not finite")
    return result
