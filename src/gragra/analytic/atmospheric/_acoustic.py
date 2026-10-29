"""Phasor gravity of a rigid-reflection acoustic halfspace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable

from gragra._analytic_evaluation import AnalyticEvaluation, NativeEvaluation
from gragra._arrays import _check_no_bool
from gragra._interactions import Newtonian
from gragra.geometry._frame import Frame, _lengths_m
from gragra.geometry._halfspace import HalfSpace
from gragra.results import GravityFieldDefinition
from gragra.results._context import EvaluationContext
from gragra.results._coords import normalize_gps
from gragra.results._gen1_definitions import gen1_field_definition
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints


def _real_scalar(
    name: str, value: object, unit: u.UnitBase, *, positive: bool
) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    _check_no_bool(value, name)
    try:
        raw = np.asarray(value.to_value(unit))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise u.UnitConversionError(f"{name} must convert to {unit}") from exc
    if raw.shape != ():
        raise ValueError(f"{name} must be a scalar Quantity")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    number = float(raw)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _pressure_scalar(value: object) -> complex:
    if not isinstance(value, u.Quantity):
        raise TypeError("surface_pressure must be an Astropy Quantity")
    _check_no_bool(value, "surface_pressure")
    try:
        raw = np.asarray(value.to_value(u.Pa))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise u.UnitConversionError("surface_pressure must convert to Pa") from exc
    if raw.shape != ():
        raise ValueError("surface_pressure must be a scalar Quantity")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError("surface_pressure must be real or complex numeric")
    pressure = complex(raw)
    if not np.isfinite(pressure):
        raise ValueError("surface_pressure must be finite")
    return pressure


@dataclass(frozen=True, eq=False, slots=True, init=False)
class AcousticHalfspaceSpec:
    """Owned physical and Fourier conventions for an acoustic halfspace."""

    geometry: HalfSpace
    surface_pressure_pa: complex
    sound_speed_m_s: float
    frequency_hz: float
    incidence_angle_rad: float
    azimuth_rad: float
    _phase_origin_tuple_m: tuple[float, float, float]
    reference_time_gps: tuple[int, int]
    wavenumber_rad_m: float
    horizontal_wavenumber_rad_m: float
    vertical_wavenumber_rad_m: float
    field_definition: GravityFieldDefinition
    interaction: Newtonian
    parameters: QTable
    formulas: pd.DataFrame
    references: pd.DataFrame

    def __init__(
        self,
        geometry: HalfSpace,
        surface_pressure: u.Quantity,
        *,
        sound_speed: u.Quantity,
        frequency: u.Quantity,
        incidence_angle: u.Quantity,
        azimuth: u.Quantity,
        phase_origin: u.Quantity,
        reference_time: object,
    ) -> None:
        if not isinstance(geometry, HalfSpace):
            raise TypeError("halfspace must be a geometry.HalfSpace")
        pressure_pa = _pressure_scalar(surface_pressure)
        speed = _real_scalar("sound_speed", sound_speed, u.m / u.s, positive=True)
        frequency_hz = _real_scalar("frequency", frequency, u.Hz, positive=True)
        theta = _real_scalar("incidence_angle", incidence_angle, u.rad, positive=False)
        if theta <= 0.0 or theta > np.pi / 2.0:
            raise ValueError("incidence_angle must satisfy 0 < theta <= pi/2")
        azimuth_rad = _real_scalar("azimuth", azimuth, u.rad, positive=False)
        origin = _lengths_m("phase_origin", phase_origin, (3,))
        reference_gps = normalize_gps(reference_time)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            wavenumber = float(2.0 * np.pi * frequency_hz / speed)
            horizontal = float(wavenumber * np.sin(theta))
            vertical = (
                0.0 if theta == np.pi / 2.0 else float(wavenumber * np.cos(theta))
            )
        if not np.isfinite([wavenumber, horizontal, vertical]).all():
            raise ValueError("derived acoustic wavenumbers must be finite")
        if wavenumber <= 0.0 or horizontal <= 0.0:
            raise ValueError("acoustic halfspace requires k > 0 and q > 0")
        definition = gen1_field_definition("acoustic_halfspace")
        parameters = QTable(
            {
                "surface_pressure_total": [pressure_pa * u.Pa],
                "sound_speed": [speed * u.m / u.s],
                "frequency": [frequency_hz * u.Hz],
                "incidence_angle": [theta * u.rad],
                "azimuth": [azimuth_rad * u.rad],
                "phase_origin_x": [origin[0] * u.m],
                "phase_origin_y": [origin[1] * u.m],
                "phase_origin_z": [origin[2] * u.m],
                "k": [wavenumber / u.m],
                "q": [horizontal / u.m],
                "k_z": [vertical / u.m],
                "reference_time_gps_seconds": [reference_gps[0]],
                "reference_time_gps_nanoseconds": [reference_gps[1]],
                "pressure_convention": ["total_surface_pressure_phasor"],
                "halfspace_frame": [geometry.frame.name],
            }
        )
        formulas = pd.DataFrame(
            [
                {
                    "observable": observable,
                    "expression": expression,
                    "convention": "G-free rigid-reflection positive-q Green solution; "
                    "G applied once; total surface pressure",
                }
                for observable, expression in (
                    ("potential", "4*pi*surface_pressure/(c_s**2*k**2) * E * ell"),
                    ("acceleration", "-gradient(Phi); air and ground branches"),
                    ("gradient", "spatial derivative of acceleration; row-major"),
                )
            ]
        )
        references = pd.DataFrame(
            [
                {
                    "source": "docs/design/gen1/formula-ledger.md#acoustic-halfspace",
                    "detail": (
                        "approved G-free air and ground solutions and interface "
                        "conditions"
                    ),
                },
                {
                    "source": "gragra.constants.G_SI",
                    "detail": "applied exactly once by the shared analytic bridge",
                },
            ]
        )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "surface_pressure_pa", pressure_pa)
        object.__setattr__(self, "sound_speed_m_s", speed)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "incidence_angle_rad", theta)
        object.__setattr__(self, "azimuth_rad", azimuth_rad)
        object.__setattr__(
            self, "_phase_origin_tuple_m", tuple(float(x) for x in origin)
        )
        object.__setattr__(self, "reference_time_gps", reference_gps)
        object.__setattr__(self, "wavenumber_rad_m", wavenumber)
        object.__setattr__(self, "horizontal_wavenumber_rad_m", horizontal)
        object.__setattr__(self, "vertical_wavenumber_rad_m", vertical)
        object.__setattr__(self, "field_definition", definition)
        object.__setattr__(self, "interaction", Newtonian())
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "formulas", formulas)
        object.__setattr__(self, "references", references)

    @property
    def phase_origin_m(self) -> np.ndarray:
        return np.asarray(self._phase_origin_tuple_m, dtype=np.float64)

    @property
    def phase_origin(self) -> u.Quantity:
        result = self.phase_origin_m * u.m
        result.setflags(write=False)
        return result


@dataclass(frozen=True, eq=False, slots=True)
class AcousticHalfspaceModel:
    """Single-frequency high-level halfspace evaluation factory."""

    spec: AcousticHalfspaceSpec

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = ("potential", "acceleration", "gradient"),
        output_frame: Frame | None = None,
    ) -> AnalyticEvaluation:
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be high-level TargetPoints")
        output = targets.frame if output_frame is None else output_frame
        if not isinstance(output, Frame):
            raise TypeError("output_frame must be a Frame")
        positions = targets.frame.transform_points_m(
            targets.positions_m, self.spec.geometry.frame
        )
        input_positions = targets.positions_m
        context = EvaluationContext(
            model_spec=self.spec,
            targets=targets,
            frame=output,
            observables=observables,
            representation=PhasorRepresentation(),
            frequencies=np.array([self.spec.frequency_hz]) * u.Hz,
            reference_time=self.spec.reference_time_gps,
        )

        def validate_targets(target_positions_m: np.ndarray) -> None:
            if len(target_positions_m) == 0:
                return
            for index, (local, original) in enumerate(
                zip(target_positions_m, input_positions, strict=True)
            ):
                tolerance = _frame_conversion_tolerance(
                    targets.frame, original, self.spec.geometry.frame
                )
                if abs(float(local[2])) <= tolerance:
                    raise ValueError(
                        f"target point at index {index} lies on the halfspace boundary"
                    )

        def evaluate_native(
            target_positions_m: np.ndarray, names: tuple[str, ...]
        ) -> NativeEvaluation:
            return NativeEvaluation(
                _acoustic_native_values(self.spec, target_positions_m, names)
            )

        return AnalyticEvaluation(
            context=context,
            computation_frame=self.spec.geometry.frame,
            target_positions_m=positions,
            evaluate_native=evaluate_native,
            validate_targets=validate_targets,
        )


def _frame_conversion_tolerance(
    input_frame: Frame, point_m: np.ndarray, output_frame: Frame
) -> float:
    """Bound only roundoff accumulated while converting this one point.

    Points already in the halfspace frame receive an exact z=0 check. For a
    converted point, its own frame trace bounds the arithmetic scale; unrelated
    targets in the same batch cannot widen the boundary tolerance.
    """
    trace = input_frame.trace_point_m(point_m, output_frame)
    if trace.edge_count == 0:
        return 0.0
    operations_budget = 16.0 * trace.edge_count * np.finfo(np.float64).eps
    return operations_budget * trace.scale_m


def _acoustic_native_values(
    spec: AcousticHalfspaceSpec,
    targets_m: np.ndarray,
    observables: tuple[str, ...],
) -> Mapping[str, np.ndarray]:
    q = spec.horizontal_wavenumber_rad_m
    p = spec.vertical_wavenumber_rad_m
    k = spec.wavenumber_rad_m
    horizontal = q * np.array(
        [np.cos(spec.azimuth_rad), np.sin(spec.azimuth_rad)], dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        phase = np.exp(1j * ((targets_m[:, :2] - spec.phase_origin_m[:2]) @ horizontal))
        coefficient = (
            4.0 * np.pi * spec.surface_pressure_pa / (spec.sound_speed_m_s**2 * k**2)
        )
        air = targets_m[:, 2] > 0.0
        z = targets_m[:, 2]
        potential_factor = np.empty(len(z), dtype=np.float64)
        first_factor = np.empty(len(z), dtype=np.float64)
        second_factor = np.empty(len(z), dtype=np.float64)
        potential_factor[air] = np.exp(-q * z[air]) / 2.0 - np.cos(p * z[air])
        first_factor[air] = -q * np.exp(-q * z[air]) / 2.0 + p * np.sin(p * z[air])
        second_factor[air] = q**2 * np.exp(-q * z[air]) / 2.0 + p**2 * np.cos(
            p * z[air]
        )
        potential_factor[~air] = -np.exp(q * z[~air]) / 2.0
        first_factor[~air] = q * potential_factor[~air]
        second_factor[~air] = q**2 * potential_factor[~air]

        potential = coefficient * phase * potential_factor
        acceleration = np.empty((len(z), 3), dtype=np.complex128)
        acceleration[:, :2] = (
            -1j
            * coefficient
            * phase[:, None]
            * horizontal[None, :]
            * potential_factor[:, None]
        )
        acceleration[:, 2] = -coefficient * phase * first_factor
        gradient = np.empty((len(z), 3, 3), dtype=np.complex128)
        gradient[:, :2, :2] = (
            coefficient
            * phase[:, None, None]
            * horizontal[None, :, None]
            * horizontal[None, None, :]
            * potential_factor[:, None, None]
        )
        cross = -1j * coefficient * phase[:, None] * horizontal * first_factor[:, None]
        gradient[:, :2, 2] = cross
        gradient[:, 2, :2] = cross
        gradient[:, 2, 2] = -coefficient * phase * second_factor

    values: dict[str, np.ndarray] = {}
    if "potential" in observables:
        values["potential"] = potential[:, None, None]
    if "acceleration" in observables:
        values["acceleration"] = acceleration[:, :, None]
    if "gradient" in observables:
        values["gradient"] = gradient.reshape(len(z), 9)[:, :, None]
    return MappingProxyType(values)


def acoustic_halfspace(
    halfspace: HalfSpace,
    surface_pressure: u.Quantity,
    *,
    sound_speed: u.Quantity,
    frequency: u.Quantity,
    incidence_angle: u.Quantity,
    azimuth: u.Quantity,
    phase_origin: u.Quantity,
    reference_time: object,
) -> AcousticHalfspaceModel:
    """Create a single-frequency total-surface-pressure acoustic model."""
    return AcousticHalfspaceModel(
        AcousticHalfspaceSpec(
            halfspace,
            surface_pressure,
            sound_speed=sound_speed,
            frequency=frequency,
            incidence_angle=incidence_angle,
            azimuth=azimuth,
            phase_origin=phase_origin,
            reference_time=reference_time,
        )
    )


__all__ = [
    "AcousticHalfspaceSpec",
    "AcousticHalfspaceModel",
    "acoustic_halfspace",
]
