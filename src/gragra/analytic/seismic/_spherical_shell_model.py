"""High-level phasor model for a finite spherical shell P/S perturbation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesMatrix

from gragra._arrays import _check_no_bool
from gragra._interactions import Newtonian
from gragra._phasor_matrix import _phasor_matrices
from gragra.analytic.seismic._phase_speed import (
    _spherical_phase_speed_to_wavenumbers,
)
from gragra.constants import G_SI
from gragra.geometry._frame import Frame, _lengths_m
from gragra.geometry._spherical_shell import SphericalShell
from gragra.results import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
    GravityFieldDefinition,
    GravityResult,
    PhasorRepresentation,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
)
from gragra.results._context import EvaluationContext
from gragra.results._coords import _frequencies_hz, normalize_gps
from gragra.results._evaluation import EvaluationResult
from gragra.results._record import OBSERVABLES, EvaluationRecord
from gragra.targets._high_points import TargetPoints

_PARTS = ("bulk", "inner_wall", "outer_boundary", "total")
_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}
_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}


def _density_kg_m3(value: u.Quantity) -> float:
    if not isinstance(value, u.Quantity):
        raise TypeError("density must be an astropy Quantity")
    raw = np.asarray(value.to_value(u.kg / u.m**3))
    if raw.shape != () or np.iscomplexobj(raw):
        raise ValueError("density must be a real scalar")
    result = float(raw)
    if not np.isfinite(result) or result < 0:
        raise ValueError("density must be finite and nonnegative")
    return result


def _real_vector(name: str, value: object) -> tuple[float, float, float]:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (3,):
        raise ValueError(f"{name} must be a real three-vector")
    vector = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError(f"{name} must be finite and nonzero")
    return tuple(float(component) for component in vector)


def _unit_vector(name: str, value: object) -> tuple[float, float, float]:
    vector = np.asarray(_real_vector(name, value), dtype=np.float64)
    scaled = vector / np.max(np.abs(vector))
    normalized = scaled / np.linalg.norm(scaled)
    return tuple(float(component) for component in normalized)


def _shell_field_definition(contribution: str) -> GravityFieldDefinition:
    if contribution == "total":
        selected = Contribution("total", _PARTS[:-1])
    else:
        selected = Contribution("selected", (contribution,))
    change = ChangeDefinition(
        ChangeForm.LINEARIZED_RESPONSE,
        ReferenceDefinition(
            ReferenceStatus.DEFINED,
            ReferenceTemporalKind.STATE,
            "zero_displacement_spherical_shell",
        ),
        changed_input="displacement",
        order=1,
        fixed_conditions=(
            "geometry",
            "density",
            "phase_speed",
            "wave_type",
            "propagation_direction",
            "polarization",
            "phase_origin",
        ),
    )
    return GravityFieldDefinition.field_change(
        source_kind="spherical_shell_perturbation",
        contribution=selected,
        potential_reference=PotentialReference.ZERO_AT_INFINITY,
        change=change,
    )


@dataclass(frozen=True, eq=False, init=False)
class SphericalShellModelSpec:
    """Owned inputs and interpretation for one spherical-shell phasor model."""

    geometry: SphericalShell
    density_kg_m3: float
    frequencies_hz: tuple[float, ...]
    wavenumbers_rad_m: tuple[float, ...]
    _displacement_bytes: bytes
    displacement_name: str | None
    displacement_channel: str | None
    displacement_epoch_gps: tuple[int, int] | None
    displacement_input_unit: u.UnitBase
    representation: PhasorRepresentation
    reference_time_gps: tuple[int, int]
    _phase_origin_tuple_m: tuple[float, float, float]
    propagation_direction: tuple[float, float, float]
    polarization: tuple[float, float, float]
    wave_type: str
    contribution_selector: str
    field_definition: GravityFieldDefinition
    interaction: Newtonian
    g_convention: str
    gravitational_constant_si: float

    def __init__(
        self,
        geometry: SphericalShell,
        density: u.Quantity,
        phase_speed: u.Quantity | FrequencySeries,
        displacement: FrequencySeries,
        representation: PhasorRepresentation,
        reference_time: object,
        phase_origin: u.Quantity,
        propagation_direction: object,
        polarization: object,
        wave_type: str,
        contribution: str,
    ) -> None:
        if not isinstance(geometry, SphericalShell):
            raise TypeError("geometry must be a SphericalShell")
        rho = _density_kg_m3(density)
        if not isinstance(displacement, FrequencySeries):
            raise TypeError("displacement must be a GWexpy FrequencySeries")
        frequencies = _frequencies_hz(displacement.frequencies)
        amplitudes = np.asarray(displacement.to_value(u.m))
        if amplitudes.dtype.kind not in "iufc" or amplitudes.shape != frequencies.shape:
            raise ValueError(
                "displacement must have numeric values matching frequencies"
            )
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        if not np.isfinite(amplitudes).all():
            raise ValueError("displacement amplitudes must be finite")
        if not isinstance(representation, PhasorRepresentation):
            raise TypeError("representation must be a PhasorRepresentation")
        reference_time_gps = normalize_gps(reference_time)
        origin = _lengths_m("phase_origin", phase_origin, (3,))
        direction = _unit_vector("propagation_direction", propagation_direction)
        pol = _unit_vector("polarization", polarization)
        if wave_type not in ("P", "S"):
            raise ValueError("wave_type must be 'P' or 'S'")
        if contribution not in _PARTS:
            raise ValueError(
                "contribution must be one of bulk, inner_wall, outer_boundary, total"
            )

        wavenumbers = _spherical_phase_speed_to_wavenumbers(
            phase_speed,
            displacement.frequencies,
            mode=wave_type,
        )
        if displacement.epoch is None:
            epoch = None
        else:
            epoch = normalize_gps(displacement.epoch)

        for name, value in (
            ("geometry", geometry),
            ("density_kg_m3", rho),
            ("frequencies_hz", tuple(float(value) for value in frequencies)),
            (
                "wavenumbers_rad_m",
                tuple(float(value) for value in wavenumbers.to_value(u.rad / u.m)),
            ),
            ("_displacement_bytes", amplitudes.copy().tobytes()),
            ("displacement_name", displacement.name),
            (
                "displacement_channel",
                None if displacement.channel is None else str(displacement.channel),
            ),
            ("displacement_epoch_gps", epoch),
            ("displacement_input_unit", displacement.unit),
            ("representation", representation),
            ("reference_time_gps", reference_time_gps),
            ("_phase_origin_tuple_m", tuple(float(value) for value in origin)),
            ("propagation_direction", direction),
            ("polarization", pol),
            ("wave_type", wave_type),
            ("contribution_selector", contribution),
            ("field_definition", _shell_field_definition(contribution)),
            ("interaction", Newtonian()),
            ("g_convention", "SI_G_applied_once"),
            ("gravitational_constant_si", G_SI),
        ):
            object.__setattr__(self, name, value)

    @property
    def density(self) -> u.Quantity:
        return self.density_kg_m3 * u.kg / u.m**3

    @property
    def displacement_m(self) -> np.ndarray:
        return np.frombuffer(self._displacement_bytes, dtype=np.complex128)

    @property
    def displacement(self) -> FrequencySeries:
        from gwpy.time import LIGOTimeGPS

        epoch = (
            None
            if self.displacement_epoch_gps is None
            else LIGOTimeGPS(*self.displacement_epoch_gps)
        )
        return FrequencySeries(
            self.displacement_m.copy(),
            frequencies=np.asarray(self.frequencies_hz) * u.Hz,
            unit=u.m,
            name=self.displacement_name,
            channel=self.displacement_channel,
            epoch=epoch,
        )

    @property
    def phase_origin_m(self) -> np.ndarray:
        return np.asarray(self._phase_origin_tuple_m, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class _SphericalShellModel:
    spec: SphericalShellModelSpec

    @property
    def geometry(self) -> SphericalShell:
        return self.spec.geometry

    def at(
        self,
        targets: TargetPoints,
        *,
        output_frame: Frame | None = None,
    ) -> SphericalShellEvaluationSetup:
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        frame = targets.frame if output_frame is None else output_frame
        resolved = _SphericalShellResolvedEvaluation(self, targets, frame)
        return SphericalShellEvaluationSetup(resolved)


@dataclass(frozen=True, eq=False, init=False)
class _SphericalShellResolvedEvaluation:
    model: _SphericalShellModel
    model_spec: SphericalShellModelSpec
    targets: TargetPoints
    computation_frame: Frame
    output_frame: Frame
    _target_positions_bytes: bytes

    def __init__(
        self,
        model: _SphericalShellModel,
        targets: TargetPoints,
        output_frame: Frame,
    ) -> None:
        if not isinstance(output_frame, Frame):
            raise TypeError("output_frame must be a Frame")
        if len(targets) == 0:
            raise ValueError("targets must be nonempty")
        computation_frame = model.spec.geometry.frame
        targets.frame._path_to(computation_frame)
        targets.frame._path_to(output_frame)
        positions = targets.frame.transform_points_m(
            targets.positions_m, computation_frame
        )
        for name, value in (
            ("model", model),
            ("model_spec", model.spec),
            ("targets", targets),
            ("computation_frame", computation_frame),
            ("output_frame", output_frame),
            ("_target_positions_bytes", positions.tobytes()),
        ):
            object.__setattr__(self, name, value)

    @property
    def target_positions_m(self) -> np.ndarray:
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            len(self.targets), 3
        )


@dataclass(frozen=True)
class SphericalShellValidationReport:
    valid: bool
    observables: tuple[str, ...]
    target_count: int


class SphericalShellEvaluationSetup:
    __slots__ = ("_resolved", "_context")

    def __init__(self, resolved: _SphericalShellResolvedEvaluation) -> None:
        if not isinstance(resolved, _SphericalShellResolvedEvaluation):
            raise TypeError("resolved must be a spherical-shell resolved evaluation")
        self._resolved = resolved
        spec = resolved.model_spec
        self._context = EvaluationContext(
            model_spec=spec,
            targets=resolved.targets,
            frame=resolved.output_frame,
            observables=OBSERVABLES,
            representation=spec.representation,
            frequencies=np.asarray(spec.frequencies_hz) * u.Hz,
            reference_time=spec.reference_time_gps,
            g_convention=spec.g_convention,
        )

    @property
    def context(self) -> EvaluationContext:
        return self._context

    @property
    def resolved(self) -> _SphericalShellResolvedEvaluation:
        return self._resolved

    @property
    def evaluation(self) -> _SphericalShellResolvedEvaluation:
        return self._resolved

    @property
    def parameters(self) -> QTable:
        resolved = self._resolved
        spec = resolved.model_spec
        geometry = spec.geometry
        origin = spec.phase_origin_m
        direction = spec.propagation_direction
        polarization = spec.polarization
        wavenumbers = np.asarray(spec.wavenumbers_rad_m)
        return QTable(
            {
                "center_x": [geometry.center_m[0] * u.m],
                "center_y": [geometry.center_m[1] * u.m],
                "center_z": [geometry.center_m[2] * u.m],
                "inner_radius": [geometry.inner_radius],
                "outer_radius": [geometry.outer_radius],
                "density": [spec.density],
                "wave_type": [spec.wave_type],
                "contribution": [spec.contribution_selector],
                "frequency_min": [min(spec.frequencies_hz) * u.Hz],
                "frequency_max": [max(spec.frequencies_hz) * u.Hz],
                "wavenumber_min": [float(np.min(wavenumbers)) / u.m],
                "wavenumber_max": [float(np.max(wavenumbers)) / u.m],
                "displacement_max": [float(np.max(np.abs(spec.displacement_m))) * u.m],
                "direction_x": [direction[0]],
                "direction_y": [direction[1]],
                "direction_z": [direction[2]],
                "polarization_x": [polarization[0]],
                "polarization_y": [polarization[1]],
                "polarization_z": [polarization[2]],
                "phase_origin_x": [origin[0] * u.m],
                "phase_origin_y": [origin[1] * u.m],
                "phase_origin_z": [origin[2] * u.m],
                "reference_gps_seconds": [spec.reference_time_gps[0]],
                "reference_gps_nanoseconds": [spec.reference_time_gps[1]],
                "frequency_count": [len(spec.frequencies_hz)],
                "target_count": [len(resolved.targets)],
                "geometry_frame": [geometry.frame.name],
                "target_frame": [resolved.targets.frame.name],
                "output_frame": [resolved.output_frame.name],
            }
        )

    @property
    def formulas(self) -> pd.DataFrame:
        spec = self._resolved.model_spec
        return pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": "Existing spherical-shell solid-harmonic series",
                    "convention": (
                        "Explicit reference_time and geometry-frame phase_origin; "
                        f"native {spec.contribution_selector} contribution; "
                        "G applied once"
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
                        "gragra.analytic.u10_spherical_shell_general."
                        "spherical_shell_observables_g_free"
                    ),
                    "detail": (
                        "G-free finite spherical-shell P/S values at declared bins"
                    ),
                },
                {
                    "source": "gragra.constants.G_SI",
                    "detail": "configured SI gravitational constant; applied once",
                },
            ]
        )

    def validate(self) -> SphericalShellValidationReport:
        self._context.validate()
        return SphericalShellValidationReport(
            True, OBSERVABLES, len(self._resolved.targets)
        )

    def run(self) -> EvaluationResult:
        self.validate()
        from gragra.analytic.u10_spherical_shell_general import (
            spherical_shell_observables_g_free,
        )

        resolved = self._resolved
        spec = resolved.model_spec
        parts = spherical_shell_observables_g_free(
            resolved.target_positions_m,
            inner_radius_m=spec.geometry.inner_radius_m,
            outer_radius_m=spec.geometry.outer_radius_m,
            density_kg_m3=spec.density_kg_m3,
            wavenumber_rad_m=np.asarray(spec.wavenumbers_rad_m),
            displacement_m=spec.displacement_m,
            propagation_direction=spec.propagation_direction,
            polarization=spec.polarization,
            wave_type=spec.wave_type,
            center_m=spec.geometry.center_m,
            phase_origin_m=spec.phase_origin_m,
        )
        native = parts[spec.contribution_selector]
        arrays = {
            "potential": np.asarray(native.potential) * G_SI,
            "acceleration": np.asarray(native.acceleration) * G_SI,
            "gradient": np.asarray(native.gradient) * G_SI,
        }
        rotation = resolved.computation_frame.rotation_to(resolved.output_frame)
        arrays["acceleration"] = np.einsum(
            "ij,...j->...i", rotation, arrays["acceleration"]
        )
        arrays["gradient"] = np.einsum(
            "ij,...jk,lk->...il", rotation, arrays["gradient"], rotation
        )
        matrices = _shell_phasor_matrices(
            arrays, resolved.targets.ids, spec.frequencies_hz
        )
        record = EvaluationRecord(
            backend="u10_spherical_shell_general",
            versions={
                "gragra": version("gragra"),
                "python": python_version(),
                "numpy": np.__version__,
                "astropy": version("astropy"),
                "gwexpy": version("gwexpy"),
            },
            gravitational_constant_si=G_SI,
            g_applications=1,
        )
        return EvaluationResult(
            values=GravityResult(**matrices),
            evaluation=self,
            record=record,
        )


def _shell_phasor_matrices(
    arrays: dict[str, np.ndarray],
    target_ids: tuple[str, ...],
    frequencies_hz: tuple[float, ...],
) -> dict[str, FrequencySeriesMatrix]:
    count, bins = len(target_ids), len(frequencies_hz)
    shaped = {
        "potential": np.asarray(arrays["potential"]).transpose(1, 0)[:, None, :],
        "acceleration": np.asarray(arrays["acceleration"]).transpose(1, 2, 0),
        "gradient": np.asarray(arrays["gradient"])
        .transpose(1, 2, 3, 0)
        .reshape(count, 9, bins),
    }
    if any(
        value.shape[0] != count or value.shape[-1] != bins for value in shaped.values()
    ):
        raise ValueError(
            "native spherical-shell arrays do not match targets and frequencies"
        )
    return _phasor_matrices(
        shaped,
        target_ids,
        frequencies_hz,
        units=_UNITS,
        cell_name=lambda observable, target, component, _row, _column: (
            f"{target} {component} {observable}"
        ),
        cell_channel=lambda observable, _target, _component, row, column: (
            f"GRAGRA:{observable}:{row}:{column}"
        ),
        matrix_name=lambda observable: f"Spherical shell {observable}",
    )


def spherical_shell(
    geometry: SphericalShell,
    density: u.Quantity,
    phase_speed: u.Quantity | FrequencySeries,
    displacement: FrequencySeries,
    representation: PhasorRepresentation,
    reference_time: object,
    phase_origin: u.Quantity,
    propagation_direction: object,
    polarization: object,
    wave_type: str,
    contribution: str = "total",
) -> _SphericalShellModel:
    """Create an explicitly referenced P/S displacement perturbation model."""
    return _SphericalShellModel(
        SphericalShellModelSpec(
            geometry,
            density,
            phase_speed,
            displacement,
            representation,
            reference_time,
            phase_origin,
            propagation_direction,
            polarization,
            wave_type,
            contribution,
        )
    )


__all__ = ["spherical_shell"]
