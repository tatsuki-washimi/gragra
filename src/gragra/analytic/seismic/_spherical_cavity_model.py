"""High-level spherical-cavity P/S phasor model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeries
from gwpy.time import LIGOTimeGPS

from gragra._arrays import _check_no_bool
from gragra._evaluation_setup import OBSERVABLES
from gragra._interactions import Newtonian
from gragra._phasor_matrix import _phasor_matrices
from gragra.analytic.seismic._phase_speed import (
    _spherical_phase_speed_to_wavenumbers,
)
from gragra.analytic.spherical_cavity_series import (
    spherical_cavity_observables_g_free,
)
from gragra.constants import G_SI
from gragra.geometry._frame import Frame, PointTrace, _lengths_m
from gragra.geometry._sphere import Sphere
from gragra.results._context import EvaluationContext
from gragra.results._coords import _frequencies_hz, normalize_gps
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
from gragra.results._record import EvaluationRecord
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints

_FIXED_CONDITIONS = (
    "geometry",
    "density",
    "phase_speed",
    "wave_type",
    "propagation_direction",
    "polarization",
    "phase_origin",
)
_PARTS = ("bulk", "wall", "total")
_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}
_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}


def _real_vector(name: str, value: object) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (3,):
        raise ValueError(f"{name} must be a real three-vector")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or not np.any(result):
        raise ValueError(f"{name} must be finite and nonzero")
    return result


def _unit_vector(name: str, value: object) -> np.ndarray:
    vector = _real_vector(name, value)
    scaled = vector / np.max(np.abs(vector))
    return scaled / np.linalg.norm(scaled)


def _displacement_values(value: FrequencySeries) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, FrequencySeries):
        raise TypeError("displacement must be a GWexpy FrequencySeries")
    frequencies = _frequencies_hz(value.frequencies)
    raw = np.asarray(value.value)
    if raw.dtype.kind not in "iufc" or raw.ndim != 1 or raw.shape != frequencies.shape:
        raise ValueError(
            "displacement must be a numeric one-dimensional series "
            "matching its frequency axis"
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            amplitudes = np.asarray(value.to_value(u.m), dtype=np.complex128)
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError("displacement must have units compatible with meters") from exc
    if not np.isfinite(amplitudes).all():
        raise ValueError("displacement amplitudes must be finite")
    return frequencies, amplitudes.copy()


def _field_definition(part: str, wave_type: str) -> GravityFieldDefinition:
    reference = ReferenceDefinition(
        ReferenceStatus.DEFINED,
        ReferenceTemporalKind.STATE,
        "zero_displacement_spherical_cavity",
    )
    change = ChangeDefinition(
        ChangeForm.LINEARIZED_RESPONSE,
        reference,
        changed_input="displacement",
        order=1,
        fixed_conditions=_FIXED_CONDITIONS,
    )
    potential_reference = (
        PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY
        if wave_type == "P" and part in ("bulk", "total")
        else PotentialReference.ZERO_AT_INFINITY
    )
    contribution = (
        Contribution("total", ("bulk", "wall"))
        if part == "total"
        else Contribution("selected", (part,))
    )
    return GravityFieldDefinition.field_change(
        source_kind="spherical_cavity_perturbation",
        contribution=contribution,
        potential_reference=potential_reference,
        change=change,
    )


@dataclass(frozen=True, eq=False, init=False)
class SphericalCavityModelSpec:
    """Owned inputs and interpretation for one spherical-cavity evaluation."""

    geometry: Sphere
    density_kg_m3: float
    frequencies_hz: tuple[float, ...]
    _wavenumber_bytes: bytes
    _displacement_bytes: bytes
    displacement_name: str | None
    displacement_channel: str | None
    displacement_epoch_gps: tuple[int, int] | None
    displacement_input_unit: u.UnitBase
    representation: PhasorRepresentation
    reference_time_gps: tuple[int, int]
    wave_type: str
    _propagation_direction_tuple: tuple[float, float, float]
    _polarization_tuple: tuple[float, float, float]
    _phase_origin_tuple_m: tuple[float, float, float]
    part: str
    field_definition: GravityFieldDefinition
    interaction: Newtonian
    g_convention: str
    gravitational_constant_si: float

    def __init__(
        self,
        geometry: Sphere,
        density: u.Quantity,
        phase_speed: u.Quantity | FrequencySeries,
        displacement: FrequencySeries,
        representation: PhasorRepresentation,
        reference_time: object,
        wave_type: str,
        propagation_direction: object,
        polarization: object,
        phase_origin: u.Quantity,
        part: str,
    ) -> None:
        if not isinstance(geometry, Sphere):
            raise TypeError("geometry must be a high-level Sphere")
        if not isinstance(density, u.Quantity):
            raise TypeError("density must be an Astropy Quantity")
        raw_density = np.asarray(density.to_value(u.kg / u.m**3))
        if raw_density.shape != () or np.iscomplexobj(raw_density):
            raise ValueError("density must be a real scalar")
        density_si = float(raw_density)
        if not np.isfinite(density_si) or density_si < 0:
            raise ValueError("density must be finite and nonnegative")
        if wave_type not in ("P", "S"):
            raise ValueError("wave_type must be 'P' or 'S'")
        if part not in _PARTS:
            raise ValueError("part must be 'bulk', 'wall', or 'total'")
        if not isinstance(representation, PhasorRepresentation):
            raise TypeError("representation must be a PhasorRepresentation")

        frequencies_hz, amplitudes_m = _displacement_values(displacement)
        wavenumbers = _spherical_phase_speed_to_wavenumbers(
            phase_speed,
            np.array(frequencies_hz, copy=True) * u.Hz,
            mode=wave_type,
        ).to_value(u.rad / u.m)
        gps = normalize_gps(reference_time)
        origin_m = _lengths_m("phase_origin", phase_origin, (3,))
        direction = _unit_vector("propagation_direction", propagation_direction)
        polarization_vector = _unit_vector("polarization", polarization)
        epoch = (
            None if displacement.epoch is None else normalize_gps(displacement.epoch)
        )

        for name, value in (
            ("geometry", geometry),
            ("density_kg_m3", density_si),
            ("frequencies_hz", tuple(float(x) for x in frequencies_hz)),
            ("_wavenumber_bytes", np.asarray(wavenumbers, dtype=np.float64).tobytes()),
            ("_displacement_bytes", amplitudes_m.tobytes()),
            ("displacement_name", displacement.name),
            (
                "displacement_channel",
                None if displacement.channel is None else str(displacement.channel),
            ),
            ("displacement_epoch_gps", epoch),
            ("displacement_input_unit", displacement.unit),
            ("representation", representation),
            ("reference_time_gps", gps),
            ("wave_type", wave_type),
            ("_propagation_direction_tuple", tuple(float(x) for x in direction)),
            ("_polarization_tuple", tuple(float(x) for x in polarization_vector)),
            ("_phase_origin_tuple_m", tuple(float(x) for x in origin_m)),
            ("part", part),
            ("field_definition", _field_definition(part, wave_type)),
            ("interaction", Newtonian()),
            ("g_convention", "SI_G_applied_once"),
            ("gravitational_constant_si", G_SI),
        ):
            object.__setattr__(self, name, value)

    @property
    def density(self) -> u.Quantity:
        return self.density_kg_m3 * u.kg / u.m**3

    @property
    def wavenumber_rad_m(self) -> np.ndarray:
        return np.frombuffer(self._wavenumber_bytes, dtype=np.float64)

    @property
    def displacement_m(self) -> np.ndarray:
        return np.frombuffer(self._displacement_bytes, dtype=np.complex128)

    @property
    def displacement(self) -> FrequencySeries:
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
    def propagation_direction(self) -> np.ndarray:
        return np.asarray(self._propagation_direction_tuple)

    @property
    def polarization(self) -> np.ndarray:
        return np.asarray(self._polarization_tuple)

    @property
    def phase_origin_m(self) -> np.ndarray:
        return np.asarray(self._phase_origin_tuple_m)


@dataclass(frozen=True, eq=False)
class SphericalCavityModel:
    """Spherical cavity model whose ``at`` method fixes targets and frame."""

    spec: SphericalCavityModelSpec

    @property
    def geometry(self) -> Sphere:
        return self.spec.geometry

    def at(
        self,
        targets: TargetPoints,
        *,
        output_frame: Frame | None = None,
    ) -> SphericalCavityEvaluationSetup:
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        resolved = SphericalCavityResolvedEvaluation(
            self,
            targets,
            targets.frame if output_frame is None else output_frame,
        )
        return SphericalCavityEvaluationSetup(resolved)


@dataclass(frozen=True, eq=False, init=False)
class SphericalCavityResolvedEvaluation:
    """Fixed target positions transformed into the spherical geometry frame."""

    model: SphericalCavityModel
    model_spec: SphericalCavityModelSpec
    targets: TargetPoints
    computation_frame: Frame
    output_frame: Frame
    observables: tuple[str, ...]
    target_traces: tuple[PointTrace, ...]
    _target_positions_bytes: bytes

    def __init__(
        self, model: SphericalCavityModel, targets: TargetPoints, output_frame: Frame
    ) -> None:
        if not isinstance(model, SphericalCavityModel):
            raise TypeError("model must be a SphericalCavityModel")
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        if not len(targets):
            raise ValueError("targets must be nonempty")
        if not isinstance(output_frame, Frame):
            raise TypeError("output_frame must be a Frame")
        geometry_frame = model.spec.geometry.frame
        targets.frame._path_to(geometry_frame)
        geometry_frame._path_to(output_frame)
        traces = tuple(
            targets.frame.trace_point_m(point, geometry_frame)
            for point in targets.positions_m
        )
        positions = np.stack([trace.point_m for trace in traces])
        for name, value in (
            ("model", model),
            ("model_spec", model.spec),
            ("targets", targets),
            ("computation_frame", geometry_frame),
            ("output_frame", output_frame),
            ("observables", OBSERVABLES),
            ("target_traces", traces),
            ("_target_positions_bytes", positions.tobytes()),
        ):
            object.__setattr__(self, name, value)

    @property
    def target_positions_m(self) -> np.ndarray:
        return np.frombuffer(self._target_positions_bytes, dtype=np.float64).reshape(
            len(self.targets), 3
        )

    @property
    def parameters(self) -> QTable:
        spec = self.model_spec
        origin = spec.phase_origin_m
        direction = spec.propagation_direction
        polarization = spec.polarization
        return QTable(
            {
                "density": [spec.density],
                "radius": [spec.geometry.radius_m * u.m],
                "wave_type": [spec.wave_type],
                "part": [spec.part],
                "frequency_min": [min(spec.frequencies_hz) * u.Hz],
                "frequency_max": [max(spec.frequencies_hz) * u.Hz],
                "wavenumber_min": [float(np.min(spec.wavenumber_rad_m)) / u.m],
                "wavenumber_max": [float(np.max(spec.wavenumber_rad_m)) / u.m],
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
                    "expression": "Existing spherical-cavity solid-harmonic series",
                    "convention": (
                        "exp(+i k d·(s-phase_origin)); exp(-2πif(t-reference_time)); "
                        f"native {self.model_spec.part} contribution; G applied once"
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
                        "gragra.analytic.spherical_cavity_series."
                        "spherical_cavity_observables_g_free"
                    ),
                    "detail": "G-free P/S cavity values at the declared frequency bins",
                }
            ]
        )


class SphericalCavityEvaluationSetup:
    """Validate, evaluate, and bind one spherical-cavity request."""

    __slots__ = ("_resolved", "_context")

    def __init__(self, resolved: SphericalCavityResolvedEvaluation) -> None:
        if not isinstance(resolved, SphericalCavityResolvedEvaluation):
            raise TypeError("resolved must be a SphericalCavityResolvedEvaluation")
        self._resolved = resolved
        spec = resolved.model_spec
        self._context = EvaluationContext(
            model_spec=spec,
            targets=resolved.targets,
            frame=resolved.output_frame,
            observables=resolved.observables,
            representation=spec.representation,
            frequencies=np.asarray(spec.frequencies_hz) * u.Hz,
            reference_time=spec.reference_time_gps,
            g_convention=spec.g_convention,
        )

    @property
    def context(self) -> EvaluationContext:
        return self._context

    @property
    def resolved(self) -> SphericalCavityResolvedEvaluation:
        return self._resolved

    @property
    def evaluation(self) -> SphericalCavityResolvedEvaluation:
        return self._resolved

    @property
    def parameters(self) -> QTable:
        return self._resolved.parameters

    @property
    def formulas(self) -> pd.DataFrame:
        return self._resolved.formulas

    @property
    def references(self) -> pd.DataFrame:
        return self._resolved.references

    def validate(self) -> None:
        self._context.validate()

    def run(self):
        """Evaluate the requested native contribution and apply SI G once."""
        self.validate()
        resolved = self._resolved
        spec = resolved.model_spec
        native = spherical_cavity_observables_g_free(
            resolved.target_positions_m,
            radius_m=spec.geometry.radius_m,
            density_kg_m3=spec.density_kg_m3,
            wavenumber_rad_m=spec.wavenumber_rad_m,
            displacement_m=spec.displacement_m,
            propagation_direction=spec.propagation_direction,
            polarization=spec.polarization,
            wave_type=spec.wave_type,
            center_m=spec.geometry.center_m,
            phase_origin_m=spec.phase_origin_m,
        )
        selected = native[spec.part]
        potential = np.asarray(selected.potential, dtype=np.complex128) * G_SI
        acceleration = np.asarray(selected.acceleration, dtype=np.complex128) * G_SI
        gradient = np.asarray(selected.gradient, dtype=np.complex128) * G_SI
        rotation = resolved.computation_frame.rotation_to(resolved.output_frame)
        acceleration = np.einsum("ij,ktj->kti", rotation, acceleration)
        gradient = np.einsum("ij,ktjl,ml->ktim", rotation, gradient, rotation)
        arrays = {
            "potential": potential.T[:, None, :],
            "acceleration": acceleration.transpose(1, 2, 0),
            "gradient": gradient.reshape(
                len(spec.frequencies_hz), len(resolved.targets), 9
            ).transpose(1, 2, 0),
        }
        matrices = _phasor_matrices(
            arrays,
            resolved.targets.ids,
            spec.frequencies_hz,
            units=_UNITS,
            cell_name=lambda name, target, component, _row, _column: (
                f"{target} {component} {name}"
            ),
            cell_channel=lambda name, _target, _component, row, column: (
                f"GRAGRA:{name}:{row}:{column}"
            ),
            matrix_name=lambda name: f"Spherical cavity {name}",
        )
        from gragra.results import GravityResult
        from gragra.results._evaluation import EvaluationResult

        record = EvaluationRecord(
            backend="spherical_cavity_series",
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


def spherical_cavity(
    geometry: Sphere,
    density: u.Quantity,
    phase_speed: u.Quantity | FrequencySeries,
    displacement: FrequencySeries,
    representation: PhasorRepresentation,
    reference_time: object,
    wave_type: str,
    propagation_direction: object,
    polarization: object,
    phase_origin: u.Quantity,
    part: str = "total",
) -> SphericalCavityModel:
    """Create a spherical-cavity phasor model with an explicit phase reference.

    ``displacement`` is a GWexpy ``FrequencySeries`` interpreted as a complex
    peak phasor under ``representation``. ``reference_time`` is mandatory and
    supplies its GPS phase reference; a series epoch is retained only as
    provenance and does not trigger rephasing. A frequency-series ``phase_speed``
    must have the exact displacement frequency axis.

    ``phase_origin`` is a position in ``geometry.frame``. The components of
    ``propagation_direction`` and ``polarization`` are also expressed in that
    frame. ``part`` selects ``"bulk"``, ``"wall"``, or the native ``"total"``.
    """
    return SphericalCavityModel(
        SphericalCavityModelSpec(
            geometry,
            density,
            phase_speed,
            displacement,
            representation,
            reference_time,
            wave_type,
            propagation_direction,
            polarization,
            phase_origin,
            part,
        )
    )


__all__ = ["spherical_cavity"]
