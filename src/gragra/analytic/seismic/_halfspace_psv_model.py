"""High-level phasor model for incident P/SV waves at a flat half-space."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from gwexpy.frequencyseries import FrequencySeries
from gwpy.time import LIGOTimeGPS

from gragra._interactions import Newtonian
from gragra.analytic.bodywave_literature import (
    _validate_azimuth,
    _validate_incidence_angle,
)
from gragra.analytic.halfspace_psv import (
    halfspace_psv_observables_g_free,
    halfspace_psv_reflection,
)
from gragra.constants import G_SI
from gragra.geometry._frame import Frame, PointTrace, _lengths_m
from gragra.geometry._halfspace import HalfSpace
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
from gragra.results._record import OBSERVABLES, EvaluationRecord
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints
from gragra.waves.rayleigh import _validate_speeds

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


def _density_si(value: u.Quantity) -> float:
    rho = _scalar_si("density", value, u.kg / u.m**3)
    if rho < 0:
        raise ValueError("density must be nonnegative")
    return rho


def _field_definition(
    potential_reference: PotentialReference,
) -> GravityFieldDefinition:
    change = ChangeDefinition(
        ChangeForm.LINEARIZED_RESPONSE,
        ReferenceDefinition(
            ReferenceStatus.DEFINED,
            ReferenceTemporalKind.STATE,
            "zero_displacement_halfspace_psv",
        ),
        changed_input="displacement",
        order=1,
        fixed_conditions=(
            "geometry",
            "density",
            "vp",
            "vs",
            "wave_type",
            "frequency",
            "incidence_angle",
            "azimuth",
            "phase_origin",
        ),
    )
    return GravityFieldDefinition.field_change(
        source_kind="halfspace_psv_perturbation",
        contribution=Contribution("total"),
        potential_reference=potential_reference,
        change=change,
    )


@dataclass(frozen=True, eq=False, init=False)
class _HalfspacePSVModelSpec:
    """Owned wave, material, and phase inputs for one PSV half-space model."""

    geometry: HalfSpace
    density_kg_m3: float
    vp_m_s: float
    vs_m_s: float
    wave_type: Literal["P", "SV"]
    frequencies_hz: tuple[float, ...]
    _displacement_bytes: bytes
    displacement_name: str | None
    displacement_channel: str | None
    displacement_epoch_gps: tuple[int, int] | None
    displacement_unit: u.UnitBase
    displacement_input_unit: u.UnitBase
    representation: PhasorRepresentation
    reference_time_gps: tuple[int, int]
    _phase_origin_tuple_m: tuple[float, float, float]
    incidence_angle_rad: float
    azimuth_rad: float
    field_definition: GravityFieldDefinition
    interaction: Newtonian
    g_convention: str

    def __init__(
        self,
        geometry: HalfSpace,
        density: u.Quantity,
        vp: u.Quantity,
        vs: u.Quantity,
        displacement: FrequencySeries,
        representation: PhasorRepresentation,
        reference_time: object,
        wave_type: Literal["P", "SV"],
        incidence_angle_rad: float,
        azimuth_rad: float,
        phase_origin: u.Quantity,
    ) -> None:
        if not isinstance(geometry, HalfSpace):
            raise TypeError("geometry must be a HalfSpace")
        rho = _density_si(density)
        vp_si = _scalar_si("vp", vp, u.m / u.s)
        vs_si = _scalar_si("vs", vs, u.m / u.s)
        vp_si, vs_si = _validate_speeds(vp_si, vs_si)
        if not isinstance(displacement, FrequencySeries):
            raise TypeError("displacement must be a GWexpy FrequencySeries")
        frequencies = _frequencies_hz(displacement.frequencies)
        amplitudes = np.asarray(displacement.to_value(u.m), dtype=np.complex128)
        if amplitudes.shape != frequencies.shape or not np.isfinite(amplitudes).all():
            raise ValueError(
                "displacement must have finite amplitudes matching frequencies"
            )
        if not isinstance(representation, PhasorRepresentation):
            raise TypeError("representation must be a PhasorRepresentation")
        reference_gps = normalize_gps(reference_time)
        origin = _lengths_m("phase_origin", phase_origin, (3,))
        theta = _validate_incidence_angle("incidence_angle_rad", incidence_angle_rad)
        azimuth = _validate_azimuth("azimuth_rad", azimuth_rad)
        reflection = halfspace_psv_reflection(
            wave_type=wave_type,
            v_p_m_s=vp_si,
            v_s_m_s=vs_si,
            incidence_angle_rad=theta,
        )

        # Keep the potential convention no broader than the D1-supported
        # normal-incidence family. D1's continuous-q limit establishes the
        # convention for positive-frequency P/SV waves with traction-free
        # B=-A, D=-C; all q>0 cases use its Fourier Green matching conditions.
        ks = 2 * np.pi * frequencies / vs_si
        horizontal_wavenumbers = ks * reflection.q_over_ks
        if not np.isfinite(horizontal_wavenumbers).all():
            raise ValueError("derived horizontal wavenumbers must be finite")
        if theta == 0.0:
            vertical_ratio = (
                reflection.p_over_ks if wave_type == "P" else reflection.s_over_ks
            )
            vertical_wavenumbers = ks * vertical_ratio
            vertical_wavenumbers = np.asarray(vertical_wavenumbers)
            normal_family = (
                np.all(horizontal_wavenumbers == 0.0)
                and np.all(np.imag(vertical_wavenumbers) == 0.0)
                and np.all(np.real(vertical_wavenumbers) > 0.0)
                and np.isfinite(vertical_wavenumbers).all()
                and (
                    reflection.reflected_p_potential_ratio == -1 + 0j
                    and reflection.reflected_sv_potential_ratio == 0j
                    if wave_type == "P"
                    else reflection.reflected_p_potential_ratio == 0j
                    and reflection.reflected_sv_potential_ratio == -1 + 0j
                )
            )
            if not normal_family:
                raise ValueError(
                    "q=0 is supported only for the positive-frequency normal-"
                    "incidence traction-free P/SV family"
                )
            potential_reference = PotentialReference.SPHERICAL_MEAN_ZERO_AT_INFINITY
        else:
            if np.any(horizontal_wavenumbers <= 0.0):
                raise ValueError(
                    "positive incidence requires positive derived "
                    "horizontal wavenumbers"
                )
            potential_reference = PotentialReference.HALFSPACE_FOURIER_GREEN_POSITIVE_Q

        epoch = (
            None if displacement.epoch is None else normalize_gps(displacement.epoch)
        )
        for name, value in (
            ("geometry", geometry),
            ("density_kg_m3", rho),
            ("vp_m_s", vp_si),
            ("vs_m_s", vs_si),
            ("wave_type", wave_type),
            ("frequencies_hz", tuple(float(value) for value in frequencies)),
            ("_displacement_bytes", amplitudes.copy().tobytes()),
            ("displacement_name", displacement.name),
            (
                "displacement_channel",
                None if displacement.channel is None else str(displacement.channel),
            ),
            ("displacement_epoch_gps", epoch),
            ("displacement_unit", u.m),
            ("displacement_input_unit", displacement.unit),
            ("representation", representation),
            ("reference_time_gps", reference_gps),
            ("_phase_origin_tuple_m", tuple(float(value) for value in origin)),
            ("incidence_angle_rad", theta),
            ("azimuth_rad", azimuth),
            ("field_definition", _field_definition(potential_reference)),
            ("interaction", Newtonian()),
            ("g_convention", "SI_G_applied_once"),
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
        epoch = (
            None
            if self.displacement_epoch_gps is None
            else LIGOTimeGPS(*self.displacement_epoch_gps)
        )
        return FrequencySeries(
            self.displacement_m.copy(),
            frequencies=np.asarray(self.frequencies_hz) * u.Hz,
            unit=self.displacement_unit,
            name=self.displacement_name,
            channel=self.displacement_channel,
            epoch=epoch,
        )

    @property
    def phase_origin_m(self) -> np.ndarray:
        return np.asarray(self._phase_origin_tuple_m)


@dataclass(frozen=True, eq=False)
class _HalfspacePSVModel:
    """Half-space geometry and owned input spectrum."""

    spec: _HalfspacePSVModelSpec

    @property
    def geometry(self) -> HalfSpace:
        return self.spec.geometry

    def at(
        self,
        targets: TargetPoints,
        *,
        output_frame: Frame | None = None,
    ) -> _HalfspacePSVEvaluationSetup:
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        resolved = _HalfspacePSVResolvedEvaluation(
            self,
            targets,
            targets.frame if output_frame is None else output_frame,
        )
        return _HalfspacePSVEvaluationSetup(resolved)


@dataclass(frozen=True, eq=False, init=False)
class _HalfspacePSVResolvedEvaluation:
    """One target request with coordinates resolved in the surface frame."""

    model: _HalfspacePSVModel
    model_spec: _HalfspacePSVModelSpec
    targets: TargetPoints
    computation_frame: Frame
    output_frame: Frame
    observables: tuple[str, ...]
    target_traces: tuple[PointTrace, ...]
    _target_positions_bytes: bytes

    def __init__(
        self, model: _HalfspacePSVModel, targets: TargetPoints, output_frame: Frame
    ) -> None:
        if type(model.spec) is not _HalfspacePSVModelSpec:
            raise TypeError("model.spec must be a HalfspacePSVModelSpec")
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be TargetPoints")
        if not len(targets):
            raise ValueError("targets must be nonempty")
        if not isinstance(output_frame, Frame):
            raise TypeError("output_frame must be a Frame")
        computation_frame = model.spec.geometry.frame
        computation_frame._path_to(output_frame)
        traces = tuple(
            targets.frame.trace_point_m(point, computation_frame)
            for point in targets.positions_m
        )
        positions = np.asarray([trace.point_m for trace in traces], dtype=np.float64)
        if not traces:
            positions = positions.reshape(0, 3)
        for name, value in (
            ("model", model),
            ("model_spec", model.spec),
            ("targets", targets),
            ("computation_frame", computation_frame),
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
        return QTable(
            {
                "wave_type": [spec.wave_type],
                "density": [spec.density],
                "vp": [spec.vp_m_s * u.m / u.s],
                "vs": [spec.vs_m_s * u.m / u.s],
                "incidence_angle": [spec.incidence_angle_rad * u.rad],
                "azimuth": [spec.azimuth_rad * u.rad],
                "frequency_min": [min(spec.frequencies_hz) * u.Hz],
                "frequency_max": [max(spec.frequencies_hz) * u.Hz],
                "frequency_count": [len(spec.frequencies_hz)],
                "displacement_max": [float(np.max(np.abs(spec.displacement_m))) * u.m],
                "phase_origin_x": [spec.phase_origin_m[0] * u.m],
                "phase_origin_y": [spec.phase_origin_m[1] * u.m],
                "phase_origin_z": [spec.phase_origin_m[2] * u.m],
                "reference_gps_seconds": [spec.reference_time_gps[0]],
                "reference_gps_nanoseconds": [spec.reference_time_gps[1]],
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
                        "D1 P/SV half-space potential and analytic derivatives"
                    ),
                    "convention": (
                        "surface-frame incidence and azimuth; fixed phase origin; "
                        "total bulk plus surface; G applied once"
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
                        "gragra.analytic.halfspace_psv.halfspace_psv_observables_g_free"
                    ),
                    "detail": (
                        "D1 traction-free P/SV half-space; Fourier matching for q>0; "
                        "documented continuous-q normal-incidence family for q=0"
                    ),
                }
            ]
        )


@dataclass(frozen=True)
class _HalfspacePSVValidationReport:
    valid: bool
    observables: tuple[str, ...]
    target_count: int
    target_regions: Mapping[str, str]


def _native_matrices(
    arrays: Mapping[str, np.ndarray],
    target_ids: tuple[str, ...],
    frequencies_hz: tuple[float, ...],
) -> dict[str, object]:
    count, bins = len(target_ids), len(frequencies_hz)
    result = {}
    for observable, components in _COMPONENTS.items():
        raw = np.asarray(arrays[observable], dtype=np.complex128)
        expected = (count, len(components), bins)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(
                f"{observable} must have finite phasor shape {expected}, "
                f"got {raw.shape}"
            )
        names = np.asarray(
            [
                [f"{target} {component} {observable}" for component in components]
                for target in target_ids
            ],
            dtype=object,
        )
        channels = np.asarray(
            [
                [
                    f"GRAGRA:{observable}:{row}:{column}"
                    for column, _ in enumerate(components)
                ]
                for row, _ in enumerate(target_ids)
            ],
            dtype=object,
        )
        from gwexpy.frequencyseries import FrequencySeriesMatrix

        result[observable] = FrequencySeriesMatrix(
            raw.copy(),
            frequencies=np.asarray(frequencies_hz) * u.Hz,
            units=np.full((count, len(components)), _UNITS[observable], dtype=object),
            names=names,
            channels=channels,
            rows=list(target_ids),
            cols=list(components),
            name=f"Half-space PSV {observable}",
        )
    return result


def _generation_record() -> EvaluationRecord:
    return EvaluationRecord(
        backend="halfspace_psv_g_free",
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


class _HalfspacePSVEvaluationSetup:
    __slots__ = ("_resolved", "_context")

    def __init__(self, resolved: _HalfspacePSVResolvedEvaluation) -> None:
        if not isinstance(resolved, _HalfspacePSVResolvedEvaluation):
            raise TypeError("resolved must be a HalfspacePSVResolvedEvaluation")
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
    def resolved(self) -> _HalfspacePSVResolvedEvaluation:
        return self._resolved

    @property
    def evaluation(self) -> _HalfspacePSVResolvedEvaluation:
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

    def validate(self) -> _HalfspacePSVValidationReport:
        regions = {}
        for target_id, point in zip(
            self._resolved.targets.ids,
            self._resolved.target_positions_m,
            strict=True,
        ):
            z = float(point[2])
            if z == 0.0:
                raise ValueError(
                    f"target ID={target_id!r} lies exactly on the free surface z == 0"
                )
            regions[target_id] = "above" if z > 0.0 else "below"
        return _HalfspacePSVValidationReport(
            True,
            OBSERVABLES,
            len(self._resolved.targets),
            MappingProxyType(regions),
        )

    def run(self):
        """Evaluate G-free native values, rotate components, and apply G once."""
        self.validate()
        resolved = self._resolved
        spec = resolved.model_spec
        values = halfspace_psv_observables_g_free(
            resolved.target_positions_m,
            wave_type=spec.wave_type,
            density_kg_m3=spec.density_kg_m3,
            v_p_m_s=spec.vp_m_s,
            v_s_m_s=spec.vs_m_s,
            frequency_hz=np.asarray(spec.frequencies_hz),
            incidence_angle_rad=spec.incidence_angle_rad,
            azimuth_rad=spec.azimuth_rad,
            incident_displacement_m=spec.displacement_m,
            phase_origin_m=spec.phase_origin_m,
        )
        rotation = resolved.computation_frame.rotation_to(resolved.output_frame)
        arrays = {
            "potential": G_SI
            * np.asarray(values.potential, dtype=np.complex128).T[:, None, :],
            "acceleration": G_SI
            * np.einsum("ij,kmj->kmi", rotation, values.acceleration).transpose(
                1, 2, 0
            ),
            "gradient": G_SI
            * np.einsum("ia,kmab,jb->kmij", rotation, values.gradient, rotation)
            .reshape(len(spec.frequencies_hz), len(resolved.targets), 9)
            .transpose(1, 2, 0),
        }
        from gragra.results import GravityResult
        from gragra.results._evaluation import EvaluationResult

        matrices = _native_matrices(arrays, resolved.targets.ids, spec.frequencies_hz)
        return EvaluationResult(
            values=GravityResult(**matrices),
            evaluation=self,
            record=_generation_record(),
        )


def halfspace_psv(
    geometry: HalfSpace,
    density: u.Quantity,
    vp: u.Quantity,
    vs: u.Quantity,
    displacement: FrequencySeries,
    representation: PhasorRepresentation,
    reference_time: object,
    wave_type: Literal["P", "SV"],
    incidence_angle_rad: float,
    azimuth_rad: float,
    phase_origin: u.Quantity,
) -> _HalfspacePSVModel:
    """Create a P/SV half-space phasor model with explicit phase reference.

    ``displacement`` is an incident displacement ``FrequencySeries`` in meters
    (or any length unit). ``reference_time`` is required and defines temporal
    phasor interpretation; ``displacement.epoch`` is retained as provenance
    and never triggers an implicit rephase. ``phase_origin`` is a position in
    ``geometry.frame``. Incidence and polarization use that same local basis,
    with azimuth measured from local +x toward local +y.
    """
    return _HalfspacePSVModel(
        _HalfspacePSVModelSpec(
            geometry=geometry,
            density=density,
            vp=vp,
            vs=vs,
            displacement=displacement,
            representation=representation,
            reference_time=reference_time,
            wave_type=wave_type,
            incidence_angle_rad=incidence_angle_rad,
            azimuth_rad=azimuth_rad,
            phase_origin=phase_origin,
        )
    )


__all__ = ["halfspace_psv"]
