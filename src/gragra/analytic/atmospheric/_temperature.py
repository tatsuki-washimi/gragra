"""Analytic gravity responses to isobaric temperature perturbations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from scipy.special import erf

from gragra._analytic_evaluation import AnalyticEvaluation, NativeEvaluation
from gragra._arrays import _check_no_bool
from gragra._interactions import Newtonian
from gragra.analytic.cavity_cuboid import _is_on_cuboid_boundary
from gragra.analytic.cavity_cylinder import _reject_cylinder_boundary_targets
from gragra.analytic.cylinder_potential import cylinder_analytic_observables_g_free
from gragra.analytic.uniform_sphere import uniform_sphere_observables_g_free
from gragra.fields._temperature import TemperatureFourierMode, TemperatureGaussian
from gragra.geometry._cuboid import Cuboid
from gragra.geometry._cylinder import Cylinder
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.results import GravityFieldDefinition
from gragra.results._context import EvaluationContext
from gragra.results._coords import normalize_gps
from gragra.results._gen1_definitions import gen1_field_definition
from gragra.results._representation import StaticRepresentation
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints

_TemperatureProfile = TemperatureFourierMode | TemperatureGaussian
_FiniteProfile = Sphere | Cuboid | Cylinder
_Profile = _TemperatureProfile | _FiniteProfile
_OBSERVABLES = ("potential", "acceleration", "gradient")


def _scalar_quantity(
    name: str,
    value: object,
    unit: u.UnitBase,
    *,
    positive: bool = False,
    allow_complex: bool = False,
) -> float | complex:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    _check_no_bool(value, name)
    try:
        raw = np.asarray(value.to_value(unit))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise u.UnitConversionError(f"{name} must convert to {unit}") from exc
    if raw.shape != ():
        raise ValueError(f"{name} must be a scalar Quantity")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if np.iscomplexobj(raw):
        if not allow_complex:
            raise ValueError(f"{name} must be real")
        result: float | complex = complex(raw.item())
    else:
        result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _frame_of(profile: _Profile) -> Frame:
    return profile.frame


def _kind_of(profile: _Profile) -> str:
    if isinstance(profile, TemperatureFourierMode):
        return "plane_wave"
    if isinstance(profile, TemperatureGaussian):
        return "gaussian"
    if isinstance(profile, Sphere):
        return "sphere"
    if isinstance(profile, Cuboid):
        return "cuboid"
    if isinstance(profile, Cylinder):
        return "cylinder"
    raise TypeError("profile must be a Temperature profile or finite geometry")


@dataclass(frozen=True, eq=False, slots=True, init=False)
class TemperatureModelSpec:
    """Owned thermodynamic and spatial assumptions for one temperature model."""

    profile: _Profile
    delta_temperature_k: float | complex
    rho0_kg_m3: float
    T0_k: float
    frequency_hz: float | None
    reference_time_gps: tuple[int, int] | None
    field_definition: GravityFieldDefinition
    interaction: Newtonian
    _parameters: QTable
    _formulas: pd.DataFrame
    _references: pd.DataFrame

    def __init__(
        self,
        profile: _Profile,
        delta_temperature: u.Quantity,
        *,
        rho0: u.Quantity,
        T0: u.Quantity,  # noqa: N803 - public Gen1 spelling
        frequency: u.Quantity | None,
        reference_time: object | None,
    ) -> None:
        kind = _kind_of(profile)
        delta = _scalar_quantity(
            "delta_temperature",
            delta_temperature,
            u.K,
            allow_complex=frequency is not None,
        )
        density = float(_scalar_quantity("rho0", rho0, u.kg / u.m**3, positive=True))
        background_temperature = float(_scalar_quantity("T0", T0, u.K, positive=True))
        if frequency is None:
            if reference_time is not None:
                raise ValueError("reference_time requires frequency")
            frequency_hz = None
            time_gps = None
        else:
            frequency_hz = float(
                _scalar_quantity("frequency", frequency, u.Hz, positive=True)
            )
            if reference_time is None:
                raise ValueError("reference_time is required for a phasor temperature")
            time_gps = normalize_gps(reference_time)

        rho_amplitude = -density * delta / background_temperature
        frame = _frame_of(profile)
        parameters: dict[str, object] = {
            "profile": [kind],
            "delta_temperature": [delta * u.K],
            "rho0": [density * u.kg / u.m**3],
            "T0": [background_temperature * u.K],
            "delta_density_amplitude": [rho_amplitude * u.kg / u.m**3],
            "support": [
                "all_space"
                if isinstance(profile, _TemperatureProfile)
                else "fixed_geometry"
            ],
            "source_frame": [frame.name],
        }
        if isinstance(profile, TemperatureFourierMode):
            parameters["wavevector"] = [profile.wavevector_rad_m * u.rad / u.m]
            parameters["phase_origin"] = [profile.phase_origin_m * u.m]
        elif isinstance(profile, TemperatureGaussian):
            parameters["center"] = [profile.center_m * u.m]
            parameters["sigma"] = [profile.sigma]
        elif isinstance(profile, Sphere):
            parameters["center"] = [profile.center_m * u.m]
            parameters["radius"] = [profile.radius]
        elif isinstance(profile, Cuboid):
            parameters["center"] = [profile.center_m * u.m]
            parameters["lengths"] = [profile.lengths]
        else:
            parameters["center"] = [profile.center_m * u.m]
            parameters["radius"] = [profile.radius]
            parameters["height"] = [profile.height]
        if frequency_hz is not None:
            parameters["frequency"] = [frequency_hz * u.Hz]
            assert time_gps is not None
            parameters["reference_time_gps_seconds"] = [time_gps[0]]
            parameters["reference_time_gps_nanoseconds"] = [time_gps[1]]

        definition = gen1_field_definition(
            "temperature",
            periodic_temperature=isinstance(profile, TemperatureFourierMode),
        )
        formulas = pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expression,
                    "convention": convention,
                }
                for name, expression, convention in self._formula_rows(
                    kind, isinstance(profile, TemperatureFourierMode)
                )
            ]
        )
        references = pd.DataFrame(
            [
                {
                    "source": "docs/design/gen1/formula-ledger.md#temperature",
                    "detail": (
                        "approved isobaric, plane-wave, Gaussian, and "
                        "finite-body formulas"
                    ),
                },
                {
                    "source": (
                        "gragra.fields.thermodynamic."
                        "isobaric_temperature_density_perturbation"
                    ),
                    "detail": "signed linearization delta_rho = -rho0*delta_T/T0",
                },
                {
                    "source": "gragra.constants.G_SI",
                    "detail": "G applied exactly once by the shared analytic bridge",
                },
            ]
        )
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "delta_temperature_k", delta)
        object.__setattr__(self, "rho0_kg_m3", density)
        object.__setattr__(self, "T0_k", background_temperature)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "reference_time_gps", time_gps)
        object.__setattr__(self, "field_definition", definition)
        object.__setattr__(self, "interaction", Newtonian())
        object.__setattr__(self, "_parameters", QTable(parameters))
        object.__setattr__(self, "_formulas", formulas)
        object.__setattr__(self, "_references", references)

    @staticmethod
    def _formula_rows(kind: str, periodic: bool) -> tuple[tuple[str, str, str], ...]:
        if periodic:
            expressions = (
                (
                    "potential",
                    "Phi=-4*pi*delta_rho_hat/|k|^2 * exp(i*k dot (x-origin))",
                ),
                ("acceleration", "a=-i*k*Phi; Static is real cosine reconstruction"),
                ("gradient", "Gamma=(k outer k)*Phi; row-major"),
            )
            convention = "G-free periodic zero-mean field; no half-amplitude factor"
        elif kind == "gaussian":
            expressions = (
                ("potential", "Phi=-M*erf(r/(sqrt(2)*sigma))/r"),
                ("acceleration", "a_r=-M*H/r^2; stable center series"),
                (
                    "gradient",
                    "radial/tangential Gaussian derivatives; stable center series",
                ),
            )
            convention = (
                "G-free Gaussian closed form with analytic center and far limits"
            )
        else:
            expressions = (
                (
                    "potential",
                    f"{kind} finite uniform-density potential; zero at infinity",
                ),
                (
                    "acceleration",
                    "negative spatial gradient of the finite-body potential",
                ),
                ("gradient", "spatial derivative of acceleration"),
            )
            convention = (
                "unit-density G-free analytic source scaled by signed delta_rho"
            )
        return tuple((name, expr, convention) for name, expr in expressions)

    @property
    def parameters(self) -> QTable:
        return self._parameters.copy(copy_data=True)

    @property
    def formulas(self) -> pd.DataFrame:
        return self._formulas.copy(deep=True)

    @property
    def references(self) -> pd.DataFrame:
        return self._references.copy(deep=True)


@dataclass(frozen=True, eq=False, slots=True)
class TemperatureModel:
    """Single static or single-frequency temperature perturbation model."""

    spec: TemperatureModelSpec

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = _OBSERVABLES,
        output_frame: Frame | None = None,
    ) -> AnalyticEvaluation:
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be high-level TargetPoints")
        computation_frame = _frame_of(self.spec.profile)
        output = targets.frame if output_frame is None else output_frame
        if not isinstance(output, Frame):
            raise TypeError("output_frame must be a Frame")
        positions = targets.frame.transform_points_m(
            targets.positions_m, computation_frame
        )
        if self.spec.frequency_hz is None:
            representation = StaticRepresentation()
            frequencies = None
            reference_time = None
        else:
            representation = PhasorRepresentation()
            frequencies = np.array([self.spec.frequency_hz]) * u.Hz
            reference_time = self.spec.reference_time_gps
        context = EvaluationContext(
            model_spec=self.spec,
            targets=targets,
            frame=output,
            observables=observables,
            representation=representation,
            frequencies=frequencies,
            reference_time=reference_time,
        )

        def validate_targets(target_positions_m: np.ndarray) -> None:
            _reject_profile_boundaries(self.spec.profile, target_positions_m)

        def evaluate_native(
            target_positions_m: np.ndarray, names: tuple[str, ...]
        ) -> NativeEvaluation:
            values = _temperature_native_values(self.spec, target_positions_m, names)
            return NativeEvaluation(values)

        return AnalyticEvaluation(
            context=context,
            computation_frame=computation_frame,
            target_positions_m=positions,
            evaluate_native=evaluate_native,
            validate_targets=validate_targets,
        )


def _reject_profile_boundaries(profile: _Profile, points_m: np.ndarray) -> None:
    if len(points_m) == 0 or isinstance(
        profile, (TemperatureFourierMode, TemperatureGaussian)
    ):
        return
    if isinstance(profile, Sphere):
        relative = points_m - profile.center_m
        distance = np.linalg.norm(relative, axis=1)
        tolerance = 1e-12 * profile.radius_m
        boundary = np.abs(distance - profile.radius_m) <= tolerance
    elif isinstance(profile, Cuboid):
        boundary = _is_on_cuboid_boundary(
            points_m - profile.center_m, profile.lengths_m / 2.0
        )
    else:
        _reject_cylinder_boundary_targets(
            points_m - profile.center_m, profile.radius_m, profile.height_m
        )
        return
    if np.any(boundary):
        index = int(np.flatnonzero(boundary)[0])
        shape = "sphere" if isinstance(profile, Sphere) else "cuboid"
        raise ValueError(
            f"target point at index {index} lies on the physical {shape} boundary"
        )


def _plane_wave_values(
    spec: TemperatureModelSpec, points_m: np.ndarray
) -> Mapping[str, np.ndarray]:
    profile = spec.profile
    assert isinstance(profile, TemperatureFourierMode)
    wavevector = profile.wavevector_rad_m
    norm_squared = float(wavevector @ wavevector)
    rho_hat = -spec.rho0_kg_m3 * spec.delta_temperature_k / spec.T0_k
    phase = (points_m - profile.phase_origin_m) @ wavevector
    coefficient = -4.0 * np.pi * rho_hat / norm_squared
    phi_phasor = coefficient * np.exp(1j * phase)
    acceleration_phasor = -1j * phi_phasor[:, None] * wavevector[None, :]
    gradient_phasor = (
        phi_phasor[:, None, None]
        * wavevector[None, :, None]
        * wavevector[None, None, :]
    )
    is_static = spec.frequency_hz is None
    potential = np.real(phi_phasor) if is_static else phi_phasor
    acceleration = np.real(acceleration_phasor) if is_static else acceleration_phasor
    gradient = np.real(gradient_phasor) if is_static else gradient_phasor
    if is_static:
        return MappingProxyType(
            {
                "potential": potential,
                "acceleration": acceleration,
                "gradient": gradient,
            }
        )
    return MappingProxyType(
        {
            "potential": potential[:, None, None],
            "acceleration": acceleration[:, :, None],
            "gradient": gradient.reshape(len(points_m), 9)[:, :, None],
        }
    )


def _gaussian_values(
    spec: TemperatureModelSpec, points_m: np.ndarray
) -> Mapping[str, np.ndarray]:
    profile = spec.profile
    assert isinstance(profile, TemperatureGaussian)
    relative = points_m - profile.center_m
    radius = np.hypot.reduce(np.abs(relative), axis=1)
    sigma = profile.sigma_m
    rho_c = -spec.rho0_kg_m3 * spec.delta_temperature_k / spec.T0_k
    dtype = np.complex128 if np.iscomplexobj(rho_c) else np.float64
    potential = np.empty(len(radius), dtype=dtype)
    acceleration = np.zeros((len(radius), 3), dtype=dtype)
    gradient = np.empty((len(radius), 3, 3), dtype=dtype)
    t_all = radius / sigma
    center_region = t_all <= 0.5
    far_or_general = ~center_region
    if np.any(center_region):
        r = radius[center_region]
        t = t_all[center_region]
        t2 = t**2
        phi_shape, h_shape = _gaussian_center_series(t)
        density_scale = 4.0 * np.pi * rho_c
        potential[center_region] = -density_scale * sigma**2 * phi_shape
        acceleration_factor = -density_scale * h_shape
        acceleration[center_region] = (
            relative[center_region] * acceleration_factor[:, None]
        )
        tangential = -density_scale * h_shape
        radial = density_scale * (2.0 * h_shape - np.exp(-0.5 * t2))
        gradient[center_region] = tangential[:, None, None] * np.eye(3)
        nonzero = r > 0.0
        if np.any(nonzero):
            unit = relative[center_region][nonzero] / r[nonzero, None]
            difference = radial[nonzero] - tangential[nonzero]
            gradient_center = gradient[center_region]
            gradient_center[nonzero] += difference[:, None, None] * (
                unit[:, :, None] * unit[:, None, :]
            )
            gradient[center_region] = gradient_center
    if np.any(far_or_general):
        r = radius[far_or_general]
        t = t_all[far_or_general]
        exp_term = np.exp(-0.5 * t**2)
        erf_term = erf(t / np.sqrt(2.0))
        h = erf_term - np.sqrt(2.0 / np.pi) * t * exp_term
        inverse_t = 1.0 / t
        phi_shape = np.sqrt(np.pi / 2.0) * erf_term * inverse_t
        acceleration_shape = np.sqrt(np.pi / 2.0) * h * inverse_t**2
        h_shape = np.sqrt(np.pi / 2.0) * h * inverse_t**3
        density_scale = 4.0 * np.pi * rho_c
        phi = -density_scale * sigma**2 * phi_shape
        acceleration_factor = -density_scale * sigma * acceleration_shape
        gamma_radial = density_scale * (2.0 * h_shape - exp_term)
        gamma_tangential = -density_scale * h_shape
        unit = relative[far_or_general] / r[:, None]
        potential[far_or_general] = phi
        acceleration[far_or_general] = unit * acceleration_factor[:, None]
        gradient[far_or_general] = gamma_tangential[:, None, None] * np.eye(3)
        difference = gamma_radial - gamma_tangential
        gradient[far_or_general] += difference[:, None, None] * (
            unit[:, :, None] * unit[:, None, :]
        )
    if spec.frequency_hz is None:
        return MappingProxyType(
            {
                "potential": potential.real,
                "acceleration": acceleration.real,
                "gradient": gradient.real,
            }
        )
    return MappingProxyType(
        {
            "potential": potential[:, None, None],
            "acceleration": acceleration[:, :, None],
            "gradient": gradient.reshape(len(points_m), 9)[:, :, None],
        }
    )


def _gaussian_center_series(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized erf(u)/t and H(t)/t^3 from center expansions.

    The outputs are sqrt(pi/2)*erf(t/sqrt(2))/t and
    sqrt(pi/2)*H(t)/t^3. The branch is restricted to |t| <= 0.5 and retains
    25 terms (the leading term plus 24 recurrence updates). The Taylor
    truncation contribution is below 1e-48 relative to the leading term; this
    bound excludes float64 roundoff. The series avoids subtracting nearly equal
    erf and exponential terms in H near the Gaussian center.
    """
    t2 = t**2
    phi_shape = np.ones_like(t)
    phi_term = np.ones_like(t)
    h_shape = np.full_like(t, 1.0 / 3.0)
    h_term = np.full_like(t, 1.0 / 3.0)
    for index in range(1, 25):
        phi_term *= -t2 * (2 * index - 1) / (2 * index * (2 * index + 1))
        phi_shape += phi_term
        h_term *= -t2 * (2 * index + 1) / (2 * index * (2 * index + 3))
        h_shape += h_term
    return phi_shape, h_shape


def _finite_body_values(
    spec: TemperatureModelSpec, points_m: np.ndarray
) -> Mapping[str, np.ndarray]:
    profile = spec.profile
    rho_amplitude = -spec.rho0_kg_m3 * spec.delta_temperature_k / spec.T0_k
    if isinstance(profile, Sphere):
        unit_values = uniform_sphere_observables_g_free(
            profile.center_m, profile.radius_m, 1.0, points_m
        )
    elif isinstance(profile, Cuboid):
        from gragra.analytic.cavity_cuboid import cuboid_observables_g_free

        unit_values = cuboid_observables_g_free(
            profile.center_m, profile.lengths_m, 1.0, points_m
        )
    else:
        unit_values = cylinder_analytic_observables_g_free(
            profile.center_m,
            profile.radius_m,
            profile.height_m,
            1.0,
            points_m,
        )
    values = {
        name: np.asarray(getattr(unit_values, name)) * rho_amplitude
        for name in _OBSERVABLES
    }
    if spec.frequency_hz is None:
        return MappingProxyType({name: np.real(values[name]) for name in _OBSERVABLES})
    return MappingProxyType(
        {
            "potential": values["potential"][:, None, None],
            "acceleration": values["acceleration"][:, :, None],
            "gradient": values["gradient"].reshape(len(points_m), 9)[:, :, None],
        }
    )


def _temperature_native_values(
    spec: TemperatureModelSpec,
    points_m: np.ndarray,
    observables: tuple[str, ...],
) -> Mapping[str, np.ndarray]:
    if isinstance(spec.profile, TemperatureFourierMode):
        values = _plane_wave_values(spec, points_m)
    elif isinstance(spec.profile, TemperatureGaussian):
        values = _gaussian_values(spec, points_m)
    else:
        values = _finite_body_values(spec, points_m)
    if spec.frequency_hz is None:
        return MappingProxyType({name: values[name] for name in observables})
    return MappingProxyType(
        {name: np.asarray(values[name], dtype=np.complex128) for name in observables}
    )


def isobaric_temperature(
    profile: _Profile,
    delta_temperature: u.Quantity,
    *,
    rho0: u.Quantity,
    T0: u.Quantity,  # noqa: N803 - public Gen1 spelling
    frequency: u.Quantity | None = None,
    reference_time: object | None = None,
) -> TemperatureModel:
    """Create a static or single-frequency isobaric temperature model.

    ``profile`` is a TemperatureFourierMode, TemperatureGaussian, Sphere,
    Cuboid, or Cylinder. TemperaturePerturbationGrid is intentionally evaluated via
    the existing ``point_sum`` route and is not an analytic profile.
    """
    spec = TemperatureModelSpec(
        profile,
        delta_temperature,
        rho0=rho0,
        T0=T0,
        frequency=frequency,
        reference_time=reference_time,
    )
    return TemperatureModel(spec)


__all__ = [
    "TemperatureModelSpec",
    "TemperatureModel",
    "isobaric_temperature",
]
