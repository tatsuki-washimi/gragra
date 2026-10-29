"""High-level finite acoustic cavity model descriptors and evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from scipy.optimize import brentq
from scipy.special import hyp0f1, spherical_jn

from gragra._analytic_evaluation import AnalyticEvaluation, NativeEvaluation
from gragra._arrays import _check_no_bool
from gragra._interactions import Newtonian
from gragra.analytic.atmospheric._cavity._sphere_harmonics import (
    _solid_harmonic_values,
)
from gragra.geometry._cuboid import Cuboid
from gragra.geometry._cylinder import Cylinder
from gragra.geometry._frame import Frame
from gragra.geometry._sphere import Sphere
from gragra.results._context import EvaluationContext
from gragra.results._coords import normalize_gps
from gragra.results._gen1_definitions import gen1_field_definition
from gragra.results._spectral import PhasorRepresentation
from gragra.targets._high_points import TargetPoints

if TYPE_CHECKING:
    from gragra.analytic.atmospheric._cavity._cuboid import CuboidAcousticCavityModel
    from gragra.analytic.atmospheric._cavity._cylinder import (
        CylinderAcousticCavityModel,
    )


def _mode_index(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


@dataclass(frozen=True, slots=True)
class SphereAcousticMode:
    """Real spherical Neumann mode indices ``(l, m, n)``."""

    l: int  # noqa: E741 - public mode spelling follows the approved (l, m, n) API
    m: int
    n: int

    def __post_init__(self) -> None:
        degree_index = _mode_index("l", self.l, minimum=0)
        m = _mode_index("m", self.m, minimum=-degree_index)
        if m > degree_index:
            raise ValueError("m must satisfy -l <= m <= l")
        n = _mode_index("n", self.n, minimum=1)
        object.__setattr__(self, "l", degree_index)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "n", n)


@dataclass(frozen=True, slots=True)
class CuboidAcousticMode:
    """Nonnegative cuboid Neumann indices, excluding the constant mode."""

    nx: int
    ny: int
    nz: int

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            object.__setattr__(
                self, name, _mode_index(name, getattr(self, name), minimum=0)
            )
        if self.nx == self.ny == self.nz == 0:
            raise ValueError("the constant cuboid mode is not supported")


@dataclass(frozen=True, slots=True)
class CylinderAcousticMode:
    """Real cylindrical Neumann mode indices and azimuthal convention."""

    m: int
    n: int
    q: int
    azimuth: str = "cos"

    def __post_init__(self) -> None:
        object.__setattr__(self, "m", _mode_index("m", self.m, minimum=0))
        object.__setattr__(self, "n", _mode_index("n", self.n, minimum=0))
        object.__setattr__(self, "q", _mode_index("q", self.q, minimum=0))
        if self.n == 0 and (self.m != 0 or self.q == 0):
            raise ValueError("n=0 is valid only for the radial-constant m=0, q>0 mode")
        if self.azimuth not in {"cos", "sin"}:
            raise ValueError("azimuth must be 'cos' or 'sin'")
        if self.m == 0 and self.azimuth == "sin":
            raise ValueError("sin azimuth is invalid for m=0")


@dataclass(frozen=True, slots=True, kw_only=True)
class AcousticCavityNumerics:
    """Owned resource limits for cavity kernels that require integration."""

    max_evaluations: int = 200_000
    max_refinements: int = 12

    def __post_init__(self) -> None:
        for name in ("max_evaluations", "max_refinements"):
            value = _mode_index(name, getattr(self, name), minimum=1)
            object.__setattr__(self, name, value)


_CavityMode: TypeAlias = SphereAcousticMode | CuboidAcousticMode | CylinderAcousticMode
_CavityGeometry: TypeAlias = Sphere | Cuboid | Cylinder
_TargetValidator = Callable[[np.ndarray, np.ndarray], object]
_NativeEvaluator = Callable[[np.ndarray, tuple[str, ...]], NativeEvaluation]


def _transform_targets_to_cavity_frame(
    targets: TargetPoints, geometry: _CavityGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Return local coordinates and per-target roundoff from frame conversion.

    The error width is based on each point's own Frame trace. Same-frame
    coordinates retain exact boundary classification; cross-frame conversion
    carries only the local floating-point uncertainty for that point.
    """
    traces = [
        targets.frame.trace_point_m(point, geometry.frame)
        for point in targets.positions_m
    ]
    positions = (
        np.asarray([trace.point_m for trace in traces], dtype=np.float64)
        if traces
        else np.empty((0, 3), dtype=np.float64)
    )
    if isinstance(geometry, Sphere):
        characteristic_size = geometry.radius_m
    elif isinstance(geometry, Cuboid):
        characteristic_size = float(np.max(geometry.lengths_m)) / 2.0
    else:
        characteristic_size = max(geometry.radius_m, geometry.height_m / 2.0)
    center_scale = float(np.max(np.abs(geometry.center_m)))
    boundary_error = np.asarray(
        [
            0.0
            if targets.frame is geometry.frame
            else 64.0
            * np.finfo(np.float64).eps
            * (trace.edge_count + 1)
            * max(
                trace.scale_m,
                center_scale,
                characteristic_size,
                float(np.nextafter(0.0, 1.0)),
            )
            for trace in traces
        ],
        dtype=np.float64,
    )
    boundary_error.setflags(write=False)
    return positions, boundary_error


def _build_cavity_evaluation(
    model_spec: object,
    targets: TargetPoints,
    *,
    observables: tuple[str, ...],
    output_frame: Frame | None,
    evaluate_native: _NativeEvaluator,
    validate_targets: _TargetValidator | None = None,
) -> AnalyticEvaluation:
    """Build the common single-frequency cavity adapter around a native kernel.

    Shape ``at`` methods close over ``targets.ids`` in ``evaluate_native`` and
    keep their boundary rule in ``validate_targets``. The common builder owns
    target Frame conversion, phasor context, G-once bridge, and packaging.
    """
    if not isinstance(targets, TargetPoints):
        raise TypeError("targets must be high-level TargetPoints")
    geometry = getattr(model_spec, "geometry", None)
    if not isinstance(geometry, (Sphere, Cuboid, Cylinder)):
        raise TypeError("model_spec.geometry must be a supported cavity geometry")
    if not callable(evaluate_native):
        raise TypeError("evaluate_native must be callable")
    if validate_targets is not None and not callable(validate_targets):
        raise TypeError("validate_targets must be callable or None")
    output = targets.frame if output_frame is None else output_frame
    if not isinstance(output, Frame):
        raise TypeError("output_frame must be a Frame")
    positions, boundary_error = _transform_targets_to_cavity_frame(targets, geometry)
    context = EvaluationContext(
        model_spec=model_spec,
        targets=targets,
        frame=output,
        observables=observables,
        representation=PhasorRepresentation(),
        frequencies=np.array([float(model_spec.frequency_hz)]) * u.Hz,
        reference_time=model_spec.reference_time_gps,
    )

    def validate_with_frame_error(target_positions_m: np.ndarray) -> None:
        if validate_targets is not None:
            validate_targets(target_positions_m, boundary_error)

    return AnalyticEvaluation(
        context=context,
        computation_frame=geometry.frame,
        target_positions_m=positions,
        evaluate_native=evaluate_native,
        validate_targets=(
            validate_with_frame_error if validate_targets is not None else None
        ),
    )


def _pressure_amplitude(value: object) -> complex:
    if not isinstance(value, u.Quantity):
        raise TypeError("pressure_amplitude must be an Astropy Quantity")
    _check_no_bool(value, "pressure_amplitude")
    try:
        raw = np.asarray(value.to_value(u.Pa))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise u.UnitConversionError("pressure_amplitude must convert to Pa") from exc
    if raw.shape != ():
        raise ValueError("pressure_amplitude must be a scalar Quantity")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError("pressure_amplitude must be real or complex numeric")
    pressure = complex(raw.item())
    if not np.isfinite(pressure):
        raise ValueError("pressure_amplitude must be finite")
    return pressure


def _positive_scalar_quantity(name: str, value: object, unit: u.UnitBase) -> float:
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
    result = float(raw)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _potential_coefficient(
    pressure_pa: complex, sound_speed_m_s: float, normalization: float, k: float
) -> complex:
    """Compute ``4*pi*p*N/(c**2*k**2)`` without silent scale underflow."""
    component_scale = max(abs(pressure_pa.real), abs(pressure_pa.imag))
    if component_scale == 0.0:
        return 0.0j
    real_scaled = pressure_pa.real / component_scale
    imag_scaled = pressure_pa.imag / component_scale
    scaled_norm = math.hypot(real_scaled, imag_scaled)
    log_magnitude = (
        math.log(component_scale)
        + math.log(scaled_norm)
        + math.log(4.0 * math.pi)
        + math.log(normalization)
        - 2.0 * math.log(sound_speed_m_s)
        - 2.0 * math.log(k)
    )
    smallest = float(np.nextafter(0.0, 1.0))
    if log_magnitude < math.log(smallest):
        raise ValueError("nonzero pressure scale underflow in sphere coefficient")
    if log_magnitude > math.log(np.finfo(np.float64).max):
        raise ValueError("nonzero pressure scale overflow in sphere coefficient")
    scaled_magnitude = math.exp(log_magnitude)
    coefficient = complex(
        real_scaled / scaled_norm * scaled_magnitude,
        imag_scaled / scaled_norm * scaled_magnitude,
    )
    if coefficient == 0.0j or not np.isfinite(coefficient):
        raise ValueError(
            "nonzero pressure scale is not representable in sphere coefficient"
        )
    return coefficient


def _positive_derivative_root(degree_index: int, n: int) -> float:
    """Return the nth strictly positive root of spherical ``j_l'``.

    A turning-point start avoids scanning the exponentially small forbidden
    region for large l. Subsequent sign changes are isolated with a step below
    the minimum root spacing; there is no fixture-specific index ceiling.
    """
    degree_index = _mode_index("l", degree_index, minimum=0)
    n = _mode_index("n", n, minimum=1)
    try:
        degree = float(degree_index)
    except OverflowError as exc:
        raise ValueError("l is too large for floating-point root evaluation") from exc
    if not np.isfinite(degree):
        raise ValueError("l is too large for floating-point root evaluation")

    step = np.pi / 8.0
    left = max(1e-8, degree - 2.0 * np.cbrt(max(degree, 1.0)))

    def derivative(x: float) -> float:
        return float(spherical_jn(degree_index, x, derivative=True))

    f_left = derivative(left)
    if not np.isfinite(f_left):
        raise ValueError("spherical derivative root could not be evaluated")
    roots: list[float] = []
    while len(roots) < n:
        right = left + step
        if not right > left or not np.isfinite(right):
            raise ValueError(
                "spherical derivative root is unresolved at float precision"
            )
        f_right = derivative(right)
        if not np.isfinite(f_right):
            raise ValueError("spherical derivative root could not be evaluated")
        if f_left == 0.0:
            root = left
        elif f_right == 0.0:
            root = right
        elif np.signbit(f_left) != np.signbit(f_right):
            root = float(
                brentq(
                    derivative,
                    left,
                    right,
                    xtol=np.nextafter(0.0, 1.0),
                    rtol=8.0 * np.finfo(np.float64).eps,
                )
            )
        else:
            left, f_left = right, f_right
            continue
        roots.append(root)
        # Move off a found root while staying well inside its new-sign lobe.
        left = root + step / 4.0
        f_left = derivative(left)
        if not np.isfinite(f_left):
            raise ValueError("spherical derivative root could not be evaluated")
    root = roots[-1]
    if not np.isfinite(root) or root <= 0.0:
        raise ValueError("spherical derivative root must be finite and positive")
    return root


@dataclass(frozen=True, eq=False, slots=True, init=False)
class SphereCavitySpec:
    """Immutable physical snapshot for one sphere Neumann-mode response."""

    geometry: Sphere
    mode: SphereAcousticMode
    pressure_amplitude_pa: complex
    sound_speed_m_s: float
    potential_coefficient: complex
    reference_time_gps: tuple[int, int]
    root: float
    wavenumber_rad_m: float
    normalization: float
    frequency_hz: float
    numerics: AcousticCavityNumerics
    field_definition: object
    interaction: Newtonian
    parameters: QTable
    formulas: pd.DataFrame
    references: pd.DataFrame

    def __init__(
        self,
        geometry: Sphere,
        mode: SphereAcousticMode,
        pressure_amplitude: object,
        *,
        sound_speed: object,
        reference_time: object,
        numerics: AcousticCavityNumerics,
    ) -> None:
        pressure = _pressure_amplitude(pressure_amplitude)
        speed = _positive_scalar_quantity("sound_speed", sound_speed, u.m / u.s)
        if not isinstance(numerics, AcousticCavityNumerics):
            raise TypeError("numerics must be an AcousticCavityNumerics")
        reference_gps = normalize_gps(reference_time)

        root = _positive_derivative_root(mode.l, mode.n)
        radius = geometry.radius_m
        with np.errstate(
            over="ignore", under="ignore", invalid="ignore", divide="ignore"
        ):
            wavenumber = float(root / radius)
            root_bessel = float(spherical_jn(mode.l, root))
            root_gap = float((root - mode.l) * (root + mode.l) - mode.l)
            relative_gap = root_gap / root**2
            normalization = float(
                math.sqrt(8.0 * math.pi / 3.0)
                / (abs(root_bessel) * math.sqrt(relative_gap))
            )
            frequency = float(speed * wavenumber / (2.0 * np.pi))
        derived = np.asarray(
            [wavenumber, root_bessel, root_gap, normalization, frequency]
        )
        if (
            not np.isfinite(derived).all()
            or wavenumber <= 0.0
            or root_gap <= 0.0
            or normalization <= 0.0
            or frequency <= 0.0
        ):
            raise ValueError("sphere mode eigenvalue and normalization must be finite")
        potential_coefficient = _potential_coefficient(
            pressure, speed, normalization, wavenumber
        )

        definition = gen1_field_definition("acoustic_cavity")
        center = geometry.center_m
        parameters = QTable(
            {
                "geometry": ["sphere"],
                "center_x": [center[0] * u.m],
                "center_y": [center[1] * u.m],
                "center_z": [center[2] * u.m],
                "radius": [radius * u.m],
                "l": [mode.l],
                "m": [mode.m],
                "n": [mode.n],
                "positive_derivative_root_index": [mode.n],
                "root": [root * u.dimensionless_unscaled],
                "wavenumber": [wavenumber / u.m],
                "normalization": [normalization],
                "pressure_amplitude": [pressure * u.Pa],
                "pressure_convention": ["spatial-RMS peak phasor"],
                "sound_speed": [speed * u.m / u.s],
                "frequency": [frequency * u.Hz],
                "reference_time_gps_seconds": [reference_gps[0]],
                "reference_time_gps_nanoseconds": [reference_gps[1]],
                "numerical_route": ["closed_form_sphere_v1"],
                "accuracy_policy": ["gen1_cavity_v1"],
                "absolute_scale_factor": [1e-12],
                "relative_phi": [1e-10],
                "relative_acceleration": [1e-10],
                "relative_gradient": [1e-9],
                "runtime_budget_fraction": [0.25],
                "max_evaluations": [numerics.max_evaluations],
                "max_refinements": [numerics.max_refinements],
                "tolerance_policy": [
                    "gen1_cavity_v1 metadata; analytic stop test not applicable"
                ],
                "source_frame": [geometry.frame.name],
            }
        )
        formulas = pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expression,
                    "convention": (
                        "G-free closed-form sphere mode; bridge applies G once"
                    ),
                }
                for name, expression in (
                    ("potential", "Phi = C * f(r^2) * H_lm"),
                    ("acceleration", "a = -gradient(Phi)"),
                    ("gradient", "Gamma = -Hessian(Phi)"),
                )
            ]
        )
        references = pd.DataFrame(
            [
                {
                    "source": (
                        "docs/design/gen1/formula-ledger.md#sphere-acoustic-cavity"
                    ),
                    "detail": (
                        "approved eigenmode, RMS normalization, potential, and "
                        "Cartesian derivatives"
                    ),
                },
                {
                    "source": "gragra.constants.G_SI",
                    "detail": "applied exactly once by the shared analytic bridge",
                },
            ]
        )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "pressure_amplitude_pa", pressure)
        object.__setattr__(self, "sound_speed_m_s", speed)
        object.__setattr__(self, "potential_coefficient", potential_coefficient)
        object.__setattr__(self, "reference_time_gps", reference_gps)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "wavenumber_rad_m", wavenumber)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "numerics", numerics)
        object.__setattr__(self, "field_definition", definition)
        object.__setattr__(self, "interaction", Newtonian())
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "formulas", formulas)
        object.__setattr__(self, "references", references)


@dataclass(frozen=True, slots=True)
class SphereAcousticCavityModel:
    """Factory for a single-frequency sphere cavity field evaluation."""

    spec: SphereCavitySpec

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = ("potential", "acceleration", "gradient"),
        output_frame: Frame | None = None,
    ) -> AnalyticEvaluation:
        def validate_targets(
            target_positions_m: np.ndarray, boundary_error: np.ndarray
        ) -> None:
            if len(target_positions_m) == 0:
                return
            relative = target_positions_m - self.spec.geometry.center_m
            distances = np.hypot(
                np.hypot(relative[:, 0], relative[:, 1]), relative[:, 2]
            )
            boundary = np.abs(distances - self.spec.geometry.radius_m) <= boundary_error
            if np.any(boundary):
                first = int(np.flatnonzero(boundary)[0])
                raise ValueError(
                    f"target point at index {first} lies on the sphere boundary"
                )

        def evaluate_native(
            target_positions_m: np.ndarray, names: tuple[str, ...]
        ) -> NativeEvaluation:
            return NativeEvaluation(
                _sphere_native_values(self.spec, target_positions_m, names)
            )

        return _build_cavity_evaluation(
            self.spec,
            targets,
            observables=observables,
            output_frame=output_frame,
            evaluate_native=evaluate_native,
            validate_targets=validate_targets,
        )


def _radial_series(
    degree_index: int, k: float, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return j_l(k*r)/r**l and its first two derivatives in s=r**2."""
    log_double_factorial = (
        math.lgamma(2 * degree_index + 2)
        - degree_index * math.log(2.0)
        - math.lgamma(degree_index + 1)
    )
    log_base = degree_index * math.log(k) - log_double_factorial
    log_base_s = (degree_index + 2) * math.log(k) - log_double_factorial
    log_base_ss = (degree_index + 4) * math.log(k) - log_double_factorial
    try:
        base = math.exp(log_base)
        base_s = math.exp(log_base_s)
        base_ss = math.exp(log_base_ss)
    except OverflowError as exc:
        raise ValueError("sphere radial series is not representable") from exc
    if base == 0.0 or base_s == 0.0 or base_ss == 0.0:
        raise ValueError("nonzero pressure radial scale underflow in sphere series")
    b = degree_index + 1.5
    argument = -(k**2) * s / 4.0
    radial = base * hyp0f1(b, argument)
    radial_s = -base_s / (4.0 * b) * hyp0f1(b + 1.0, argument)
    radial_ss = base_ss / (16.0 * b * (b + 1.0)) * hyp0f1(b + 2.0, argument)
    values = tuple(
        np.asarray(item, dtype=np.float64) for item in (radial, radial_s, radial_ss)
    )
    if any(not np.isfinite(item).all() for item in values):
        raise ValueError("sphere radial series evaluation is not finite")
    return values  # type: ignore[return-value]


def _sphere_native_values(
    spec: SphereCavitySpec, targets_m: np.ndarray, observables: tuple[str, ...]
) -> dict[str, np.ndarray]:
    count = len(targets_m)
    shape = {
        "potential": (count, 1, 1),
        "acceleration": (count, 3, 1),
        "gradient": (count, 9, 1),
    }
    dtype = np.complex128
    values = {name: np.zeros(shape[name], dtype=dtype) for name in observables}
    if not count or spec.pressure_amplitude_pa == 0.0:
        return values

    relative = targets_m - spec.geometry.center_m
    distance = np.hypot(np.hypot(relative[:, 0], relative[:, 1]), relative[:, 2])
    inside = distance < spec.geometry.radius_m
    degree_index, m = spec.mode.l, spec.mode.m
    harmonic, harmonic_gradient, harmonic_hessian = _solid_harmonic_values(
        degree_index, m, relative
    )
    coefficient = spec.potential_coefficient
    f = np.zeros(count, dtype=np.complex128)
    f_s = np.zeros(count, dtype=np.complex128)
    f_ss = np.zeros(count, dtype=np.complex128)
    if np.any(inside):
        s = distance[inside] ** 2
        jreg, jreg_s, jreg_ss = _radial_series(degree_index, spec.wavenumber_rad_m, s)
        radius_power = math.exp(-degree_index * math.log(spec.geometry.radius_m))
        regular_constant = (
            (degree_index + 1)
            * spherical_jn(degree_index, spec.root)
            / (2 * degree_index + 1)
            * radius_power
        )
        if coefficient != 0.0 and radius_power == 0.0:
            raise ValueError("nonzero pressure radial scale underflow at sphere radius")
        f[inside] = coefficient * (-jreg + regular_constant)
        f_s[inside] = -coefficient * jreg_s
        f_ss[inside] = -coefficient * jreg_ss

    outside = ~inside
    if np.any(outside) and degree_index:
        radial_distance = distance[outside]
        log_factor = (degree_index + 1) * math.log(spec.geometry.radius_m) - (
            2 * degree_index + 1
        ) * np.log(radial_distance)
        try:
            factor = np.exp(log_factor)
        except FloatingPointError as exc:
            raise ValueError("exterior sphere field is not representable") from exc
        prefactor = (
            -coefficient
            * degree_index
            * spherical_jn(degree_index, spec.root)
            / (2 * degree_index + 1)
        )
        if coefficient != 0.0 and prefactor == 0.0:
            raise ValueError("nonzero pressure radial scale underflow outside sphere")
        if coefficient != 0.0 and np.any(factor == 0.0):
            raise ValueError("nonzero pressure exterior field underflow outside sphere")
        f[outside] = prefactor * factor
        power = degree_index + 0.5
        f_s[outside] = -(power / distance[outside] ** 2) * f[outside]
        f_ss[outside] = power * (power + 1.0) / distance[outside] ** 4 * f[outside]

    potential_native = f * harmonic
    acceleration_native = -(
        2.0 * f_s[:, None] * relative * harmonic[:, None]
        + f[:, None] * harmonic_gradient
    )
    identity = np.eye(3)
    gradient_native = -(
        (
            4.0 * f_ss[:, None, None] * relative[:, :, None] * relative[:, None, :]
            + 2.0 * f_s[:, None, None] * identity[None, :, :]
        )
        * harmonic[:, None, None]
        + 2.0
        * f_s[:, None, None]
        * (
            relative[:, :, None] * harmonic_gradient[:, None, :]
            + harmonic_gradient[:, :, None] * relative[:, None, :]
        )
        + f[:, None, None] * harmonic_hessian
    )

    if "potential" in values:
        values["potential"][:, 0, 0] = potential_native
    if "acceleration" in values:
        values["acceleration"][:, :, 0] = acceleration_native
    if "gradient" in values:
        values["gradient"][:, :, 0] = gradient_native.reshape(count, 9)
    if any(not np.isfinite(item).all() for item in values.values()):
        raise ValueError("sphere acoustic field is not finite")
    return values


def acoustic_cavity_mode(
    geometry: _CavityGeometry,
    mode: _CavityMode,
    pressure_amplitude: u.Quantity,
    *,
    sound_speed: u.Quantity,
    reference_time: object,
    numerics: AcousticCavityNumerics = AcousticCavityNumerics(),  # noqa: B008 - approved API default
) -> (
    SphereAcousticCavityModel | CuboidAcousticCavityModel | CylinderAcousticCavityModel
):
    """Build a cavity gravity model by dispatching on its geometry.

    Sphere geometry uses the in-module evaluator. Cuboid and Cylinder geometry
    use their lazily imported shape-specific builders. Each geometry requires
    its matching acoustic mode descriptor.
    """
    if isinstance(geometry, Sphere):
        if not isinstance(mode, SphereAcousticMode):
            raise TypeError("a Sphere requires a SphereAcousticMode")
        return SphereAcousticCavityModel(
            SphereCavitySpec(
                geometry,
                mode,
                pressure_amplitude,
                sound_speed=sound_speed,
                reference_time=reference_time,
                numerics=numerics,
            )
        )
    if isinstance(geometry, Cuboid):
        if not isinstance(mode, CuboidAcousticMode):
            raise TypeError("a Cuboid requires a CuboidAcousticMode")
        from gragra.analytic.atmospheric._cavity._cuboid import build_cuboid_cavity

        return build_cuboid_cavity(
            geometry,
            mode,
            pressure_amplitude,
            sound_speed=sound_speed,
            reference_time=reference_time,
            numerics=numerics,
        )
    if isinstance(geometry, Cylinder):
        if not isinstance(mode, CylinderAcousticMode):
            raise TypeError("a Cylinder requires a CylinderAcousticMode")
        from gragra.analytic.atmospheric._cavity._cylinder import (
            build_cylinder_cavity,
        )

        return build_cylinder_cavity(
            geometry,
            mode,
            pressure_amplitude,
            sound_speed=sound_speed,
            reference_time=reference_time,
            numerics=numerics,
        )
    raise TypeError("geometry must be a Sphere, Cuboid, or Cylinder")


__all__ = [
    "SphereAcousticMode",
    "CuboidAcousticMode",
    "CylinderAcousticMode",
    "AcousticCavityNumerics",
    "SphereCavitySpec",
    "SphereAcousticCavityModel",
    "acoustic_cavity_mode",
]
