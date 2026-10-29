"""One dimensional Gaussian-transform kernel for cuboid acoustic modes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_CEILING,
    ROUND_FLOOR,
    Decimal,
    localcontext,
)
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from scipy.special import erfc, erfcx

from gragra._analytic_evaluation import AnalyticEvaluation, NativeEvaluation
from gragra._interactions import Newtonian
from gragra.analytic.atmospheric._cavity._model import _build_cavity_evaluation
from gragra.analytic.atmospheric._cavity._numerics import integrate_transform
from gragra.constants import G_SI
from gragra.exceptions import ConvergenceError, NumericalDiagnostic
from gragra.geometry._cuboid import Cuboid
from gragra.geometry._frame import Frame
from gragra.results._gen1_definitions import gen1_field_definition
from gragra.targets._high_points import TargetPoints

if TYPE_CHECKING:
    from gragra.analytic.atmospheric._cavity._model import (
        AcousticCavityNumerics,
        CuboidAcousticMode,
    )

_MIN_SUBNORMAL = float(np.nextafter(0.0, 1.0))
_METHOD = "cuboid_volume_gaussian_v1"
_RELATIVE_TOLERANCES = {"potential": 1e-10, "acceleration": 1e-10, "gradient": 1e-9}
_OBSERVABLE_COMPONENTS = {
    "potential": slice(0, 1),
    "acceleration": slice(1, 4),
    "gradient": slice(4, 13),
}
_OBSERVABLE_NORMS = {
    "potential": "absolute",
    "acceleration": "euclidean",
    "gradient": "frobenius",
}


def _scaled_complex(value: complex, logarithmic_scale: float, label: str) -> complex:
    magnitude = math.hypot(value.real, value.imag)
    if magnitude == 0.0:
        return 0.0j
    log_magnitude = math.log(magnitude) + logarithmic_scale
    if log_magnitude < math.log(_MIN_SUBNORMAL):
        raise ValueError(f"nonzero pressure scale underflow in cuboid {label}")
    if log_magnitude > math.log(np.finfo(float).max):
        raise ValueError(f"nonzero pressure scale overflow in cuboid {label}")
    output_magnitude = math.exp(log_magnitude)
    result = value * (output_magnitude / magnitude)
    if result == 0.0 or not np.isfinite(result):
        raise ValueError(
            f"nonzero pressure scale is not representable in cuboid {label}"
        )
    return complex(result)


def _upward_float(value: Decimal) -> float | None:
    if value.is_nan() or value.is_infinite() or value < 0:
        return None
    if value == 0:
        return 0.0
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return None
    if not np.isfinite(converted):
        return None
    if converted == 0.0:
        return _MIN_SUBNORMAL
    if Decimal.from_float(converted) < value:
        converted = float(np.nextafter(converted, np.inf))
    return converted if np.isfinite(converted) and converted > 0.0 else None


def _fraction_decimal(value: Fraction, *, upward: bool) -> Decimal:
    rounding = ROUND_CEILING if upward else ROUND_FLOOR
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.Emax = MAX_EMAX
        ctx.Emin = MIN_EMIN
        ctx.rounding = rounding
        result = Decimal(value.numerator) / Decimal(value.denominator)
        represented = Fraction(*result.as_integer_ratio())
        if upward and represented < value:
            result = result.next_plus(context=ctx)
        elif not upward and represented > value:
            result = result.next_minus(context=ctx)
        return result


@dataclass(frozen=True, eq=False, slots=True, init=False)
class CuboidCavitySpec:
    """Immutable physical and numerical snapshot for one cuboid mode."""

    geometry: Cuboid
    mode: CuboidAcousticMode
    pressure_amplitude_pa: complex
    sound_speed_m_s: float
    reference_time_gps: tuple[int, int]
    frequency_hz: float
    betas_rad_m: tuple[float, float, float]
    wavenumber_rad_m: float
    normalization: float
    density_amplitude: complex
    green_coefficient: complex
    geometric_length_m: float
    pressure_scale_si: float
    scales_si: tuple[float, float, float]
    numerics: AcousticCavityNumerics
    field_definition: object
    interaction: Newtonian
    parameters: QTable
    formulas: pd.DataFrame
    references: pd.DataFrame

    def __init__(
        self,
        geometry: Cuboid,
        mode: CuboidAcousticMode,
        pressure_amplitude: object,
        *,
        sound_speed: object,
        reference_time: object,
        numerics: AcousticCavityNumerics,
    ) -> None:
        from gragra.analytic.atmospheric._cavity._model import (
            AcousticCavityNumerics,
            _positive_scalar_quantity,
            _pressure_amplitude,
        )
        from gragra.results._coords import normalize_gps

        pressure = _pressure_amplitude(pressure_amplitude)
        speed = _positive_scalar_quantity("sound_speed", sound_speed, u.m / u.s)
        if not isinstance(numerics, AcousticCavityNumerics):
            raise TypeError("numerics must be an AcousticCavityNumerics")
        reference_gps = normalize_gps(reference_time)

        lengths = geometry.lengths_m
        beta = np.asarray(mode_indices := (mode.nx, mode.ny, mode.nz), dtype=float)
        betas = beta * np.pi / lengths
        if not np.isfinite(betas).all() or np.any(betas < 0.0):
            raise ValueError("cuboid modal wavenumbers must be finite")
        wavenumber = float(np.hypot(np.hypot(betas[0], betas[1]), betas[2]))
        k_squared = wavenumber * wavenumber
        frequency = speed * wavenumber / (2.0 * np.pi)
        nonzero_count = sum(int(index != 0) for index in mode_indices)
        normalization = math.exp(0.5 * nonzero_count * math.log(2.0))
        derived = np.asarray([wavenumber, k_squared, frequency, normalization])
        if (
            not np.isfinite(derived).all()
            or wavenumber <= 0.0
            or k_squared <= 0.0
            or frequency <= 0.0
        ):
            raise ValueError("cuboid mode eigenvalue must be finite and nonzero")

        if pressure == 0.0:
            density_amplitude = 0.0j
            green_coefficient = 0.0j
        else:
            density_amplitude = _scaled_complex(
                pressure, -2.0 * math.log(speed), "density amplitude"
            )
            green_coefficient = _scaled_complex(
                density_amplitude, -math.log(k_squared), "Green coefficient"
            )

        geometric_length = math.exp(float(np.mean(np.log(lengths))))
        if not np.isfinite(geometric_length) or geometric_length <= 0.0:
            raise ValueError("cuboid geometric length is not representable")
        density_scale = math.hypot(density_amplitude.real, density_amplitude.imag)
        scales = (
            G_SI * density_scale * geometric_length**2,
            G_SI * density_scale * geometric_length,
            G_SI * density_scale,
        )
        if pressure != 0.0 and (
            not np.isfinite(scales).all() or any(value == 0.0 for value in scales)
        ):
            raise ValueError("nonzero cuboid physical scale is not representable")

        center = geometry.center_m
        definition = gen1_field_definition("acoustic_cavity")
        parameters = QTable(
            {
                "geometry": ["cuboid"],
                "center_x": [center[0] * u.m],
                "center_y": [center[1] * u.m],
                "center_z": [center[2] * u.m],
                "length_x": [lengths[0] * u.m],
                "length_y": [lengths[1] * u.m],
                "length_z": [lengths[2] * u.m],
                "nx": [mode.nx],
                "ny": [mode.ny],
                "nz": [mode.nz],
                "normalization": [normalization],
                "pressure_amplitude": [pressure * u.Pa],
                "pressure_convention": ["spatial-RMS peak phasor"],
                "sound_speed": [speed * u.m / u.s],
                "wavenumber": [wavenumber / u.m],
                "frequency": [frequency * u.Hz],
                "reference_time_gps_seconds": [reference_gps[0]],
                "reference_time_gps_nanoseconds": [reference_gps[1]],
                "numerical_route": [_METHOD],
                "accuracy_policy": ["gen1_cavity_v1"],
                "absolute_scale_factor": [1e-12],
                "relative_phi": [1e-10],
                "relative_acceleration": [1e-10],
                "relative_gradient": [1e-9],
                "runtime_budget_fraction": [0.25],
                "max_evaluations": [numerics.max_evaluations],
                "max_refinements": [numerics.max_refinements],
                "tolerance_policy": ["gen1_cavity_v1 runtime budget"],
                "source_frame": [geometry.frame.name],
            }
        )
        formulas = pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expression,
                    "convention": (
                        "G-free separable volume Gaussian transform; "
                        "bridge applies G once"
                    ),
                }
                for name, expression in (
                    (
                        "potential",
                        "Phi/G = -2*A*N/sqrt(pi)*integral(F_x*F_y*F_z, t=0..inf)",
                    ),
                    (
                        "acceleration",
                        "a_i/G = 2*A*N/sqrt(pi)*integral(d_i(F_x*F_y*F_z), t=0..inf)",
                    ),
                    (
                        "gradient",
                        "Gamma_ij/G = 2*A*N/sqrt(pi)*integral("
                        "d_ij(F_x*F_y*F_z), t=0..inf)",
                    ),
                )
            ]
        )
        references = pd.DataFrame(
            [
                {
                    "source": "docs/design/gen1/cavity-transforms.md#cuboid-six-faces",
                    "detail": (
                        "equivalent separable source-volume Gaussian transform; "
                        "interval derivatives and integrated-by-parts tails"
                    ),
                },
                {
                    "source": "docs/design/gen1/acceptance-error-budget.md",
                    "detail": (
                        "gen1_cavity_v1 runtime estimates and "
                        "componentwise rigorous tails"
                    ),
                },
            ]
        )
        for name, value in (
            ("geometry", geometry),
            ("mode", mode),
            ("pressure_amplitude_pa", pressure),
            ("sound_speed_m_s", speed),
            ("reference_time_gps", reference_gps),
            ("frequency_hz", frequency),
            ("betas_rad_m", tuple(float(value) for value in betas)),
            ("wavenumber_rad_m", wavenumber),
            ("normalization", normalization),
            ("density_amplitude", density_amplitude),
            ("green_coefficient", green_coefficient),
            ("geometric_length_m", geometric_length),
            ("pressure_scale_si", density_scale),
            ("scales_si", scales),
            ("numerics", numerics),
            ("field_definition", definition),
            ("interaction", Newtonian()),
            ("parameters", parameters),
            ("formulas", formulas),
            ("references", references),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CuboidAcousticCavityModel:
    """Factory for evaluation of a single-frequency cuboid cavity mode."""

    spec: CuboidCavitySpec

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = ("potential", "acceleration", "gradient"),
        output_frame: Frame | None = None,
    ) -> AnalyticEvaluation:
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be high-level TargetPoints")

        def validate_targets(
            target_positions_m: np.ndarray, boundary_error: np.ndarray
        ) -> None:
            lengths = self.spec.geometry.lengths_m
            relative = target_positions_m - self.spec.geometry.center_m
            half = lengths / 2.0
            within = np.all(
                np.abs(relative) <= half[None, :] + boundary_error[:, None], axis=1
            )
            on_face = np.any(
                np.abs(np.abs(relative) - half[None, :]) <= boundary_error[:, None],
                axis=1,
            )
            boundary = within & on_face
            if np.any(boundary):
                first = int(np.flatnonzero(boundary)[0])
                raise ValueError(
                    f"target point at index {first} lies on the cuboid boundary"
                )

        target_ids = tuple(targets.ids)

        def evaluate_native(
            target_positions_m: np.ndarray, names: tuple[str, ...]
        ) -> NativeEvaluation:
            return _cuboid_native_evaluation(
                self.spec, target_positions_m, names, target_ids
            )

        return _build_cavity_evaluation(
            self.spec,
            targets,
            observables=observables,
            output_frame=output_frame,
            evaluate_native=evaluate_native,
            validate_targets=validate_targets,
        )


def build_cuboid_cavity(
    geometry: Cuboid,
    mode: CuboidAcousticMode,
    pressure_amplitude: object,
    *,
    sound_speed: object,
    reference_time: object,
    numerics: AcousticCavityNumerics,
) -> CuboidAcousticCavityModel:
    """Construct the validated cuboid mode snapshot and evaluation factory."""
    from gragra.analytic.atmospheric._cavity._model import (
        CuboidAcousticMode,
    )

    if not isinstance(geometry, Cuboid):
        raise TypeError("geometry must be a Cuboid")
    if not isinstance(mode, CuboidAcousticMode):
        raise TypeError("mode must be a CuboidAcousticMode")
    return CuboidAcousticCavityModel(
        CuboidCavitySpec(
            geometry,
            mode,
            pressure_amplitude,
            sound_speed=sound_speed,
            reference_time=reference_time,
            numerics=numerics,
        )
    )


def _interval_cosine_transform(
    t: float, x: float, length: float, mode_index: int
) -> tuple[float, float, float, float]:
    """Return F, G, F_x and F_xx from the finite-interval erfcx transform."""
    left = -0.5 * length
    right = 0.5 * length
    beta = mode_index * np.pi / length
    left_delta = left - x
    right_delta = right - x
    if t <= 0.0:
        if mode_index == 0:
            f = length
            g = 0.0
        else:
            f = 0.0
            g = length * (1.0 - (-1.0) ** mode_index) / (mode_index * np.pi)
        return f, g, 0.0, 0.0

    if mode_index == 0:
        left_arg = t * left_delta
        right_arg = t * right_delta
        if x < left:
            difference = erfc(left_arg) - erfc(right_arg)
        elif x > right:
            difference = erfc(-right_arg) - erfc(-left_arg)
        else:
            difference = 2.0 - erfc(right_arg) - erfc(-left_arg)
        f = math.sqrt(math.pi) * difference / (2.0 * t)
        g = 0.0
    else:
        inverse_frequency = beta / (2.0 * t)
        if not np.isfinite(inverse_frequency):
            # This is the t -> 0 limit; the omitted O(t^2) region is below
            # the first representable transform coordinate for this mode.
            f = 0.0
            g = length * (1.0 - (-1.0) ** mode_index) / (mode_index * np.pi)
            return f, g, 0.0, 0.0
        z_left = t * left_delta - 1j * inverse_frequency
        z_right = t * right_delta - 1j * inverse_frequency
        e_left = np.exp(-((t * left_delta) ** 2) + 1j * beta * left)
        e_right = np.exp(-((t * right_delta) ** 2) + 1j * beta * right)
        if x < left:
            scaled_integral = e_left * erfcx(z_left) - e_right * erfcx(z_right)
        elif x > right:
            scaled_integral = e_right * erfcx(-z_right) - e_left * erfcx(-z_left)
        else:
            central = np.exp(-(inverse_frequency**2) + 1j * beta * x)
            scaled_integral = (
                2.0 * central - e_right * erfcx(z_right) - e_left * erfcx(-z_left)
            )
        integral = math.sqrt(math.pi) * scaled_integral / (2.0 * t)
        phase = np.exp(0.5j * mode_index * np.pi)
        projected = phase * integral
        f, g = float(projected.real), float(projected.imag)

    exp_left = math.exp(-((t * left_delta) ** 2))
    exp_right = math.exp(-((t * right_delta) ** 2))
    parity = (-1.0) ** mode_index
    f_x = exp_left - parity * exp_right - beta * g
    f_xx = 2.0 * t**2 * (left_delta * exp_left - parity * right_delta * exp_right)
    f_xx -= beta**2 * f
    if x == 0.0:
        if mode_index % 2:
            # The odd centered mode is antisymmetric against the even Gaussian.
            f = 0.0
            f_xx = 0.0
        else:
            # Its first target derivative is odd against the centered interval.
            g = 0.0
            f_x = 0.0
    return f, g, f_x, f_xx


def _cuboid_transform_integrand(
    spec: CuboidCavitySpec, relative_position: np.ndarray, t: float
) -> np.ndarray:
    lengths = spec.geometry.lengths_m
    half = lengths / 2.0
    indices = (spec.mode.nx, spec.mode.ny, spec.mode.nz)
    result = np.zeros(13, dtype=np.complex128)
    gaussian_factor = -4.0 * spec.normalization / math.sqrt(math.pi) * t**2

    for normal_axis in range(3):
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]
        first_tangent, second_tangent = tangent_axes
        for sigma in (-1.0, 1.0):
            endpoint = sigma * half[normal_axis]
            phase = 1.0 if sigma < 0.0 else (-1.0) ** indices[normal_axis]
            distance = endpoint - relative_position[normal_axis]
            exponential = math.exp(-((t * distance) ** 2))
            q = sigma * phase * distance * exponential
            q_x = sigma * phase * (-1.0 + 2.0 * t**2 * distance**2) * exponential
            q_xx = (
                sigma
                * phase
                * 2.0
                * t**2
                * distance
                * (-3.0 + 2.0 * t**2 * distance**2)
                * exponential
            )
            tangent_values = [
                _interval_cosine_transform(
                    t,
                    float(relative_position[axis]),
                    float(lengths[axis]),
                    indices[axis],
                )
                for axis in tangent_axes
            ]
            factors = [q, tangent_values[0][0], tangent_values[1][0]]
            first = [q_x, tangent_values[0][2], tangent_values[1][2]]
            second = [q_xx, tangent_values[0][3], tangent_values[1][3]]
            axes = [normal_axis, first_tangent, second_tangent]
            other_product = factors[0] * factors[1] * factors[2]
            term = gaussian_factor * other_product
            result[0] += -spec.green_coefficient * term
            for axis in range(3):
                slot = axes.index(axis)
                product = first[slot]
                for other_slot, factor in enumerate(factors):
                    if other_slot != slot:
                        product *= factor
                result[1 + axis] += spec.green_coefficient * gaussian_factor * product
                for other_axis in range(3):
                    if axis == other_axis:
                        hessian_product = second[slot]
                        for other_slot, factor in enumerate(factors):
                            if other_slot != slot:
                                hessian_product *= factor
                    else:
                        other_slot = axes.index(other_axis)
                        hessian_product = first[slot] * first[other_slot]
                        for remaining_slot, factor in enumerate(factors):
                            if remaining_slot != slot and remaining_slot != other_slot:
                                hessian_product *= factor
                    result[4 + 3 * axis + other_axis] += (
                        spec.green_coefficient * gaussian_factor * hessian_product
                    )
    if not np.isfinite(result).all():
        return np.full(13, np.nan + 1j * np.nan)
    return result


def _cuboid_volume_transform_integrand(
    spec: CuboidCavitySpec, relative_position: np.ndarray, t: float
) -> np.ndarray:
    """G-free field integrand from the separable source-volume Gaussian form.

    The identity 1/r = 2/sqrt(pi) * integral(exp(-t**2*r**2), t=0..infinity)
    separates the rectangular source into three one-dimensional cosine
    transforms.  Differentiating those transforms gives acceleration and
    gradient without the small-wavenumber cancellation in the six-face Green
    representation.
    """
    lengths = spec.geometry.lengths_m
    indices = (spec.mode.nx, spec.mode.ny, spec.mode.nz)
    intervals = [
        _interval_cosine_transform(
            t,
            float(relative_position[axis]),
            float(lengths[axis]),
            indices[axis],
        )
        for axis in range(3)
    ]
    f = np.asarray([item[0] for item in intervals], dtype=np.float64)
    f_x = np.asarray([item[2] for item in intervals], dtype=np.float64)
    f_xx = np.asarray([item[3] for item in intervals], dtype=np.float64)
    prefactor = 2.0 * spec.density_amplitude * spec.normalization / math.sqrt(math.pi)
    result = np.empty(13, dtype=np.complex128)
    result[0] = -prefactor * np.prod(f)
    for axis in range(3):
        other_axes = [index for index in range(3) if index != axis]
        result[1 + axis] = prefactor * f_x[axis] * f[other_axes[0]] * f[other_axes[1]]
        for other_axis in range(3):
            if axis == other_axis:
                value = f_xx[axis] * f[other_axes[0]] * f[other_axes[1]]
            else:
                remaining = next(
                    index for index in range(3) if index not in (axis, other_axis)
                )
                value = f_x[axis] * f_x[other_axis] * f[remaining]
            result[4 + 3 * axis + other_axis] = prefactor * value
    if not np.isfinite(result).all():
        return np.full(13, np.nan + 1j * np.nan)
    return result


def _mode_field(
    spec: CuboidCavitySpec, relative_position: np.ndarray
) -> tuple[complex, np.ndarray, np.ndarray]:
    lengths = spec.geometry.lengths_m
    beta = np.asarray(spec.betas_rad_m)
    phase = beta * (relative_position + lengths / 2.0)
    cosine = np.cos(phase)
    sine = np.sin(phase)
    psi = spec.normalization * float(np.prod(cosine))
    gradient = np.zeros(3, dtype=np.float64)
    hessian = np.zeros((3, 3), dtype=np.float64)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        gradient[axis] = (
            -spec.normalization
            * beta[axis]
            * sine[axis]
            * cosine[other[0]]
            * cosine[other[1]]
        )
        hessian[axis, axis] = -(beta[axis] ** 2) * psi
        for other_axis in range(axis + 1, 3):
            remaining = next(
                index for index in range(3) if index not in (axis, other_axis)
            )
            value = (
                spec.normalization
                * beta[axis]
                * beta[other_axis]
                * sine[axis]
                * sine[other_axis]
                * cosine[remaining]
            )
            hessian[axis, other_axis] = value
            hessian[other_axis, axis] = value
    return complex(psi), gradient, hessian


def _polynomial_for_q(order: int, distance: Decimal) -> dict[int, Decimal]:
    if order == 0:
        return {0: abs(distance)}
    if order == 1:
        return {0: Decimal(1), 2: Decimal(2) * distance * distance}
    return {
        2: Decimal(6) * abs(distance),
        4: Decimal(4) * abs(distance) ** 3,
    }


def _polynomial_for_p(order: int, maximum_distance: Decimal) -> dict[int, Decimal]:
    if order == 0:
        return {0: Decimal(1)}
    if order == 1:
        return {2: Decimal(2) * maximum_distance}
    return {
        2: Decimal(2),
        4: Decimal(4) * maximum_distance * maximum_distance,
    }


def _multiply_polynomials(
    left: dict[int, Decimal], right: dict[int, Decimal]
) -> dict[int, Decimal]:
    output: dict[int, Decimal] = {}
    for left_power, left_value in left.items():
        for right_power, right_value in right.items():
            power = left_power + right_power
            output[power] = output.get(power, Decimal(0)) + left_value * right_value
    return output


def _gaussian_power_tail(power: int, delta: Decimal, cutoff: Decimal) -> Decimal:
    """Upper bound integral(T,infinity) t**p exp(-delta**2 t**2) dt."""
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.Emax = MAX_EMAX
        ctx.Emin = MIN_EMIN
        ctx.rounding = ROUND_FLOOR
        delta_squared = delta * delta
        exponent_argument = delta_squared * cutoff * cutoff
        with localcontext() as upper:
            upper.prec = 100
            upper.Emax = MAX_EMAX
            upper.Emin = MIN_EMIN
            upper.rounding = ROUND_CEILING
            exponential = (-exponent_argument).exp().next_plus(context=upper)
            if exponential == 0:
                # Context underflow means the exact positive exponential is
                # below 10**MIN_EMIN; that value is still a conservative upper
                # endpoint, including all finite float-derived coefficients.
                exponential = Decimal(1).scaleb(MIN_EMIN)
            denominator = Decimal(2) * delta_squared
            integrals: dict[int, Decimal] = {
                0: exponential / (denominator * cutoff),
                1: exponential / denominator,
            }
            for current in range(2, power + 1):
                first_term = cutoff ** (current - 1) * exponential / denominator
                second_term = (
                    Decimal(current - 1) * integrals[current - 2] / denominator
                )
                integrals[current] = first_term + second_term
            return integrals[power]


def _tail_transform_components(
    spec: CuboidCavitySpec, relative_position: np.ndarray, cutoff: float
) -> np.ndarray | None:
    """Outward Decimal component bounds for all 13 transformed field entries."""
    try:
        with localcontext() as floor:
            floor.prec = 100
            floor.Emax = MAX_EMAX
            floor.Emin = MIN_EMIN
            floor.rounding = ROUND_FLOOR
            with localcontext() as ceil:
                ceil.prec = 100
                ceil.Emax = MAX_EMAX
                ceil.Emin = MIN_EMIN
                ceil.rounding = ROUND_CEILING
                lengths = [
                    Decimal.from_float(float(x)) for x in spec.geometry.lengths_m
                ]
                length_fractions = [
                    Fraction.from_float(float(x)) for x in spec.geometry.lengths_m
                ]
                coordinate_fractions = [
                    Fraction.from_float(float(x)) for x in relative_position
                ]
                half_fractions = [value / 2 for value in length_fractions]
                bounds = [Decimal(0) for _ in range(13)]
                indices = (spec.mode.nx, spec.mode.ny, spec.mode.nz)
                alpha: list[tuple[int, int, int]] = [(0, 0, 0)]
                alpha.extend(
                    (1, 0, 0) if axis == 0 else (0, 1, 0) if axis == 1 else (0, 0, 1)
                    for axis in range(3)
                )
                alpha.extend(
                    tuple(int(axis == i) + int(axis == j) for axis in range(3))
                    for i in range(3)
                    for j in range(3)
                )
                cutoff_decimal = Decimal.from_float(float(cutoff))
                # 2**(n/2) <= [1, 1.415, 2, 2.83] and
                # 1/sqrt(pi) < 1/1.7, both rational bounds.
                nonzero = sum(int(index != 0) for index in indices)
                normalization_upper = (
                    Decimal("1"),
                    Decimal("1.415"),
                    Decimal("2"),
                    Decimal("2.83"),
                )[nonzero]
                prefactor = Decimal(40) * normalization_upper / Decimal(17)

                for normal_axis in range(3):
                    tangents = [axis for axis in range(3) if axis != normal_axis]
                    j, k = tangents
                    for sign in (-1, 1):
                        endpoint_fraction = sign * half_fractions[normal_axis]
                        distance_fraction = abs(
                            endpoint_fraction - coordinate_fractions[normal_axis]
                        )
                        distance = _fraction_decimal(distance_fraction, upward=True)
                        tangent_data = []
                        for axis in (j, k):
                            low = -half_fractions[axis] - coordinate_fractions[axis]
                            high = half_fractions[axis] - coordinate_fractions[axis]
                            gap_fraction = max(Fraction(0), low, -high)
                            max_distance_fraction = max(abs(low), abs(high))
                            gap = _fraction_decimal(gap_fraction, upward=False)
                            max_distance = _fraction_decimal(
                                max_distance_fraction, upward=True
                            )
                            tangent_data.append((gap, max_distance, lengths[axis]))
                        exact_delta_squared = (
                            distance_fraction**2
                            + max(
                                Fraction(0),
                                -half_fractions[j] - coordinate_fractions[j],
                                coordinate_fractions[j] - half_fractions[j],
                            )
                            ** 2
                            + max(
                                Fraction(0),
                                -half_fractions[k] - coordinate_fractions[k],
                                coordinate_fractions[k] - half_fractions[k],
                            )
                            ** 2
                        )
                        if exact_delta_squared <= 0:
                            return None
                        delta_squared = _fraction_decimal(
                            exact_delta_squared, upward=False
                        )
                        delta = delta_squared.sqrt().next_minus(context=floor)
                        q_polynomials = [
                            _polynomial_for_q(order, distance) for order in range(3)
                        ]
                        p_polynomials = [
                            [
                                {
                                    power: coefficient * length
                                    for power, coefficient in _polynomial_for_p(
                                        order, maximum_distance
                                    ).items()
                                }
                                for order in range(3)
                            ]
                            for _, maximum_distance, length in tangent_data
                        ]
                        for component, orders in enumerate(alpha):
                            coefficient_polynomial = _multiply_polynomials(
                                _multiply_polynomials(
                                    q_polynomials[orders[normal_axis]],
                                    p_polynomials[0][orders[j]],
                                ),
                                p_polynomials[1][orders[k]],
                            )
                            component_bound = Decimal(0)
                            for power, coefficient in coefficient_polynomial.items():
                                component_bound += coefficient * _gaussian_power_tail(
                                    power + 2, delta, cutoff_decimal
                                )
                            bounds[component] += prefactor * component_bound

                coefficient_real = Decimal.from_float(spec.green_coefficient.real)
                coefficient_imag = Decimal.from_float(spec.green_coefficient.imag)
                coefficient_abs = (
                    (coefficient_real**2 + coefficient_imag**2)
                    .sqrt()
                    .next_plus(context=ceil)
                )
                output: list[float] = []
                for bound in bounds:
                    value = _upward_float(bound * coefficient_abs)
                    if value is None:
                        return None
                    output.append(value)
                return np.asarray(output, dtype=np.float64)
    except (ArithmeticError, OverflowError, ValueError):
        return None


def _volume_interval_tail_terms(
    order: int,
    *,
    beta_upper: Decimal,
    sqrt_pi_upper: Decimal,
    gap_squared_lower: Decimal,
    endpoint_data: tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]],
) -> list[tuple[Decimal, int, Decimal]]:
    """Upper terms for one interval transform derivative on [T, infinity)."""
    if order == 0:
        return [(sqrt_pi_upper, -1, gap_squared_lower)]
    terms: list[tuple[Decimal, int, Decimal]] = []
    beta_factor = beta_upper if order == 1 else beta_upper * beta_upper
    if beta_factor > 0:
        terms.append((beta_factor * sqrt_pi_upper, -1, gap_squared_lower))
    if order == 1:
        terms.extend(
            (Decimal(1), 0, endpoint_squared) for _, endpoint_squared in endpoint_data
        )
    else:
        terms.extend(
            (Decimal(2) * endpoint_distance, 2, endpoint_squared)
            for endpoint_distance, endpoint_squared in endpoint_data
        )
    return terms


def _volume_exact_parity_zeros(
    spec: CuboidCavitySpec, relative_position: np.ndarray
) -> np.ndarray:
    """Mark volume-transform components identically zero by centered parity."""
    indices = (spec.mode.nx, spec.mode.ny, spec.mode.nz)
    cosine_zero = [
        float(relative_position[axis]) == 0.0 and index % 2 == 1
        for axis, index in enumerate(indices)
    ]
    derivative_zero = [
        float(relative_position[axis]) == 0.0 and index % 2 == 0
        for axis, index in enumerate(indices)
    ]
    second_derivative_zero = cosine_zero
    output = np.zeros(13, dtype=bool)
    output[0] = any(cosine_zero)
    for axis in range(3):
        other_axes = [index for index in range(3) if index != axis]
        output[1 + axis] = derivative_zero[axis] or any(
            cosine_zero[index] for index in other_axes
        )
        for other_axis in range(3):
            if axis == other_axis:
                output[4 + 3 * axis + other_axis] = second_derivative_zero[axis] or any(
                    cosine_zero[index] for index in other_axes
                )
            else:
                remaining = next(
                    index for index in range(3) if index not in (axis, other_axis)
                )
                output[4 + 3 * axis + other_axis] = (
                    derivative_zero[axis]
                    or derivative_zero[other_axis]
                    or cosine_zero[remaining]
                )
    return output


def _near_center_parity_observables(
    spec: CuboidCavitySpec, relative_position: np.ndarray
) -> tuple[str, ...]:
    """Return fields whose centered parity makes their relative budget fragile.

    Coordinates within a roundoff-sized neighborhood of a center reflection
    plane are still evaluated at their represented values. This only selects
    a more conservative starting cutoff; it never marks the field as zero.
    """
    half = spec.geometry.lengths_m / 2.0
    epsilon_scale = 64.0 * np.finfo(np.float64).eps
    # The threshold scales with the physical box dimensions and is used only
    # to choose the integration start; it is never a coordinate snap radius.
    near_axes = np.asarray(
        [
            abs(float(relative_position[axis]))
            <= epsilon_scale
            * max(float(half[axis]), abs(float(relative_position[axis])))
            for axis in range(3)
        ],
        dtype=bool,
    )
    if not np.any(near_axes):
        return ()
    symmetry_position = np.asarray(relative_position, dtype=np.float64).copy()
    symmetry_position[near_axes] = 0.0
    parity_zeros = _volume_exact_parity_zeros(spec, symmetry_position)
    return tuple(
        observable
        for observable in ("potential", "acceleration", "gradient")
        if np.all(parity_zeros[_OBSERVABLE_COMPONENTS[observable]])
    )


def _tail_within_center_parity_budget(
    spec: CuboidCavitySpec,
    observables: tuple[str, ...],
    tails: np.ndarray,
) -> bool:
    """Check a fixed absolute tail allocation for center-parity fields."""
    for observable in observables:
        tail_norm = _upper_si_norm(_native_groups(tails, observable))
        if tail_norm is None:
            return False
        absolute_budget = _budget_lower(spec, observable, Decimal(0))
        with localcontext() as ctx:
            ctx.prec = 100
            ctx.Emax = MAX_EMAX
            ctx.Emin = MIN_EMIN
            ctx.rounding = ROUND_FLOOR
            tail_allocation = absolute_budget * Decimal("0.5")
        if Decimal.from_float(tail_norm) > tail_allocation:
            return False
    return True


def _budgeted_initial_cutoff(
    spec: CuboidCavitySpec,
    relative_position: np.ndarray,
    initial_cutoff: float,
    tail_bound: Callable[[float], np.ndarray | None],
) -> float:
    """Choose a start whose parity-sensitive tails fit half B_abs when possible.

    This cutoff search is an efficiency hint. If the stricter allocation cannot
    be certified, the original finite cutoff is retained and normal integration
    acceptance decides whether the requested fields are resolved.
    """
    sensitive = _near_center_parity_observables(spec, relative_position)
    if not sensitive:
        return initial_cutoff

    lower = initial_cutoff
    try:
        tails = tail_bound(lower)
    except (ArithmeticError, OverflowError, ValueError):
        tails = None
    if tails is None:
        return initial_cutoff
    if _tail_within_center_parity_budget(spec, sensitive, tails):
        return lower

    upper = lower
    for _ in range(64):
        upper *= 2.0
        if not np.isfinite(upper) or not upper > lower:
            return initial_cutoff
        try:
            tails = tail_bound(upper)
        except (ArithmeticError, OverflowError, ValueError):
            tails = None
        if tails is None:
            return initial_cutoff
        if _tail_within_center_parity_budget(spec, sensitive, tails):
            break
        lower = upper
    else:
        return initial_cutoff

    # Keep the selected starting interval close to the required budget. The
    # final acceptance callback remains authoritative for every field.
    for _ in range(32):
        midpoint = lower + (upper - lower) * 0.5
        if not lower < midpoint < upper:
            break
        try:
            tails = tail_bound(midpoint)
        except (ArithmeticError, OverflowError, ValueError):
            tails = None
        if tails is None:
            return initial_cutoff
        if _tail_within_center_parity_budget(spec, sensitive, tails):
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _tail_volume_transform_components(
    spec: CuboidCavitySpec, relative_position: np.ndarray, cutoff: float
) -> np.ndarray | None:
    """Outward component bounds for the separable volume-transform tail.

    Integration by parts moves one or two target derivatives onto the smooth
    cosine mode.  The remaining finite-interval boundary terms are Gaussian,
    while the interior term is bounded by sqrt(pi)/t.  Their products give an
    integrable t**-3 interior tail and Gaussian tails for every boundary term.
    """
    try:
        with localcontext() as floor:
            floor.prec = 100
            floor.Emax = MAX_EMAX
            floor.Emin = MIN_EMIN
            floor.rounding = ROUND_FLOOR
            with localcontext() as ceil:
                ceil.prec = 100
                ceil.Emax = MAX_EMAX
                ceil.Emin = MIN_EMIN
                ceil.rounding = ROUND_CEILING
                lengths = [
                    Fraction.from_float(float(value))
                    for value in spec.geometry.lengths_m
                ]
                coordinates = [
                    Fraction.from_float(float(value)) for value in relative_position
                ]
                indices = (spec.mode.nx, spec.mode.ny, spec.mode.nz)
                pi_upper = Decimal(355) / Decimal(113)
                sqrt_pi_upper = pi_upper.sqrt().next_plus(context=ceil)
                interval_terms = []
                for length_fraction, coordinate, index in zip(
                    lengths, coordinates, indices, strict=True
                ):
                    half = length_fraction / 2
                    left = -half
                    right = half
                    gap = max(Fraction(0), left - coordinate, coordinate - right)
                    left_distance = abs(left - coordinate)
                    right_distance = abs(right - coordinate)
                    gap_squared_lower = _fraction_decimal(gap * gap, upward=False)
                    endpoint_data = (
                        (
                            _fraction_decimal(left_distance, upward=True),
                            _fraction_decimal(
                                left_distance * left_distance, upward=False
                            ),
                        ),
                        (
                            _fraction_decimal(right_distance, upward=True),
                            _fraction_decimal(
                                right_distance * right_distance, upward=False
                            ),
                        ),
                    )
                    length_decimal = Decimal.from_float(float(length_fraction))
                    beta = Decimal(index) * pi_upper / length_decimal
                    interval_terms.append(
                        {
                            0: _volume_interval_tail_terms(
                                0,
                                beta_upper=beta,
                                sqrt_pi_upper=sqrt_pi_upper,
                                gap_squared_lower=gap_squared_lower,
                                endpoint_data=endpoint_data,
                            ),
                            1: _volume_interval_tail_terms(
                                1,
                                beta_upper=beta,
                                sqrt_pi_upper=sqrt_pi_upper,
                                gap_squared_lower=gap_squared_lower,
                                endpoint_data=endpoint_data,
                            ),
                            2: _volume_interval_tail_terms(
                                2,
                                beta_upper=beta,
                                sqrt_pi_upper=sqrt_pi_upper,
                                gap_squared_lower=gap_squared_lower,
                                endpoint_data=endpoint_data,
                            ),
                        }
                    )

                modes = (spec.mode.nx, spec.mode.ny, spec.mode.nz)
                nonzero = sum(int(index != 0) for index in modes)
                normalization_upper = (
                    Decimal("1"),
                    Decimal("1.415"),
                    Decimal(2),
                    Decimal("2.83"),
                )[nonzero]
                density_real = Decimal.from_float(spec.density_amplitude.real)
                density_imag = Decimal.from_float(spec.density_amplitude.imag)
                density_magnitude = (
                    (density_real * density_real + density_imag * density_imag)
                    .sqrt()
                    .next_plus(context=ceil)
                )
                prefactor = (
                    density_magnitude * normalization_upper * Decimal(20) / Decimal(17)
                )
                cutoff_decimal = Decimal.from_float(float(cutoff))
                alpha: list[tuple[int, int, int]] = [(0, 0, 0)]
                alpha.extend(
                    (1, 0, 0) if axis == 0 else (0, 1, 0) if axis == 1 else (0, 0, 1)
                    for axis in range(3)
                )
                alpha.extend(
                    tuple(int(axis == i) + int(axis == j) for axis in range(3))
                    for i in range(3)
                    for j in range(3)
                )
                bounds = [Decimal(0) for _ in range(13)]
                for component, orders in enumerate(alpha):
                    axis_options = [
                        interval_terms[axis][order] for axis, order in enumerate(orders)
                    ]
                    for first in axis_options[0]:
                        for second in axis_options[1]:
                            for third in axis_options[2]:
                                coefficient = first[0] * second[0] * third[0]
                                power = first[1] + second[1] + third[1]
                                with localcontext() as lower_sum:
                                    lower_sum.prec = 100
                                    lower_sum.Emax = MAX_EMAX
                                    lower_sum.Emin = MIN_EMIN
                                    lower_sum.rounding = ROUND_FLOOR
                                    delta_squared = first[2] + second[2] + third[2]
                                if delta_squared == 0:
                                    if power != -3:
                                        return None
                                    integral = Decimal(1) / (
                                        Decimal(2) * cutoff_decimal * cutoff_decimal
                                    )
                                else:
                                    delta = delta_squared.sqrt().next_minus(
                                        context=floor
                                    )
                                    integral = _gaussian_power_tail(
                                        max(power, 0), delta, cutoff_decimal
                                    )
                                    if power < 0:
                                        integral *= cutoff_decimal**power
                                bounds[component] += coefficient * integral

                exact_zero = _volume_exact_parity_zeros(spec, relative_position)
                output: list[float] = []
                for component, bound in enumerate(bounds):
                    if exact_zero[component]:
                        output.append(0.0)
                        continue
                    converted = _upward_float(prefactor * bound)
                    if converted is None:
                        return None
                    output.append(converted)
                return np.asarray(output, dtype=np.float64)
    except (ArithmeticError, OverflowError, ValueError):
        return None


def _mode_native_at_point(
    spec: CuboidCavitySpec, relative_position: np.ndarray
) -> np.ndarray:
    psi, gradient, hessian = _mode_field(spec, relative_position)
    output = np.empty(13, dtype=np.complex128)
    output[0] = -4.0 * np.pi * spec.green_coefficient * psi
    output[1:4] = 4.0 * np.pi * spec.green_coefficient * gradient
    output[4:13] = 4.0 * np.pi * spec.green_coefficient * hessian.reshape(9)
    return output


def _local_field_roundoff_bounds(spec: CuboidCavitySpec) -> np.ndarray | None:
    """Outward componentwise estimate for local Helmholtz arithmetic."""
    try:
        with localcontext() as ctx:
            ctx.prec = 100
            ctx.Emax = MAX_EMAX
            ctx.Emin = MIN_EMIN
            ctx.rounding = ROUND_CEILING
            real = Decimal.from_float(spec.green_coefficient.real)
            imag = Decimal.from_float(spec.green_coefficient.imag)
            coefficient = (real * real + imag * imag).sqrt().next_plus(context=ctx)
            count = sum(
                int(index != 0) for index in (spec.mode.nx, spec.mode.ny, spec.mode.nz)
            )
            normalization = (Decimal(1), Decimal("1.415"), Decimal(2), Decimal("2.83"))[
                count
            ]
            # 355/113 is an outward rational upper bound for pi.
            base = (
                Decimal(4) * Decimal(355) / Decimal(113) * normalization * coefficient
            )
            beta_upper = [
                Decimal(index)
                * Decimal(355)
                / Decimal(113)
                / Decimal.from_float(float(length))
                for index, length in zip(
                    (spec.mode.nx, spec.mode.ny, spec.mode.nz),
                    spec.geometry.lengths_m,
                    strict=True,
                )
            ]
            derivative_factors = [Decimal(1), *beta_upper]
            derivative_factors.extend(
                beta_upper[row] * beta_upper[column]
                for row in range(3)
                for column in range(3)
            )
            unit_roundoff = Decimal.from_float(16.0 * np.finfo(float).eps)
            bounds: list[float] = []
            for factor in derivative_factors:
                bound = _upward_float(base * factor * unit_roundoff)
                if bound is None:
                    return None
                bounds.append(bound)
            return np.asarray(bounds, dtype=np.float64)
    except (ArithmeticError, OverflowError, ValueError):
        return None


def _combined_component_errors(
    values: np.ndarray | None,
    integration_errors: np.ndarray | None,
    local_values: np.ndarray,
    local_roundoff: np.ndarray,
) -> np.ndarray | None:
    """Combine quadrature, local-mode and final-addition errors outwardly."""
    if values is None or integration_errors is None:
        return None
    try:
        with localcontext() as ctx:
            ctx.prec = 100
            ctx.Emax = MAX_EMAX
            ctx.Emin = MIN_EMIN
            ctx.rounding = ROUND_CEILING
            unit_roundoff = Decimal.from_float(8.0 * np.finfo(float).eps)
            output: list[float] = []
            for value, local, error, local_error in zip(
                values,
                local_values,
                integration_errors,
                local_roundoff,
                strict=True,
            ):
                addition_scale = sum(
                    abs(Decimal.from_float(float(component)))
                    for item in (value, local)
                    for component in (complex(item).real, complex(item).imag)
                )
                total = (
                    Decimal.from_float(float(error))
                    + Decimal.from_float(float(local_error))
                    + unit_roundoff * addition_scale
                )
                bound = _upward_float(total)
                if bound is None:
                    return None
                output.append(bound)
            return np.asarray(output, dtype=np.float64)
    except (ArithmeticError, OverflowError, ValueError):
        return None


def _native_groups(values: np.ndarray, observable: str) -> np.ndarray:
    return values[_OBSERVABLE_COMPONENTS[observable]]


def _float_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if np.iscomplexobj(values):
        parts = [
            component for value in values.flat for component in (value.real, value.imag)
        ]
        return math.hypot(*[float(value) for value in parts])
    return math.hypot(*[float(value) for value in values.flat])


def _decimal_norm(values: np.ndarray, *, multiply_by_g: bool, rounding: str) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.Emax = MAX_EMAX
        ctx.Emin = MIN_EMIN
        ctx.rounding = rounding
        g = Decimal.from_float(float(G_SI)) if multiply_by_g else Decimal(1)
        total = Decimal(0)
        for item in np.asarray(values).flat:
            if np.iscomplexobj(values):
                real = Decimal.from_float(float(np.real(item)))
                imag = Decimal.from_float(float(np.imag(item)))
                total += (real * g) ** 2 + (imag * g) ** 2
            else:
                component = Decimal.from_float(float(item)) * g
                total += component**2
        root = total.sqrt()
        if root == 0:
            return root
        return (
            root.next_minus(context=ctx)
            if rounding == ROUND_FLOOR
            else root.next_plus(context=ctx)
        )


def _scale_si_decimal(spec: CuboidCavitySpec, observable: str) -> Decimal:
    """Compute the prescribed physical source scale with outward Decimal math."""
    lengths = [Decimal.from_float(float(value)) for value in spec.geometry.lengths_m]
    pressure_real = Decimal.from_float(spec.pressure_amplitude_pa.real)
    pressure_imag = Decimal.from_float(spec.pressure_amplitude_pa.imag)
    speed = Decimal.from_float(spec.sound_speed_m_s)
    power = {"potential": 2, "acceleration": 1, "gradient": 0}[observable]
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.Emax = MAX_EMAX
        ctx.Emin = MIN_EMIN
        ctx.rounding = ROUND_FLOOR
        volume = lengths[0] * lengths[1] * lengths[2]
        log_volume = volume.ln().next_minus(context=ctx)
        log_length = (log_volume / Decimal(3)).next_minus(context=ctx)
        length = log_length.exp().next_minus(context=ctx)
        pressure_magnitude = (
            (pressure_real**2 + pressure_imag**2).sqrt().next_minus(context=ctx)
        )
        density_scale = pressure_magnitude / (speed**2)
        g = Decimal.from_float(float(G_SI))
        scale = g * density_scale * (length**power)
        return +scale


def _budget_lower(
    spec: CuboidCavitySpec, observable: str, value_norm_lower: Decimal
) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.Emax = MAX_EMAX
        ctx.Emin = MIN_EMIN
        ctx.rounding = ROUND_FLOOR
        relative = Decimal.from_float(_RELATIVE_TOLERANCES[observable])
        absolute_scale = Decimal("1e-12") * _scale_si_decimal(spec, observable)
        return Decimal("0.25") * (absolute_scale + relative * value_norm_lower)


def _upper_si_norm(components: np.ndarray) -> float | None:
    """Outward G multiplication and Euclidean/Frobenius aggregation."""
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.Emax = MAX_EMAX
        ctx.Emin = MIN_EMIN
        ctx.rounding = ROUND_CEILING
        g = Decimal.from_float(float(G_SI))
        squared = Decimal(0)
        for component in np.asarray(components, dtype=np.float64).flat:
            value = Decimal.from_float(float(component)) * g
            squared += value * value
        root = squared.sqrt()
        if root > 0:
            root = root.next_plus(context=ctx)
        return _upward_float(root)


def _value_norm_lower(native_values: np.ndarray) -> Decimal:
    """Lower bound on the represented SI observable norm."""
    si_values = np.asarray(native_values) * G_SI
    return _decimal_norm(si_values, multiply_by_g=False, rounding=ROUND_FLOOR)


def _acceptance_callback(
    spec: CuboidCavitySpec,
    requested: tuple[str, ...],
    local_values: np.ndarray,
    local_roundoff: np.ndarray,
):
    def accept(values: np.ndarray, errors: np.ndarray, tails: np.ndarray) -> bool:
        full_values = values + local_values
        full_errors = _combined_component_errors(
            values, errors, local_values, local_roundoff
        )
        if full_errors is None:
            return False
        for observable in requested:
            value_group = _native_groups(full_values, observable)
            error_group = _native_groups(full_errors, observable)
            tail_group = _native_groups(tails, observable)
            value_norm = _value_norm_lower(value_group)
            error_norm_si = _upper_si_norm(error_group)
            tail_norm_si = _upper_si_norm(tail_group)
            if error_norm_si is None or tail_norm_si is None:
                return False
            with localcontext() as ctx:
                ctx.prec = 100
                ctx.Emax = MAX_EMAX
                ctx.Emin = MIN_EMIN
                ctx.rounding = ROUND_CEILING
                combined_error = Decimal.from_float(error_norm_si) + Decimal.from_float(
                    tail_norm_si
                )
            if combined_error > _budget_lower(spec, observable, value_norm):
                return False
        return True

    return accept


def _diagnostic_records(
    spec: CuboidCavitySpec,
    target_index: int,
    target_id: str,
    requested: tuple[str, ...],
    values: np.ndarray | None,
    errors: np.ndarray | None,
    tails: np.ndarray | None,
    evaluations: int,
    refinements: int,
    termination: str,
) -> tuple[NumericalDiagnostic, ...]:
    records = []
    for observable in requested:
        component_values = (
            None if values is None else _native_groups(values, observable)
        )
        component_errors = (
            None if errors is None else _native_groups(errors, observable)
        )
        component_tails = None if tails is None else _native_groups(tails, observable)
        scale = spec.scales_si[
            {"potential": 0, "acceleration": 1, "gradient": 2}[observable]
        ]
        if termination == "exact_zero":
            value_norm = tolerance = error_norm = tail_norm = 0.0
        else:
            if component_values is None:
                value_norm = None
            else:
                value_norm_decimal = _decimal_norm(
                    np.asarray(component_values) * G_SI,
                    multiply_by_g=False,
                    rounding=ROUND_CEILING,
                )
                value_norm = _upward_float(value_norm_decimal)
            error_norm = (
                None if component_errors is None else _upper_si_norm(component_errors)
            )
            tail_norm = (
                None if component_tails is None else _upper_si_norm(component_tails)
            )
            if value_norm is None:
                tolerance = None
            else:
                tolerance = _upward_float(
                    _budget_lower(spec, observable, _value_norm_lower(component_values))
                )
        records.append(
            NumericalDiagnostic(
                method=_METHOD,
                target_index=target_index,
                target_id=target_id,
                observable=observable,
                norm=_OBSERVABLE_NORMS[observable],
                accuracy_policy="gen1_cavity_v1",
                scale_si=scale,
                value_norm_si=value_norm,
                requested_tolerance_si=tolerance,
                error_estimate_si=error_norm,
                rigorous_tail_bound_si=tail_norm,
                evaluations=evaluations,
                refinements=refinements,
                termination=termination,
            )
        )
    return tuple(records)


def _cuboid_native_evaluation(
    spec: CuboidCavitySpec,
    targets_m: np.ndarray,
    observables: tuple[str, ...],
    target_ids: tuple[str, ...],
) -> NativeEvaluation:
    count = len(targets_m)
    values = {
        "potential": np.zeros((count, 1, 1), dtype=np.complex128),
        "acceleration": np.zeros((count, 3, 1), dtype=np.complex128),
        "gradient": np.zeros((count, 9, 1), dtype=np.complex128),
    }
    if count == 0:
        return NativeEvaluation({name: values[name] for name in observables})

    if spec.pressure_amplitude_pa == 0.0:
        diagnostics = tuple(
            diagnostic
            for index, target_id in enumerate(target_ids)
            for diagnostic in _diagnostic_records(
                spec,
                index,
                target_id,
                observables,
                np.zeros(13, dtype=np.complex128),
                np.zeros(13),
                np.zeros(13),
                0,
                0,
                "exact_zero",
            )
        )
        return NativeEvaluation(
            {name: values[name] for name in observables}, diagnostics
        )

    lengths = spec.geometry.lengths_m
    half = lengths / 2.0
    center = spec.geometry.center_m
    diagnostics = []
    for target_index, (target, target_id) in enumerate(
        zip(targets_m, target_ids, strict=True)
    ):
        relative = target - center
        local_values = np.zeros(13, dtype=np.complex128)
        local_roundoff = np.zeros(13, dtype=np.float64)
        # A strict positive face-distance lower bound is certified from the
        # represented target and exact binary64 geometry values.
        face_distances = []
        for normal_axis in range(3):
            tangents = [axis for axis in range(3) if axis != normal_axis]
            d = abs(half[normal_axis] - abs(relative[normal_axis]))
            gaps = [max(0.0, abs(relative[axis]) - half[axis]) for axis in tangents]
            face_distances.append(math.hypot(d, gaps[0], gaps[1]))
        minimum_distance = min(face_distances)
        if not np.isfinite(minimum_distance) or minimum_distance <= 0.0:
            raise ValueError("cuboid target transform gap is not positive")
        # The direct volume transform has an algebraic t**-3 interior tail.
        # Start farther out than the surface representation so its certified
        # tail does not consume the bounded refinement rounds before the
        # quadrature has a chance to converge.
        base_cutoff = 64.0 / minimum_distance
        if not np.isfinite(base_cutoff) or base_cutoff <= 0.0:
            estimate = None
        else:
            tail_cache: dict[float, np.ndarray | None] = {}

            def tail_bound(
                cutoff_value: float,
                *,
                _tail_cache: dict[float, np.ndarray | None] = tail_cache,
                _relative: np.ndarray = relative,
            ) -> np.ndarray | None:
                if cutoff_value not in _tail_cache:
                    _tail_cache[cutoff_value] = _tail_volume_transform_components(
                        spec, _relative, cutoff_value
                    )
                return _tail_cache[cutoff_value]

            cutoff = _budgeted_initial_cutoff(spec, relative, base_cutoff, tail_bound)
            native_component_scales = np.asarray(
                [
                    abs(spec.density_amplitude) * spec.geometric_length_m**2,
                    *([abs(spec.density_amplitude) * spec.geometric_length_m] * 3),
                    *([abs(spec.density_amplitude)] * 9),
                ],
                dtype=np.float64,
            )
            estimate = integrate_transform(
                lambda t, _relative=relative: _cuboid_volume_transform_integrand(
                    spec, _relative, t
                ),
                tail_bound,
                initial_cutoff=cutoff,
                component_scales=native_component_scales,
                acceptance=_acceptance_callback(
                    spec, observables, local_values, local_roundoff
                ),
                numerics=spec.numerics,
            )

        if estimate is None:
            termination = "uncertainty_unresolved"
            estimate_values = estimate_errors = estimate_tails = None
            evaluations = refinements = 0
        else:
            termination = estimate.termination
            estimate_values = estimate.values
            estimate_errors = estimate.component_errors
            estimate_tails = estimate.tail_bounds
            evaluations = estimate.evaluations
            refinements = estimate.refinements
        total_values = (
            None if estimate_values is None else estimate_values + local_values
        )
        total_errors = _combined_component_errors(
            estimate_values, estimate_errors, local_values, local_roundoff
        )
        if termination != "converged":
            diagnostics.extend(
                _diagnostic_records(
                    spec,
                    target_index,
                    target_id,
                    observables,
                    total_values,
                    total_errors,
                    estimate_tails,
                    evaluations,
                    refinements,
                    termination,
                )
            )
            raise ConvergenceError(
                f"cuboid transform failed for target {target_id!r}: {termination}",
                diagnostics,
            )
        if total_values is None or total_errors is None or estimate_tails is None:
            unresolved = _diagnostic_records(
                spec,
                target_index,
                target_id,
                observables,
                None,
                None,
                None,
                evaluations,
                refinements,
                "uncertainty_unresolved",
            )
            raise ConvergenceError(
                "cuboid transform lost numerical evidence", unresolved
            )
        if not np.isfinite(total_values).all():
            unresolved = _diagnostic_records(
                spec,
                target_index,
                target_id,
                observables,
                None,
                None,
                None,
                evaluations,
                refinements,
                "uncertainty_unresolved",
            )
            raise ConvergenceError("cuboid field is not representable", unresolved)
        diagnostics.extend(
            _diagnostic_records(
                spec,
                target_index,
                target_id,
                observables,
                total_values,
                total_errors,
                estimate_tails,
                evaluations,
                refinements,
                "converged",
            )
        )
        values["potential"][target_index, 0, 0] = total_values[0]
        values["acceleration"][target_index, :, 0] = total_values[1:4]
        values["gradient"][target_index, :, 0] = total_values[4:13]

    return NativeEvaluation(
        {name: values[name] for name in observables}, tuple(diagnostics)
    )


__all__ = [
    "CuboidCavitySpec",
    "CuboidAcousticCavityModel",
    "build_cuboid_cavity",
]
