"""Gap-selected spectral kernel for finite cylindrical acoustic cavities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from scipy.special import ive, jnp_zeros, jv, jvp, kve

from gragra._analytic_evaluation import AnalyticEvaluation, NativeEvaluation
from gragra._interactions import Newtonian
from gragra.constants import G_SI
from gragra.exceptions import ConvergenceError, NumericalDiagnostic
from gragra.geometry._cylinder import Cylinder
from gragra.geometry._frame import Frame
from gragra.results._coords import normalize_gps
from gragra.results._gen1_definitions import gen1_field_definition
from gragra.targets._high_points import TargetPoints

if TYPE_CHECKING:
    from gragra.analytic.atmospheric._cavity._model import (
        AcousticCavityNumerics,
        CylinderAcousticMode,
    )

_METHOD = "cylinder_gap_selected_spectral_v1"
_EPS = np.finfo(np.float64).eps
_M1_CARTESIAN_LADDER_ARGUMENT_MAX = 1e-3
_MIN_FLOAT = float(np.nextafter(0.0, 1.0))
_PI_DECIMAL = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
)
_OBSERVABLES = ("potential", "acceleration", "gradient")
_OBSERVABLE_BUDGETS = {
    "potential": (1e-10, 2),
    "acceleration": (1e-10, 1),
    "gradient": (1e-9, 0),
}


def _d(value: float | int) -> Decimal:
    return (
        Decimal.from_float(float(value))
        if not isinstance(value, int)
        else Decimal(value)
    )


def _decimal_to_float_up(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        return result
    if Decimal.from_float(result) < value:
        result = float(np.nextafter(result, np.inf))
    if result == 0.0 and value > 0:
        return _MIN_FLOAT
    return result


def _decimal_to_float_nearest(value: Decimal, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or (result == 0.0 and value != 0):
        raise ValueError(f"{label} is outside finite float64 range")
    return result


def _sqrt_norm_upper(values: object) -> float:
    raw = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(raw).all():
        return math.inf
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        total = Decimal(0)
        for item in raw:
            value = Decimal.from_float(abs(float(item)))
            total = (total + value * value).next_plus()
        return _decimal_to_float_up(total.sqrt().next_plus())


def _sum_upper(values: object) -> float:
    raw = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(raw).all():
        return math.inf
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        total = Decimal(0)
        for item in raw:
            total = (total + Decimal.from_float(float(item))).next_plus()
        return _decimal_to_float_up(total)


def _complex_abs_upper(value: complex) -> Decimal:
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        real = Decimal.from_float(float(value.real))
        imag = Decimal.from_float(float(value.imag))
        square = (real * real + imag * imag).next_plus()
        return square.sqrt().next_plus()


def _decimal_lower_gap(value: float, boundary: float) -> Decimal:
    """Positive lower bound for the represented absolute coordinate gap."""
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_FLOOR
        gap = abs(
            Decimal.from_float(float(value)) - Decimal.from_float(float(boundary))
        )
        return gap.next_minus()


def _cylinder_volume_upper(spec: CylinderCavitySpec) -> Decimal:
    """Return the physical pi*R**2*H volume in outward Decimal arithmetic."""
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        radius = Decimal.from_float(spec.geometry.radius_m)
        height = Decimal.from_float(spec.geometry.height_m)
        return (_PI_DECIMAL * radius**2 * height).next_plus()


def _radial_gap_lower(x: float, y: float, radius: float) -> Decimal:
    """Outward lower bound on ``abs(hypot(x, y) - radius)``."""
    if x == 0.0 and y == 0.0:
        with localcontext() as context:
            context.prec = 100
            context.rounding = ROUND_FLOOR
            return (Decimal.from_float(float(radius))).next_minus()
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_FLOOR
        xd = Decimal.from_float(float(x))
        yd = Decimal.from_float(float(y))
        square_lower = (xd * xd + yd * yd).next_minus()
        if square_lower <= 0:
            square_lower = Decimal(0)
        radial_lower = square_lower.sqrt().next_minus()
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING
        square_upper = (
            Decimal.from_float(float(x)) ** 2 + Decimal.from_float(float(y)) ** 2
        ).next_plus()
        radial_upper = square_upper.sqrt().next_plus()
    rd = Decimal.from_float(float(radius))
    if radial_lower > rd:
        with localcontext() as context:
            context.prec = 100
            context.rounding = ROUND_FLOOR
            return (radial_lower - rd).next_minus()
    if radial_upper < rd:
        with localcontext() as context:
            context.prec = 100
            context.rounding = ROUND_FLOOR
            return (rd - radial_upper).next_minus()
    return Decimal(0)


def _power_series_triplet(
    order: int, x: float, *, modified: bool
) -> tuple[float, float, float]:
    """Stable Bessel value and first two argument derivatives near zero."""
    sign = 1.0 if modified else -1.0
    coefficients = [
        sign**j
        / (math.factorial(j) * math.factorial(order + j) * 2.0 ** (order + 2 * j))
        for j in range(5)
    ]
    value = first = second = 0.0
    for j, coefficient in enumerate(coefficients):
        power = order + 2 * j
        term = coefficient * x**power
        value += term
        if power:
            first += power * coefficient * x ** (power - 1)
        if power >= 2:
            second += power * (power - 1) * coefficient * x ** (power - 2)
    if modified:
        scale = math.exp(-x)
        # The exponential scaling does not alter the underlying Bessel ODE;
        # callers use these as derivatives divided by exp(x).
        return scale * value, scale * first, scale * second
    return value, first, second


def _j_triplet(order: int, x: float) -> tuple[float, float, float]:
    if x < 1e-4:
        return _power_series_triplet(order, x, modified=False)
    value = float(jv(order, x))
    first = float(jvp(order, x, 1))
    second = -first / x - (1.0 - (order / x) ** 2) * value
    return value, first, second


def _i_scaled_triplet(order: int, x: float) -> tuple[float, float, float]:
    if x < 1e-4:
        return _power_series_triplet(order, x, modified=True)
    value = float(ive(order, x))
    first = 0.5 * (float(ive(abs(order - 1), x)) + float(ive(order + 1, x)))
    second = -first / x + (1.0 + (order / x) ** 2) * value
    return value, first, second


def _k_scaled_triplet(order: int, x: float) -> tuple[float, float, float]:
    value = float(kve(order, x))
    first = -0.5 * (float(kve(abs(order - 1), x)) + float(kve(order + 1, x)))
    second = -first / x + (1.0 + (order / x) ** 2) * value
    return value, first, second


def _bessel_derivative_series(
    order: int, root: float, root_value: float, delta: float
) -> float:
    """Evaluate J'_m(root+delta)/delta without a near-root subtraction."""
    coefficients = [0.0] * 12
    coefficients[0] = root_value
    coefficients[1] = 0.0
    for index in range(10):
        previous = coefficients[index - 1] if index >= 1 else 0.0
        previous2 = coefficients[index - 2] if index >= 2 else 0.0
        numerator = (
            root * (index + 1) * (2 * index + 1) * coefficients[index + 1]
            + (index**2 + root**2 - order**2) * coefficients[index]
            + 2.0 * root * previous
            + previous2
        )
        coefficients[index + 2] = -numerator / (root**2 * (index + 2) * (index + 1))
    value = 0.0
    for power in range(2, len(coefficients)):
        value += power * coefficients[power] * delta ** (power - 2)
    return value


def _radial_overlap_j(spec: CylinderCavitySpec, lam: float) -> float:
    radius = spec.geometry.radius_m
    x = lam * radius
    if spec.mode.n == 0:
        if x == 0.0:
            return radius**2 / 2.0
        return radius**2 * float(jv(1, x)) / x
    alpha = spec.alpha
    delta = x - alpha
    if abs(delta) <= 2e-4 * max(1.0, alpha):
        derivative_over_delta = _bessel_derivative_series(
            spec.mode.m, alpha, spec.root_bessel, delta
        )
        ratio = -x * derivative_over_delta / (2.0 * alpha + delta)
    else:
        denominator = alpha**2 - x**2
        ratio = x * float(jvp(spec.mode.m, x, 1)) / denominator
    return radius**2 * spec.root_bessel * ratio


def _radial_overlap_i_scaled(spec: CylinderCavitySpec, kappa: float) -> float:
    radius = spec.geometry.radius_m
    x = kappa * radius
    if spec.mode.n == 0:
        if x == 0.0:
            return radius**2 / 2.0
        return radius**2 * float(ive(1, x)) / x
    _, derivative, _ = _i_scaled_triplet(spec.mode.m, x)
    return radius**2 * spec.root_bessel * x * derivative / (spec.alpha**2 + x**2)


def _axial_overlap(
    q: int, height: float, z: float, wave_number: float
) -> tuple[float, float, float]:
    """Stable C_q and its first two target-z derivatives."""
    beta = q * math.pi / height
    half = height / 2.0
    if wave_number == 0.0:
        return (height if q == 0 else 0.0), 0.0, 0.0

    denominator = wave_number**2 - beta**2
    away_from_resonance = abs(denominator) > 1e-4 * (wave_number**2 + beta**2)
    if q == 0:
        amplitude = 2.0 * math.sin(wave_number * half) / wave_number
    elif away_from_resonance:
        numerator = (
            math.sin(wave_number * half) if q % 2 == 0 else math.cos(wave_number * half)
        )
        amplitude = 2.0 * wave_number * numerator / denominator
    else:
        lower = (wave_number - beta) * half / math.pi
        upper = (wave_number + beta) * half / math.pi
        if q % 2 == 0:
            parity = -1.0 if (q // 2) % 2 else 1.0
            amplitude = parity * half * (float(np.sinc(lower)) + float(np.sinc(upper)))
        else:
            parity = -1.0 if ((q + 1) // 2) % 2 else 1.0
            amplitude = parity * half * (float(np.sinc(lower)) - float(np.sinc(upper)))

    if q % 2 == 0:
        value = amplitude * math.cos(wave_number * z)
        first = -wave_number * amplitude * math.sin(wave_number * z)
    else:
        value = amplitude * math.sin(wave_number * z)
        first = wave_number * amplitude * math.cos(wave_number * z)
    second = -(wave_number**2) * value
    return value, first, second


def _angular_projection(value: object, m: int, phi: float, azimuth: str):
    phase = np.exp(1j * m * phi)
    complex_value = phase * np.asarray(value, dtype=np.complex128)
    return complex_value.real if azimuth == "cos" else complex_value.imag


def _spatial_derivatives(
    m: int,
    azimuth: str,
    radius: float,
    phi: float,
    q: float,
    qr: float,
    qz: float,
    qrr: float,
    qrz: float,
    qzz: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    if radius == 0.0:
        raise ValueError("axis derivatives require the analytic axis expansion")
    if m == 1:
        return _m1_cartesian_derivatives(azimuth, radius, phi, q, qr, qz, qrr, qrz, qzz)
    phase = np.exp(1j * m * phi)
    radial_gradient = np.array([qr, 1j * m * q / radius, qz], dtype=np.complex128)
    cylindrical_hessian = np.array(
        [
            [qrr, 1j * m * (qr / radius - q / radius**2), qrz],
            [
                1j * m * (qr / radius - q / radius**2),
                qr / radius - m**2 * q / radius**2,
                1j * m * qz / radius,
            ],
            [qrz, 1j * m * qz / radius, qzz],
        ],
        dtype=np.complex128,
    )
    cosine, sine = math.cos(phi), math.sin(phi)
    basis = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    projected_value = _angular_projection(q, m, phi, azimuth)
    projected_gradient = _angular_projection(
        phase * (basis @ radial_gradient), 0, 0.0, azimuth
    )
    projected_hessian = _angular_projection(
        phase * (basis @ cylindrical_hessian @ basis.T), 0, 0.0, azimuth
    )
    return (
        float(projected_value),
        np.asarray(projected_gradient, dtype=np.float64),
        np.asarray(projected_hessian, dtype=np.float64),
    )


def _m1_cartesian_derivatives(
    azimuth: str,
    radius: float,
    phi: float,
    q: float,
    qr: float,
    qz: float,
    qrr: float,
    qrz: float,
    qzz: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate an m=1 radial field in Cartesian form without inference.

    Decimal arithmetic evaluates radial derivative numerators from the
    represented inputs before division. Bessel call sites with known small
    arguments use exact neighboring-order Cartesian ladders instead.
    """
    cosine, sine = math.cos(phi), math.sin(phi)
    direction = np.array([cosine, sine], dtype=np.float64)
    transverse = np.array([radius * cosine, radius * sine], dtype=np.float64)
    complex_position = complex(transverse[0], transverse[1])
    with localcontext() as context:
        context.prec = 80
        r_decimal = Decimal.from_float(radius)
        q_decimal = Decimal.from_float(q)
        qr_decimal = Decimal.from_float(qr)
        qrr_decimal = Decimal.from_float(qrr)
        qz_decimal = Decimal.from_float(qz)
        qrz_decimal = Decimal.from_float(qrz)
        h_r = float((r_decimal * qr_decimal - q_decimal) / r_decimal**2)
        h_rr = float(
            (
                r_decimal**2 * qrr_decimal
                - Decimal(2) * r_decimal * qr_decimal
                + Decimal(2) * q_decimal
            )
            / r_decimal**3
        )
        h_rz = float((r_decimal * qrz_decimal - qz_decimal) / r_decimal**2)
    h = q / radius
    h_z = qz / radius
    h_zz = qzz / radius
    unit_gradient = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    complex_gradient = np.empty(3, dtype=np.complex128)
    complex_gradient[:2] = unit_gradient * h + complex_position * h_r * direction
    complex_gradient[2] = complex_position * h_z

    complex_hessian = np.zeros((3, 3), dtype=np.complex128)
    transverse_hessian = np.empty((2, 2), dtype=np.complex128)
    h_r_over_r = h_r / radius
    for row in range(2):
        for column in range(2):
            transverse_hessian[row, column] = (
                unit_gradient[row] * h_r * direction[column]
                + unit_gradient[column] * h_r * direction[row]
                + complex_position
                * (
                    h_rr * direction[row] * direction[column]
                    + h_r_over_r
                    * (
                        (1.0 if row == column else 0.0)
                        - direction[row] * direction[column]
                    )
                )
            )
    complex_hessian[:2, :2] = transverse_hessian
    complex_hessian[:2, 2] = unit_gradient * h_z + complex_position * h_rz * direction
    complex_hessian[2, :2] = complex_hessian[:2, 2]
    complex_hessian[2, 2] = complex_position * h_zz

    if azimuth == "cos":
        return (
            float(np.real(complex_position * h)),
            complex_gradient.real,
            complex_hessian.real,
        )
    return (
        float(np.imag(complex_position * h)),
        complex_gradient.imag,
        complex_hessian.imag,
    )


def _m1_bessel_cartesian(
    spec: CylinderCavitySpec,
    radius: float,
    phi: float,
    wave_number: float,
    family: Literal["j", "i"],
    radial_scale: float,
    axial: tuple[float, float, float],
    *,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Use exact neighboring-order Bessel ladders for an m=1 radial factor."""
    argument = wave_number * radius

    def harmonic(order: int) -> complex:
        if family == "j":
            radial = float(jv(order, argument))
        else:
            radial = float(ive(abs(order), argument))
        return radial * np.exp(1j * order * phi)

    previous2 = harmonic(-1)
    previous = harmonic(0)
    center = harmonic(1)
    following = harmonic(2)
    following2 = harmonic(3)
    if family == "j":
        gradient_x = 0.5 * wave_number * (previous - following)
        gradient_y = 0.5j * wave_number * (previous + following)
        hessian_xx = wave_number**2 * (previous2 - 2.0 * center + following2) / 4.0
        hessian_yy = -(wave_number**2) * (previous2 + 2.0 * center + following2) / 4.0
    else:
        gradient_x = 0.5 * wave_number * (previous + following)
        gradient_y = 0.5j * wave_number * (previous - following)
        hessian_xx = wave_number**2 * (previous2 + 2.0 * center + following2) / 4.0
        hessian_yy = -(wave_number**2) * (previous2 - 2.0 * center + following2) / 4.0
    hessian_xy = 0.25j * wave_number**2 * (previous2 - following2)

    axial_value, axial_z, axial_zz = axial
    value = center * axial_value
    gradient = np.array(
        [gradient_x * axial_value, gradient_y * axial_value, center * axial_z]
    )
    hessian = np.array(
        [
            [hessian_xx * axial_value, hessian_xy * axial_value, gradient_x * axial_z],
            [hessian_xy * axial_value, hessian_yy * axial_value, gradient_y * axial_z],
            [gradient_x * axial_z, gradient_y * axial_z, center * axial_zz],
        ],
        dtype=np.complex128,
    )
    if spec.mode.azimuth == "cos":
        value = complex(value).real
        gradient = np.asarray(gradient).real
        hessian = hessian.real
    else:
        value = complex(value).imag
        gradient = np.asarray(gradient).imag
        hessian = hessian.imag
    coefficient = spec.pressure_over_c2k2 * radial_scale * scale_factor
    fields = np.empty(13, dtype=np.complex128)
    fields[0] = -coefficient * value
    fields[1:4] = coefficient * gradient
    fields[4:13] = coefficient * hessian.reshape(9)
    return fields


def _j_bessel_cartesian_native_vector(
    spec: CylinderCavitySpec,
    relative: np.ndarray,
    radius: float,
    axial: tuple[float, float, float],
    *,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Evaluate a J_m radial mode and its derivatives in Cartesian form.

    Angular harmonics are formed from the represented x/r and y/r values.
    This preserves tiny nonzero sine phases on either side of the x axis,
    where an ``atan2`` angle can round to exactly +/-pi.
    """
    x, y = float(relative[0]), float(relative[1])
    x_unit, y_unit = x / radius, y / radius
    wave_number = spec.radial_wavenumber_rad_m
    argument = wave_number * radius
    order = spec.mode.m

    def harmonic(index: int) -> complex:
        cosine, sine, _cosine_error, _sine_error = _multiple_angle_data(
            index, x_unit, y_unit
        )
        magnitude = abs(index)
        radial = _j_triplet(magnitude, argument)[0]
        if index < 0 and magnitude % 2:
            radial = -radial
        return radial * complex(cosine, sine)

    lower_two = harmonic(order - 2)
    lower_one = harmonic(order - 1)
    center = harmonic(order)
    upper_one = harmonic(order + 1)
    upper_two = harmonic(order + 2)
    gradient_x = 0.5 * wave_number * (lower_one - upper_one)
    gradient_y = 0.5j * wave_number * (lower_one + upper_one)
    hessian_xx = wave_number**2 * (lower_two - 2.0 * center + upper_two) / 4.0
    hessian_yy = -(wave_number**2) * (lower_two + 2.0 * center + upper_two) / 4.0
    hessian_xy = 0.25j * wave_number**2 * (lower_two - upper_two)
    axial_value, axial_z, axial_zz = axial
    complex_components = np.array(
        [
            -center * axial_value,
            gradient_x * axial_value,
            gradient_y * axial_value,
            center * axial_z,
            hessian_xx * axial_value,
            hessian_xy * axial_value,
            gradient_x * axial_z,
            hessian_xy * axial_value,
            hessian_yy * axial_value,
            gradient_y * axial_z,
            gradient_x * axial_z,
            gradient_y * axial_z,
            center * axial_zz,
        ],
        dtype=np.complex128,
    )
    spatial_components = (
        complex_components.real
        if spec.mode.azimuth == "cos"
        else complex_components.imag
    )
    coefficient = (
        spec.pressure_over_c2k2 * (4.0 * math.pi) * spec.normalization * scale_factor
    )
    return np.asarray(coefficient * spatial_components, dtype=np.complex128)


def _axis_derivatives(
    m: int,
    azimuth: str,
    radial_coefficient: float,
    next_radial_coefficient: float,
    axial: float,
    axial_z: float,
    axial_zz: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    value = 0.0
    gradient = np.zeros(3, dtype=np.complex128)
    hessian = np.zeros((3, 3), dtype=np.complex128)
    if m == 0:
        value = radial_coefficient * axial
        gradient[2] = radial_coefficient * axial_z
        hessian[0, 0] = hessian[1, 1] = 2.0 * next_radial_coefficient * axial
        hessian[2, 2] = radial_coefficient * axial_zz
    elif m == 1:
        gradient[0] = radial_coefficient * axial
        gradient[1] = 1j * radial_coefficient * axial
        hessian[0, 2] = hessian[2, 0] = radial_coefficient * axial_z
        hessian[1, 2] = hessian[2, 1] = 1j * radial_coefficient * axial_z
    elif m == 2:
        hessian[0, 0] = 2.0 * radial_coefficient * axial
        hessian[0, 1] = hessian[1, 0] = 2j * radial_coefficient * axial
        hessian[1, 1] = -2.0 * radial_coefficient * axial
    if azimuth == "cos":
        return float(np.real(value)), gradient.real, hessian.real
    return float(np.imag(value)), gradient.imag, hessian.imag


def _pressure_over_c2k2(pressure: complex, sound_speed: float, k: float) -> complex:
    if pressure == 0.0j:
        return 0.0j
    with localcontext() as context:
        context.prec = 90
        denominator = Decimal.from_float(sound_speed) ** 2 * Decimal.from_float(k) ** 2
        real = Decimal.from_float(float(pressure.real)) / denominator
        imag = Decimal.from_float(float(pressure.imag)) / denominator
    return complex(
        _decimal_to_float_nearest(real, "pressure/(c_s^2*k^2)"),
        _decimal_to_float_nearest(imag, "pressure/(c_s^2*k^2)"),
    )


@dataclass(frozen=True, eq=False, slots=True, init=False)
class CylinderCavitySpec:
    """Immutable physical snapshot for one cylindrical Neumann mode."""

    geometry: Cylinder
    mode: CylinderAcousticMode
    pressure_amplitude_pa: complex
    sound_speed_m_s: float
    pressure_over_c2k2: complex
    reference_time_gps: tuple[int, int]
    alpha: float
    root_bessel: float
    radial_wavenumber_rad_m: float
    axial_wavenumber_rad_m: float
    wavenumber_rad_m: float
    normalization: float
    frequency_hz: float
    root_relative_error_estimate: float
    numerics: AcousticCavityNumerics
    field_definition: object
    interaction: Newtonian
    parameters: QTable
    formulas: pd.DataFrame
    references: pd.DataFrame

    def __init__(
        self,
        geometry: Cylinder,
        mode: CylinderAcousticMode,
        pressure_amplitude: object,
        *,
        sound_speed: object,
        reference_time: object,
        numerics: AcousticCavityNumerics,
    ) -> None:
        from gragra.analytic.atmospheric._cavity._model import (
            AcousticCavityNumerics as _CavityNumerics,
        )
        from gragra.analytic.atmospheric._cavity._model import (
            CylinderAcousticMode as _CylinderAcousticMode,
        )
        from gragra.analytic.atmospheric._cavity._model import (
            _positive_scalar_quantity,
            _pressure_amplitude,
        )

        if not isinstance(geometry, Cylinder):
            raise TypeError("geometry must be a Cylinder")
        if not isinstance(mode, _CylinderAcousticMode):
            raise TypeError("mode must be a CylinderAcousticMode")
        if not isinstance(numerics, _CavityNumerics):
            raise TypeError("numerics must be an AcousticCavityNumerics")
        pressure = _pressure_amplitude(pressure_amplitude)
        speed = _positive_scalar_quantity("sound_speed", sound_speed, u.m / u.s)
        reference_gps = normalize_gps(reference_time)

        if mode.n == 0:
            alpha = 0.0
            root_bessel = 1.0
            radial_wave_number = 0.0
            radial_factor = 1.0
        else:
            try:
                alpha = float(jnp_zeros(mode.m, mode.n)[-1])
            except (ValueError, OverflowError, FloatingPointError) as exc:
                raise ValueError(
                    "cylindrical radial Neumann root is unresolved"
                ) from exc
            root_bessel = float(jv(mode.m, alpha))
            radial_wave_number = alpha / geometry.radius_m
            radial_gap = (
                1.0 if mode.m == 0 else ((alpha - mode.m) * (alpha + mode.m)) / alpha**2
            )
            radial_factor = root_bessel**2 * radial_gap
        beta = mode.q * math.pi / geometry.height_m
        wave_number = math.hypot(radial_wave_number, beta)
        eta_q = 1.0 if mode.q == 0 else 0.5
        eta_m = 1.0 if mode.m == 0 else 0.5
        normalization = (
            math.sqrt(2.0)
            if mode.n == 0
            else 1.0 / math.sqrt(eta_q * eta_m * radial_factor)
        )
        frequency = speed * wave_number / (2.0 * math.pi)
        derived = np.asarray(
            [
                alpha,
                root_bessel,
                radial_wave_number,
                beta,
                wave_number,
                normalization,
                frequency,
            ]
        )
        if (
            (mode.n == 0 and mode.q == 0)
            or not np.isfinite(derived).all()
            or wave_number <= 0
            or normalization <= 0
        ):
            raise ValueError(
                "cylinder mode eigenvalue and normalization must be finite"
            )
        coefficient = _pressure_over_c2k2(pressure, speed, wave_number)
        if pressure != 0.0j and coefficient == 0.0j:
            raise ValueError("nonzero pressure scale underflow in cylinder coefficient")

        root_residual = abs(float(jvp(mode.m, alpha, 1))) if mode.n else 0.0
        root_curvature = abs(float(jvp(mode.m, alpha, 2))) if mode.n else 1.0
        root_error = root_residual / root_curvature + 8.0 * _EPS * max(alpha, 1.0)
        root_relative = root_error / max(alpha, 1.0)
        if not np.isfinite(root_relative):
            raise ValueError("cylinder root uncertainty estimate is not finite")

        center = geometry.center_m
        definition = gen1_field_definition("acoustic_cavity")
        parameters = QTable(
            {
                "geometry": ["cylinder"],
                "center_x": [center[0] * u.m],
                "center_y": [center[1] * u.m],
                "center_z": [center[2] * u.m],
                "radius": [geometry.radius_m * u.m],
                "height": [geometry.height_m * u.m],
                "m": [mode.m],
                "n": [mode.n],
                "q": [mode.q],
                "azimuth": [mode.azimuth],
                "positive_radial_derivative_root_index": [mode.n],
                "alpha": [alpha * u.dimensionless_unscaled],
                "root_bessel": [root_bessel],
                "radial_wavenumber": [radial_wave_number / u.m],
                "axial_wavenumber": [beta / u.m],
                "wavenumber": [wave_number / u.m],
                "normalization": [normalization],
                "pressure_amplitude": [pressure * u.Pa],
                "pressure_convention": ["spatial-RMS peak phasor"],
                "sound_speed": [speed * u.m / u.s],
                "frequency": [frequency * u.Hz],
                "reference_time_gps_seconds": [reference_gps[0]],
                "reference_time_gps_nanoseconds": [reference_gps[1]],
                "numerical_route": ["gap_selected_cylindrical_spectral_v1"],
                "max_evaluations": [numerics.max_evaluations],
                "max_refinements": [numerics.max_refinements],
                "accuracy_policy": ["gen1_cavity_v1"],
                "absolute_scale_factor": [1e-12],
                "relative_phi": [1e-10],
                "relative_acceleration": [1e-10],
                "relative_gradient": [1e-9],
                "runtime_budget_fraction": [0.25],
                "source_frame": [geometry.frame.name],
            }
        )
        formulas = pd.DataFrame(
            [
                {
                    "observable": name,
                    "expression": expression,
                    "convention": (
                        "G-free Green boundary transform plus interior Helmholtz term; "
                        "bridge applies G once"
                    ),
                }
                for name, expression in (
                    ("potential", "Phi = -A/k^2 * (S + 4*pi*Psi inside)"),
                    ("acceleration", "a = +A/k^2 * grad(S + 4*pi*Psi inside)"),
                    ("gradient", "Gamma = +A/k^2 * Hessian(S + 4*pi*Psi inside)"),
                )
            ]
        )
        references = pd.DataFrame(
            [
                {
                    "source": (
                        "docs/design/gen1/cavity-transforms.md"
                        "#cylinder-modes-and-exact-overlaps"
                    ),
                    "detail": (
                        "real-mode normalization, exact overlaps, and "
                        "gap-selected transforms"
                    ),
                },
                {
                    "source": (
                        "docs/design/gen1/cavity-transforms.md"
                        "#cylinder-rigorous-tail-coefficients"
                    ),
                    "detail": (
                        "outward omitted-integral bounds for each Cartesian derivative"
                    ),
                },
                {
                    "source": "gragra.constants.G_SI",
                    "detail": "applied exactly once by the shared analytic bridge",
                },
            ]
        )
        for name, value in (
            ("geometry", geometry),
            ("mode", mode),
            ("pressure_amplitude_pa", pressure),
            ("sound_speed_m_s", speed),
            ("pressure_over_c2k2", coefficient),
            ("reference_time_gps", reference_gps),
            ("alpha", alpha),
            ("root_bessel", root_bessel),
            ("radial_wavenumber_rad_m", radial_wave_number),
            ("axial_wavenumber_rad_m", beta),
            ("wavenumber_rad_m", wave_number),
            ("normalization", normalization),
            ("frequency_hz", frequency),
            ("root_relative_error_estimate", root_relative),
            ("numerics", numerics),
            ("field_definition", definition),
            ("interaction", Newtonian()),
            ("parameters", parameters),
            ("formulas", formulas),
            ("references", references),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CylinderAcousticCavityModel:
    """High-level single-frequency cylindrical cavity evaluator."""

    spec: CylinderCavitySpec

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = _OBSERVABLES,
        output_frame: Frame | None = None,
    ) -> AnalyticEvaluation:
        if not isinstance(targets, TargetPoints):
            raise TypeError("targets must be high-level TargetPoints")
        output = targets.frame if output_frame is None else output_frame
        if not isinstance(output, Frame):
            raise TypeError("output_frame must be a Frame")

        def validate_targets(
            target_positions_m: np.ndarray, boundary_error: np.ndarray
        ) -> None:
            _validate_cylinder_targets(
                self.spec.geometry,
                target_positions_m,
                boundary_error,
            )

        def evaluate_native(
            target_positions_m: np.ndarray, names: tuple[str, ...]
        ) -> NativeEvaluation:
            return _evaluate_cylinder(self.spec, target_positions_m, targets.ids, names)

        from gragra.analytic.atmospheric._cavity._model import (
            _build_cavity_evaluation,
        )

        return _build_cavity_evaluation(
            self.spec,
            targets,
            observables=observables,
            output_frame=output,
            evaluate_native=evaluate_native,
            validate_targets=validate_targets,
        )


def build_cylinder_cavity(
    geometry: Cylinder,
    mode: CylinderAcousticMode,
    pressure_amplitude: object,
    *,
    sound_speed: object,
    reference_time: object,
    numerics: AcousticCavityNumerics,
) -> CylinderAcousticCavityModel:
    return CylinderAcousticCavityModel(
        CylinderCavitySpec(
            geometry,
            mode,
            pressure_amplitude,
            sound_speed=sound_speed,
            reference_time=reference_time,
            numerics=numerics,
        )
    )


def _validate_cylinder_targets(
    geometry: Cylinder,
    target_positions_m: np.ndarray,
    boundary_error: np.ndarray,
) -> None:
    center = geometry.center_m
    half_height = geometry.height_m / 2.0
    relative_positions = target_positions_m - center
    if boundary_error.shape != (len(target_positions_m),):
        raise ValueError("boundary_error must have one value per target")
    for index, point in enumerate(relative_positions):
        radius = math.hypot(float(point[0]), float(point[1]))
        tolerance = float(boundary_error[index])
        side = (
            abs(radius - geometry.radius_m) <= tolerance
            and abs(float(point[2])) <= half_height + tolerance
        )
        cap = (
            abs(abs(float(point[2])) - half_height) <= tolerance
            and radius <= geometry.radius_m + tolerance
        )
        if side or cap:
            raise ValueError(
                f"target point at index {index} lies on the cylinder boundary"
            )


def _radial_axis_coefficients(
    family: Literal["j", "i"], order: int, wave_number: float
) -> tuple[float, float]:
    leading = (wave_number / 2.0) ** order / math.factorial(order)
    sign = -1.0 if family == "j" else 1.0
    next_coefficient = sign * leading * wave_number**2 / (4.0 * (order + 1))
    return leading, next_coefficient


def _native_vector(
    spec: CylinderCavitySpec,
    radius: float,
    phi: float,
    derivatives: tuple[float, float, float, float, float, float],
    *,
    axis_coefficients: tuple[float, ...] | None = None,
    scale_factor: float = 1.0,
) -> np.ndarray:
    q, qr, qz, qrr, qrz, qzz = derivatives
    if axis_coefficients is None:
        spatial_value, spatial_gradient, spatial_hessian = _spatial_derivatives(
            spec.mode.m,
            spec.mode.azimuth,
            radius,
            phi,
            q,
            qr,
            qz,
            qrr,
            qrz,
            qzz,
        )
    else:
        spatial_value, spatial_gradient, spatial_hessian = _axis_derivatives(
            spec.mode.m,
            spec.mode.azimuth,
            axis_coefficients[0],
            axis_coefficients[1],
            q,
            qz,
            qzz,
        )
    coefficient = spec.pressure_over_c2k2 * scale_factor
    fields = np.empty(13, dtype=np.complex128)
    fields[0] = -coefficient * spatial_value
    fields[1:4] = coefficient * spatial_gradient
    fields[4:13] = coefficient * np.asarray(spatial_hessian).reshape(9)
    return fields


def _integer_exponential_tail(delta: Decimal, cutoff: float, power: int) -> Decimal:
    """Upper bound on integral_K^inf t**power exp(-delta*t) dt."""
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_FLOOR
        d_lower = delta
        k_exact = Decimal.from_float(float(cutoff))
        exponent_lower = (d_lower * k_exact).next_minus()
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        exponential = (-exponent_lower).exp().next_plus()
        d_lower = delta
        k_exact = Decimal.from_float(float(cutoff))
        total = Decimal(0)
        factorial = math.factorial(power)
        for index in range(power + 1):
            k_power = power - index
            coefficient = Decimal(factorial // math.factorial(k_power))
            numerator = coefficient * (k_exact**k_power)
            denominator = d_lower ** (index + 1)
            term = (numerator / denominator).next_plus()
            total = (total + term).next_plus()
        return (exponential * total).next_plus()


def _modified_tail_factor(
    *,
    normalization: float,
    radius: float,
    height: float,
    order: int,
    cutoff: float,
    cap: bool,
) -> Decimal:
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_FLOOR
        k = Decimal.from_float(float(cutoff))
        r = Decimal.from_float(float(radius))
        denominator = (2 * k * r).next_minus()
        exponent_upper = (Decimal((order + 3) ** 2) / denominator).next_plus()
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        exponential_upper = exponent_upper.exp().next_plus()
        pi_upper = Decimal(22) / Decimal(7)
        sqrt_factor = (pi_upper / (2 * Decimal.from_float(radius))).sqrt().next_plus()
        norm = Decimal.from_float(float(normalization)).next_plus()
        r_decimal = Decimal.from_float(float(radius))
        if cap:
            prefactor = Decimal(2) * norm * r_decimal**2 * sqrt_factor
        else:
            prefactor = (
                Decimal(4)
                * norm
                * r_decimal
                * Decimal.from_float(float(height))
                * sqrt_factor
            )
        return (prefactor * exponential_upper).next_plus()


def _tail_vector(
    spec: CylinderCavitySpec,
    route: str,
    cutoff: float,
    gap_lower: Decimal,
) -> np.ndarray:
    tail = np.zeros(13, dtype=np.float64)
    if spec.pressure_amplitude_pa == 0.0j:
        return tail
    if not np.isfinite(cutoff) or cutoff <= 0.0 or gap_lower <= 0:
        return np.full(13, math.inf)
    coefficient = _complex_abs_upper(spec.pressure_over_c2k2)
    values: list[float] = []
    for derivative_order in range(3):
        power = derivative_order + 1
        if route == "cap_inner":
            with localcontext() as context:
                context.prec = 90
                context.rounding = ROUND_FLOOR
                radius_lower = Decimal.from_float(spec.geometry.radius_m).next_minus()
                cutoff_exact = Decimal.from_float(float(cutoff))
                cutoff_lower = cutoff_exact.next_minus()
            with localcontext() as context:
                context.prec = 90
                context.rounding = ROUND_CEILING
                radius_upper = Decimal.from_float(spec.geometry.radius_m).next_plus()
                norm_upper = Decimal.from_float(spec.normalization).next_plus()
                pi_upper = Decimal(22) / Decimal(7)
                sqrt_factor = (
                    (pi_upper / (Decimal(2) * radius_lower)).sqrt().next_plus()
                )
                exponent = (
                    Decimal((spec.mode.m + 1) ** 2)
                    / (Decimal(2) * radius_lower * cutoff_lower)
                ).next_plus()
                exponent_factor = exponent.exp().next_plus()
                coefficient_bound = (
                    Decimal(4)
                    * norm_upper
                    * radius_upper
                    * sqrt_factor
                    * exponent_factor
                ).next_plus()
            for derivative_order in range(3):
                moment = _integer_exponential_tail(gap_lower, cutoff, derivative_order)
                with localcontext() as context:
                    context.prec = 90
                    context.rounding = ROUND_CEILING
                    sqrt_cutoff_lower = cutoff_lower.sqrt().next_minus()
                    moment_factor = (moment / sqrt_cutoff_lower).next_plus()
                    bound = (
                        coefficient * coefficient_bound * moment_factor
                    ).next_plus()
                values.append(_decimal_to_float_up(bound))
            tail[0] = values[0]
            tail[1:4] = values[1]
            tail[4:13] = values[2]
            return tail
        if route in {"side_modified", "cap_modified"}:
            base = _modified_tail_factor(
                normalization=spec.normalization,
                radius=spec.geometry.radius_m,
                height=spec.geometry.height_m,
                order=spec.mode.m,
                cutoff=cutoff,
                cap=route == "cap_modified",
            )
            moment = _integer_exponential_tail(gap_lower, cutoff, power)
            with localcontext() as context:
                context.prec = 90
                context.rounding = ROUND_CEILING
                moment_sqrt = Decimal.from_float(float(cutoff)).sqrt().next_minus()
                moment_factor = (moment / moment_sqrt).next_plus()
                bound = (coefficient * base * moment_factor).next_plus()
        else:
            moment = _integer_exponential_tail(gap_lower, cutoff, power)
            with localcontext() as context:
                context.prec = 90
                context.rounding = ROUND_CEILING
                n = Decimal.from_float(spec.normalization)
                radius = Decimal.from_float(spec.geometry.radius_m)
                if route == "cap_hankel":
                    prefactor = Decimal(22) / Decimal(7) * n * radius**2
                else:
                    prefactor = (
                        Decimal(2)
                        * Decimal(22)
                        / Decimal(7)
                        * n
                        * radius
                        * Decimal.from_float(spec.geometry.height_m)
                    )
                bound = (coefficient * prefactor * moment).next_plus()
        values.append(_decimal_to_float_up(bound))
    tail[0] = values[0]
    tail[1:4] = values[1]
    tail[4:13] = values[2]
    return tail


def _run_budget_abs_si(spec: CylinderCavitySpec, observable: str) -> Decimal:
    """The fixed absolute part of B_run in outward SI arithmetic."""
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        pressure_abs = _complex_abs_upper(spec.pressure_amplitude_pa)
        speed = Decimal.from_float(spec.sound_speed_m_s)
        volume = _cylinder_volume_upper(spec)
        length = volume ** (Decimal(1) / Decimal(3))
        length = length.next_plus()
        g = Decimal.from_float(float(G_SI))
        scale_power = _OBSERVABLE_BUDGETS[observable][1]
        scale = g * pressure_abs / speed**2 * length**scale_power
        return (Decimal("0.25") * Decimal("1e-12") * scale).next_plus()


def _observable_indices(observable: str) -> slice:
    return {
        "potential": slice(0, 1),
        "acceleration": slice(1, 4),
        "gradient": slice(4, 13),
    }[observable]


def _decimal_norm(values: np.ndarray) -> Decimal:
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        total = Decimal(0)
        for value in np.asarray(values, dtype=np.float64).ravel():
            item = Decimal.from_float(abs(float(value)))
            total = (total + item * item).next_plus()
        return total.sqrt().next_plus()


def _complex_norm_upper(values: np.ndarray) -> Decimal:
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        total = Decimal(0)
        for value in np.asarray(values, dtype=np.complex128).ravel():
            magnitude = _complex_abs_upper(complex(value))
            total = (total + magnitude * magnitude).next_plus()
        return total.sqrt().next_plus()


def _is_accepted(
    errors: np.ndarray,
    tails: np.ndarray,
    observables: tuple[str, ...],
    budgets: dict[str, Decimal],
) -> bool:
    g = Decimal.from_float(float(G_SI))
    for observable in observables:
        indices = _observable_indices(observable)
        with localcontext() as context:
            context.prec = 90
            context.rounding = ROUND_CEILING
            total = (g * _decimal_norm(errors[indices])).next_plus()
            total += (g * _decimal_norm(tails[indices])).next_plus()
            total = total.next_plus()
        if total > budgets[observable]:
            return False
    return True


def _dec_diag(value: Decimal | None) -> float | None:
    return None if value is None else _decimal_to_float_up(value)


def _local_mode_vector(spec: CylinderCavitySpec, relative: np.ndarray) -> np.ndarray:
    radius = math.hypot(float(relative[0]), float(relative[1]))
    phi = math.atan2(float(relative[1]), float(relative[0])) if radius else 0.0
    z = float(relative[2])
    beta = spec.axial_wavenumber_rad_m
    axial_argument = beta * (z + spec.geometry.height_m / 2.0)
    axial = math.cos(axial_argument)
    axial_z = -beta * math.sin(axial_argument)
    axial_zz = -(beta**2) * axial
    if radius == 0.0:
        if spec.mode.n == 0:
            radial_leading, radial_next = 1.0, 0.0
        else:
            radial_leading, radial_next = _radial_axis_coefficients(
                "j", spec.mode.m, spec.radial_wavenumber_rad_m
            )
        return _native_vector(
            spec,
            radius,
            phi,
            (axial, 0.0, axial_z, 0.0, 0.0, axial_zz),
            axis_coefficients=(
                spec.normalization * radial_leading,
                spec.normalization * radial_next,
            ),
            scale_factor=4.0 * math.pi,
        )
    if (
        spec.mode.m == 1
        and spec.mode.n != 0
        and spec.radial_wavenumber_rad_m * radius <= _M1_CARTESIAN_LADDER_ARGUMENT_MAX
    ):
        return _m1_bessel_cartesian(
            spec,
            radius,
            phi,
            spec.radial_wavenumber_rad_m,
            "j",
            spec.normalization,
            (axial, axial_z, axial_zz),
            scale_factor=4.0 * math.pi,
        )
    if spec.mode.n == 0:
        radial, radial_first, radial_second = 1.0, 0.0, 0.0
    else:
        value, first, second = _j_triplet(
            spec.mode.m, spec.radial_wavenumber_rad_m * radius
        )
        radial = value
        radial_first = spec.radial_wavenumber_rad_m * first
        radial_second = spec.radial_wavenumber_rad_m**2 * second
    n = spec.normalization
    derivatives = (
        n * radial * axial,
        n * radial_first * axial,
        n * radial * axial_z,
        n * radial_second * axial,
        n * radial_first * axial_z,
        n * radial * axial_zz,
    )
    return _native_vector(
        spec,
        radius,
        phi,
        derivatives,
        scale_factor=4.0 * math.pi,
    )


def _local_inner_cap_pair_axial(
    spec: CylinderCavitySpec, z: float
) -> tuple[float, float, float]:
    """Return local-plus-two-inner-cap axial factor and its z derivatives."""
    half = spec.geometry.height_m / 2.0
    radial_wave_number = spec.radial_wavenumber_rad_m
    beta = spec.axial_wavenumber_rad_m
    if not -half < z < half:
        raise ValueError("paired inner-cap terms require an interior target")

    lower_decay = math.exp(-radial_wave_number * (half - z))
    upper_decay = math.exp(-radial_wave_number * (half + z))
    if spec.mode.q % 2 == 0:
        parity = -1.0 if (spec.mode.q // 2) % 2 else 1.0
        phase = beta * z
        local_offset = (
            -2.0 * math.sin(phase / 2.0) ** 2
            if parity > 0.0
            else -2.0 * math.cos(phase / 2.0) ** 2
        )
        bulk_offset = -0.5 * (
            math.expm1(-radial_wave_number * (half - z))
            + math.expm1(-radial_wave_number * (half + z))
        )
        value = local_offset + bulk_offset
        first = -parity * beta * math.sin(phase) - 0.5 * radial_wave_number * (
            lower_decay - upper_decay
        )
        second = -parity * beta**2 * math.cos(phase) - 0.5 * radial_wave_number**2 * (
            lower_decay + upper_decay
        )
        return value, first, second

    parity = -1.0 if ((spec.mode.q + 1) // 2) % 2 else 1.0
    phase = beta * z
    value = parity * math.sin(phase) + 0.5 * (lower_decay - upper_decay)
    first = parity * beta * math.cos(phase) + 0.5 * radial_wave_number * (
        lower_decay + upper_decay
    )
    second = -parity * beta**2 * math.sin(phase) + 0.5 * radial_wave_number**2 * (
        lower_decay - upper_decay
    )
    return value, first, second


def _local_inner_cap_pair_vector(
    spec: CylinderCavitySpec, relative: np.ndarray
) -> np.ndarray:
    """Evaluate the local mode and both inner-cap bulk terms as one function."""
    radius = math.hypot(float(relative[0]), float(relative[1]))
    phi = math.atan2(float(relative[1]), float(relative[0])) if radius else 0.0
    axial, axial_z, axial_zz = _local_inner_cap_pair_axial(spec, float(relative[2]))
    radial_wave_number = spec.radial_wavenumber_rad_m
    if radius == 0.0:
        if spec.mode.n == 0:
            radial_leading, radial_next = 1.0, 0.0
        else:
            radial_leading, radial_next = _radial_axis_coefficients(
                "j", spec.mode.m, radial_wave_number
            )
        return _native_vector(
            spec,
            radius,
            phi,
            (axial, 0.0, axial_z, 0.0, 0.0, axial_zz),
            axis_coefficients=(
                4.0 * math.pi * spec.normalization * radial_leading,
                4.0 * math.pi * spec.normalization * radial_next,
            ),
        )

    if spec.mode.n == 0:
        return _native_vector(
            spec,
            radius,
            phi,
            (
                spec.normalization * axial,
                0.0,
                spec.normalization * axial_z,
                0.0,
                0.0,
                spec.normalization * axial_zz,
            ),
            scale_factor=4.0 * math.pi,
        )
    return _j_bessel_cartesian_native_vector(
        spec, relative, radius, (axial, axial_z, axial_zz)
    )


def _inner_cap_bulk_vector(
    spec: CylinderCavitySpec, relative: np.ndarray, route: _TransformRoute
) -> np.ndarray:
    """Evaluate the closed Fourier bulk term for an inner-cap route."""
    radius = math.hypot(float(relative[0]), float(relative[1]))
    phi = math.atan2(float(relative[1]), float(relative[0])) if radius else 0.0
    z = float(relative[2])
    cap_z = route.cap_sign * spec.geometry.height_m / 2.0
    gap = z - cap_z
    if gap == 0.0:
        raise ValueError("inner-cap bulk term is undefined on the cap plane")
    direction = 1.0 if gap > 0.0 else -1.0
    epsilon = 1.0 if route.cap_sign < 0 else (-1.0 if spec.mode.q % 2 else 1.0)
    radial_wave_number = spec.radial_wavenumber_rad_m
    bulk_scale = (
        2.0
        * math.pi
        * spec.normalization
        * route.cap_sign
        * epsilon
        * direction
        * math.exp(-radial_wave_number * abs(gap))
    )
    axial = (
        1.0,
        -direction * radial_wave_number,
        radial_wave_number**2,
    )
    if radius == 0.0:
        leading, next_coefficient = _radial_axis_coefficients(
            "j", spec.mode.m, radial_wave_number
        )
        return _native_vector(
            spec,
            radius,
            phi,
            (1.0, 0.0, axial[1], 0.0, 0.0, axial[2]),
            axis_coefficients=(bulk_scale * leading, bulk_scale * next_coefficient),
        )
    gap_decay = math.exp(-radial_wave_number * abs(gap))
    return _j_bessel_cartesian_native_vector(
        spec,
        relative,
        radius,
        tuple(value * gap_decay for value in axial),
        scale_factor=(route.cap_sign * epsilon * direction / 2.0),
    )


def _inner_cap_bulk_error_vector(
    spec: CylinderCavitySpec,
    relative: np.ndarray,
    route: _TransformRoute | None = None,
) -> np.ndarray:
    """Bound one-cap bulk value/root sensitivity for represented components.

    A missing route requests the componentwise maximum over both caps for the
    focused internal diagnostics that do not have a selected route.
    """
    if route is not None:
        if route.kind != "cap_inner":
            raise ValueError("inner-cap bulk error requires a cap_inner route")
        return _single_inner_cap_bulk_error_vector(spec, relative, route.cap_sign)

    bounds = []
    for cap_sign in (-1, 1):
        gap = float(relative[2]) - cap_sign * spec.geometry.height_m / 2.0
        if gap != 0.0:
            bounds.append(_single_inner_cap_bulk_error_vector(spec, relative, cap_sign))
    return np.maximum.reduce(bounds) if bounds else np.zeros(13, dtype=np.float64)


def _single_inner_cap_bulk_error_vector(
    spec: CylinderCavitySpec, relative: np.ndarray, cap_sign: int
) -> np.ndarray:
    """Propagate root and evaluation errors through one Cartesian cap term."""
    x, y, z = (float(value) for value in relative)
    cap_z = cap_sign * spec.geometry.height_m / 2.0
    gap = z - cap_z
    if gap == 0.0:
        raise ValueError("inner-cap bulk error is undefined on the cap plane")
    distance = abs(gap)
    direction = 1.0 if gap > 0.0 else -1.0
    radial_wave_number = spec.radial_wavenumber_rad_m
    radial_data = _j_cartesian_radial_alpha_data(spec, np.asarray(relative))

    axial_data: dict[int, tuple[float, float, float, float]] = {}
    decay = math.exp(-radial_wave_number * distance)
    delta_a = math.ulp(radial_wave_number) if radial_wave_number else 0.0
    delta_gap = math.ulp(gap)
    exponent_error = (
        distance * delta_a
        + radial_wave_number * delta_gap
        + 4.0 * _EPS * abs(radial_wave_number * distance)
    )
    decay_error = decay * (16.0 * _EPS + exponent_error)
    for order in range(3):
        sign = (-direction) ** order
        power = radial_wave_number**order
        value = sign * power * decay
        if order == 0:
            polynomial = -distance
            polynomial_a = 0.0
            polynomial_gap = -1.0
        elif order == 1:
            polynomial = 1.0 - distance * radial_wave_number
            polynomial_a = -distance
            polynomial_gap = -radial_wave_number
        else:
            polynomial = 2.0 * radial_wave_number - distance * radial_wave_number**2
            polynomial_a = 2.0 - 2.0 * distance * radial_wave_number
            polynomial_gap = -(radial_wave_number**2)
        alpha_derivative = sign * decay * polynomial / spec.geometry.radius_m
        derivative_a = (
            sign
            * decay
            * (polynomial_a - distance * polynomial)
            / spec.geometry.radius_m
        )
        derivative_gap = (
            sign
            * decay
            * (polynomial_gap - radial_wave_number * polynomial)
            / spec.geometry.radius_m
        )
        value_error = (
            power * decay_error
            + (
                order * radial_wave_number ** (order - 1) * delta_a * decay
                if order
                else 0.0
            )
            + 16.0 * _EPS * abs(value)
        )
        derivative_error = (
            abs(derivative_a) * delta_a
            + abs(derivative_gap) * delta_gap
            + 32.0 * _EPS * abs(alpha_derivative)
        )
        axial_data[order] = (
            value,
            alpha_derivative,
            float(np.nextafter(value_error, np.inf)),
            float(np.nextafter(derivative_error, np.inf)),
        )

    if spec.mode.n == 0:
        coefficient_log_derivative = 0.0
        delta_alpha = 0.0
        coefficient_log_error = 0.0
    else:
        root_value = float(jv(spec.mode.m, spec.alpha))
        root_first = float(jvp(spec.mode.m, spec.alpha, 1))
        coefficient_log_derivative = -root_first / root_value
        if spec.mode.m:
            coefficient_log_derivative -= spec.mode.m**2 / (
                spec.alpha * (spec.alpha**2 - spec.mode.m**2)
            )
        coefficient_log_derivative -= (
            2.0 * spec.alpha / (spec.geometry.radius_m**2 * spec.wavenumber_rad_m**2)
        )
        coefficient_log_error = 64.0 * _EPS * (1.0 + abs(coefficient_log_derivative))
        delta_alpha = spec.root_relative_error_estimate * max(spec.alpha, 1.0)

    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        coefficient = _decimal_to_float_up(
            (
                Decimal(2)
                * _PI_DECIMAL.next_plus()
                * Decimal.from_float(spec.normalization).next_plus()
                * _complex_abs_upper(spec.pressure_over_c2k2)
            ).next_plus()
        )
    component_terms = (
        ((0, 0), 0, -1.0),
        ((1, 0), 0, 1.0),
        ((0, 1), 0, 1.0),
        ((0, 0), 1, 1.0),
        ((2, 0), 0, 1.0),
        ((1, 1), 0, 1.0),
        ((1, 0), 1, 1.0),
        ((1, 1), 0, 1.0),
        ((0, 2), 0, 1.0),
        ((0, 1), 1, 1.0),
        ((1, 0), 1, 1.0),
        ((0, 1), 1, 1.0),
        ((0, 0), 2, 1.0),
    )
    result = np.empty(13, dtype=np.float64)
    for index, (derivative, axial_order, _sign) in enumerate(component_terms):
        radial, radial_alpha, radial_error, radial_alpha_error = radial_data[derivative]
        axial, axial_alpha, axial_error, axial_alpha_error = axial_data[axial_order]
        product = radial * axial
        sensitivity_terms = (
            coefficient_log_derivative * product,
            radial_alpha * axial,
            radial * axial_alpha,
        )
        sensitivity = math.fsum(sensitivity_terms)
        sensitivity_error = (
            abs(coefficient_log_derivative)
            * (radial_error * abs(axial) + abs(radial) * axial_error)
            + radial_alpha_error * abs(axial)
            + abs(radial_alpha) * axial_error
            + radial_error * abs(axial_alpha)
            + abs(radial) * axial_alpha_error
            + coefficient_log_error * abs(product)
            + 32.0 * _EPS * math.fsum(abs(value) for value in sensitivity_terms)
        )
        field_error = (
            radial_error * abs(axial)
            + abs(radial) * axial_error
            + radial_error * axial_error
            + 32.0 * _EPS * abs(product)
        )
        total = delta_alpha * (abs(sensitivity) + sensitivity_error) + field_error
        with localcontext() as context:
            context.prec = 90
            context.rounding = ROUND_CEILING
            total_decimal = (
                Decimal.from_float(coefficient)
                * Decimal.from_float(float(np.nextafter(total, np.inf)))
            ).next_plus()
            result[index] = _decimal_to_float_up(total_decimal)

    if math.hypot(x, y) == 0.0:
        # A single cap has exponential axial dependence, so total-mode q
        # parity at the cylinder center does not apply. At the axis, retain
        # only radial components whose value and alpha derivative both vanish.
        exact_zero = np.asarray(
            [
                radial_data[derivative][0] == 0.0 and radial_data[derivative][1] == 0.0
                for derivative, _, _ in component_terms
            ],
            dtype=bool,
        )
    else:
        exact_zero = _x_axis_zero_component_mask(spec, relative)
    result[exact_zero] = 0.0
    return result


def _j_cartesian_radial_alpha_data(
    spec: CylinderCavitySpec, relative: np.ndarray
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """Return projected J-mode Cartesian derivatives and alpha sensitivities."""
    x, y = float(relative[0]), float(relative[1])
    radius = math.hypot(x, y)
    radial_wave_number = spec.radial_wavenumber_rad_m
    order = spec.mode.m
    if radius == 0.0:
        keys = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
        values = dict.fromkeys(keys, 0.0)
        derivatives = values.copy()
        if order == 0 and spec.mode.azimuth == "cos":
            values[(0, 0)] = 1.0
            values[(2, 0)] = -(radial_wave_number**2) / 2.0
            values[(0, 2)] = -(radial_wave_number**2) / 2.0
            derivatives[(2, 0)] = -radial_wave_number / spec.geometry.radius_m
            derivatives[(0, 2)] = -radial_wave_number / spec.geometry.radius_m
        elif order == 1:
            key = (1, 0) if spec.mode.azimuth == "cos" else (0, 1)
            values[key] = radial_wave_number / 2.0
            derivatives[key] = 1.0 / (2.0 * spec.geometry.radius_m)
        elif order == 2:
            if spec.mode.azimuth == "cos":
                values[(2, 0)] = radial_wave_number**2 / 4.0
                values[(0, 2)] = -(radial_wave_number**2) / 4.0
                derivatives[(2, 0)] = radial_wave_number / (
                    2.0 * spec.geometry.radius_m
                )
                derivatives[(0, 2)] = -radial_wave_number / (
                    2.0 * spec.geometry.radius_m
                )
            else:
                values[(1, 1)] = radial_wave_number**2 / 4.0
                derivatives[(1, 1)] = radial_wave_number / (
                    2.0 * spec.geometry.radius_m
                )
        return {
            key: (
                value,
                derivatives[key],
                (
                    float(np.nextafter(32.0 * _EPS * abs(value), np.inf))
                    if value
                    else 0.0
                ),
                (
                    float(np.nextafter(32.0 * _EPS * abs(derivatives[key]), np.inf))
                    if derivatives[key]
                    else 0.0
                ),
            )
            for key, value in values.items()
        }

    x_unit, y_unit = x / radius, y / radius
    argument = radial_wave_number * radius
    delta_argument = math.ulp(argument)
    ladders = {
        (0, 0): ((0, 1.0 + 0.0j),),
        (1, 0): ((-1, 0.5 + 0.0j), (1, -0.5 + 0.0j)),
        (0, 1): ((-1, 0.5j), (1, 0.5j)),
        (2, 0): ((-2, 0.25 + 0.0j), (0, -0.5 + 0.0j), (2, 0.25 + 0.0j)),
        (1, 1): ((-2, 0.25j), (2, -0.25j)),
        (0, 2): ((-2, -0.25 + 0.0j), (0, -0.5 + 0.0j), (2, -0.25 + 0.0j)),
    }
    output: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for derivative, terms in ladders.items():
        degree = sum(derivative)
        value_terms = []
        derivative_terms = []
        value_error_terms = []
        derivative_error_terms = []
        for offset, coefficient in terms:
            bessel_order = order + offset
            cosine, sine, cosine_error, sine_error = _multiple_angle_data(
                bessel_order, x_unit, y_unit
            )
            if spec.mode.azimuth == "cos":
                projection = coefficient.real * cosine - coefficient.imag * sine
                projection_error = (
                    abs(coefficient.real) * cosine_error
                    + abs(coefficient.imag) * sine_error
                )
            else:
                projection = coefficient.real * sine + coefficient.imag * cosine
                projection_error = (
                    abs(coefficient.real) * sine_error
                    + abs(coefficient.imag) * cosine_error
                )
            magnitude = abs(bessel_order)
            sign = -1.0 if bessel_order < 0 and magnitude % 2 else 1.0
            bessel, first, second = _j_triplet(magnitude, argument)
            bessel *= sign
            first *= sign
            second *= sign
            value_terms.append(projection * bessel)
            derivative_terms.append(projection * first)
            value_error_terms.append(
                abs(projection)
                * (32.0 * _EPS * abs(bessel) + abs(first) * delta_argument)
                + projection_error * abs(bessel)
            )
            derivative_error_terms.append(
                abs(projection)
                * (32.0 * _EPS * abs(first) + abs(second) * delta_argument)
                + projection_error * abs(first)
            )

        value_sum = math.fsum(value_terms)
        derivative_sum = math.fsum(derivative_terms)
        value_abs_sum = math.fsum(abs(value) for value in value_terms)
        derivative_abs_sum = math.fsum(abs(value) for value in derivative_terms)
        scale = radial_wave_number**degree
        radial_value = scale * value_sum
        value_error = scale * (
            math.fsum(value_error_terms) + 16.0 * _EPS * value_abs_sum
        ) + 16.0 * _EPS * abs(radial_value)
        if degree:
            first_scale = degree * radial_wave_number ** (degree - 1)
            first_scale /= spec.geometry.radius_m
        else:
            first_scale = 0.0
        second_scale = scale * radius / spec.geometry.radius_m
        radial_alpha = first_scale * value_sum + second_scale * derivative_sum
        radial_alpha_error = (
            abs(first_scale)
            * (math.fsum(value_error_terms) + 16.0 * _EPS * value_abs_sum)
            + abs(second_scale)
            * (math.fsum(derivative_error_terms) + 16.0 * _EPS * derivative_abs_sum)
            + 16.0
            * _EPS
            * (abs(first_scale * value_sum) + abs(second_scale * derivative_sum))
        )
        output[derivative] = (
            radial_value,
            radial_alpha,
            float(np.nextafter(value_error, np.inf)),
            float(np.nextafter(radial_alpha_error, np.inf)),
        )
    return output


def _local_inner_cap_pair_axial_a_derivatives(
    spec: CylinderCavitySpec, z: float
) -> tuple[float, float, float]:
    """Differentiate the paired axial factors with respect to radial k."""
    a = spec.radial_wavenumber_rad_m
    half = spec.geometry.height_m / 2.0
    lower_length = half - z
    upper_length = half + z
    lower = math.exp(-a * lower_length)
    upper = math.exp(-a * upper_length)
    if spec.mode.q % 2 == 0:
        return (
            0.5 * (lower_length * lower + upper_length * upper),
            -0.5 * (lower - upper + a * (-lower_length * lower + upper_length * upper)),
            -0.5
            * (
                2.0 * a * (lower + upper)
                - a**2 * (lower_length * lower + upper_length * upper)
            ),
        )

    return (
        0.5 * (-lower_length * lower + upper_length * upper),
        0.5 * (lower + upper + a * (-lower_length * lower - upper_length * upper)),
        0.5
        * (
            2.0 * a * (lower - upper)
            + a**2 * (-lower_length * lower + upper_length * upper)
        ),
    )


def _paired_inner_cap_bessel_error_vector(
    spec: CylinderCavitySpec, relative: np.ndarray
) -> np.ndarray:
    """Estimate paired-term root and evaluation error per Cartesian component.

    Neighboring-order Bessel ladders are projected onto the requested real
    azimuthal mode before absolute errors are formed. This retains exact
    symmetry zeros and small represented sine-plane values.
    """
    errors = np.zeros(13, dtype=np.float64)
    if spec.mode.n == 0:
        return errors

    x, y, z = (float(value) for value in relative)
    radius = math.hypot(x, y)
    x_unit, y_unit = (1.0, 0.0) if radius == 0.0 else (x / radius, y / radius)
    radial_wave_number = spec.radial_wavenumber_rad_m
    argument = radial_wave_number * radius
    axial = _local_inner_cap_pair_axial(spec, z)
    axial_a = _local_inner_cap_pair_axial_a_derivatives(spec, z)
    axial_errors, axial_a_errors = _paired_axial_evaluation_errors(spec, z)

    ladders = {
        (0, 0): ((0, 1.0 + 0.0j),),
        (1, 0): ((-1, 0.5 + 0.0j), (1, -0.5 + 0.0j)),
        (0, 1): ((-1, 0.5j), (1, 0.5j)),
        (2, 0): ((-2, 0.25 + 0.0j), (0, -0.5 + 0.0j), (2, 0.25 + 0.0j)),
        (1, 1): ((-2, 0.25j), (2, -0.25j)),
        (0, 2): ((-2, -0.25 + 0.0j), (0, -0.5 + 0.0j), (2, -0.25 + 0.0j)),
    }
    angular: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    order = spec.mode.m
    for derivative, terms in ladders.items():
        projected = []
        for offset, coefficient in terms:
            bessel_order = order + offset
            cosine, sine, cosine_error, sine_error = _multiple_angle_data(
                bessel_order, x_unit, y_unit
            )
            if spec.mode.azimuth == "cos":
                value = coefficient.real * cosine - coefficient.imag * sine
                error = (
                    abs(coefficient.real) * cosine_error
                    + abs(coefficient.imag) * sine_error
                )
            else:
                value = coefficient.real * sine + coefficient.imag * cosine
                error = (
                    abs(coefficient.real) * sine_error
                    + abs(coefficient.imag) * cosine_error
                )
            projected.append((bessel_order, value, error))
        angular[derivative] = tuple(projected)  # type: ignore[assignment]

    bessel_data: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for derivative, terms in angular.items():
        degree = sum(derivative)
        value_terms: list[float] = []
        derivative_terms: list[float] = []
        value_error_terms: list[float] = []
        derivative_error_terms: list[float] = []
        value_absolute_terms: list[float] = []
        derivative_absolute_terms: list[float] = []
        delta_argument = math.ulp(argument)
        for bessel_order, projection, projection_error in terms:
            bessel_value = float(jv(bessel_order, argument))
            bessel_first = float(jvp(bessel_order, argument, 1))
            bessel_second = float(jvp(bessel_order, argument, 2))
            value_term = projection * bessel_value
            derivative_term = projection * bessel_first
            value_terms.append(value_term)
            derivative_terms.append(derivative_term)
            value_absolute_terms.append(abs(value_term))
            derivative_absolute_terms.append(abs(derivative_term))
            value_error_terms.append(
                abs(projection)
                * (32.0 * _EPS * abs(bessel_value) + abs(bessel_first) * delta_argument)
                + projection_error * abs(bessel_value)
            )
            derivative_error_terms.append(
                abs(projection)
                * (
                    32.0 * _EPS * abs(bessel_first)
                    + abs(bessel_second) * delta_argument
                )
                + projection_error * abs(bessel_first)
            )

        value_sum = math.fsum(value_terms)
        derivative_sum = math.fsum(derivative_terms)
        scale = radial_wave_number**degree
        value = scale * value_sum
        value_error = scale * (
            math.fsum(value_error_terms) + 16.0 * _EPS * math.fsum(value_absolute_terms)
        )
        derivative_scale = (
            degree * radial_wave_number ** (degree - 1) if degree else 0.0
        )
        derivative_value = (
            derivative_scale * value_sum + scale * radius * derivative_sum
        )
        derivative_error = (
            abs(derivative_scale)
            * (
                math.fsum(value_error_terms)
                + 16.0 * _EPS * math.fsum(value_absolute_terms)
            )
            + abs(scale * radius)
            * (
                math.fsum(derivative_error_terms)
                + 16.0 * _EPS * math.fsum(derivative_absolute_terms)
            )
            + 16.0
            * _EPS
            * (abs(derivative_scale * value_sum) + abs(scale * radius * derivative_sum))
        )
        bessel_data[derivative] = (
            value,
            derivative_value,
            float(np.nextafter(value_error, np.inf)),
            float(np.nextafter(derivative_error, np.inf)),
        )

    alpha = spec.alpha
    radius_m = spec.geometry.radius_m
    k_squared = spec.wavenumber_rad_m**2
    root_value = float(jv(order, alpha))
    root_first = float(jvp(order, alpha, 1))
    normalization_log_derivative = -root_first / root_value
    if order:
        normalization_log_derivative -= order**2 / (alpha * (alpha**2 - order**2))
    coefficient_log_derivative = normalization_log_derivative - 2.0 * alpha / (
        radius_m**2 * k_squared
    )
    delta_alpha = spec.root_relative_error_estimate * max(alpha, 1.0)
    coefficient_abs = 4.0 * math.pi * abs(spec.pressure_over_c2k2) * spec.normalization

    component_terms = (
        ((0, 0), 0, -1.0),
        ((1, 0), 0, 1.0),
        ((0, 1), 0, 1.0),
        ((0, 0), 1, 1.0),
        ((2, 0), 0, 1.0),
        ((1, 1), 0, 1.0),
        ((1, 0), 1, 1.0),
        ((1, 1), 0, 1.0),
        ((0, 2), 0, 1.0),
        ((0, 1), 1, 1.0),
        ((1, 0), 1, 1.0),
        ((0, 1), 1, 1.0),
        ((0, 0), 2, 1.0),
    )
    for index, (derivative, axial_order, _sign) in enumerate(component_terms):
        bessel_value, bessel_a, bessel_error, bessel_a_error = bessel_data[derivative]
        axial_value = axial[axial_order]
        axial_a_value = axial_a[axial_order]
        root_derivative = (
            bessel_a * axial_value
            + bessel_value * axial_a_value
            + radius_m * coefficient_log_derivative * bessel_value * axial_value
        ) / radius_m
        root_error = delta_alpha * abs(root_derivative)
        evaluation_error = (
            abs(axial_value) * bessel_error
            + abs(bessel_value) * axial_errors[axial_order]
            + bessel_a_error * abs(axial_a_value) * _EPS
            + abs(bessel_value) * axial_a_errors[axial_order] * _EPS
            + 32.0 * _EPS * abs(bessel_value * axial_value)
        )
        total = coefficient_abs * (root_error + evaluation_error)
        errors[index] = float(np.nextafter(total, np.inf))

    exact_zero = _exact_zero_component_mask(spec, relative)
    errors[exact_zero] = 0.0
    return errors


def _multiple_angle_data(
    order: int, x_unit: float, y_unit: float
) -> tuple[float, float, float, float]:
    """Evaluate cos(n*phi), sin(n*phi) with error scaling for tiny sine phases."""
    magnitude = abs(order)
    if magnitude == 0:
        return 1.0, 0.0, 0.0, 0.0

    cosine_previous, cosine = 1.0, x_unit
    for _ in range(2, magnitude + 1):
        cosine_previous, cosine = cosine, 2.0 * x_unit * cosine - cosine_previous

    if magnitude == 1:
        sine_polynomial = 1.0
    else:
        previous, sine_polynomial = 1.0, 2.0 * x_unit
        for _ in range(2, magnitude):
            previous, sine_polynomial = (
                sine_polynomial,
                2.0 * x_unit * sine_polynomial - previous,
            )
    sine = y_unit * sine_polynomial
    if order < 0:
        sine = -sine

    cosine_error = 16.0 * _EPS * magnitude**2 * max(abs(cosine), 1.0)
    sine_error = (
        abs(y_unit) * 16.0 * _EPS * magnitude**2 * max(abs(sine_polynomial), 1.0)
    )
    sine_error += 8.0 * _EPS * abs(sine)
    return cosine, sine, cosine_error, sine_error


def _paired_axial_evaluation_errors(
    spec: CylinderCavitySpec,
    z: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Estimate paired axial arithmetic error without flooring parity zeros."""
    a = spec.radial_wavenumber_rad_m
    beta = spec.axial_wavenumber_rad_m
    half = spec.geometry.height_m / 2.0
    lower = math.exp(-a * (half - z))
    upper = math.exp(-a * (half + z))
    phase = beta * z
    if spec.mode.q % 2 == 0:
        parity = -1.0 if (spec.mode.q // 2) % 2 else 1.0
        axial_scales = (
            abs(
                2.0 * math.sin(phase / 2.0) ** 2
                if parity > 0
                else 2.0 * math.cos(phase / 2.0) ** 2
            )
            + 0.5
            * (abs(math.expm1(-a * (half - z))) + abs(math.expm1(-a * (half + z)))),
            abs(beta * math.sin(phase)) + 0.5 * a * (lower + upper),
            beta**2 * abs(math.cos(phase)) + 0.5 * a**2 * (lower + upper),
        )
        sinh = 0.5 * (lower - upper)
        cosh = 0.5 * (lower + upper)
        axial_a_scales = (
            half * cosh + abs(z * sinh),
            abs(a * half - 1.0) * abs(sinh) + abs(a * z * cosh),
            abs(a * (a * half - 2.0)) * cosh + abs(a**2 * z * sinh),
        )
    else:
        axial_scales = (
            abs(math.sin(phase)) + 0.5 * (lower + upper),
            beta * abs(math.cos(phase)) + 0.5 * a * (lower + upper),
            beta**2 * abs(math.sin(phase)) + 0.5 * a**2 * (lower + upper),
        )
        sinh = 0.5 * (lower - upper)
        cosh = 0.5 * (lower + upper)
        axial_a_scales = (
            abs(z * cosh) + half * abs(sinh),
            abs(1.0 - a * half) * cosh + abs(a * z * sinh),
            abs(a * (2.0 - a * half)) * abs(sinh) + abs(a**2 * z * cosh),
        )

    axial_errors = [32.0 * _EPS * value for value in axial_scales]
    axial_a_errors = [32.0 * _EPS * value for value in axial_a_scales]
    for index in range(3):
        if z == 0.0 and (spec.mode.q + index) % 2:
            axial_errors[index] = 0.0
            axial_a_errors[index] = 0.0
    return tuple(axial_errors), tuple(axial_a_errors)


def _compensated_complex_sum(terms: tuple[np.ndarray, ...]) -> np.ndarray:
    if not terms:
        return np.zeros(13, dtype=np.complex128)
    return np.asarray(
        [
            complex(
                math.fsum(float(term[index].real) for term in terms),
                math.fsum(float(term[index].imag) for term in terms),
            )
            for index in range(13)
        ],
        dtype=np.complex128,
    )


def _closed_term_error_vector(
    terms: tuple[np.ndarray, ...],
    relative_errors: tuple[float, ...],
    special_errors: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Accumulate closed-form errors and compensated-sum roundoff outward."""
    result = np.empty(13, dtype=np.float64)
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        summation_factor = Decimal(4) * Decimal.from_float(_EPS)
        for index in range(13):
            total = Decimal(0)
            magnitude_sum = Decimal(0)
            for term, relative_error in zip(terms, relative_errors, strict=True):
                magnitude = _complex_abs_upper(complex(term[index]))
                magnitude_sum = (magnitude_sum + magnitude).next_plus()
                total = (
                    total + magnitude * Decimal.from_float(relative_error)
                ).next_plus()
            for errors in special_errors:
                total = (total + Decimal.from_float(float(errors[index]))).next_plus()
            total = (total + summation_factor * magnitude_sum).next_plus()
            result[index] = _decimal_to_float_up(total)
    return result


def _complex_add_roundoff_upper(left: complex, right: complex) -> float:
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        magnitude = _complex_abs_upper(left) + _complex_abs_upper(right)
        return _decimal_to_float_up(
            (Decimal(4) * Decimal.from_float(_EPS) * magnitude).next_plus()
        )


def _component_scales(spec: CylinderCavitySpec) -> np.ndarray:
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        scale = (
            _complex_abs_upper(spec.pressure_over_c2k2)
            * Decimal.from_float(spec.normalization)
        ).next_plus()
        length = Decimal.from_float(min(spec.geometry.radius_m, spec.geometry.height_m))
        values = [scale]
        values.extend([scale / length] * 3)
        values.extend([scale / length**2] * 9)
    raw = np.asarray(
        [_decimal_to_float_up(value) for value in values], dtype=np.float64
    )
    if np.any(raw <= 0.0) or not np.isfinite(raw).all():
        raise ValueError("cylinder integration scales are not representable")
    return raw


def _failure_diagnostic(
    spec: CylinderCavitySpec,
    target_index: int,
    target_id: str,
    observable: str,
    termination: str,
    evaluations: int,
    refinements: int,
    *,
    values: np.ndarray | None = None,
    values_available: bool = False,
    errors: np.ndarray | None = None,
    tails: np.ndarray | None = None,
    complete_estimate: bool = False,
) -> NumericalDiagnostic:
    power = _OBSERVABLE_BUDGETS[observable][1]
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        pressure_abs = _complex_abs_upper(spec.pressure_amplitude_pa)
        length = _cylinder_volume_upper(spec) ** (Decimal(1) / Decimal(3))
        scale_si = (
            Decimal.from_float(float(G_SI))
            * pressure_abs
            / Decimal.from_float(spec.sound_speed_m_s) ** 2
            * length**power
        ).next_plus()
        g = Decimal.from_float(float(G_SI))
        indices = _observable_indices(observable)
        value_si = (
            (g * _complex_norm_upper(values[indices])).next_plus()
            if values_available and values is not None
            else None
        )
        error_si = (
            (g * _decimal_norm(errors[indices])).next_plus()
            if errors is not None
            else None
        )
        tail_si = (
            (g * _decimal_norm(tails[indices])).next_plus()
            if tails is not None
            else None
        )
        tolerance_si = None
        if complete_estimate and value_si is not None:
            relative = Decimal.from_float(_OBSERVABLE_BUDGETS[observable][0])
            tolerance_si = (
                Decimal("0.25") * (Decimal("1e-12") * scale_si + relative * value_si)
            ).next_plus()
    return NumericalDiagnostic(
        method=_METHOD,
        target_index=target_index,
        target_id=target_id,
        observable=observable,
        norm={
            "potential": "absolute",
            "acceleration": "euclidean",
            "gradient": "frobenius",
        }[observable],
        accuracy_policy="gen1_cavity_v1",
        scale_si=_decimal_to_float_up(scale_si),
        value_norm_si=_dec_diag(value_si),
        requested_tolerance_si=_dec_diag(tolerance_si),
        error_estimate_si=_dec_diag(error_si),
        rigorous_tail_bound_si=_dec_diag(tail_si),
        evaluations=evaluations,
        refinements=refinements,
        termination=termination,
    )


def _success_diagnostics(
    spec: CylinderCavitySpec,
    target_index: int,
    target_id: str,
    observables: tuple[str, ...],
    values: np.ndarray,
    errors: np.ndarray,
    tails: np.ndarray,
    evaluations: int,
    refinements: int,
) -> tuple[NumericalDiagnostic, ...]:
    pressure_abs = _complex_abs_upper(spec.pressure_amplitude_pa)
    volume = _cylinder_volume_upper(spec)
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        length = volume ** (Decimal(1) / Decimal(3))
        g = Decimal.from_float(float(G_SI))
    diagnostics = []
    for observable in observables:
        indices = _observable_indices(observable)
        error_gfree = _decimal_norm(errors[indices])
        tail_gfree = _decimal_norm(tails[indices])
        value_gfree = _complex_norm_upper(values[indices])
        power = _OBSERVABLE_BUDGETS[observable][1]
        relative = Decimal.from_float(_OBSERVABLE_BUDGETS[observable][0])
        with localcontext() as context:
            context.prec = 90
            context.rounding = ROUND_CEILING
            scale_si = (
                g
                * pressure_abs
                / Decimal.from_float(spec.sound_speed_m_s) ** 2
                * length**power
            ).next_plus()
            value_si = (g * value_gfree).next_plus()
            tolerance = (
                Decimal("0.25") * (Decimal("1e-12") * scale_si + relative * value_si)
            ).next_plus()
            error_si = (g * error_gfree).next_plus()
            tail_si = (g * tail_gfree).next_plus()
        diagnostics.append(
            NumericalDiagnostic(
                method=_METHOD,
                target_index=target_index,
                target_id=target_id,
                observable=observable,
                norm={
                    "potential": "absolute",
                    "acceleration": "euclidean",
                    "gradient": "frobenius",
                }[observable],
                accuracy_policy="gen1_cavity_v1",
                scale_si=_decimal_to_float_up(scale_si),
                value_norm_si=_decimal_to_float_up(value_si),
                requested_tolerance_si=_decimal_to_float_up(tolerance),
                error_estimate_si=_decimal_to_float_up(error_si),
                rigorous_tail_bound_si=_decimal_to_float_up(tail_si),
                evaluations=evaluations,
                refinements=refinements,
                termination="converged",
            )
        )
    return tuple(diagnostics)


def _integrate_routes(
    spec: CylinderCavitySpec,
    relative: np.ndarray,
    routes: tuple[_TransformRoute, ...],
    observables: tuple[str, ...],
    baseline_values: np.ndarray,
    local_errors: np.ndarray,
):
    from gragra.analytic.atmospheric._cavity._numerics import integrate_transform

    if not routes:
        from gragra.analytic.atmospheric._cavity._numerics import TransformEstimate

        zeros = np.zeros(13, dtype=np.float64)
        return TransformEstimate(
            values=np.zeros(13, dtype=np.complex128),
            component_errors=zeros,
            tail_bounds=zeros,
            evaluations=0,
            refinements=0,
            cutoff=None,
            termination="converged",
        )

    route_integrands = tuple(
        _route_integrand(spec, relative, route) for route in routes
    )
    exact_zero = _exact_zero_component_mask(spec, relative)

    def integrand(wave_number: float) -> np.ndarray:
        terms = tuple(function(wave_number) for function in route_integrands)
        # Compensate the sum of side/cap contributions before the adaptive
        # controller estimates their combined transform.
        real = [
            math.fsum(float(term[index].real) for term in terms) for index in range(13)
        ]
        imag = [
            math.fsum(float(term[index].imag) for term in terms) for index in range(13)
        ]
        result = np.asarray(real, dtype=np.float64) + 1j * np.asarray(
            imag, dtype=np.float64
        )
        result[exact_zero] = 0.0j
        return result

    def tail_bound(cutoff: float) -> np.ndarray:
        route_tails = tuple(
            _tail_vector(spec, route.kind, cutoff, route.gap_lower) for route in routes
        )
        bounds = np.asarray(
            [_sum_upper([term[index] for term in route_tails]) for index in range(13)],
            dtype=np.float64,
        )
        bounds[exact_zero] = 0.0
        return bounds

    def acceptance(
        values: np.ndarray, component_errors: np.ndarray, tail_bounds: np.ndarray
    ) -> bool:
        g = Decimal.from_float(float(G_SI))
        current_values = baseline_values + np.asarray(values, dtype=np.complex128)
        budgets: dict[str, Decimal] = {}
        for observable in observables:
            indices = _observable_indices(observable)
            value_si = (g * _complex_norm_upper(current_values[indices])).next_plus()
            local_error_si = (g * _decimal_norm(local_errors[indices])).next_plus()
            relative = Decimal.from_float(_OBSERVABLE_BUDGETS[observable][0])
            run_budget = (
                _run_budget_abs_si(spec, observable) + relative * value_si / Decimal(4)
            ).next_plus()
            budgets[observable] = max(Decimal(0), run_budget - local_error_si)
        return _is_accepted(
            np.asarray(component_errors, dtype=np.float64),
            np.asarray(tail_bounds, dtype=np.float64),
            observables,
            budgets,
        )

    component_scales = _target_component_scales(spec, relative, exact_zero)
    combined_scales = np.asarray(
        [_sum_upper([value] * len(routes)) for value in component_scales],
        dtype=np.float64,
    )
    # Keep the controller's positive-scale contract while avoiding a generic
    # nonzero roundoff floor for components whose cylindrical-axis derivatives
    # vanish exactly by azimuthal order.
    combined_scales[exact_zero] = np.finfo(np.float64).tiny
    phase_length = _maximum_route_phase_length(spec, relative, routes)
    # Limit the phase span of every actually selected route to pi per panel.
    # This only seeds/refines the controller's grid; its error estimator,
    # acceptance rule, and resource limits remain authoritative.
    max_panel_width = math.pi / phase_length if phase_length > 0.0 else None
    return integrate_transform(
        integrand,
        tail_bound,
        initial_cutoff=max(route.initial_cutoff for route in routes),
        component_scales=combined_scales,
        acceptance=acceptance,
        numerics=spec.numerics,
        max_panel_width=max_panel_width,
        refined_orders=(16, 32, 64),
        promote_initial_panels=True,
        origin_subdivisions=8,
        filter_panel_roundoff=True,
        _q2_band=(138.4615384615381, 415.38461538461434),
        _q2_phase_length=phase_length,
        _q2_exact_zero_mask=exact_zero,
    )


def _maximum_route_phase_length(
    spec: CylinderCavitySpec,
    relative: np.ndarray,
    routes: tuple[_TransformRoute, ...],
) -> float:
    """Largest dimensional Fourier phase length among selected routes."""
    radius = math.hypot(float(relative[0]), float(relative[1]))
    half_height = spec.geometry.height_m / 2.0
    z = float(relative[2])
    phase_lengths = []
    for route in routes:
        if route.kind in {"side_hankel", "cap_hankel"}:
            # The oscillatory Hankel routes multiply target and source Bessel
            # factors, whose largest asymptotic phase rate is r + R.
            phase_lengths.append(radius + spec.geometry.radius_m)
        elif route.kind == "side_modified":
            # The centered axial transform contains phases at z +/- H/2.
            phase_lengths.append(half_height + abs(z))
        elif route.kind in {"cap_inner", "cap_modified"}:
            # These cap transforms contain the axial phase from the target to
            # the selected cap. The inner-cap route has the same phase in its
            # exponentially damped residual after its closed bulk term.
            phase_lengths.append(abs(z - route.cap_sign * half_height))
    return max(phase_lengths, default=0.0)


def _target_component_scales(
    spec: CylinderCavitySpec,
    relative: np.ndarray,
    exact_zero: np.ndarray,
) -> np.ndarray:
    """Return positive scales that follow the represented target's mode."""
    component_scales = _component_scales(spec)
    radius = math.hypot(float(relative[0]), float(relative[1]))
    length = min(spec.geometry.radius_m, spec.geometry.height_m)
    radial_orders = np.asarray([0, 1, 1, 0, 2, 2, 1, 2, 2, 1, 1, 1, 0], dtype=np.int8)
    if spec.mode.n != 0 and 0.0 < radius < length:
        radial_degree = spec.mode.m + 2 * np.maximum(
            (radial_orders - spec.mode.m + 1) // 2, 0
        )
        radial_power = radial_degree - radial_orders
        component_scales *= (radius / length) ** radial_power

    angular_zero = _x_axis_parity_component_mask(spec)
    if angular_zero.any() and radius > 0.0:
        sine = abs(float(relative[1])) / radius
        angular_factor = min(
            1.0,
            32.0 * (spec.mode.m + 1) ** 2 * (sine + 64.0 * _EPS),
        )
        component_scales[angular_zero] *= angular_factor

    component_scales[exact_zero] = np.finfo(np.float64).tiny
    return np.maximum(component_scales, np.finfo(np.float64).tiny)


def _x_axis_zero_component_mask(
    spec: CylinderCavitySpec, relative: np.ndarray
) -> np.ndarray:
    """Return components forced to zero by exact x-axis angular parity."""
    if (
        math.hypot(float(relative[0]), float(relative[1])) == 0.0
        or float(relative[1]) != 0.0
    ):
        return np.zeros(13, dtype=bool)
    return _x_axis_parity_component_mask(spec)


def _x_axis_parity_component_mask(spec: CylinderCavitySpec) -> np.ndarray:
    """Return components whose angular mode is odd under y reflection."""
    mask = np.zeros(13, dtype=bool)
    if spec.mode.azimuth == "sin":
        mask[[0, 1, 3, 4, 6, 8, 10, 12]] = True
    else:
        mask[[2, 5, 7, 9, 11]] = True
    return mask


def _exact_zero_component_mask(
    spec: CylinderCavitySpec, relative: np.ndarray
) -> np.ndarray:
    """Identify exact axis and angular-plane zeros at represented coordinates."""
    mask = np.zeros(13, dtype=bool)
    radius = math.hypot(float(relative[0]), float(relative[1]))
    if radius == 0.0:
        radial_orders = np.asarray(
            [0, 1, 1, 0, 2, 2, 1, 2, 2, 1, 1, 1, 0], dtype=np.int8
        )
        axial_orders = np.asarray(
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 2], dtype=np.int8
        )
        order = spec.mode.m
        mask[:] = (radial_orders < order) | ((radial_orders - order) % 2 != 0)
        if float(relative[2]) == 0.0:
            axial_order = spec.mode.q % 2
            mask |= (axial_orders < axial_order) | (
                (axial_orders - axial_order) % 2 != 0
            )
        return mask
    return _x_axis_zero_component_mask(spec, relative)


def _evaluate_cylinder(
    spec: CylinderCavitySpec,
    target_positions_m: np.ndarray,
    target_ids: tuple[str, ...],
    observables: tuple[str, ...],
) -> NativeEvaluation:
    count = len(target_positions_m)
    output_shapes = {
        "potential": (count, 1, 1),
        "acceleration": (count, 3, 1),
        "gradient": (count, 9, 1),
    }
    arrays = {
        name: np.zeros(shape, dtype=np.complex128)
        for name, shape in output_shapes.items()
        if name in observables
    }
    if count == 0:
        return NativeEvaluation(arrays)
    if spec.pressure_amplitude_pa == 0.0j:
        diagnostics = tuple(
            NumericalDiagnostic(
                method=_METHOD,
                target_index=index,
                target_id=target_ids[index],
                observable=name,
                norm={
                    "potential": "absolute",
                    "acceleration": "euclidean",
                    "gradient": "frobenius",
                }[name],
                accuracy_policy="gen1_cavity_v1",
                scale_si=0.0,
                value_norm_si=0.0,
                requested_tolerance_si=0.0,
                error_estimate_si=0.0,
                rigorous_tail_bound_si=0.0,
                evaluations=0,
                refinements=0,
                termination="exact_zero",
            )
            for index in range(count)
            for name in observables
        )
        return NativeEvaluation(arrays, diagnostics)

    center = spec.geometry.center_m
    geometry_scale = min(spec.geometry.radius_m, spec.geometry.height_m)
    all_diagnostics: list[NumericalDiagnostic] = []
    for target_index, target in enumerate(target_positions_m):
        relative = target - center
        routes = _select_routes(spec, relative)
        accumulated = np.zeros(13, dtype=np.complex128)
        inside = (
            math.hypot(float(relative[0]), float(relative[1])) < spec.geometry.radius_m
            and abs(float(relative[2])) < spec.geometry.height_m / 2.0
        )
        local = (
            _local_mode_vector(spec, relative)
            if inside
            else np.zeros(13, dtype=np.complex128)
        )
        local_relative_error = 64.0 * _EPS + spec.root_relative_error_estimate * (
            1.0 + (spec.wavenumber_rad_m * geometry_scale) ** 2
        )
        inner_caps = tuple(route for route in routes if route.kind == "cap_inner")
        pair_inner_caps = (
            inside
            and len(inner_caps) == 2
            and {route.cap_sign for route in inner_caps} == {-1, 1}
        )
        if pair_inner_caps:
            closed_terms = [_local_inner_cap_pair_vector(spec, relative)]
            paired_partials = (
                local,
                *(
                    _inner_cap_bulk_vector(spec, relative, route)
                    for route in inner_caps
                ),
            )
            pair_extent = (
                spec.geometry.radius_m
                + spec.geometry.height_m / 2.0
                + math.hypot(float(relative[0]), float(relative[1]))
            )
            pair_relative_error = 64.0 * _EPS + spec.root_relative_error_estimate * (
                1.0 + (spec.wavenumber_rad_m * pair_extent) ** 2
            )
            pair_roundoff = _closed_term_error_vector(
                paired_partials, (0.0,) * len(paired_partials), ()
            )
            pair_roundoff = np.asarray(
                [_sum_upper([float(value)] * 16) for value in pair_roundoff],
                dtype=np.float64,
            )
            pair_bessel_error = _paired_inner_cap_bessel_error_vector(spec, relative)
            closed_relative_errors = (pair_relative_error,)
            closed_term_errors = [pair_roundoff, pair_bessel_error]
        else:
            closed_terms = [local]
            closed_relative_errors = (local_relative_error,)
            closed_term_errors = []
        if not pair_inner_caps:
            for route in routes:
                if route.kind == "cap_inner":
                    closed_terms.append(_inner_cap_bulk_vector(spec, relative, route))
                    closed_term_errors.append(
                        _inner_cap_bulk_error_vector(spec, relative, route)
                    )
            closed_relative_errors = (local_relative_error,) + (0.0,) * (
                len(closed_terms) - 1
            )
        accumulated = _compensated_complex_sum(tuple(closed_terms))
        component_errors = _closed_term_error_vector(
            tuple(closed_terms),
            closed_relative_errors,
            tuple(closed_term_errors),
        )
        exact_zero = _exact_zero_component_mask(spec, relative)
        accumulated[exact_zero] = 0.0j
        component_errors[exact_zero] = 0.0
        component_tails = np.zeros(13, dtype=np.float64)
        estimate = _integrate_routes(
            spec,
            relative,
            routes,
            observables,
            accumulated,
            component_errors,
        )
        total_evaluations = int(estimate.evaluations)
        total_refinements = int(estimate.refinements)
        termination = str(estimate.termination)
        if estimate.values is not None:
            transform_values = np.asarray(estimate.values, dtype=np.complex128)
            baseline_values = accumulated.copy()
            accumulated = _compensated_complex_sum((baseline_values, transform_values))
        if estimate.component_errors is not None:
            component_errors = np.asarray(
                [
                    _sum_upper(
                        (
                            left,
                            right,
                            _complex_add_roundoff_upper(
                                complex(baseline_values[index]),
                                complex(transform_values[index]),
                            )
                            if estimate.values is not None
                            else 0.0,
                        )
                    )
                    for index, (left, right) in enumerate(
                        zip(
                            component_errors,
                            np.asarray(estimate.component_errors, dtype=np.float64),
                            strict=True,
                        )
                    )
                ],
                dtype=np.float64,
            )
        if estimate.tail_bounds is not None:
            component_tails = np.array(
                estimate.tail_bounds, dtype=np.float64, copy=True
            )
        accumulated[exact_zero] = 0.0j
        component_errors[exact_zero] = 0.0
        component_tails[exact_zero] = 0.0
        if termination != "converged":
            values_available = estimate.values is not None
            errors_available = estimate.component_errors is not None
            tails_available = estimate.tail_bounds is not None
            complete_evidence = (
                values_available and errors_available and tails_available
            )
            failures = tuple(
                _failure_diagnostic(
                    spec,
                    target_index,
                    target_ids[target_index],
                    observable,
                    termination,
                    total_evaluations,
                    total_refinements,
                    values=accumulated if values_available else None,
                    values_available=values_available,
                    errors=component_errors if errors_available else None,
                    tails=component_tails if tails_available else None,
                    complete_estimate=values_available,
                )
                for observable in observables
            )
            if complete_evidence:
                exceeded = tuple(
                    item
                    for item in failures
                    if Decimal.from_float(item.error_estimate_si or 0.0)
                    + Decimal.from_float(item.rigorous_tail_bound_si or 0.0)
                    > Decimal.from_float(item.requested_tolerance_si or 0.0)
                )
                if exceeded:
                    failures = exceeded
            raise ConvergenceError(
                f"cylinder combined transform terminated as {termination}",
                failures,
            )
        diagnostics = _success_diagnostics(
            spec,
            target_index,
            target_ids[target_index],
            observables,
            accumulated,
            component_errors,
            component_tails,
            total_evaluations,
            total_refinements,
        )
        failed = [
            item
            for item in diagnostics
            if Decimal.from_float(item.error_estimate_si or 0.0)
            + Decimal.from_float(item.rigorous_tail_bound_si or 0.0)
            > Decimal.from_float(item.requested_tolerance_si or 0.0)
        ]
        if failed:
            failure_items = tuple(
                NumericalDiagnostic(
                    method=item.method,
                    target_index=item.target_index,
                    target_id=item.target_id,
                    observable=item.observable,
                    norm=item.norm,
                    accuracy_policy=item.accuracy_policy,
                    scale_si=item.scale_si,
                    value_norm_si=item.value_norm_si,
                    requested_tolerance_si=item.requested_tolerance_si,
                    error_estimate_si=item.error_estimate_si,
                    rigorous_tail_bound_si=item.rigorous_tail_bound_si,
                    evaluations=item.evaluations,
                    refinements=item.refinements,
                    termination="uncertainty_unresolved",
                )
                for item in diagnostics
            )
            raise ConvergenceError(
                "cylinder total error estimate plus rigorous tail exceeds B_run",
                failure_items,
            )
        all_diagnostics.extend(diagnostics)
        if "potential" in arrays:
            arrays["potential"][target_index, 0, 0] = accumulated[0]
        if "acceleration" in arrays:
            arrays["acceleration"][target_index, :, 0] = accumulated[1:4]
        if "gradient" in arrays:
            arrays["gradient"][target_index, :, 0] = accumulated[4:13]
    return NativeEvaluation(arrays, tuple(all_diagnostics))


@dataclass(frozen=True, slots=True)
class _TransformRoute:
    kind: str
    gap_lower: Decimal
    gap_m: float
    initial_cutoff: float
    cap_sign: int = 0


def _make_route(
    kind: str,
    gap_lower: Decimal,
    gap_m: float,
    *,
    cap_sign: int = 0,
    length_scale: float,
) -> _TransformRoute:
    if gap_lower <= 0 or gap_m <= 0.0 or not math.isfinite(gap_m):
        raise ValueError("cylinder transform gap is not certified positive")
    cutoff = max(1.0 / length_scale, 1.0 / gap_m)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cylinder transform initial cutoff is not finite")
    return _TransformRoute(kind, gap_lower, gap_m, cutoff, cap_sign)


def _select_routes(
    spec: CylinderCavitySpec, relative: np.ndarray
) -> tuple[_TransformRoute, ...]:
    radius = math.hypot(float(relative[0]), float(relative[1]))
    height = spec.geometry.height_m
    half_height = height / 2.0
    z = float(relative[2])
    radial_gap = abs(radius - spec.geometry.radius_m)
    radial_lower = _radial_gap_lower(
        float(relative[0]), float(relative[1]), spec.geometry.radius_m
    )
    routes: list[_TransformRoute] = []
    if abs(z) > half_height:
        axial_gap = abs(z) - half_height
        if radial_gap >= axial_gap:
            routes.append(
                _make_route(
                    "side_modified",
                    radial_lower,
                    radial_gap,
                    length_scale=min(spec.geometry.radius_m, height),
                )
            )
        else:
            axial_lower = _decimal_lower_gap(z, math.copysign(half_height, z))
            routes.append(
                _make_route(
                    "side_hankel",
                    axial_lower,
                    axial_gap,
                    length_scale=min(spec.geometry.radius_m, height),
                )
            )
    else:
        routes.append(
            _make_route(
                "side_modified",
                radial_lower,
                radial_gap,
                length_scale=min(spec.geometry.radius_m, height),
            )
        )

    for cap_sign in (-1, 1):
        cap_z = cap_sign * half_height
        axial_gap = abs(z - cap_z)
        axial_lower = _decimal_lower_gap(z, cap_z)
        if radius < spec.geometry.radius_m and radial_lower > axial_lower:
            routes.append(
                _make_route(
                    "cap_inner",
                    radial_lower,
                    spec.geometry.radius_m - radius,
                    cap_sign=cap_sign,
                    length_scale=min(spec.geometry.radius_m, height),
                )
            )
        elif radius > spec.geometry.radius_m and radial_lower > axial_lower:
            routes.append(
                _make_route(
                    "cap_modified",
                    radial_lower,
                    radial_gap,
                    cap_sign=cap_sign,
                    length_scale=min(spec.geometry.radius_m, height),
                )
            )
        else:
            routes.append(
                _make_route(
                    "cap_hankel",
                    axial_lower,
                    axial_gap,
                    cap_sign=cap_sign,
                    length_scale=min(spec.geometry.radius_m, height),
                )
            )
    return tuple(routes)


def _side_hankel_axial(
    spec: CylinderCavitySpec, wave_number: float, z: float
) -> tuple[float, float, float]:
    half = spec.geometry.height_m / 2.0
    q = spec.mode.q
    beta = spec.axial_wavenumber_rad_m
    parity = -1.0 if q % 2 else 1.0
    if wave_number == 0.0:
        return (spec.geometry.height_m if q == 0 else 0.0, 0.0, 0.0)
    denominator = wave_number**2 + beta**2
    if z > half:
        gap = z - half
        decay_h = math.exp(-wave_number * spec.geometry.height_m)
        difference = (parity - 1.0) - math.expm1(-wave_number * spec.geometry.height_m)
        axial = wave_number * math.exp(-wave_number * gap) * difference / denominator
    else:
        gap = -z - half
        decay_h = math.exp(-wave_number * spec.geometry.height_m)
        if parity == 1.0:
            difference = -math.expm1(-wave_number * spec.geometry.height_m)
        else:
            difference = 1.0 + decay_h
        axial = wave_number * math.exp(-wave_number * gap) * difference / denominator
    direction = 1.0 if z > 0 else -1.0
    return axial, -direction * wave_number * axial, wave_number**2 * axial


def _cap_inner_zero_endpoint_vector(
    spec: CylinderCavitySpec, relative: np.ndarray, route: _TransformRoute
) -> np.ndarray:
    """Return the explicit kappa=0 limit for the radial-constant cap residual."""
    if spec.mode.n != 0 or spec.mode.m != 0:
        return np.zeros(13, dtype=np.complex128)
    cap_z = route.cap_sign * spec.geometry.height_m / 2.0
    gap = float(relative[2]) - cap_z
    if gap == 0.0:
        raise ValueError("inner-cap residual endpoint requires nonzero axial gap")
    epsilon = 1.0 if route.cap_sign < 0 else (-1.0 if spec.mode.q % 2 else 1.0)
    factor = 4.0 * spec.normalization * route.cap_sign * epsilon
    q = -factor * gap
    qz = -factor
    radius = math.hypot(float(relative[0]), float(relative[1]))
    phi = math.atan2(float(relative[1]), float(relative[0])) if radius else 0.0
    derivatives = (q, 0.0, qz, 0.0, 0.0, 0.0)
    if radius == 0.0:
        return _native_vector(
            spec,
            radius,
            phi,
            derivatives,
            axis_coefficients=(1.0, 0.0),
        )
    return _native_vector(spec, radius, phi, derivatives)


def _route_integrand(
    spec: CylinderCavitySpec,
    relative: np.ndarray,
    route: _TransformRoute,
):
    radius = math.hypot(float(relative[0]), float(relative[1]))
    phi = math.atan2(float(relative[1]), float(relative[0])) if radius else 0.0
    z = float(relative[2])
    m = spec.mode.m
    geometry = spec.geometry
    source_radial = 1.0 if spec.mode.n == 0 else spec.root_bessel

    def evaluate(wave_number: float) -> np.ndarray:
        if wave_number < 0.0:
            raise ValueError("cylinder transform wavenumber must be nonnegative")
        if wave_number == 0.0:
            if route.kind == "cap_inner":
                return _cap_inner_zero_endpoint_vector(spec, relative, route)
            return np.zeros(13, dtype=np.complex128)
        if route.kind == "side_modified":
            x_target = wave_number * radius
            x_source = wave_number * geometry.radius_m
            axial = _axial_overlap(spec.mode.q, geometry.height_m, z, wave_number)
            if radius < geometry.radius_m:
                target = _i_scaled_triplet(m, x_target)
                _, source_derivative, _ = _k_scaled_triplet(m, x_source)
                source_factor = source_derivative * math.exp(
                    -wave_number * (geometry.radius_m - radius)
                )
                radial = tuple(
                    wave_number**order * target[order] * source_factor
                    for order in range(3)
                )
            else:
                _, source_derivative, _ = _i_scaled_triplet(m, x_source)
                target = _k_scaled_triplet(m, x_target)
                source_factor = source_derivative * math.exp(
                    -wave_number * (radius - geometry.radius_m)
                )
                radial = tuple(
                    wave_number**order * target[order] * source_factor
                    for order in range(3)
                )
            prefactor = (
                4.0
                * spec.normalization
                * geometry.radius_m
                * source_radial
                * wave_number
            )
            if (
                m == 1
                and radius > 0.0
                and radius < geometry.radius_m
                and x_target <= _M1_CARTESIAN_LADDER_ARGUMENT_MAX
            ):
                return _m1_bessel_cartesian(
                    spec,
                    radius,
                    phi,
                    wave_number,
                    "i",
                    prefactor * source_factor,
                    axial,
                )
            if radius == 0.0:
                leading, next_coefficient = _radial_axis_coefficients(
                    "i", m, wave_number
                )
                if spec.mode.n == 0:
                    _, source_derivative, _ = _k_scaled_triplet(0, x_source)
                    source_factor = source_derivative * math.exp(
                        -wave_number * geometry.radius_m
                    )
                axis_prefactor = prefactor * source_factor
                return _native_vector(
                    spec,
                    radius,
                    phi,
                    (axial[0], 0.0, axial[1], 0.0, 0.0, axial[2]),
                    axis_coefficients=(
                        leading * axis_prefactor,
                        next_coefficient * axis_prefactor,
                    ),
                )
            derivatives = (
                prefactor * radial[0] * axial[0],
                prefactor * radial[1] * axial[0],
                prefactor * radial[0] * axial[1],
                prefactor * radial[2] * axial[0],
                prefactor * radial[1] * axial[1],
                prefactor * radial[0] * axial[2],
            )
            return _native_vector(spec, radius, phi, derivatives)

        if route.kind == "cap_inner":
            cap_z = route.cap_sign * geometry.height_m / 2.0
            gap = z - cap_z
            if gap == 0.0:
                raise ValueError("inner-cap residual requires nonzero axial gap")
            epsilon = 1.0 if route.cap_sign < 0 else (-1.0 if spec.mode.q % 2 else 1.0)
            argument = wave_number * radius
            target = _i_scaled_triplet(m, argument)
            kappa_radius = wave_number * geometry.radius_m
            _, source_derivative, _ = _k_scaled_triplet(m, kappa_radius)
            radial_gap = geometry.radius_m - radius
            radial_wave_number = spec.radial_wavenumber_rad_m
            ratio = (
                1.0
                if radial_wave_number == 0.0
                else (wave_number / math.hypot(wave_number, radial_wave_number)) ** 2
            )
            prefactor = (
                4.0
                * spec.normalization
                * route.cap_sign
                * epsilon
                * geometry.radius_m
                * spec.root_bessel
                * ratio
                * source_derivative
                * math.exp(-wave_number * radial_gap)
            )
            axial = (
                math.sin(wave_number * gap),
                wave_number * math.cos(wave_number * gap),
                -(wave_number**2) * math.sin(wave_number * gap),
            )
            if radius == 0.0:
                leading, next_coefficient = _radial_axis_coefficients(
                    "i", m, wave_number
                )
                return _native_vector(
                    spec,
                    radius,
                    phi,
                    (axial[0], 0.0, axial[1], 0.0, 0.0, axial[2]),
                    axis_coefficients=(
                        prefactor * leading,
                        prefactor * next_coefficient,
                    ),
                )
            if m == 1 and argument <= _M1_CARTESIAN_LADDER_ARGUMENT_MAX:
                return _m1_bessel_cartesian(
                    spec,
                    radius,
                    phi,
                    wave_number,
                    "i",
                    prefactor,
                    axial,
                )
            derivatives = (
                prefactor * target[0] * axial[0],
                prefactor * wave_number * target[1] * axial[0],
                prefactor * target[0] * axial[1],
                prefactor * wave_number**2 * target[2] * axial[0],
                prefactor * wave_number * target[1] * axial[1],
                prefactor * target[0] * axial[2],
            )
            return _native_vector(spec, radius, phi, derivatives)

        if route.kind == "side_hankel":
            axial = _side_hankel_axial(spec, wave_number, z)
            radial = _j_triplet(m, wave_number * radius)
            source_derivative = float(jvp(m, wave_number * geometry.radius_m, 1))
            prefactor = (
                2.0
                * math.pi
                * spec.normalization
                * geometry.radius_m
                * source_radial
                * wave_number
                * source_derivative
            )
            direction = 1.0 if z > 0 else -1.0
            axial_derivatives = (
                axial[0],
                -direction * wave_number * axial[0],
                wave_number**2 * axial[0],
            )
            if (
                m == 1
                and radius > 0.0
                and wave_number * radius <= _M1_CARTESIAN_LADDER_ARGUMENT_MAX
            ):
                return _m1_bessel_cartesian(
                    spec,
                    radius,
                    phi,
                    wave_number,
                    "j",
                    prefactor,
                    axial_derivatives,
                )
            if radius == 0.0:
                leading, next_coefficient = _radial_axis_coefficients(
                    "j", m, wave_number
                )
                return _native_vector(
                    spec,
                    radius,
                    phi,
                    (
                        axial_derivatives[0],
                        0.0,
                        axial_derivatives[1],
                        0.0,
                        0.0,
                        axial_derivatives[2],
                    ),
                    axis_coefficients=(
                        prefactor * leading,
                        prefactor * next_coefficient,
                    ),
                )
            derivatives = (
                prefactor * radial[0] * axial_derivatives[0],
                prefactor * wave_number * radial[1] * axial_derivatives[0],
                prefactor * radial[0] * axial_derivatives[1],
                prefactor * wave_number**2 * radial[2] * axial_derivatives[0],
                prefactor * wave_number * radial[1] * axial_derivatives[1],
                prefactor * radial[0] * axial_derivatives[2],
            )
            return _native_vector(spec, radius, phi, derivatives)

        if route.kind == "cap_hankel":
            cap_sign = route.cap_sign
            cap_z = cap_sign * geometry.height_m / 2.0
            direction = 1.0 if z > cap_z else -1.0
            epsilon = 1.0 if cap_sign < 0 else (-1.0 if spec.mode.q % 2 else 1.0)
            axial_prefactor = (
                2.0
                * math.pi
                * spec.normalization
                * cap_sign
                * epsilon
                * direction
                * wave_number
                * math.exp(-wave_number * abs(z - cap_z))
                * _radial_overlap_j(spec, wave_number)
            )
            axial_derivatives = (
                axial_prefactor,
                -direction * wave_number * axial_prefactor,
                wave_number**2 * axial_prefactor,
            )
            if (
                m == 1
                and radius > 0.0
                and wave_number * radius <= _M1_CARTESIAN_LADDER_ARGUMENT_MAX
            ):
                return _m1_bessel_cartesian(
                    spec,
                    radius,
                    phi,
                    wave_number,
                    "j",
                    1.0,
                    axial_derivatives,
                )
            radial = _j_triplet(m, wave_number * radius)
            if radius == 0.0:
                leading, next_coefficient = _radial_axis_coefficients(
                    "j", m, wave_number
                )
                return _native_vector(
                    spec,
                    radius,
                    phi,
                    (
                        1.0,
                        0.0,
                        -direction * wave_number,
                        0.0,
                        0.0,
                        wave_number**2,
                    ),
                    axis_coefficients=(
                        leading * axial_derivatives[0],
                        next_coefficient * axial_derivatives[0],
                    ),
                )
            derivatives = (
                radial[0] * axial_derivatives[0],
                wave_number * radial[1] * axial_derivatives[0],
                radial[0] * axial_derivatives[1],
                wave_number**2 * radial[2] * axial_derivatives[0],
                wave_number * radial[1] * axial_derivatives[1],
                radial[0] * axial_derivatives[2],
            )
            return _native_vector(spec, radius, phi, derivatives)

        if route.kind == "cap_modified":
            cap_sign = route.cap_sign
            cap_z = cap_sign * geometry.height_m / 2.0
            epsilon = 1.0 if cap_sign < 0 else (-1.0 if spec.mode.q % 2 else 1.0)
            w_scaled = _radial_overlap_i_scaled(spec, wave_number)
            target = _k_scaled_triplet(m, wave_number * radius)
            radial_factor = w_scaled * math.exp(
                -wave_number * (radius - geometry.radius_m)
            )
            radial = tuple(
                wave_number**order * target[order] * radial_factor for order in range(3)
            )
            axial = (
                math.sin(wave_number * (z - cap_z)),
                wave_number * math.cos(wave_number * (z - cap_z)),
                -(wave_number**2) * math.sin(wave_number * (z - cap_z)),
            )
            prefactor = 4.0 * spec.normalization * cap_sign * epsilon * wave_number
            derivatives = (
                prefactor * radial[0] * axial[0],
                prefactor * radial[1] * axial[0],
                prefactor * radial[0] * axial[1],
                prefactor * radial[2] * axial[0],
                prefactor * radial[1] * axial[1],
                prefactor * radial[0] * axial[2],
            )
            return _native_vector(spec, radius, phi, derivatives)
        raise ValueError(f"unsupported cylinder transform route {route.kind!r}")

    return evaluate
