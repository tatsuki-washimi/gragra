"""Independent Newton-projection Sphere oracle; never imports production code.

Roots, normalization, radial Newton integrals and Cartesian finite differences
are computed at arbitrary precision. Refinement differences are estimates.
"""

import copy
import math
from functools import cache

import mpmath as mp

from .acceptance import OracleUncertainty
from .sphere_fixtures import SphereFixture


def spherical_j(degree, argument):
    if argument == 0:
        return mp.mpf(1 if degree == 0 else 0)
    return mp.sqrt(mp.pi / (2 * argument)) * mp.besselj(
        degree + mp.mpf("0.5"), argument
    )


def spherical_j_prime(degree, argument):
    if degree == 0:
        return -spherical_j(1, argument)
    return (
        spherical_j(degree - 1, argument)
        - (degree + 1) * spherical_j(degree, argument) / argument
    )


@cache
def derivative_root(degree, radial_index, digits):
    with mp.workdps(digits):
        left = mp.mpf("0.01")
        step = mp.pi / 12
        found = 0
        while left < (radial_index + degree + 5) * mp.pi:
            right = left + step
            if spherical_j_prime(degree, left) * spherical_j_prime(degree, right) < 0:
                found += 1
                if found == radial_index:
                    return +mp.findroot(
                        lambda value: spherical_j_prime(degree, value),
                        (left, right),
                        solver="anderson",
                    )
            left = right
    raise ArithmeticError("positive derivative root not independently bracketed")


@cache
def gauss_nodes(order, digits):
    with mp.workdps(digits):
        nodes, weights = mp.gauss_quadrature(order, "legendre")
        return tuple(zip(nodes, weights, strict=True))


def solid_harmonic(degree, signed_order, point):
    """Approved real harmonics from differentiated Legendre polynomials."""
    order = abs(signed_order)
    x, y, z = point
    squared_radius = x * x + y * y + z * z
    polynomial = mp.mpf(0)
    for index in range((degree - order) // 2 + 1):
        power = degree - 2 * index
        coefficient = mp.mpf((-1) ** index * math.factorial(2 * degree - 2 * index))
        coefficient /= (
            2**degree
            * math.factorial(index)
            * math.factorial(degree - index)
            * math.factorial(power - order)
        )
        polynomial += coefficient * z ** (power - order) * squared_radius**index
    factor = mp.sqrt(
        (2 * degree + 1)
        * mp.mpf(math.factorial(degree - order))
        / (4 * mp.pi * math.factorial(degree + order))
    )
    if order:
        azimuth = (x + mp.j * y) ** order
        azimuth = azimuth.real if signed_order > 0 else azimuth.imag
        return mp.sqrt(2) * factor * polynomial * azimuth
    return factor * polynomial


class NewtonSphere:
    def __init__(self, fixture, gravitational_constant, digits=80, quadrature_order=48):
        self.fixture = fixture
        self.digits = digits
        self.quadrature_order = quadrature_order
        self.degree = fixture.degree
        with mp.workdps(digits):
            self.radius = mp.mpf(fixture.radius_m)
            self.root = derivative_root(fixture.degree, fixture.radial_index, digits)
            self.wave_number = self.root / self.radius
            radial_squared = self.integrate(
                lambda s: s * s * spherical_j(self.degree, self.wave_number * s) ** 2,
                mp.mpf(0),
                self.radius,
            )
            self.normalization = mp.sqrt(
                4 * mp.pi * self.radius**3 / (3 * radial_squared)
            )
            pressure = mp.mpc(fixture.pressure_real_pa, fixture.pressure_imag_pa)
            density_amplitude = pressure / mp.mpf(fixture.sound_speed_m_s) ** 2
            self.coefficient = (
                -4
                * mp.pi
                * mp.mpf(gravitational_constant)
                * density_amplitude
                * self.normalization
                / (2 * self.degree + 1)
            )
            self.full_inner = self.integrate(
                lambda s: (
                    s ** (self.degree + 2)
                    * spherical_j(self.degree, self.wave_number * s)
                ),
                mp.mpf(0),
                self.radius,
            )
        self.radial_cache = {}

    def integrate(self, function, lower, upper):
        if lower == upper:
            return mp.mpf(0)
        midpoint = (lower + upper) / 2
        half_width = (upper - lower) / 2
        return half_width * mp.fsum(
            weight * function(midpoint + half_width * node)
            for node, weight in gauss_nodes(self.quadrature_order, self.digits)
        )

    def radial_factor(self, radius, crosscheck=False):
        key = (radius, crosscheck)
        if key in self.radial_cache:
            return self.radial_cache[key]
        integrate = (
            (lambda function, lower, upper: mp.quad(function, [lower, upper]))
            if crosscheck
            else self.integrate
        )
        if radius >= self.radius:
            inner = (
                integrate(
                    lambda s: (
                        s ** (self.degree + 2)
                        * spherical_j(self.degree, self.wave_number * s)
                    ),
                    mp.mpf(0),
                    self.radius,
                )
                if crosscheck
                else self.full_inner
            )
            value = inner / radius ** (2 * self.degree + 1)
        else:
            inner = integrate(
                lambda s: (
                    s ** (self.degree + 2)
                    * spherical_j(self.degree, self.wave_number * s)
                ),
                mp.mpf(0),
                radius,
            )
            outer = integrate(
                lambda s: (
                    s ** (1 - self.degree)
                    * spherical_j(self.degree, self.wave_number * s)
                    if s
                    else mp.mpf(0)
                ),
                radius,
                self.radius,
            )
            value = outer + (inner / radius ** (2 * self.degree + 1) if radius else 0)
        self.radial_cache[key] = value
        return value

    def potential(self, point, crosscheck=False):
        radius = mp.sqrt(mp.fsum(value * value for value in point))
        return (
            self.coefficient
            * solid_harmonic(self.degree, self.fixture.order, point)
            * self.radial_factor(radius, crosscheck)
        )

    def observables(self, point, step):
        first_weights = tuple(
            mp.mpf(value) / 60 for value in (-1, 9, -45, 0, 45, -9, 1)
        )
        second_weights = tuple(
            mp.mpf(value) / 180 for value in (2, -27, 270, -490, 270, -27, 2)
        )
        axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        def directional(direction):
            values = [
                self.potential(
                    tuple(point[i] + offset * step * direction[i] for i in range(3))
                )
                for offset in range(-3, 4)
            ]
            return (
                mp.fsum(w * v for w, v in zip(first_weights, values, strict=True))
                / step,
                mp.fsum(w * v for w, v in zip(second_weights, values, strict=True))
                / step**2,
            )

        axis_derivatives = [directional(axis) for axis in axes]
        hessian = [[mp.mpc(0) for _ in range(3)] for _ in range(3)]
        for i in range(3):
            hessian[i][i] = axis_derivatives[i][1]
            for j in range(i):
                direction = tuple(axes[i][k] + axes[j][k] for k in range(3))
                mixed = (
                    directional(direction)[1]
                    - axis_derivatives[i][1]
                    - axis_derivatives[j][1]
                ) / 2
                hessian[i][j] = hessian[j][i] = mixed
        return {
            "potential": (self.potential(point),),
            "acceleration": tuple(-value[0] for value in axis_derivatives),
            "gradient": tuple(-value for row in hessian for value in row),
        }


def vector_norm(values):
    return mp.sqrt(mp.fsum(abs(value) ** 2 for value in values))


def differences(left, right):
    return {
        name: vector_norm(
            tuple(a - b for a, b in zip(left[name], right[name], strict=True))
        )
        for name in left
    }


@cache
def radial_engine(
    degree,
    index,
    radius,
    sound_speed,
    pressure_real,
    pressure_imag,
    gravitational_constant,
    digits,
    order,
):
    fixture = SphereFixture(
        degree,
        0,
        index,
        "radial_cache",
        (0.0, 0.0, 0.0),
        radius,
        sound_speed,
        pressure_real,
        pressure_imag,
    )
    return NewtonSphere(fixture, gravitational_constant, digits, order)


def fixture_engine(fixture, gravitational_constant, digits, order):
    engine = copy.copy(
        radial_engine(
            fixture.degree,
            fixture.radial_index,
            fixture.radius_m,
            fixture.sound_speed_m_s,
            fixture.pressure_real_pa,
            fixture.pressure_imag_pa,
            gravitational_constant,
            digits,
            order,
        )
    )
    engine.fixture = fixture
    return engine


def sphere_reference(fixture, gravitational_constant):
    """Refine solely against the frozen reference budget, never production values."""
    high = fixture_engine(fixture, gravitational_constant, 80, 48)
    middle = fixture_engine(fixture, gravitational_constant, 80, 40)
    coarse = fixture_engine(fixture, gravitational_constant, 80, 32)
    low = fixture_engine(fixture, gravitational_constant, 50, 48)
    with mp.workdps(80):
        point = tuple(mp.mpf(value) for value in fixture.target_m)
        radius = mp.sqrt(mp.fsum(value * value for value in point))
        gap = abs(radius - high.radius)
        if gap == 0:
            raise ValueError("physical boundary rejected")
        step = min(high.radius / 50, gap / 8)
        amplitude = abs(mp.mpc(fixture.pressure_real_pa, fixture.pressure_imag_pa))
        amplitude /= mp.mpf(fixture.sound_speed_m_s) ** 2
        length = (4 * mp.pi / 3) ** (mp.mpf(1) / 3) * high.radius
        scales = {
            "potential": mp.mpf(gravitational_constant) * amplitude * length**2,
            "acceleration": mp.mpf(gravitational_constant) * amplitude * length,
            "gradient": mp.mpf(gravitational_constant) * amplitude,
        }
        previous = None
        prior_change = None
        resolved = False
        for refinement in range(13):
            current = high.observables(point, step)
            change = differences(current, previous) if previous else None
            if change and prior_change:
                derivative = {
                    name: 10 * max(change[name], prior_change[name]) for name in current
                }
                budgets = {
                    name: mp.mpf("1e-12") * scales[name]
                    + mp.mpf("1e-9" if name == "gradient" else "1e-10")
                    * vector_norm(current[name])
                    for name in current
                }
                if all(derivative[name] <= budgets[name] / 40 for name in current):
                    resolved = True
                    break
            prior_change = change
            previous = current
            if refinement < 12:
                step /= 2
        with mp.workdps(50):
            low_values = low.observables(point, step)
        propagated = differences(current, low_values)
        middle_values = middle.observables(point, step)
        coarse_values = coarse.observables(point, step)
        last_quadrature_change = differences(current, middle_values)
        prior_quadrature_change = differences(middle_values, coarse_values)
        crosscheck = abs(high.potential(point, True) - current["potential"][0])
        output = {}
        for name, values in current.items():
            converted = tuple(complex(value) for value in values)
            rounding = vector_norm(
                tuple(
                    value - mp.mpc(actual)
                    for value, actual in zip(values, converted, strict=True)
                )
            )
            components = (
                ("derivative_refinement", float(derivative[name])),
                (
                    "quadrature_root_normalization_refinement",
                    float(
                        10
                        * max(
                            last_quadrature_change[name], prior_quadrature_change[name]
                        )
                    ),
                ),
                (
                    "root_normalization_arithmetic_precision",
                    float(10 * propagated[name]),
                ),
                (
                    "independent_potential_quadrature",
                    float(10 * crosscheck) if name == "potential" else 0.0,
                ),
                ("float_conversion", float(rounding)),
            )
            total = math.fsum(value for _, value in components)
            qualified = resolved and total <= float(budgets[name] / 20)
            uncertainty = OracleUncertainty(
                total if qualified else None,
                None,
                total if qualified else None,
                "estimate" if qualified else None,
                "Newton projection / seven-point differences / precision refinement",
                components,
            )
            output[name] = {"values_si": converted, "uncertainty": uncertainty}
        return output, {
            "root": str(high.root),
            "normalization": str(high.normalization),
            "digits": [50, 80],
            "quadrature_orders": [32, 40, 48],
            "precision_comparison_fixed_order": 48,
            "refinement": refinement,
            "step_m": str(step),
            "scales_si": {name: float(value) for name, value in scales.items()},
        }
