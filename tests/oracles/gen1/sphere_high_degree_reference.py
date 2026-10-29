"""High-precision discovery-following regression; no production dependencies."""

import copy
import math
from functools import cache

import mpmath as mp

from .acceptance import OracleUncertainty
from .sphere_reference import (
    NewtonSphere,
    differences,
    spherical_j,
    spherical_j_prime,
    vector_norm,
)


@cache
def first_derivative_root(degree, digits):
    """Independently indexed bracket, not a production/discovery root seed."""
    if degree < 1:
        raise ValueError("this high-degree first-root bracket requires l>=1")
    with mp.workdps(digits):
        lower = mp.sqrt(degree * (degree + 1))
        upper = mp.besseljzero(mp.mpf(degree) + mp.mpf("0.5"), 1)
        assert spherical_j_prime(degree, lower) > 0
        assert spherical_j_prime(degree, upper) < 0
        root = mp.findroot(
            lambda value: spherical_j_prime(degree, value),
            (lower, upper),
            solver="anderson",
        )
        assert lower < root < upper
        return +root


class HighDegreeNewtonSphere(NewtonSphere):
    def __init__(self, fixture, gravitational_constant, digits, quadrature_order):
        if fixture.radial_index != 1:
            raise ValueError("predeclared high-degree matrix contains n=1 only")
        self.fixture = fixture
        self.digits = digits
        self.quadrature_order = quadrature_order
        self.degree = fixture.degree
        with mp.workdps(digits):
            self.radius = mp.mpf(fixture.radius_m)
            self.root = first_derivative_root(self.degree, digits)
            self.wave_number = self.root / self.radius
            radial_squared = self.integrate(
                lambda radius: (
                    radius**2 * spherical_j(self.degree, self.wave_number * radius) ** 2
                ),
                mp.mpf(0),
                self.radius,
            )
            self.normalization = mp.sqrt(
                4 * mp.pi * self.radius**3 / (3 * radial_squared)
            )
            density = mp.mpc(fixture.pressure_real_pa, fixture.pressure_imag_pa)
            density /= mp.mpf(fixture.sound_speed_m_s) ** 2
            self.coefficient = (
                -4
                * mp.pi
                * mp.mpf(gravitational_constant)
                * density
                * self.normalization
                / (2 * self.degree + 1)
            )
            self.full_inner = self.integrate(
                lambda radius: (
                    radius ** (self.degree + 2)
                    * spherical_j(self.degree, self.wave_number * radius)
                ),
                mp.mpf(0),
                self.radius,
            )
        self.radial_cache = {}


@cache
def radial_engine(fixture, gravitational_constant, digits, order):
    # The fixture is frozen/hashable. Remove angular/target fields from the key
    # in fixture_engine to share all radial kernels across signed orders.
    return HighDegreeNewtonSphere(fixture, gravitational_constant, digits, order)


def fixture_engine(fixture, gravitational_constant, digits, order):
    from dataclasses import replace

    radial_fixture = replace(
        fixture, order=0, target_kind="radial_cache", target_m=(0.0, 0.0, 0.0)
    )
    result = copy.copy(
        radial_engine(radial_fixture, gravitational_constant, digits, order)
    )
    result.fixture = fixture
    return result


@cache
def alternate_normalization(fixture, gravitational_constant):
    high = fixture_engine(fixture, gravitational_constant, 130, 96)
    with mp.workdps(130):
        radial = mp.quad(
            lambda radius: (
                radius**2 * spherical_j(high.degree, high.wave_number * radius) ** 2
            ),
            [0, high.radius],
        )
        return +mp.sqrt(4 * mp.pi * high.radius**3 / (3 * radial))


def exact_zero_certificate(fixture):
    if fixture.degree <= 2:
        return None
    if all(value == 0 for value in fixture.target_m):
        return "regular degree-l solid harmonic times analytic even radial factor; l>2"
    if (
        fixture.target_m[0] == fixture.target_m[1] == 0
        and abs(fixture.order) == fixture.degree
        and fixture.degree > 2
    ):
        return "sectoral (x+i*y)^l transverse degree exceeds two on polar axis"
    return None


def sphere_high_degree_reference(fixture, gravitational_constant):
    high = fixture_engine(fixture, gravitational_constant, 130, 96)
    with mp.workdps(130):
        point = tuple(mp.mpf(value) for value in fixture.target_m)
        radius = mp.sqrt(mp.fsum(value * value for value in point))
        gap = abs(radius - high.radius)
        if gap == 0:
            raise ValueError("physical boundary rejected")
        amplitude = abs(mp.mpc(fixture.pressure_real_pa, fixture.pressure_imag_pa))
        amplitude /= mp.mpf(fixture.sound_speed_m_s) ** 2
        length = (4 * mp.pi / 3) ** (mp.mpf(1) / 3) * high.radius
        scales = {
            "potential": mp.mpf(gravitational_constant) * amplitude * length**2,
            "acceleration": mp.mpf(gravitational_constant) * amplitude * length,
            "gradient": mp.mpf(gravitational_constant) * amplitude,
        }
        metadata = {
            "root": mp.nstr(high.root, 130),
            "normalization": mp.nstr(high.normalization, 130),
            "root_indexing": (
                "unique first derivative root in sqrt(l(l+1))..first J zero"
            ),
            "digits": [90, 130],
            "quadrature_orders": [64, 80, 96],
            "precision_comparison_fixed_order": 96,
            "scales_si": {name: float(value) for name, value in scales.items()},
        }
        certificate = exact_zero_certificate(fixture)
        if certificate:
            metadata["exact_zero_certificate"] = certificate
            metadata["step_m"] = None
            metadata["refinement"] = None
            return {
                name: {
                    "values_si": (0j,) * size,
                    "uncertainty": OracleUncertainty(
                        0.0,
                        0.0,
                        0.0,
                        "rigorous_bound",
                        certificate,
                        (("exact_degree_certificate", 0.0),),
                    ),
                }
                for name, size in (
                    ("potential", 1),
                    ("acceleration", 3),
                    ("gradient", 9),
                )
            }, metadata
        step = min(high.radius / 50, gap / 8)
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
        middle = fixture_engine(fixture, gravitational_constant, 130, 80)
        coarse = fixture_engine(fixture, gravitational_constant, 130, 64)
        low = fixture_engine(fixture, gravitational_constant, 90, 96)
        with mp.workdps(90):
            low_values = low.observables(point, step)
        precision = differences(current, low_values)
        middle_values = middle.observables(point, step)
        last_quadrature_change = differences(current, middle_values)
        prior_quadrature_change = differences(
            middle_values, coarse.observables(point, step)
        )
        crosscheck = abs(high.potential(point, True) - current["potential"][0])
        normalization_relative = abs(
            alternate_normalization(fixture, gravitational_constant)
            / high.normalization
            - 1
        )
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
                    float(10 * precision[name]),
                ),
                (
                    "independent_potential_quadrature",
                    float(10 * crosscheck) if name == "potential" else 0.0,
                ),
                (
                    "independent_radial_normalization_quadrature",
                    float(10 * normalization_relative * vector_norm(values)),
                ),
                ("float_conversion", float(rounding)),
            )
            total = math.fsum(value for _, value in components)
            qualified = resolved and total <= float(budgets[name] / 20)
            output[name] = {
                "values_si": converted,
                "uncertainty": OracleUncertainty(
                    total if qualified else None,
                    None,
                    total if qualified else None,
                    "estimate" if qualified else None,
                    "high-precision Newton projection / seven-point target differences",
                    components,
                ),
            }
        metadata["step_m"] = mp.nstr(step, 130)
        metadata["refinement"] = refinement
        metadata["alternate_normalization_relative_difference"] = mp.nstr(
            normalization_relative, 130
        )
        return output, metadata
