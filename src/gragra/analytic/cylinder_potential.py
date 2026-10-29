"""Analytic finite-cylinder potential; no spatial quadrature.

The logarithmic special function below uses convergent, analytically
integrated elliptic series. It is not a finite elliptic closed form.
Domains, truncation bounds, and provenance: docs/design/cylinder-potential.md.
"""

from math import comb

import numpy as np

from gragra.analytic.cavity_cylinder import (
    _cylinder_end_fields,
    _cylinder_far_field_values,
    _cylinder_far_field_values_scaled,
    _cylinder_is_far_field_point,
    _cylinder_is_far_field_scaled,
    _cylinder_local_targets,
    _cylinder_rotation,
    _reject_cylinder_boundary_targets,
    _validate_cylinder_inputs,
    cylinder_acceleration_gradient_g_free,
    cylinder_potential_axis_g_free,
)
from gragra.analytic.values import GFreeGravityValues


def _log_moments(d: np.longdouble, x: np.longdouble, count: int) -> np.ndarray:
    """Exact moments integral_0^1 v^(2p) log(d²+x²v²) dv, p=0,... ."""
    p = np.arange(count, dtype=np.longdouble)
    denominator = 2 * p + 1
    log_end = np.log(d * d + x * x)
    if d == 0:
        return (log_end - 2 / denominator) / denominator
    if x >= d:
        ratio = (d / x) ** 2
        rational = np.arctan(x / d) / (d / x)
        out = np.empty(count, dtype=np.longdouble)
        for index in range(count):
            rational = 1 / np.longdouble(2 * index + 1) - ratio * rational
            out[index] = (log_end - 2 * rational) / (2 * index + 1)
        return out
    # beta <= 1/2; positive beta-series avoids an unstable upward recurrence.
    beta = x * x / (d * d + x * x)
    term = np.ones(count, dtype=np.longdouble)
    total = np.zeros(count, dtype=np.longdouble)
    for k in range(1, 81):
        term *= beta * (2 * k) / (denominator + 2 * k)
        total += term / k
    return (log_end - total) / denominator


def _logarithmic_local(d: np.longdouble, x: np.longdouble) -> np.longdouble:
    """Complementary K series for 0<=d<1/4, 0<x<1/4.

    32 complementary terms and 64 binomial terms; these are series
    coefficients, not samples of a spatial or spectral integrand.
    """
    count = 95
    log_coeff = np.zeros(count, dtype=np.longdouble)
    plain_coeff = np.zeros(count, dtype=np.longdouble)
    alpha = np.longdouble(1)
    harmonic = np.longdouble(0)
    for n in range(32):
        polynomial = np.array(
            [comb(n, k) * d ** (2 * (n - k)) * x ** (2 * k) for k in range(n + 1)],
            dtype=np.longdouble,
        )
        binomial = np.ones(64, dtype=np.longdouble)
        derivative = np.zeros(64, dtype=np.longdouble)
        for j in range(1, 64):
            binomial[j] = -binomial[j - 1] * (n + j - 0.5) / j * x * x
            derivative[j] = derivative[j - 1] + 1 / np.longdouble(n + j - 0.5)
        product = alpha * np.convolve(polynomial, binomial)
        log_coeff[: len(product)] += product
        product = alpha * np.convolve(
            polynomial,
            binomial * (np.log(np.longdouble(4)) - harmonic - derivative / 2),
        )
        plain_coeff[: len(product)] += product
        alpha *= (np.longdouble(2 * n + 1) / (2 * n + 2)) ** 2
        harmonic += 1 / np.longdouble((n + 1) * (2 * n + 1))
    denominators = 2 * np.arange(count, dtype=np.longdouble) + 1
    return (
        4
        * x
        * np.sum(plain_coeff / denominators - log_coeff * _log_moments(d, x, count) / 2)
    )


def _cylinder_logarithmic_function(radius: float, r: float, u: float) -> np.longdouble:
    """J = integral_0^(2pi) asinh(u/sqrt(R²+r²-2Rr cos(phi))) dphi.

    Real odd branch, J(0)=0, including the continuous r=R endpoint limit.
    Public gravity APIs still reject the physical rim. All production
    evaluations are analytic series, with a domain cover bounded away
    from the slowly convergent complete-elliptic endpoint.
    """
    radius, r, u = map(np.longdouble, (radius, r, u))
    if u == 0:
        return np.longdouble(0)
    c = radius + r
    d = abs(radius - r) / c
    x = abs(u) / c
    pi = np.arccos(np.longdouble(-1))
    if d < 0.25 and x < 0.25:
        return np.sign(u) * _logarithmic_local(d, x)
    t = 4 * radius * r / c**2
    y = x / np.hypot(1, x)
    v = 1 / (1 + x * x)
    alpha = np.longdouble(1)
    power = np.longdouble(1)
    total = np.arcsinh(x)
    if d >= 0.25:
        # t <= 15/16. I_n = integral_0^y (1-s²)^(n-1) ds.
        integral = y
        for n in range(1, 1025):
            if n > 1:
                integral = ((2 * n - 2) * integral + y * v ** (n - 1)) / (2 * n - 1)
            alpha *= (np.longdouble(2 * n - 1) / (2 * n)) ** 2
            power *= t
            total += alpha * power * integral
            if alpha * power * t * y / (1 - t) < 2e-19:
                break
    else:
        # v <= 16/17. Backward recurrence for T_n=I_n(1)-I_n(y)
        # avoids cancellation between two nearly equal complete primitives.
        tail = np.zeros(1025, dtype=np.longdouble)
        for n in range(1024, 1, -1):
            tail[n - 1] = ((2 * n - 1) * tail[n] + y * v ** (n - 1)) / (2 * n - 2)
        total += np.log(c / max(radius, r))
        for n in range(1, 1025):
            alpha *= (np.longdouble(2 * n - 1) / (2 * n)) ** 2
            power *= t
            total -= alpha * power * tail[n]
    return np.sign(u) * 2 * pi * total


def cylinder_analytic_observables_g_free(
    center_m: list | tuple | np.ndarray,
    radius_m: float,
    height_m: float,
    density_kg_m3: float,
    targets_m: list | tuple | np.ndarray,
    *,
    rotation_matrix: list | tuple | np.ndarray | None = None,
) -> GFreeGravityValues:
    """Analytic potential, acceleration, and gradient of a finite cylinder.

    Potential has the infinity-zero gauge. All outputs are read-only,
    G-free float64 arrays; units are kg/m, kg/m², and kg/m³. The cylinder's
    local axis is z and ``rotation_matrix`` maps local vectors to world.
    Material-boundary targets are rejected, matching the gradient contract.
    Uses elliptic functions and convergent analytic series, not numerical
    spatial integration. The existing ``cylinder_observables_g_free`` remains
    the separately named numerical boundary-quadrature diagnostic.
    """
    center, radius, height, density, targets = _validate_cylinder_inputs(
        center_m, radius_m, height_m, density_kg_m3, targets_m
    )
    rotation = (
        np.eye(3) if rotation_matrix is None else _cylinder_rotation(rotation_matrix)
    )
    points, scaled_mask, coordinate_scales, scaled_points = _cylinder_local_targets(
        center, targets, rotation
    )
    _reject_cylinder_boundary_targets(
        points,
        radius,
        height,
        scaled_mask=scaled_mask,
        coordinate_scales=coordinate_scales,
        scaled_points=scaled_points,
    )
    acceleration, gradient = cylinder_acceleration_gradient_g_free(
        center, radius, height, density, targets, rotation_matrix=rotation
    )
    potential = np.zeros(len(points))
    for index, point in enumerate(points):
        if density == 0:
            continue
        if scaled_mask[index]:
            if not _cylinder_is_far_field_scaled(
                coordinate_scales[index], scaled_points[index], radius, height
            ):
                raise OverflowError(
                    "non-far target exceeds float64 local-coordinate range"
                )
            potential[index] = _cylinder_far_field_values_scaled(
                radius,
                height,
                density,
                coordinate_scales[index],
                scaled_points[index],
            )[0]
            continue
        if _cylinder_is_far_field_point(points[index], radius, height):
            potential[index], _, _ = _cylinder_far_field_values(
                radius, height, density, points[index]
            )
            continue
        x, y, z = point
        r = np.hypot(x, y)
        if r == 0:
            potential[index] = cylinder_potential_axis_g_free(
                [0, 0, 0], radius, height, density, [[0, 0, z]]
            )[0]
            continue
        endpoints = []
        for u in (-height / 2 - z, height / 2 - z):
            radial, cap, _, _ = _cylinder_end_fields(radius, r, u)
            log_term = _cylinder_logarithmic_function(radius, r, u)
            endpoints.append(radius**2 * log_term + r * radial + u * cap)
        potential[index] = -density / 2 * (endpoints[1] - endpoints[0])
    return GFreeGravityValues(potential, acceleration, gradient)
