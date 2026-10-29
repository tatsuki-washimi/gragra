"""Finite-interval solid harmonics and analytic Cartesian second-order jets.

All coordinates/radii here are scaled by the outer radius. Taylor coefficients
are integrated analytically, never sampled as a spatial quadrature. See the U10
design for derivation, signs and roundoff/truncation diagnostics.
"""

import numpy as np

_REAL = np.longdouble
_COMPLEX = np.clongdouble
_EPS = np.finfo(_REAL).eps


def jet(value=0, gradient=None, hessian=None):
    out = np.zeros(13, dtype=_REAL)
    out[0] = value
    if gradient is not None:
        out[1:4] = gradient
    if hessian is not None:
        out[4:] = np.asarray(hessian).ravel()
    return out


def multiply(a, b):
    cross = np.outer(a[1:4], b[1:4])
    return jet(
        a[0] * b[0],
        a[0] * b[1:4] + b[0] * a[1:4],
        a[0] * b[4:].reshape(3, 3) + b[0] * a[4:].reshape(3, 3) + cross + cross.T,
    )


def norm(a):
    scale = np.max(np.abs(a))
    return _REAL(0) if scale == 0 else scale * np.sqrt(np.sum(np.abs(a / scale) ** 2))


def power_difference(lo, hi, exponent, *, relative_gap=None):
    """hi**n-lo**n without subtracting nearby powers, positive endpoints."""
    if exponent == 0 or (lo == hi and relative_gap is None):
        return _REAL(0)
    if lo == 0:
        return hi**exponent
    gap = (hi - lo) / hi if relative_gap is None else relative_gap
    return -(hi**exponent) * np.expm1(exponent * np.log1p(-gap))


def inner_power(exponent, thickness):
    """(1-thickness)**exponent without first rounding the inner radius."""
    return np.exp(exponent * np.log1p(-thickness))


def coefficients(q, degree):
    leading = _REAL(1)
    for ell in range(degree):
        leading *= q / (2 * ell + 3)
    terms = [leading]
    absolute = abs(leading)
    for m in range(1, 256):
        term = terms[-1] * (-q * q / (2 * m * (2 * degree + 2 * m + 1)))
        terms.append(term)
        absolute += abs(term)
        if m > q + 4 and abs(term) <= _EPS * absolute / 1024:
            return np.asarray(terms, dtype=_REAL)
    raise ArithmeticError("radial Taylor expansion reached 256 terms")


def radial_jet(value, first, second, point, radius):
    unit = point / radius
    return jet(
        value,
        first * unit,
        (first / radius) * np.eye(3) + (second - first / radius) * np.outer(unit, unit),
    )


def degree_terms(point, inner, thickness, q, d, p, projection, max_degree):
    """Yield four potential jets and absolute Taylor sums for each degree."""
    radius = norm(point)
    scale = max(_REAL(1), radius)
    z = jet(d @ (point / scale), d / scale)
    tangent = jet((p - projection * d) @ (point / scale), (p - projection * d) / scale)
    square = jet(
        (point / scale) @ (point / scale),
        2 * point / scale**2,
        2 * np.eye(3) / scale**2,
    )
    previous, before = jet(1), jet()
    angular_previous, angular_before = jet(), jet()
    for ell in range(max_degree + 1):
        if ell == 0:
            harmonic, angular = previous, angular_previous
        else:
            factor = _REAL(2 * ell - 1) / ell
            harmonic = factor * multiply(z, previous) - _REAL(ell - 1) / ell * multiply(
                square, before
            )
            angular = factor * (
                multiply(tangent, previous) + multiply(z, angular_previous)
            ) - _REAL(ell - 1) / ell * multiply(square, angular_before)
            before, previous, angular_before, angular_previous = (
                previous,
                harmonic,
                angular_previous,
                angular,
            )
        coeff = coefficients(q, ell)
        bulk = jet()
        bulk_abs = jet()
        walls = np.zeros((2, 13), dtype=_REAL)
        walls_abs = walls.copy()
        difference = jet()
        difference_abs = jet()
        if radius > inner:
            exponent = -2 * ell - 1
            # Harmonic is scaled by radius**ell only for exterior queries.
            radial_value = radius ** (-ell - 1) if radius > 1 else radius**exponent
            outside = radial_jet(
                radial_value,
                exponent * radial_value / radius,
                exponent * (exponent - 1) * radial_value / radius**2,
                point,
                radius,
            )
            outside_h, outside_a = (
                multiply(harmonic, outside),
                multiply(angular, outside),
            )
        for m, c in enumerate(coeff):
            if projection != 0:
                if radius <= inner:
                    integrated = power_difference(
                        inner, _REAL(1), 2 * m + 2, relative_gap=thickness
                    ) / (2 * m + 2)
                    term = c * integrated * harmonic
                elif radius >= 1:
                    integrated = power_difference(
                        inner,
                        _REAL(1),
                        2 * ell + 2 * m + 3,
                        relative_gap=thickness,
                    ) / (2 * ell + 2 * m + 3)
                    term = c * integrated * outside_h
                else:
                    exponent = 2 * ell + 2 * m + 3
                    inside = (
                        radius ** (2 * m + 2)
                        * power_difference(
                            inner / radius,
                            _REAL(1),
                            exponent,
                            relative_gap=((radius - 1) + thickness) / radius,
                        )
                        / exponent
                    )
                    upper = power_difference(radius, _REAL(1), 2 * m + 2) / (2 * m + 2)
                    first = -(2 * ell + 1) * inside / radius
                    second = (2 * ell + 1) * (2 * ell + 2) * inside / radius**2 - (
                        2 * ell + 1
                    ) * radius ** (2 * m)
                    term = c * multiply(
                        harmonic,
                        radial_jet(inside + upper, first, second, point, radius),
                    )
                bulk += term
                bulk_abs += abs(term)
            base = projection * (ell + 2 * m) * harmonic + angular
            for side, h in enumerate((inner, _REAL(1))):
                if radius < h:
                    wall = (
                        c
                        / q
                        * (inner_power(2 * m, thickness) if side == 0 else 1)
                        * base
                    )
                else:
                    wall = (
                        c
                        / q
                        * (
                            inner_power(2 * ell + 2 * m + 1, thickness)
                            if side == 0
                            else 1
                        )
                        * (projection * (ell + 2 * m) * outside_h + outside_a)
                    )
                walls[side] += wall
                walls_abs[side] += abs(wall)
            if radius < inner:
                term = (
                    -c
                    / q
                    * power_difference(inner, _REAL(1), 2 * m, relative_gap=thickness)
                    * base
                )
            elif radius > 1:
                term = (
                    -c
                    / q
                    * power_difference(
                        inner,
                        _REAL(1),
                        2 * ell + 2 * m + 1,
                        relative_gap=thickness,
                    )
                    * (projection * (ell + 2 * m) * outside_h + outside_a)
                )
            else:
                # Different radial branches in material; longdouble preserves
                # cancellation and its magnitude is included in the estimate.
                term = (
                    c
                    / q
                    * (
                        inner_power(2 * ell + 2 * m + 1, thickness)
                        * (projection * (ell + 2 * m) * outside_h + outside_a)
                        - base
                    )
                )
            difference += term
            difference_abs += abs(term)
        phase = _COMPLEX(1j) ** ell
        bulk_complex = 1j * phase * q * projection * bulk
        wall_phase = -1j * phase
        terms = np.asarray(
            [
                bulk_complex,
                wall_phase * walls[0],
                -wall_phase * walls[1],
                bulk_complex + wall_phase * difference,
            ],
            dtype=_COMPLEX,
        )
        absolute_bulk = q * abs(projection) * bulk_abs
        absolute = np.asarray(
            [absolute_bulk, walls_abs[0], walls_abs[1], absolute_bulk + difference_abs],
            dtype=_REAL,
        )
        yield ell, terms, absolute, len(coeff)
