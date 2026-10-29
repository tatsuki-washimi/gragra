"""Stable real spherical solid harmonics and Cartesian derivatives."""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

_Jet = tuple[np.ndarray, np.ndarray, np.ndarray]


def _constant_jet(value: complex, count: int) -> _Jet:
    return (
        np.full(count, value, dtype=np.clongdouble),
        np.zeros((count, 3), dtype=np.clongdouble),
        np.zeros((count, 3, 3), dtype=np.clongdouble),
    )


def _jet_product(left: _Jet, right: _Jet) -> _Jet:
    left_value, left_gradient, left_hessian = left
    right_value, right_gradient, right_hessian = right
    value = left_value * right_value
    gradient = (
        left_gradient * right_value[:, None] + right_gradient * left_value[:, None]
    )
    hessian = (
        left_hessian * right_value[:, None, None]
        + right_hessian * left_value[:, None, None]
        + left_gradient[:, :, None] * right_gradient[:, None, :]
        + right_gradient[:, :, None] * left_gradient[:, None, :]
    )
    return value, gradient, hessian


def _scaled_sum(left: _Jet, left_scale: float, right: _Jet, right_scale: float) -> _Jet:
    return tuple(
        left_scale * left_part + right_scale * right_part
        for left_part, right_part in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _base_normalization(order: int) -> float:
    """Return normalized ``P_m^m`` coefficient without factorial products."""
    if order == 0:
        return math.exp(-0.5 * math.log(4.0 * math.pi))
    log_normalization = 0.5 * (
        math.log(2 * order + 1) - math.log(4.0 * math.pi) - math.lgamma(2 * order + 1)
    ) + 0.5 * math.log(2.0)
    log_double_factorial = (
        math.lgamma(2 * order + 1) - order * math.log(2.0) - math.lgamma(order + 1)
    )
    coefficient = math.exp(log_normalization + log_double_factorial)
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("real spherical harmonic normalization is not representable")
    return coefficient


def _recurrence_coefficients(degree: int, order: int) -> tuple[float, float]:
    """Normalized associated-Legendre recurrence coefficients."""
    first = math.sqrt(
        ((2 * degree - 1) / (degree - order)) * ((2 * degree + 1) / (degree + order))
    )
    if degree == order + 1:
        return first, 0.0
    second = math.sqrt(
        ((2 * degree + 1) / (2 * degree - 3))
        * (
            ((degree - order - 1) * (degree + order - 1))
            / ((degree - order) * (degree + order))
        )
    )
    return first, second


def _solid_harmonic_values(
    degree_index: int, m: int, positions_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return real ``r**l Y_lm`` values, gradients, and Hessians.

    The normalized recurrence advances in degree for fixed ``|m|``. Its base
    is the normalized sectoral polynomial ``(x + i*y)**|m|``; each recurrence
    step multiplies jets by ``z`` and ``x*x + y*y + z*z``. Propagating value,
    first derivative, and second derivative together avoids expanding a
    high-degree polynomial into large coefficients that later cancel.

    Positive ``m`` selects the cosine projection and negative ``m`` selects
    the sine projection, matching the model's real spherical-harmonic
    convention. Normalization is the orthonormal spherical-harmonic
    normalization with the extra ``sqrt(2)`` for nonzero real orders.
    """
    if isinstance(degree_index, (bool, np.bool_)) or not isinstance(
        degree_index, Integral
    ):
        raise TypeError("degree_index must be an integer")
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise TypeError("m must be an integer")
    degree = int(degree_index)
    signed_order = int(m)
    order = abs(signed_order)
    if degree < 0:
        raise ValueError("degree_index must be nonnegative")
    if order > degree:
        raise ValueError("m must satisfy -degree_index <= m <= degree_index")

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (N, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("positions_m must contain only finite values")

    coordinates = np.asarray(positions, dtype=np.longdouble)
    count = len(coordinates)
    w_value = np.asarray(
        coordinates[:, 0] + 1j * coordinates[:, 1], dtype=np.clongdouble
    )
    w_gradient = np.zeros((count, 3), dtype=np.clongdouble)
    w_gradient[:, 0] = 1.0
    w_gradient[:, 1] = 1.0j
    w_jet: _Jet = (
        w_value,
        w_gradient,
        np.zeros((count, 3, 3), dtype=np.clongdouble),
    )

    base = _constant_jet(_base_normalization(order), count)
    for _ in range(order):
        base = _jet_product(base, w_jet)

    if degree == order:
        complex_jet = base
    else:
        z_gradient = np.zeros((count, 3), dtype=np.clongdouble)
        z_gradient[:, 2] = 1.0
        z_jet: _Jet = (
            np.asarray(coordinates[:, 2], dtype=np.clongdouble),
            z_gradient,
            np.zeros((count, 3, 3), dtype=np.clongdouble),
        )
        radial_squared_jet: _Jet = (
            np.asarray(
                np.einsum("ij,ij->i", coordinates, coordinates),
                dtype=np.clongdouble,
            ),
            np.asarray(2.0 * coordinates, dtype=np.clongdouble),
            np.broadcast_to(2.0 * np.eye(3, dtype=np.longdouble), (count, 3, 3)).astype(
                np.clongdouble, copy=True
            ),
        )

        previous = base
        previous_previous: _Jet | None = None
        for current_degree in range(order + 1, degree + 1):
            first, second = _recurrence_coefficients(current_degree, order)
            first_term = _jet_product(z_jet, previous)
            if previous_previous is None:
                current = tuple(first * item for item in first_term)  # type: ignore[assignment]
            else:
                second_term = _jet_product(radial_squared_jet, previous_previous)
                current = _scaled_sum(first_term, first, second_term, -second)
            previous_previous, previous = previous, current
        complex_jet = previous

    component = 0 if signed_order >= 0 else 1
    outputs: list[np.ndarray] = []
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for item in complex_jet:
            real_component = item.real if component == 0 else item.imag
            output = np.asarray(real_component, dtype=np.float64)
            if not np.isfinite(output).all():
                raise ValueError("spherical harmonic evaluation is not finite")
            outputs.append(output)
    return outputs[0], outputs[1], outputs[2]


__all__ = ["_solid_harmonic_values"]
