"""Finite full-cap Cartesian checks versus direct 2D Newton surface integrals.

Independent of production and frozen computational references. This is a
formula counterexample search at fixed finite points, not scientific adoption.
"""

import math

import numpy as np
from scipy import integrate, special


def cartesian(values, order, r, phi, sine):
    q, qr, qz, qrr, qrz, qzz = values
    phase = np.exp(1j * order * phi)
    c, s = np.cos(phi), np.sin(phi)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    gradient = phase * np.array([qr, 1j * order * q / r, qz])
    hessian = phase * np.array(
        [
            [qrr, 1j * order * (qr / r - q / r**2), qrz],
            [
                1j * order * (qr / r - q / r**2),
                qr / r - order**2 * q / r**2,
                1j * order * qz / r,
            ],
            [qrz, 1j * order * qz / r, qzz],
        ]
    )
    output = np.r_[
        phase * q, rotation @ gradient, (rotation @ hessian @ rotation.T).ravel()
    ]
    return output.imag if sine else output.real


def cap_spectrum(radius, alpha, order, point, normalization, sigma, endpoint, sine):
    r = math.hypot(point[0], point[1])
    phi = math.atan2(point[1], point[0])
    g, a = point[2], alpha / radius
    sign = math.copysign(1, g)
    coefficient = (
        4 * normalization * sigma * endpoint * radius * special.jv(order, alpha)
    )
    cutoff = 80 / (radius - r)
    prefactor = (
        2 * np.pi * normalization * sigma * endpoint * sign * np.exp(-a * abs(g))
    )
    pairs = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    bulk = np.array(
        [
            prefactor * a**d * special.jvp(order, a * r, d) * (-sign * a) ** e
            for d, e in pairs
        ]
    )

    def integrand(kappa):
        if kappa == 0:
            return (
                np.array(
                    [
                        -4 * normalization * sigma * endpoint * g,
                        0,
                        0,
                        -4 * normalization * sigma * endpoint,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
                if a == 0
                else np.zeros(13)
            )
        scaled_k_prime = (
            -(
                special.kve(order - 1, kappa * radius)
                + special.kve(order + 1, kappa * radius)
            )
            / 2
        )
        scaled_i = (
            special.ive(order, kappa * r),
            (special.ive(order - 1, kappa * r) + special.ive(order + 1, kappa * r)) / 2,
            (
                special.ive(order - 2, kappa * r)
                + 2 * special.ive(order, kappa * r)
                + special.ive(order + 2, kappa * r)
            )
            / 4,
        )
        factor = coefficient * kappa**2 / (kappa**2 + a**2) * scaled_k_prime
        factor *= np.exp(-kappa * (radius - r))
        values = np.array(
            [
                factor
                * kappa ** (d + e)
                * scaled_i[d]
                * np.sin(kappa * g + e * np.pi / 2)
                for d, e in pairs
            ]
        )
        return cartesian(values, order, r, phi, sine)

    residual, _ = integrate.quad_vec(integrand, 0, cutoff, epsabs=2e-12, epsrel=2e-12)
    return residual + cartesian(bulk, order, r, phi, sine)


def direct_cap(
    radius, alpha, order, point, normalization, sigma, endpoint, sine, nodes
):
    u, weights = np.polynomial.legendre.leggauss(nodes)
    radial = radius * (u + 1) / 2
    radial_weights = radius * weights / 2
    angular = np.arange(4 * nodes) * (2 * np.pi / (4 * nodes))
    rr, angle = np.meshgrid(radial, angular, indexing="ij")
    sources = np.array([rr * np.cos(angle), rr * np.sin(angle), np.zeros_like(rr)])
    density = special.jv(order, alpha * rr / radius)
    density *= np.sin(order * angle) if sine else np.cos(order * angle)
    weight = normalization * endpoint * density * rr * radial_weights[:, None]
    weight *= 2 * np.pi / len(angular)
    differences = np.array(point)[:, None, None] - sources
    squared = np.sum(differences**2, axis=0)
    flux = sigma * differences[2]
    values = [np.sum(weight * flux / squared**1.5)]
    for i in range(3):
        kernel = (sigma if i == 2 else 0) / squared**1.5
        kernel -= 3 * flux * differences[i] / squared**2.5
        values.append(np.sum(weight * kernel))
    for i in range(3):
        for j in range(3):
            numerator = sigma * differences[j] if i == 2 else 0
            numerator += sigma * differences[i] if j == 2 else 0
            numerator += flux if i == j else 0
            kernel = -3 * numerator / squared**2.5
            kernel += 15 * flux * differences[i] * differences[j] / squared**3.5
            values.append(np.sum(weight * kernel))
    return np.array(values)


def main():
    cases = 0
    radius, phi = 1.3, 0.37
    for order, constant in ((0, True), (0, False), (1, False), (2, False), (3, False)):
        alpha = 0.0 if constant else float(special.jnp_zeros(order, 1)[0])
        eta_m = 1.0 if order == 0 else 0.5
        normalization = (
            math.sqrt(2)
            if constant
            else 1
            / math.sqrt(
                0.5 * eta_m * special.jv(order, alpha) ** 2 * (1 - order**2 / alpha**2)
            )
        )
        for sine in (False,) if order == 0 else (False, True):
            for g in (-0.17 * radius, 0.08 * radius):
                point = (
                    0.23 * radius * math.cos(phi),
                    0.23 * radius * math.sin(phi),
                    g,
                )
                for sigma in (-1, 1):
                    endpoint = 1 if sigma == -1 else -1
                    predicted = cap_spectrum(
                        radius,
                        alpha,
                        order,
                        point,
                        normalization,
                        sigma,
                        endpoint,
                        sine,
                    )
                    direct = direct_cap(
                        radius,
                        alpha,
                        order,
                        point,
                        normalization,
                        sigma,
                        endpoint,
                        sine,
                        192,
                    )
                    coarse = direct_cap(
                        radius,
                        alpha,
                        order,
                        point,
                        normalization,
                        sigma,
                        endpoint,
                        sine,
                        128,
                    )
                    for section in (slice(0, 1), slice(1, 4), slice(4, 13)):
                        scale = max(np.linalg.norm(direct[section]), 1e-10)
                        assert (
                            np.linalg.norm(direct[section] - coarse[section])
                            <= 1e-9 * scale
                        )
                        assert (
                            np.linalg.norm(predicted[section] - direct[section])
                            <= 1e-9 * scale
                        )
                    cases += 1
    print(f"Finite full-cap Newton S/Cartesian gradient/Hessian cases={cases} PASS")


if __name__ == "__main__":
    main()
