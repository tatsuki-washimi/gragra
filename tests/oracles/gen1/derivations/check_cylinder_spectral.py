"""Independent finite numerical check of full S, gradient and Hessian spectra.

Production-free spectral integrals are compared with surface Newton kernels,
not with a production implementation. These are formula checks, not Gen1
scientific acceptance. Geometry R=1,H=1.4, normalized real modes, float64.
"""

import numpy as np
from scipy import integrate, special


def scaled_i(m, x, d=0):
    if d == 0:
        return special.ive(m, x)
    if d == 1:
        return (special.ive(m - 1, x) + special.ive(m + 1, x)) / 2
    return (special.ive(m - 2, x) + 2 * special.ive(m, x) + special.ive(m + 2, x)) / 4


def scaled_k(m, x, d=0):
    if d == 0:
        return special.kve(m, x)
    if d == 1:
        return -(special.kve(m - 1, x) + special.kve(m + 1, x)) / 2
    return (special.kve(m - 2, x) + 2 * special.kve(m, x) + special.kve(m + 2, x)) / 4


def radial_overlap(lam, m, alpha, r_symbol):
    if alpha == 0:
        return (
            r_symbol * special.jv(1, lam * r_symbol) / lam
            if lam
            else r_symbol * r_symbol / 2
        )
    v = lam * r_symbol
    if abs(v - alpha) < 1e-8:
        return (
            r_symbol * r_symbol / 2 * special.jv(m, alpha) ** 2 * (1 - m * m / alpha**2)
        )
    return (
        r_symbol
        * r_symbol
        * special.jv(m, alpha)
        * v
        * special.jvp(m, v)
        / (alpha * alpha - v * v)
    )


def cartesian(qs, m, r, phi, imaginary=False):
    q_symbol, qr_symbol, qz_symbol, qrr_symbol, qrz_symbol, qzz_symbol = qs
    phase = np.exp(1j * m * phi)
    gradient = phase * np.array(
        [qr_symbol, 1j * m * q_symbol / r, qz_symbol], dtype=complex
    )
    hessian = phase * np.array(
        [
            [qrr_symbol, 1j * m * (qr_symbol / r - q_symbol / r**2), qrz_symbol],
            [
                1j * m * (qr_symbol / r - q_symbol / r**2),
                qr_symbol / r - m * m * q_symbol / r**2,
                1j * m * qz_symbol / r,
            ],
            [qrz_symbol, 1j * m * qz_symbol / r, qzz_symbol],
        ],
        dtype=complex,
    )
    rot = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    values = np.r_[phase * q_symbol, rot @ gradient, (rot @ hessian @ rot.T).ravel()]
    return values.imag if imaginary else values.real


def spectra(point, m, alpha, q, r_symbol, h_symbol, n_symbol, cutoff=180, debug=False):
    r = np.hypot(*point[:2])
    z = point[2]
    beta = q * np.pi / h_symbol
    j_symbol = special.jv(m, alpha)

    def axial(kappa, j):
        ft = (
            h_symbol
            / 2
            * (
                np.exp(1j * beta * h_symbol / 2)
                * np.sinc((kappa - beta) * h_symbol / (2 * np.pi))
                + np.exp(-1j * beta * h_symbol / 2)
                * np.sinc((kappa + beta) * h_symbol / (2 * np.pi))
            )
        )
        return ((1j * kappa) ** j * np.exp(1j * kappa * z) * ft).real

    def component(radial, axial_deriv):
        return np.array(
            [
                radial[0] * axial_deriv[0],
                radial[1] * axial_deriv[0],
                radial[0] * axial_deriv[1],
                radial[2] * axial_deriv[0],
                radial[1] * axial_deriv[1],
                radial[0] * axial_deriv[2],
            ]
        )

    if abs(r - r_symbol) > abs(z) - h_symbol / 2:

        def side(kappa):
            if r < r_symbol:
                radial = np.array(
                    [
                        kappa**d
                        * scaled_i(m, kappa * r, d)
                        * scaled_k(m, kappa * r_symbol, 1)
                        for d in range(3)
                    ]
                ) * np.exp(-kappa * (r_symbol - r))
            else:
                radial = np.array(
                    [
                        kappa**d
                        * scaled_i(m, kappa * r_symbol, 1)
                        * scaled_k(m, kappa * r, d)
                        for d in range(3)
                    ]
                ) * np.exp(-kappa * (r - r_symbol))
            return (
                4
                * n_symbol
                * r_symbol
                * j_symbol
                * kappa
                * component(radial, [axial(kappa, j) for j in range(3)])
            )
    else:
        assert abs(z) > h_symbol / 2

        def side(lam):
            if z > h_symbol / 2:
                d_symbol = (
                    lam
                    * (
                        (-1) ** q * np.exp(-lam * (z - h_symbol / 2))
                        - np.exp(-lam * (z + h_symbol / 2))
                    )
                    / (lam * lam + beta * beta)
                )
            else:
                d_symbol = (
                    lam
                    * (
                        np.exp(-lam * (-z - h_symbol / 2))
                        - (-1) ** q * np.exp(-lam * (-z + h_symbol / 2))
                    )
                    / (lam * lam + beta * beta)
                )
            radial = [lam**d * special.jvp(m, lam * r, d) for d in range(3)]
            axial_deriv = [(-np.sign(z) * lam) ** j * d_symbol for j in range(3)]
            return (
                2
                * np.pi
                * n_symbol
                * r_symbol
                * j_symbol
                * lam
                * special.jvp(m, lam * r_symbol)
                * component(radial, axial_deriv)
            )

    total = integrate.quad_vec(side, 0, cutoff, epsabs=2e-11, epsrel=2e-11)[0]
    if debug:
        print("side", total)
    for sigma in [-1, 1]:
        zc = sigma * h_symbol / 2
        epsilon = 1 if sigma == -1 else (-1) ** q
        if abs(z - zc) >= max(r - r_symbol, 0):
            assert z != zc

            def cap(lam, zc=zc, sigma=sigma, epsilon=epsilon):
                overlap = radial_overlap(lam, m, alpha, r_symbol)
                radial = [lam**d * special.jvp(m, lam * r, d) for d in range(3)]
                axial_deriv = [
                    (-np.sign(z - zc) * lam) ** j * np.exp(-lam * abs(z - zc))
                    for j in range(3)
                ]
                return (
                    2
                    * np.pi
                    * n_symbol
                    * sigma
                    * epsilon
                    * np.sign(z - zc)
                    * lam
                    * overlap
                    * component(radial, axial_deriv)
                )
        else:
            assert r > r_symbol

            def cap(kappa, zc=zc, sigma=sigma, epsilon=epsilon):
                if alpha == 0:
                    wscaled_symbol = r_symbol * scaled_i(1, kappa * r_symbol) / kappa
                else:
                    wscaled_symbol = (
                        r_symbol
                        * r_symbol
                        * j_symbol
                        * kappa
                        * r_symbol
                        * scaled_i(m, kappa * r_symbol, 1)
                        / (alpha * alpha + (kappa * r_symbol) ** 2)
                    )
                radial = [
                    kappa**d
                    * scaled_k(m, kappa * r, d)
                    * wscaled_symbol
                    * np.exp(-kappa * (r - r_symbol))
                    for d in range(3)
                ]
                axial_deriv = [
                    kappa**j * np.sin(kappa * (z - zc) + j * np.pi / 2)
                    for j in range(3)
                ]
                return (
                    4
                    * n_symbol
                    * sigma
                    * epsilon
                    * kappa
                    * component(radial, axial_deriv)
                )

        cap_value = integrate.quad_vec(cap, 0, cutoff, epsabs=2e-11, epsrel=2e-11)[0]
        if debug:
            print("cap", sigma, cap_value)
        total += cap_value
    return total


def surface(point, m, alpha, q, r_symbol, h_symbol, n_symbol, order, imaginary=False):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    phi = np.pi * (nodes + 1)
    wp = np.pi * weights
    ang = np.sin(m * phi) if imaginary else np.cos(m * phi)
    angular_weight = ang * wp

    def kernel(points, normals):
        d = points - point
        dist = np.linalg.norm(d, axis=-1)
        nd = np.sum(normals * d, axis=-1)
        scalar = -nd / dist**3
        gradient = (
            normals / dist[..., None] ** 3
            - 3 * nd[..., None] * d / dist[..., None] ** 5
        )
        hessian = (
            3
            * (
                normals[..., :, None] * d[..., None, :]
                + d[..., :, None] * normals[..., None, :]
                + nd[..., None, None] * np.eye(3)
            )
            / dist[..., None, None] ** 5
        )
        hessian -= (
            15
            * nd[..., None, None]
            * d[..., :, None]
            * d[..., None, :]
            / dist[..., None, None] ** 7
        )
        return np.concatenate(
            [scalar[..., None], gradient, hessian.reshape(*dist.shape, 9)], axis=-1
        )

    zz = h_symbol / 2 * nodes
    wz = h_symbol / 2 * weights
    points = np.stack(
        np.broadcast_arrays(
            r_symbol * np.cos(phi)[:, None],
            r_symbol * np.sin(phi)[:, None],
            zz[None, :],
        ),
        axis=-1,
    )
    normals = np.stack(
        np.broadcast_arrays(
            np.cos(phi)[:, None], np.sin(phi)[:, None], np.zeros_like(zz)[None, :]
        ),
        axis=-1,
    )
    mode = (
        n_symbol
        * special.jv(m, alpha)
        * angular_weight[:, None]
        * np.cos(q * np.pi / h_symbol * (zz + h_symbol / 2))[None, :]
    )
    out = np.sum(
        kernel(points, normals) * mode[..., None] * wz[None, :, None] * r_symbol,
        axis=(0, 1),
    )
    rr = r_symbol / 2 * (nodes + 1)
    wr = r_symbol / 2 * weights
    for sigma in [-1, 1]:
        points = np.stack(
            np.broadcast_arrays(
                rr[None, :] * np.cos(phi)[:, None],
                rr[None, :] * np.sin(phi)[:, None],
                np.full((1, order), sigma * h_symbol / 2),
            ),
            axis=-1,
        )
        normals = np.zeros_like(points)
        normals[..., 2] = sigma
        mode = (
            n_symbol
            * angular_weight[:, None]
            * special.jv(m, alpha * rr / r_symbol)[None, :]
            * (1 if sigma == -1 else (-1) ** q)
        )
        out += np.sum(
            kernel(points, normals) * mode[..., None] * (rr * wr)[None, :, None],
            axis=(0, 1),
        )
    return out


def main():
    r_symbol, h_symbol = 1.0, 1.4
    points = [
        np.array([0.42, 0.16, 0.13]),
        np.array([1.0, 0.0, 1.15]),
        np.array([1.45, 0.2, h_symbol / 2]),
    ]
    for m in range(4):
        alpha = float(special.jnp_zeros(m, 1)[0])
        q = 1
        n_symbol = 1 / np.sqrt(
            0.5
            * (1 if m == 0 else 0.5)
            * special.jv(m, alpha) ** 2
            * (1 - m * m / alpha**2)
        )
        # Independently integrate the full radial and angular squared norm.
        radial = integrate.quad(
            lambda r, m=m, alpha=alpha: r * special.jv(m, alpha * r / r_symbol) ** 2,
            0,
            r_symbol,
            epsabs=1e-13,
        )[0]
        angular = integrate.quad(
            lambda p, m=m: np.cos(m * p) ** 2, 0, 2 * np.pi, epsabs=1e-13
        )[0]
        axial = integrate.quad(
            lambda z, q=q: np.cos(q * np.pi / h_symbol * (z + h_symbol / 2)) ** 2,
            -h_symbol / 2,
            h_symbol / 2,
            epsabs=1e-13,
        )[0]
        assert (
            abs(
                n_symbol
                * n_symbol
                * radial
                * angular
                * axial
                / (np.pi * r_symbol * r_symbol * h_symbol)
                - 1
            )
            < 2e-13
        )
        for point in points:
            r = np.hypot(*point[:2])
            phi = np.arctan2(point[1], point[0])
            values = spectra(point, m, alpha, q, r_symbol, h_symbol, n_symbol)
            for imaginary in [False, True] if m else [False]:
                got = cartesian(values, m, r, phi, imaginary)
                ref = surface(
                    point, m, alpha, q, r_symbol, h_symbol, n_symbol, 100, imaginary
                )
                ref2 = surface(
                    point, m, alpha, q, r_symbol, h_symbol, n_symbol, 140, imaginary
                )
                difference = np.max(np.abs(got - ref2))
                refinement = np.max(np.abs(ref - ref2))
                assert difference < 2e-9, (m, point, difference, refinement)
                assert refinement < 2e-9, (m, point, refinement)
                print(
                    f"m={m} q=1 sin={imaginary} point={point.tolist()} "
                    f"maxabs={difference:.4g} surface_refinement={refinement:.4g}"
                )
    m = alpha = 0
    q = 1
    n_symbol = np.sqrt(2.0)
    for point in points:
        qs = spectra(point, m, alpha, q, r_symbol, h_symbol, n_symbol)
        got = cartesian(qs, m, np.hypot(*point[:2]), np.arctan2(point[1], point[0]))
        ref = surface(point, m, alpha, q, r_symbol, h_symbol, n_symbol, 140)
        difference = np.max(np.abs(got - ref))
        assert difference < 2e-9
        print(f"radial constant point={point.tolist()} maxabs={difference:.4g}")
    print(
        "Cylinder full S/grad/Hessian and real volume RMS FINITE NUMERICAL CHECK PASS"
    )


if __name__ == "__main__":
    main()
