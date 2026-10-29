"""Finite independent Gaussian-transform vs surface-kernel formula checks.

No production imports; dimensionless G-free boundary S and its derivatives.
These comparisons check formulas, not the final scientific acceptance budget.
"""

import numpy as np
from scipy import integrate, special


def transform(t, x, l_symbol, n):
    a, b = -l_symbol / 2, l_symbol / 2
    da, db = a - x, b - x
    beta = n * np.pi / l_symbol
    if t == 0:
        return np.array([l_symbol if n == 0 else 0.0, 0.0, 0.0])
    if n == 0:
        if x < a:
            f_symbol = (
                np.sqrt(np.pi)
                / (2 * t)
                * (
                    np.exp(-((t * da) ** 2)) * special.erfcx(t * da)
                    - np.exp(-((t * db) ** 2)) * special.erfcx(t * db)
                )
            )
        elif x > b:
            f_symbol = (
                np.sqrt(np.pi)
                / (2 * t)
                * (
                    np.exp(-((t * db) ** 2)) * special.erfcx(-t * db)
                    - np.exp(-((t * da) ** 2)) * special.erfcx(-t * da)
                )
            )
        else:
            f_symbol = (
                np.sqrt(np.pi) / (2 * t) * (special.erf(t * db) - special.erf(t * da))
            )
        g_symbol = 0.0
    else:
        za = t * da - 1j * beta / (2 * t)
        zb = t * db - 1j * beta / (2 * t)
        ea = np.exp(-((t * da) ** 2) + 1j * beta * a)
        eb = np.exp(-((t * db) ** 2) + 1j * beta * b)
        if x < a:
            val = ea * special.erfcx(za) - eb * special.erfcx(zb)
        elif x > b:
            val = eb * special.erfcx(-zb) - ea * special.erfcx(-za)
        else:
            val = (
                2 * np.exp(-beta * beta / (4 * t * t) + 1j * beta * x)
                - eb * special.erfcx(zb)
                - ea * special.erfcx(-za)
            )
        val *= np.sqrt(np.pi) / (2 * t) * np.exp(1j * n * np.pi / 2)
        f_symbol, g_symbol = val.real, val.imag
    e_a = np.exp(-((t * da) ** 2))
    e_b = np.exp(-((t * db) ** 2))
    f1_symbol = e_a - (-1) ** n * e_b - beta * g_symbol
    f2_symbol = 2 * t * t * (da * e_a - (-1) ** n * db * e_b) - beta * beta * f_symbol
    return np.array([f_symbol, f1_symbol, f2_symbol])


def transform_surface(point, lengths, indices, cutoff=100):
    n_symbol = 2 ** (np.count_nonzero(indices) / 2)

    def integrand(t):
        fs = [
            transform(t, x, l_symbol, n)
            for x, l_symbol, n in zip(point, lengths, indices, strict=True)
        ]
        out = np.zeros(13)
        for i in range(3):
            for sigma in [-1, 1]:
                epsilon = 1 if sigma < 0 else (-1) ** indices[i]
                d = sigma * lengths[i] / 2 - point[i]
                e = np.exp(-t * t * d * d)
                qs = (
                    sigma
                    * epsilon
                    * e
                    * np.array(
                        [
                            d,
                            -1 + 2 * t * t * d * d,
                            2 * t * t * d * (-3 + 2 * t * t * d * d),
                        ]
                    )
                )
                factors = [qs if j == i else fs[j] for j in range(3)]

                def derivative(counts, factors=factors):
                    return np.prod([factors[j][counts[j]] for j in range(3)])

                out[0] += derivative([0, 0, 0])
                for j in range(3):
                    counts = [0, 0, 0]
                    counts[j] += 1
                    out[1 + j] += derivative(counts)
                    for k in range(3):
                        counts2 = counts.copy()
                        counts2[k] += 1
                        out[4 + 3 * j + k] += derivative(counts2)
        return -4 * n_symbol / np.sqrt(np.pi) * t * t * out

    return integrate.quad_vec(integrand, 0, cutoff, epsabs=1e-11, epsrel=1e-11)[0]


def direct_surface(point, lengths, indices, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    n_symbol = 2 ** (np.count_nonzero(indices) / 2)
    out = np.zeros(13)
    for i in range(3):
        tangents = [j for j in range(3) if j != i]
        j, k = tangents
        for sigma in [-1, 1]:
            points = np.empty((order, order, 3))
            points[..., i] = sigma * lengths[i] / 2
            points[..., j] = nodes[:, None] * lengths[j] / 2
            points[..., k] = nodes[None, :] * lengths[k] / 2
            normal = np.zeros(3)
            normal[i] = sigma
            mode = n_symbol * np.prod(
                np.cos((points + lengths / 2) * np.pi * indices / lengths), axis=-1
            )
            d = points - point
            dist = np.linalg.norm(d, axis=-1)
            nd = np.sum(d * normal, axis=-1)
            scalar = -nd / dist**3
            grad = (
                normal / dist[..., None] ** 3
                - 3 * nd[..., None] * d / dist[..., None] ** 5
            )
            hess = (
                3
                * (
                    normal[:, None] * d[..., None, :]
                    + d[..., :, None] * normal[None, :]
                    + nd[..., None, None] * np.eye(3)
                )
                / dist[..., None, None] ** 5
            )
            hess -= (
                15
                * nd[..., None, None]
                * d[..., :, None]
                * d[..., None, :]
                / dist[..., None, None] ** 7
            )
            values = np.concatenate(
                [scalar[..., None], grad, hess.reshape(order, order, 9)], axis=-1
            )
            weight = (
                mode * weights[:, None] * weights[None, :] * lengths[j] * lengths[k] / 4
            )
            out += np.sum(values * weight[..., None], axis=(0, 1))
    return out


def main():
    max_error = 0.0
    for n in [0, 1, 2, 5, 10]:
        for t in [1e-6, 0.01, 0.3, 2.0, 20.0, 100.0]:
            for x in [-1.7, -1.0, -0.3, 0.8, 1.0, 1.6]:
                got = transform(t, x, 2.0, n)
                direct = []
                for d in range(3):

                    def integrand(u, x=x, t=t, d=d, n=n):
                        delta = u - x
                        kernel = np.exp(-((t * delta) ** 2))
                        factor = [
                            1.0,
                            2 * t * t * delta,
                            -2 * t * t + 4 * t**4 * delta * delta,
                        ][d]
                        return kernel * factor * np.cos(n * np.pi / 2 * (u + 1))

                    value = integrate.quad(
                        integrand,
                        -1,
                        1,
                        epsabs=1e-11,
                        points=[x] if -1 < x < 1 else None,
                    )[0]
                    direct.append(value)
                error = np.max(np.abs(got - direct))
                max_error = max(error, max_error)
                assert error < 2e-9, (n, t, x, got, direct, error)
    print("Cuboid F/Fprime/Fsecond erfcx numerical check maxabs", max_error)
    lengths = np.array([2.0, 1.6, 2.4])
    for indices in [np.array([1, 1, 0]), np.array([2, 1, 1]), np.array([0, 0, 1])]:
        for point in [
            np.array([0.2, 0.1, 0.3]),
            np.array([1.45, 1.1, 0.3]),
            np.array([1.0, 1.1, 0.3]),
        ]:
            got = transform_surface(point, lengths, indices)
            ref = direct_surface(point, lengths, indices, 100)
            refined = direct_surface(point, lengths, indices, 140)
            error = np.max(np.abs(got - refined))
            uncertainty = np.max(np.abs(ref - refined))
            assert error < 2e-9, (indices, point, error)
            assert uncertainty < 2e-9, (indices, point, uncertainty)
            print(
                f"indices={indices.tolist()} point={point.tolist()} "
                f"maxabs={error:.4g} surface_refinement={uncertainty:.4g}"
            )
    print("Cuboid full S/grad/Hessian FINITE NUMERICAL CHECK PASS")


if __name__ == "__main__":
    main()
