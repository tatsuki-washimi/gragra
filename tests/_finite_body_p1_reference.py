"""Independent fixtures and literal continuum formulas for P1-density tests."""

import numpy as np

B = np.array([0.07, -0.04, 0.05])
A = np.array([[0.03, 0.01, -0.02], [-0.015, -0.02, 0.005], [0.012, 0.018, 0.025]])
TARGETS = np.array([[3.0, -2.5, 2.25], [-2.75, 2.8, 2.4]])
VTK6 = np.array(
    [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
)


def displacement(x):
    return B + np.asarray(x) @ A.T


def density(x, right):
    x = np.asarray(x)
    return (
        (5 - 0.15 * x[..., 0] + 0.08 * x[..., 1] - 0.04 * x[..., 2])
        if right
        else (2 + 0.2 * x[..., 0] + 0.1 * x[..., 1] + 0.05 * x[..., 2])
    )


def structured_p1_density_tetra(n):
    axis = np.linspace(-1, 1, n + 1)
    vertices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )

    def ix(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    cells, rho = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                corners = np.array(
                    [
                        ix(i, j, k),
                        ix(i + 1, j, k),
                        ix(i + 1, j + 1, k),
                        ix(i, j + 1, k),
                        ix(i, j, k + 1),
                        ix(i + 1, j, k + 1),
                        ix(i + 1, j + 1, k + 1),
                        ix(i, j + 1, k + 1),
                    ]
                )
                for tetra in corners[VTK6]:
                    cells.append(tetra)
                    rho.append(density(vertices[tetra], i >= n // 2))
    return vertices, np.asarray(cells), np.asarray(rho)


def tetra_polynomial_integral(v, c, linear, quadratic):
    volume = abs(np.linalg.det((v[1:] - v[0]).T)) / 6
    total = np.sum(v, axis=0)
    first = volume * total / 4
    second = volume * (np.outer(total, total) + v.T @ v) / 20
    return c * volume + linear @ first + np.sum(quadratic * second)


def affine_density_mass_and_first_moment(v, rho):
    volume = abs(np.linalg.det((v[1:] - v[0]).T)) / 6
    return volume * np.sum(rho) / 4, volume * (
        np.sum(rho) * np.sum(v, axis=0) + rho @ v
    ) / 20


def triangle_polynomial_integral(v, c, linear, quadratic):
    cross = np.cross(v[1] - v[0], v[2] - v[0])
    area = np.linalg.norm(cross) / 2
    total = np.sum(v, axis=0)
    first = area * total / 3
    second = area * (np.outer(total, total) + v.T @ v) / 12
    return c * area + linear @ first + np.sum(quadratic * second)


def _kernel(source, target, u, observable):
    d = source - target
    q = d @ d
    root = np.sqrt(q)
    z = d @ u
    if observable == "potential":
        return z / (q * root)
    if observable == "acceleration":
        return u / (q * root) - 3 * d * z / (q * q * root)
    if observable == "gradient":
        return 3 * (np.outer(u, d) + np.outer(d, u) + np.eye(3) * z) / (
            q * q * root
        ) - 15 * np.outer(d, d) * z / (q * q * q * root)
    raise ValueError(observable)


def continuum_oracle(order, targets, observable):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    shape = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}[observable]
    out = np.zeros((len(targets), *shape))
    for lo, hi, right in ((-1, 0, False), (0, 1, True)):
        xs = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        wx = 0.5 * (hi - lo) * weights
        for i, x in enumerate(xs):
            for j, y in enumerate(nodes):
                for k, z in enumerate(nodes):
                    p = np.array([x, y, z])
                    w = density(p, right) * wx[i] * weights[j] * weights[k]
                    for m, target in enumerate(targets):
                        out[m] += w * _kernel(p, target, displacement(p), observable)
    return out


def response_norm(value, observable):
    return (
        np.abs(value)
        if observable == "potential"
        else np.linalg.norm(
            value, axis=(-1,) if observable == "acceleration" else (-2, -1)
        )
    )


def target_scales(observable):
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    distance = np.linalg.norm(np.maximum(np.abs(TARGETS) - 1, 0), axis=1)
    return (
        27.3
        * np.max(np.linalg.norm(displacement(vertices), axis=1))
        / distance ** {"potential": 2, "acceleration": 3, "gradient": 4}[observable]
    )
