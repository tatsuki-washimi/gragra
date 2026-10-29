"""Independent fixtures and literal continuum formulas for finite-body tests.

This module deliberately does not import production kernels or integrators.
"""

import numpy as np

B = np.array([0.07, -0.04, 0.05], dtype=np.float64)
A = np.array(
    [
        [0.03, 0.01, -0.02],
        [-0.015, -0.02, 0.005],
        [0.012, 0.018, 0.025],
    ],
    dtype=np.float64,
)
TARGETS = np.array([[3.0, -2.5, 2.25], [-2.75, 2.8, 2.4]])
VTK6 = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=np.int64,
)


def displacement(points: np.ndarray) -> np.ndarray:
    """Return the frozen affine field ``b + A x``."""
    return B + np.asarray(points) @ A.T


def structured_two_density_tetra(n: int):
    """Return vertices, tetra, and per-tetra density for the frozen box."""
    axis = np.linspace(-1.0, 1.0, n + 1)
    vertices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )

    def vertex_index(i: int, j: int, k: int) -> int:
        return (i * (n + 1) + j) * (n + 1) + k

    tetra = []
    density = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                corners = np.array(
                    [
                        vertex_index(i, j, k),
                        vertex_index(i + 1, j, k),
                        vertex_index(i + 1, j + 1, k),
                        vertex_index(i, j + 1, k),
                        vertex_index(i, j, k + 1),
                        vertex_index(i + 1, j, k + 1),
                        vertex_index(i + 1, j + 1, k + 1),
                        vertex_index(i, j + 1, k + 1),
                    ],
                    dtype=np.int64,
                )
                tetra.extend(corners[VTK6])
                density.extend([2.0 if (axis[i] + axis[i + 1]) < 0.0 else 5.0] * 6)
    return vertices, np.asarray(tetra), np.asarray(density)


def _literal_displacement_kernel(
    source: np.ndarray, target: np.ndarray, disp: np.ndarray, observable: str
) -> np.ndarray:
    r = source - target
    q = np.dot(r, r)
    root = np.sqrt(q)
    z = np.dot(r, disp)
    if observable == "potential":
        return z / (q * root)
    if observable == "acceleration":
        return disp / (q * root) - 3.0 * r * z / (q * q * root)
    if observable == "gradient":
        return 3.0 * (np.outer(disp, r) + np.outer(r, disp) + np.eye(3) * z) / (
            q * q * root
        ) - 15.0 * np.outer(r, r) * z / (q * q * q * root)
    raise ValueError(observable)


def continuum_oracle(order: int, targets: np.ndarray, observable: str) -> np.ndarray:
    """Tensor Gauss--Legendre integral over the two constant-density half-boxes."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    component_shape = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}[
        observable
    ]
    result = np.zeros((len(targets), *component_shape), dtype=np.float64)
    for x_lo, x_hi, density in ((-1.0, 0.0, 2.0), (0.0, 1.0, 5.0)):
        xs = 0.5 * (x_hi - x_lo) * nodes + 0.5 * (x_hi + x_lo)
        wx = 0.5 * (x_hi - x_lo) * weights
        ys = nodes
        zs = nodes
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                for iz, z in enumerate(zs):
                    point = np.array([x, y, z])
                    weight = density * wx[ix] * weights[iy] * weights[iz]
                    u = displacement(point)
                    for target_index, target in enumerate(targets):
                        result[target_index] += weight * _literal_displacement_kernel(
                            point, target, u, observable
                        )
    return result


def static_piecewise_acceleration(order: int, targets: np.ndarray) -> np.ndarray:
    """Literal G-free acceleration integral for the frozen two-half box."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = np.zeros((len(targets), 3), dtype=np.float64)
    for x_lo, x_hi, density in ((-1.0, 0.0, 2.0), (0.0, 1.0, 5.0)):
        xs = 0.5 * (x_hi - x_lo) * nodes + 0.5 * (x_hi + x_lo)
        wx = 0.5 * (x_hi - x_lo) * weights
        for ix, x in enumerate(xs):
            for iy, y in enumerate(nodes):
                for iz, z in enumerate(nodes):
                    source = np.array([x, y, z])
                    quadrature_weight = density * wx[ix] * weights[iy] * weights[iz]
                    delta = source - targets
                    radius = np.linalg.norm(delta, axis=1)
                    result += quadrature_weight * delta / radius[:, np.newaxis] ** 3
    return result


def response_norm(value: np.ndarray, observable: str) -> np.ndarray:
    if observable == "potential":
        return np.abs(value)
    axes = (-1,) if observable == "acceleration" else (-2, -1)
    return np.linalg.norm(value, axis=axes)


def target_scales(observable: str) -> np.ndarray:
    vertices = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    u_max = np.max(np.linalg.norm(displacement(vertices), axis=1))
    power = {"potential": 2, "acceleration": 3, "gradient": 4}[observable]
    nearest = np.maximum(np.abs(TARGETS) - 1.0, 0.0)
    distance = np.linalg.norm(nearest, axis=1)
    return 28.0 * u_max / distance**power
