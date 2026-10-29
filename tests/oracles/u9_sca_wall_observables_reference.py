"""Independent wall-sheet Newton integration; no gragra imports.

Geometry is parameterized here, independently of production. Orders on the
Gauss and periodic coordinates can be refined separately. G is not included.
"""

import numpy as np


def _gauss_rule(order):
    """Refine Gauss roots/weights in longdouble without normalizing area."""
    nodes = np.polynomial.legendre.leggauss(order)[0].astype(np.longdouble)

    def recurrence(x):
        previous, current = np.ones_like(x), x.copy()
        for degree in range(2, order + 1):
            previous, current = (
                current,
                ((2 * degree - 1) * x * current - (degree - 1) * previous) / degree,
            )
        return current, order * (previous - x * current) / (1 - x * x)

    for _ in range(8):
        polynomial, derivative = recurrence(nodes)
        correction = polynomial / derivative
        nodes -= correction
        if np.max(abs(correction)) <= 4 * np.finfo(np.longdouble).eps:
            break
    else:
        raise ArithmeticError("REFERENCE_NOT_CONVERGED: Gauss nodes")
    _, derivative = recurrence(nodes)
    return nodes, 2 / ((1 - nodes * nodes) * derivative**2)


def surface(shape, *, order=64, angular_order=256):
    """Return local source positions, outward unit normals, and area weights.

    Fixed validation geometries: sphere a=2, box half lengths 2,3,5,
    cylinder a=2 and full height=6 (meters).
    """
    nodes, weights = _gauss_rule(order)
    pi = np.arccos(np.longdouble(-1))
    phi = 2 * pi * (np.arange(angular_order, dtype=np.longdouble) + 0.5) / angular_order
    if shape == "sphere":
        z, angle = np.meshgrid(nodes, phi, indexing="ij")
        radial = np.sqrt(1 - z**2)
        normals = np.stack(
            (radial * np.cos(angle), radial * np.sin(angle), z), axis=-1
        ).reshape(-1, 3)
        area = np.repeat(4 * weights * 2 * pi / angular_order, angular_order)
        return 2 * normals, normals, area
    points, normals, areas = [], [], []
    if shape == "cuboid":
        half = np.array([2, 3, 5])
        s, t = np.meshgrid(nodes, nodes, indexing="ij")
        for axis in range(3):
            tangent = [j for j in range(3) if j != axis]
            for sign in (-1, 1):
                p = np.zeros((*s.shape, 3), dtype=np.longdouble)
                p[..., axis] = sign * half[axis]
                p[..., tangent[0]] = half[tangent[0]] * s
                p[..., tangent[1]] = half[tangent[1]] * t
                n = np.zeros_like(p)
                n[..., axis] = sign
                w = np.outer(weights, weights) * np.prod(half[tangent])
                points.append(p.reshape(-1, 3))
                normals.append(n.reshape(-1, 3))
                areas.append(w.ravel())
    elif shape == "cylinder":
        z, angle = np.meshgrid(3 * nodes, phi, indexing="ij")
        p = np.stack((2 * np.cos(angle), 2 * np.sin(angle), z), axis=-1)
        n = np.stack((np.cos(angle), np.sin(angle), np.zeros_like(z)), axis=-1)
        points.append(p.reshape(-1, 3))
        normals.append(n.reshape(-1, 3))
        areas.append(np.repeat(6 * weights * 2 * pi / angular_order, angular_order))
        radius, angle = np.meshgrid(nodes + 1, phi, indexing="ij")
        for sign in (-1, 1):
            p = np.stack(
                (
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    np.full_like(radius, 3 * sign),
                ),
                axis=-1,
            )
            n = np.zeros_like(p)
            n[..., 2] = sign
            points.append(p.reshape(-1, 3))
            normals.append(n.reshape(-1, 3))
            areas.append(
                np.repeat((nodes + 1) * weights * 2 * pi / angular_order, angular_order)
            )
    else:
        raise ValueError(shape)
    return np.concatenate(points), np.concatenate(normals), np.concatenate(areas)


def wall_reference(
    shape,
    targets,
    displacement,
    *,
    density=2300,
    order=64,
    angular_order=256,
    center=None,
    rotation=None,
):
    """Integrate sigma=-rho*n.u using source-target R and fixed world u."""
    points, normals, areas = surface(shape, order=order, angular_order=angular_order)
    if rotation is not None:
        points, normals = points @ rotation.T, normals @ rotation.T
    if center is not None:
        points = points + center
    u = np.atleast_2d(displacement)
    mass = -density * (u @ normals.T) * areas
    out = [
        np.empty((len(u), len(targets), *suffix), dtype=mass.dtype)
        for suffix in ((), (3,), (3, 3))
    ]
    for index, target in enumerate(targets):
        delta = points - target
        inv = 1 / np.linalg.norm(delta, axis=1)
        out[0][:, index] = -mass @ inv
        out[1][:, index] = np.einsum("sn,ni,n->si", mass, delta, inv**3)
        kernel = (
            3 * delta[:, :, None] * delta[:, None, :] * inv[:, None, None] ** 5
            - np.eye(3)[None] * inv[:, None, None] ** 3
        )
        out[2][:, index] = np.einsum("sn,nij->sij", mass, kernel)
    dtype = np.complex128 if np.iscomplexobj(displacement) else np.float64
    return tuple(
        np.asarray(v[0] if np.ndim(displacement) == 1 else v, dtype=dtype) for v in out
    )


def fixtures(shape):
    """Predeclared non-singular query set, including every production switch."""
    if shape == "sphere":
        return np.array(
            [[0, 0, 0], [0.4, -0.3, 0.7], [0, 0, 0.8], [3.1, -0.8, 1.3], [0, 0, -4.0]]
        )
    if shape == "cuboid":
        return np.array(
            [
                [0, 0, 0],
                [0.4, -0.3, 0.7],
                [0, 0, 0.8],
                [4.1, -0.8, 6.3],
                [2, 3, 8],
                [-2, -3, -8],
                [2, 0.5, 8],
                [5, 3, 1],
            ]
        )
    distance = 64 * np.sqrt(13)
    direction = np.array([2.0, -1.0, 3.0]) / np.sqrt(14)
    points = [
        [0, 0, 0],
        [0.4, -0.3, 0.7],
        [0, 0, 0.8],
        [0, 0, 5],
        [2e-8, 0, 0.8],
        [2e-6, 0, 0.8],
        [2e-8, 0, 5],
        [0.009999, 0, 0.8],
        [0.010001, 0, 0.8],
        [0.009999, 0, 5],
        [0.010001, 0, 5],
        [0.001999, 0, 0.8],
        [0.002001, 0, 0.8],
        [0.099999, 0, 0.8],
        [0.100001, 0, 0.8],
        [0.099999, 0, 5],
        [0.100001, 0, 5],
        [3.1, -0.8, 4.3],
        [2, 0, 5],
        [3, 0, 3],
    ]
    points += [direction * distance * factor for factor in (0.99999, 1.00001, 4)]
    return np.array(points)


def review_points(shape):
    """Converged external edge, axis, and far-switch regressions."""
    if shape == "cuboid":
        return np.array(
            [
                [2 + 1e-6, 3, 8],
                [2 - 1e-6, 3, 8],
                [2 + 1e-7, 3, 8],
                [2 - 1e-7, 3, 8],
            ]
        )
    if shape == "cylinder":
        points = [
            [0.001999, 0, 200],
            [0.002001, 0, 200],
            [0.099999, 0, 200],
            [0.100001, 0, 200],
            [0.001999, 0, 5],
            [0.002001, 0, 5],
            [0.099999, 0, 5],
            [0.100001, 0, 5],
            [0.099, 0, 32 * np.sqrt(13) - 1e-6],
            [0.099, 0, 32 * np.sqrt(13) + 1e-6],
            [0.100001, 0, np.sqrt((32 * np.sqrt(13) - 1e-6) ** 2 - 0.100001**2)],
            [0.11, 0, np.sqrt((32 * np.sqrt(13) - 1e-6) ** 2 - 0.11**2)],
            [0.199999, 0, 0.8],
            [0.200001, 0, 0.8],
            [0.199999, 0, 5],
            [0.200001, 0, 5],
        ]
        for radial in (0.100001, 0.200001, 3.0):
            for side in (-1, 1):
                distance = 16 * np.sqrt(13) + side * 1e-6
                points.append([radial, 0, np.sqrt(distance**2 - radial**2)])
        return np.array(points)
    return np.empty((0, 3))


def acceptance(reference, shape, block):
    length = {"sphere": 2, "cuboid": np.sqrt(38), "cylinder": np.sqrt(13)}[shape]
    scale = 4 * np.pi * 2300 * 1e-9 * (length, 1, 1 / length)[block]
    return 1e-12 * scale + (1e-10, 1e-10, 1e-9)[block] * np.abs(reference)
