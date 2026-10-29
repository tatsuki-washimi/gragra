"""Secondary exterior volume and controlled potential-difference routes."""

import mpmath as mp
import numpy as np

from .sphere_reference import gauss_nodes
from .surface_reference import LD, SurfaceNewton, radial_profile


def interval_nodes(lower, upper, order):
    midpoint, half = LD(lower + upper) / 2, LD(upper - lower) / 2
    nodes, weights = zip(*gauss_nodes(order, 80), strict=True)
    return (
        np.array([midpoint + half * LD(mp.nstr(node, 80)) for node in nodes]),
        np.array([half * LD(mp.nstr(weight, 80)) for weight in weights]),
    )


def volume_values(fixture, order, gravitational_constant=6.67430e-11):
    engine = SurfaceNewton(fixture, gravitational_constant)
    if engine.inside:
        raise ValueError("unregularized volume crosscheck is exterior only")
    if engine.cuboid:
        axes = [
            interval_nodes(-length / 2, length / 2, order) for length in engine.lengths
        ]
        positions = np.stack(
            np.meshgrid(*(axis[0] for axis in axes), indexing="ij"), axis=-1
        )
        weights = (
            axes[0][1][:, None, None]
            * axes[1][1][None, :, None]
            * axes[2][1][None, None, :]
        )
        density = LD(mp.nstr(engine.normalization, 80)) * np.ones_like(weights)
        for axis in range(3):
            density *= np.cos(
                engine.pi
                * fixture.mode[axis]
                * (positions[..., axis] / engine.lengths[axis] + LD("0.5"))
            )
    else:
        r, wr = interval_nodes(0, engine.radius, order)
        phi, wphi = interval_nodes(-engine.pi, engine.pi, 2 * order)
        z, wz = interval_nodes(-engine.height / 2, engine.height / 2, order)
        positions = np.empty((r.size, phi.size, z.size, 3), dtype=LD)
        positions[..., 0] = r[:, None, None] * np.cos(phi)[None, :, None]
        positions[..., 1] = r[:, None, None] * np.sin(phi)[None, :, None]
        positions[..., 2] = z[None, None, :]
        weights = (
            r[:, None, None]
            * wr[:, None, None]
            * wphi[None, :, None]
            * wz[None, None, :]
        )
        profile = radial_profile(
            abs(fixture.order),
            mp.nstr(engine.root, 80),
            mp.nstr(mp.mpf(fixture.radius_m), 80),
            tuple(str(value) for value in r),
        )
        angular = (
            np.sin(abs(fixture.order) * phi)
            if fixture.order < 0
            else np.cos(fixture.order * phi)
        )
        axial = np.cos(
            engine.pi * fixture.axial_index * (z / engine.height + LD("0.5"))
        )
        density = (
            LD(mp.nstr(engine.normalization, 80))
            * profile[:, None, None]
            * angular[None, :, None]
            * axial[None, None, :]
        )
    displacement = engine.target - positions.reshape(-1, 3)
    weights = (weights * density).ravel()
    inverse = 1 / np.sqrt(np.sum(displacement * displacement, axis=1))
    integrals = [-np.sum(weights * inverse, dtype=LD)]
    integrals += [
        -np.sum(weights * displacement[:, i] * inverse**3, dtype=LD) for i in range(3)
    ]
    for i in range(3):
        for j in range(3):
            kernel = 3 * displacement[:, i] * displacement[:, j] * inverse**5 - (
                inverse**3 if i == j else 0
            )
            integrals.append(np.sum(weights * kernel, dtype=LD))
    return (
        engine.coefficient
        * LD(mp.nstr(engine.wave_number_squared, 80))
        * np.array(integrals)
    )


def potential_difference_values(fixture, step, quadrature_order=32):
    engine = SurfaceNewton(fixture, 6.67430e-11)
    first_weights = np.array([-1, 9, -45, 0, 45, -9, 1], dtype=LD) / 60
    second_weights = np.array([2, -27, 270, -490, 270, -27, 2], dtype=LD) / 180
    step = LD(step)

    def directional(direction):
        values = np.array(
            [
                engine.evaluate(
                    quadrature_order, engine.target + offset * step * direction
                )[0][0]
                for offset in range(-3, 4)
            ]
        )
        return np.sum(first_weights * values) / step, np.sum(
            second_weights * values
        ) / step**2

    axes = np.eye(3, dtype=LD)
    axis = [directional(direction) for direction in axes]
    hessian = np.empty((3, 3), dtype=np.clongdouble)
    for i in range(3):
        hessian[i, i] = axis[i][1]
        for j in range(i):
            hessian[i, j] = hessian[j, i] = (
                directional(axes[i] + axes[j])[1] - axis[i][1] - axis[j][1]
            ) / 2
    return np.concatenate(
        (
            [engine.evaluate(quadrature_order)[0][0]],
            [-value[0] for value in axis],
            -hessian.ravel(),
        )
    )
