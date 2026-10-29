"""Independent scalar Newton oracle; no gragra integrator/kernel imports.

Every input is first a binary64 value, then converted exactly by integer ratio.
The primary path transforms body positions at 80 decimal digits using the
supplied matrix as-is. The world-position path is diagnostic only.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

DPS = 80
COMPONENTS = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}


def exact_float64(value):
    """Return the exact rational represented by one finite binary64 value."""
    numerator, denominator = float(np.float64(value)).as_integer_ratio()
    return mp.mpf(numerator) / denominator


def _newton(world, masses, targets, observable, softening):
    shape = (len(world), len(targets)) + COMPONENTS[observable]
    result = np.empty(shape, dtype=object)
    epsilon2 = exact_float64(softening) ** 2
    for frame, positions in enumerate(world):
        for target_index, target in enumerate(targets):
            sums = [mp.mpf(0) for _ in range(int(np.prod(COMPONENTS[observable])) or 1)]
            for position, mass in zip(positions, masses, strict=True):
                if mass == 0:
                    continue
                delta = [target[axis] - position[axis] for axis in range(3)]
                r2 = sum(value * value for value in delta) + epsilon2
                if r2 == 0:
                    raise ValueError("active source/target collision in oracle")
                inverse = 1 / mp.sqrt(r2)
                if observable == "potential":
                    terms = [-mass * inverse]
                elif observable == "acceleration":
                    terms = [-mass * value * inverse**3 for value in delta]
                else:
                    terms = [
                        mass
                        * (
                            3 * delta[i] * delta[j] * inverse**5
                            - (inverse**3 if i == j else 0)
                        )
                        for i in range(3)
                        for j in range(3)
                    ]
                sums = [a + b for a, b in zip(sums, terms, strict=True)]
            result[frame, target_index] = (
                sums[0]
                if observable == "potential"
                else np.array(sums, dtype=object).reshape(COMPONENTS[observable])
            )
    return result


def rigid_motion_gfree(
    positions, masses, rotations, translations, targets, *, observable, softening_m=0.0
):
    """Primary oracle: exact input ingestion followed by MP rigid transforms."""
    with mp.workdps(DPS):
        body = [[exact_float64(x) for x in row] for row in positions]
        mass = [exact_float64(x) for x in masses]
        target = [[exact_float64(x) for x in row] for row in targets]
        world = []
        for rotation, translation in zip(rotations, translations, strict=True):
            matrix = [[exact_float64(x) for x in row] for row in rotation]
            center = [exact_float64(x) for x in translation]
            world.append(
                [
                    [
                        sum(matrix[i][j] * point[j] for j in range(3)) + center[i]
                        for i in range(3)
                    ]
                    for point in body
                ]
            )
        return _newton(world, mass, target, observable, softening_m)


def transformed_world_gfree(
    world_positions, masses, targets, *, observable, softening_m=0.0
):
    """Auxiliary diagnostic; it intentionally omits MP coordinate transforms."""
    with mp.workdps(DPS):
        world = [
            [[exact_float64(x) for x in row] for row in frame]
            for frame in world_positions
        ]
        mass = [exact_float64(x) for x in masses]
        target = [[exact_float64(x) for x in row] for row in targets]
        return _newton(world, mass, target, observable, softening_m)


def normalized_float64(values, scale):
    """Round only after MP division by the registered physical scale."""
    with mp.workdps(DPS):
        return np.array([float(value / scale) for value in values.flat]).reshape(
            values.shape
        )


def reference_scale(masses, length_m, observable):
    with mp.workdps(DPS):
        mass = sum(exact_float64(value) for value in masses if value > 0)
        return (
            mass
            / length_m ** {"potential": 1, "acceleration": 2, "gradient": 3}[observable]
        )
