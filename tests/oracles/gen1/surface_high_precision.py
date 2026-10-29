"""Arbitrary-precision direct surface fallback for cancellation-limited references."""

import math

import mpmath as mp

from .acceptance import OracleUncertainty
from .sphere_reference import gauss_nodes
from .surface_reference import SurfaceNewton


def panel_nodes_mp(lower, upper, projection, gap, order):
    if gap <= 0:
        raise ValueError("physical surface rejected")
    points = {lower, upper}
    if lower < projection < upper:
        points.add(projection)
    spacing = gap
    for _ in range(128):
        for point in (projection - spacing, projection + spacing):
            if lower < point < upper:
                points.add(point)
        if projection - spacing <= lower and projection + spacing >= upper:
            break
        spacing *= 2
    else:
        raise ArithmeticError("high precision panel resource limit")
    result = []
    points = sorted(points)
    for lower, upper in zip(points[:-1], points[1:], strict=True):
        midpoint, half = (lower + upper) / 2, (upper - lower) / 2
        result += [
            (midpoint + half * node, half * weight)
            for node, weight in gauss_nodes(order, 80)
        ]
    return result


def cylinder_sources(engine, order):
    fixture = engine.fixture
    radius, height = mp.mpf(fixture.radius_m), mp.mpf(fixture.height_m)
    target = tuple(mp.mpf(value) for value in fixture.target_m)
    radial = mp.sqrt(target[0] ** 2 + target[1] ** 2)
    angle = mp.atan2(target[1], target[0])
    degree = abs(fixture.order)
    angular = (
        (lambda value: mp.sin(degree * value))
        if fixture.order < 0
        else (lambda value: mp.cos(degree * value))
    )
    gap = mp.sqrt((radial - radius) ** 2 + max(abs(target[2]) - height / 2, 0) ** 2)
    phi = panel_nodes_mp(angle - mp.pi, angle + mp.pi, angle, gap / radius, order)
    axial = panel_nodes_mp(-height / 2, height / 2, target[2], gap, order)
    axial = [
        (z, weight * mp.cos(mp.pi * fixture.axial_index * (z / height + mp.mpf("0.5"))))
        for z, weight in axial
    ]
    wall = engine.normalization * mp.besselj(degree, engine.root)
    for angle_value, angular_weight in phi:
        cosine, sine = mp.cos(angle_value), mp.sin(angle_value)
        weight = radius * wall * angular_weight * angular(angle_value)
        for z, axial_weight in axial:
            yield (
                (radius * cosine, radius * sine, z),
                (cosine, sine, mp.mpf(0)),
                weight * axial_weight,
            )
    for sign in (-1, 1):
        z = sign * height / 2
        gap = mp.sqrt((target[2] - z) ** 2 + max(radial - radius, 0) ** 2)
        radial_nodes = panel_nodes_mp(mp.mpf(0), radius, radial, gap, order)
        phi = panel_nodes_mp(
            angle - mp.pi, angle + mp.pi, angle, gap / max(radial, radius / 8), order
        )
        phi = [
            (mp.cos(value), mp.sin(value), weight * angular(value))
            for value, weight in phi
        ]
        cap_phase = (-1) ** fixture.axial_index if sign > 0 else 1
        for r, radial_weight in radial_nodes:
            weight = (
                cap_phase
                * engine.normalization
                * r
                * radial_weight
                * mp.besselj(degree, engine.root * r / radius)
            )
            for cosine, sine, angular_weight in phi:
                yield (
                    (r * cosine, r * sine, z),
                    (mp.mpf(0), mp.mpf(0), mp.mpf(sign)),
                    weight * angular_weight,
                )


def evaluate_high_precision(fixture, order, digits, gravitational_constant):
    engine = SurfaceNewton(fixture, gravitational_constant)
    with mp.workdps(digits):
        target = tuple(mp.mpf(value) for value in fixture.target_m)
        total = [mp.mpf(0) for _ in range(13)]
        absolute = [mp.mpf(0) for _ in range(13)]
        count = 0
        sources = (
            cuboid_sources(engine, order)
            if engine.cuboid
            else cylinder_sources(engine, order)
        )
        for source, normal, weight in sources:
            displacement = tuple(x - s for x, s in zip(target, source, strict=True))
            squared = sum(value * value for value in displacement)
            inverse = 1 / mp.sqrt(squared)
            inv3, inv5, inv7 = inverse**3, inverse**5, inverse**7
            flux = sum(n * value for n, value in zip(normal, displacement, strict=True))
            kernels = [flux * inv3]
            kernels += [
                normal[i] * inv3 - 3 * flux * displacement[i] * inv5 for i in range(3)
            ]
            kernels += [
                -3
                * (
                    normal[i] * displacement[j]
                    + normal[j] * displacement[i]
                    + (flux if i == j else 0)
                )
                * inv5
                + 15 * flux * displacement[i] * displacement[j] * inv7
                for i in range(3)
                for j in range(3)
            ]
            for index, kernel in enumerate(kernels):
                contribution = weight * kernel
                total[index] += contribution
                absolute[index] += abs(contribution)
            count += 1
        if engine.inside:

            def function(x, y, z):
                return engine.mode_mp((x, y, z))

            local = [function(*target)]
            local += [
                mp.diff(function, target, tuple(int(axis == i) for axis in range(3)))
                for i in range(3)
            ]
            local += [
                mp.diff(
                    function,
                    target,
                    tuple(int(axis == i) + int(axis == j) for axis in range(3)),
                )
                for i in range(3)
                for j in range(3)
            ]
            for index, value in enumerate(local):
                total[index] += 4 * mp.pi * value
                absolute[index] += 4 * mp.pi * abs(value)
        total[0] *= -1
        return (
            tuple(engine.coefficient_mp * value for value in total),
            tuple(abs(engine.coefficient_mp) * value for value in absolute),
            count,
        )


def vector_norm(values):
    return mp.sqrt(mp.fsum(abs(value) ** 2 for value in values))


def cuboid_sources(engine, order):
    lengths = tuple(mp.mpf(value) for value in engine.fixture.lengths_m)
    target = tuple(mp.mpf(value) for value in engine.fixture.target_m)
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        for sign in (-1, 1):
            coordinate = sign * lengths[axis] / 2
            gap = mp.sqrt(
                (target[axis] - coordinate) ** 2
                + sum(
                    max(abs(target[index]) - lengths[index] / 2, 0) ** 2
                    for index in other
                )
            )
            u = panel_nodes_mp(
                -lengths[other[0]] / 2,
                lengths[other[0]] / 2,
                target[other[0]],
                gap,
                order,
            )
            v = panel_nodes_mp(
                -lengths[other[1]] / 2,
                lengths[other[1]] / 2,
                target[other[1]],
                gap,
                order,
            )
            u = [
                (
                    value,
                    weight
                    * mp.cos(
                        mp.pi
                        * engine.fixture.mode[other[0]]
                        * (value / lengths[other[0]] + mp.mpf("0.5"))
                    ),
                )
                for value, weight in u
            ]
            v = [
                (
                    value,
                    weight
                    * mp.cos(
                        mp.pi
                        * engine.fixture.mode[other[1]]
                        * (value / lengths[other[1]] + mp.mpf("0.5"))
                    ),
                )
                for value, weight in v
            ]
            normal = tuple(mp.mpf(sign if index == axis else 0) for index in range(3))
            phase = (-1) ** engine.fixture.mode[axis] if sign > 0 else 1
            for first, first_weight in u:
                for second, second_weight in v:
                    point = [mp.mpf(0)] * 3
                    point[axis], point[other[0]], point[other[1]] = (
                        coordinate,
                        first,
                        second,
                    )
                    yield (
                        tuple(point),
                        normal,
                        engine.normalization * phase * first_weight * second_weight,
                    )


def high_precision_reference(fixture, gravitational_constant, existing):
    """Override unresolved values using independent quadrature/precision checks."""
    results = [
        evaluate_high_precision(fixture, order, 50, gravitational_constant)
        for order in (16, 24, 32)
    ]
    high, absolute, count = evaluate_high_precision(
        fixture, 32, 80, gravitational_constant
    )
    slices = {
        "potential": slice(0, 1),
        "acceleration": slice(1, 4),
        "gradient": slice(4, 13),
    }
    engine = SurfaceNewton(fixture, gravitational_constant)
    with mp.workdps(80):
        for name, section in slices.items():
            if existing[name]["uncertainty"].resolved():
                continue
            values = high[section]
            differences = [
                vector_norm(
                    tuple(
                        a - b
                        for a, b in zip(
                            results[i][0][section],
                            results[i - 1][0][section],
                            strict=True,
                        )
                    )
                )
                for i in (1, 2)
            ]
            quadrature = 10 * max(differences)
            precision = 10 * vector_norm(
                tuple(
                    a - b for a, b in zip(values, results[2][0][section], strict=True)
                )
            )
            roundoff = 1000 * count * mp.mpf(10) ** -49 * vector_norm(absolute[section])
            root_norm = (
                10 * engine.root_precision_estimate * vector_norm(absolute[section])
            )
            converted = tuple(complex(value) for value in values)
            conversion = vector_norm(
                tuple(
                    value - mp.mpc(actual)
                    for value, actual in zip(values, converted, strict=True)
                )
            )
            components = tuple(
                (name, float(value))
                for name, value in (
                    ("high_precision_surface_quadrature", quadrature),
                    ("50_80_digit_precision", precision),
                    ("high_precision_roundoff_estimate", roundoff),
                    ("root_normalization_precision", root_norm),
                    ("float_conversion", conversion),
                )
            )
            total = math.fsum(value for _, value in components)
            budget = 1e-12 * engine.scales[name] + (
                1e-9 if name == "gradient" else 1e-10
            ) * float(vector_norm(values))
            qualified = total <= 0.05 * budget
            existing[name] = {
                "values_si": converted,
                "uncertainty": OracleUncertainty(
                    total if qualified else None,
                    None,
                    total if qualified else None,
                    "estimate" if qualified else None,
                    "high precision Newton surface / independent refinement",
                    components,
                ),
            }
    return existing, {
        "fallback_digits": [50, 80],
        "fallback_orders": [16, 24, 32],
        "fallback_surface_nodes": count,
    }
