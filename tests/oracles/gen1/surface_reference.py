"""Independent finite-surface Newton quadrature, no production imports."""

import math
from functools import cache

import mpmath as mp
import numpy as np

from .acceptance import OracleUncertainty
from .cavity_fixtures import CuboidFixture
from .sphere_reference import gauss_nodes

LD = np.longdouble


@cache
def cylinder_root(order, index, digits):
    if index == 0:
        if order:
            raise ValueError("only m=0 has the radial-constant branch")
        return mp.mpf(0)
    with mp.workdps(digits):

        def derivative(value):
            return mp.diff(lambda x: mp.besselj(order, x), value)

        left = mp.mpf("0.01")
        found = 0
        while left < (order + index + 5) * mp.pi:
            right = left + mp.pi / 12
            if derivative(left) * derivative(right) < 0:
                found += 1
                if found == index:
                    return +mp.findroot(derivative, (left, right), solver="anderson")
            left = right
    raise ArithmeticError("independent positive Bessel derivative root not bracketed")


@cache
def cylinder_radial_data(order, index, radius_float, axial, digits):
    with mp.workdps(digits):
        radius = mp.mpf(radius_float)
        root = cylinder_root(order, index, digits)
        radial_integral = mp.quad(
            lambda r: r * mp.besselj(order, root * r / radius) ** 2,
            [0, radius],
        )
        angular_integral = 2 * mp.pi if order == 0 else mp.pi
        axial_mean_square = mp.mpf(1) if axial == 0 else mp.mpf("0.5")
        normalization = mp.sqrt(
            mp.pi * radius**2 / (radial_integral * angular_integral * axial_mean_square)
        )
        return +root, +normalization


@cache
def radial_profile(order, root_string, radius_string, nodes_tuple):
    with mp.workdps(80):
        root, radius = mp.mpf(root_string), mp.mpf(radius_string)
        return np.array(
            [
                LD(str(mp.besselj(order, root * mp.mpf(node) / radius)))
                for node in nodes_tuple
            ],
            dtype=LD,
        )


def panel_nodes(lower, upper, projection, gap, order):
    """Finite panels clustered at the target projection, not an infinite plane."""
    lower, upper, projection, gap = map(LD, (lower, upper, projection, gap))
    if not gap > 0:
        raise ValueError("strictly positive finite-surface distance required")
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
        raise ArithmeticError("surface panel resource limit")
    points = sorted(points)
    nodes, weights = [], []
    reference = gauss_nodes(order, 80)
    for left, right in zip(points[:-1], points[1:], strict=True):
        midpoint, half = (left + right) / 2, (right - left) / 2
        for node, weight in reference:
            nodes.append(midpoint + half * LD(mp.nstr(node, 80)))
            weights.append(half * LD(mp.nstr(weight, 80)))
    return np.array(nodes, dtype=LD), np.array(weights, dtype=LD)


def norm_ld(values):
    return np.sqrt(np.sum(np.abs(np.asarray(values)) ** 2, dtype=LD))


class SurfaceNewton:
    def __init__(self, fixture, gravitational_constant):
        self.fixture = fixture
        self.cuboid = isinstance(fixture, CuboidFixture)
        self.target = np.array(fixture.target_m, dtype=LD)
        with mp.workdps(80):
            self.pressure = mp.mpc(fixture.pressure_real_pa, fixture.pressure_imag_pa)
            self.amplitude = self.pressure / mp.mpf(fixture.sound_speed_m_s) ** 2
            self.gravitational_constant = mp.mpf(gravitational_constant)
            self.root_precision_estimate = mp.mpf(0)
            if self.cuboid:
                self.lengths = np.array(fixture.lengths_m, dtype=LD)
                self.volume = mp.fprod(mp.mpf(v) for v in fixture.lengths_m)
                self.normalization = mp.sqrt(2) ** sum(
                    index != 0 for index in fixture.mode
                )
                self.wave_number_squared = mp.fsum(
                    (mp.pi * index / mp.mpf(length)) ** 2
                    for index, length in zip(
                        fixture.mode, fixture.lengths_m, strict=True
                    )
                )
                self.inside = bool(np.all(np.abs(self.target) < self.lengths / 2))
                self.characteristic = LD(max(fixture.lengths_m))
            else:
                self.radius, self.height = LD(fixture.radius_m), LD(fixture.height_m)
                self.root, self.normalization = cylinder_radial_data(
                    abs(fixture.order),
                    fixture.radial_index,
                    fixture.radius_m,
                    fixture.axial_index,
                    80,
                )
                low_root, low_norm = cylinder_radial_data(
                    abs(fixture.order),
                    fixture.radial_index,
                    fixture.radius_m,
                    fixture.axial_index,
                    50,
                )
                self.root_precision_estimate = (
                    abs(self.root - low_root) * (1 + abs(self.root))
                    + abs(self.normalization - low_norm) / self.normalization
                )
                self.volume = (
                    mp.pi * mp.mpf(fixture.radius_m) ** 2 * mp.mpf(fixture.height_m)
                )
                self.wave_number_squared = (self.root / mp.mpf(fixture.radius_m)) ** 2
                self.wave_number_squared += (
                    mp.pi * fixture.axial_index / mp.mpf(fixture.height_m)
                ) ** 2
                radial = np.hypot(self.target[0], self.target[1])
                self.inside = (
                    radial < self.radius and abs(self.target[2]) < self.height / 2
                )
                self.characteristic = max(self.radius, self.height)
            if self.wave_number_squared == 0:
                raise ValueError("global constant cavity mode forbidden")
            self.coefficient_mp = (
                self.gravitational_constant * self.amplitude / self.wave_number_squared
            )
            self.coefficient = np.clongdouble(
                str(self.coefficient_mp.real)
            ) + np.clongdouble(1j) * np.clongdouble(str(self.coefficient_mp.imag))
            length = self.volume ** (mp.mpf(1) / 3)
            self.scales = {
                "potential": float(
                    self.gravitational_constant * abs(self.amplitude) * length**2
                ),
                "acceleration": float(
                    self.gravitational_constant * abs(self.amplitude) * length
                ),
                "gradient": float(self.gravitational_constant * abs(self.amplitude)),
            }
        self.pi = (
            LD(str(mp.pi))
            if mp.mp.dps >= 80
            else LD("3.1415926535897932384626433832795028841971693993751")
        )
        self.minimum_gap = LD(np.inf)

    def mode_mp(self, point):
        if self.cuboid:
            return self.normalization * mp.fprod(
                mp.cos(mp.pi * index * (coordinate / mp.mpf(length) + mp.mpf("0.5")))
                for index, length, coordinate in zip(
                    self.fixture.mode, self.fixture.lengths_m, point, strict=True
                )
            )
        x, y, z = point
        radius = mp.sqrt(x * x + y * y)
        angle = mp.atan2(y, x)
        order = abs(self.fixture.order)
        angular = (
            mp.sin(order * angle) if self.fixture.order < 0 else mp.cos(order * angle)
        )
        return (
            self.normalization
            * mp.besselj(order, self.root * radius / mp.mpf(self.fixture.radius_m))
            * angular
            * mp.cos(
                mp.pi
                * self.fixture.axial_index
                * (z / mp.mpf(self.fixture.height_m) + mp.mpf("0.5"))
            )
        )

    def local_mode_derivatives(self, target):
        if not self.inside:
            return np.zeros(13, dtype=LD)
        with mp.workdps(80):
            point = tuple(mp.mpf(str(value)) for value in target)
            values = [self.mode_mp(point)]
            for axis in range(3):
                orders = tuple(int(index == axis) for index in range(3))
                values.append(
                    mp.diff(lambda x, y, z: self.mode_mp((x, y, z)), point, orders)
                )
            for i in range(3):
                for j in range(3):
                    orders = tuple(
                        int(index == i) + int(index == j) for index in range(3)
                    )
                    values.append(
                        mp.diff(lambda x, y, z: self.mode_mp((x, y, z)), point, orders)
                    )
            return np.array([LD(str(value)) for value in values], dtype=LD)

    def surfaces(self, order):
        if self.cuboid:
            for axis in range(3):
                other = [index for index in range(3) if index != axis]
                for sign in (-1, 1):
                    coordinate = sign * self.lengths[axis] / 2
                    normal_distance = abs(self.target[axis] - coordinate)
                    squared_gap = normal_distance**2 + sum(
                        max(abs(self.target[index]) - self.lengths[index] / 2, LD(0))
                        ** 2
                        for index in other
                    )
                    gap = np.sqrt(squared_gap)
                    self.minimum_gap = min(self.minimum_gap, gap)
                    u, wu = panel_nodes(
                        -self.lengths[other[0]] / 2,
                        self.lengths[other[0]] / 2,
                        self.target[other[0]],
                        gap,
                        order,
                    )
                    v, wv = panel_nodes(
                        -self.lengths[other[1]] / 2,
                        self.lengths[other[1]] / 2,
                        self.target[other[1]],
                        gap,
                        order,
                    )
                    points = np.zeros((u.size, v.size, 3), dtype=LD)
                    points[:, :, axis] = coordinate
                    points[:, :, other[0]] = u[:, None]
                    points[:, :, other[1]] = v[None, :]
                    mode = LD(mp.nstr(self.normalization, 80)) * np.ones(
                        (u.size, v.size), dtype=LD
                    )
                    for index in range(3):
                        mode *= np.cos(
                            self.pi
                            * self.fixture.mode[index]
                            * (points[:, :, index] / self.lengths[index] + LD("0.5"))
                        )
                    normal = np.zeros(3, dtype=LD)
                    normal[axis] = sign
                    yield (
                        points.reshape(-1, 3),
                        normal,
                        (wu[:, None] * wv[None, :] * mode).ravel(),
                    )
            return
        radial = np.hypot(self.target[0], self.target[1])
        angle = np.arctan2(self.target[1], self.target[0])
        order_m = abs(self.fixture.order)
        normalization = LD(mp.nstr(self.normalization, 80))
        side_gap = np.hypot(
            radial - self.radius, max(abs(self.target[2]) - self.height / 2, LD(0))
        )
        self.minimum_gap = min(self.minimum_gap, side_gap)
        phi, wphi = panel_nodes(
            angle - self.pi, angle + self.pi, angle, side_gap / self.radius, order
        )
        z, wz = panel_nodes(
            -self.height / 2, self.height / 2, self.target[2], side_gap, order
        )
        points = np.zeros((phi.size, z.size, 3), dtype=LD)
        points[:, :, 0] = self.radius * np.cos(phi)[:, None]
        points[:, :, 1] = self.radius * np.sin(phi)[:, None]
        points[:, :, 2] = z[None, :]
        normals = np.zeros_like(points)
        normals[:, :, 0] = np.cos(phi)[:, None]
        normals[:, :, 1] = np.sin(phi)[:, None]
        angular = (
            np.sin(order_m * phi) if self.fixture.order < 0 else np.cos(order_m * phi)
        )
        axial = np.cos(
            self.pi * self.fixture.axial_index * (z / self.height + LD("0.5"))
        )
        with mp.workdps(80):
            wall_radial = LD(str(mp.besselj(order_m, self.root)))
        mode = normalization * wall_radial * angular[:, None] * axial[None, :]
        weights = self.radius * wphi[:, None] * wz[None, :] * mode
        yield points.reshape(-1, 3), normals.reshape(-1, 3), weights.ravel()
        for sign in (-1, 1):
            height = sign * self.height / 2
            gap = np.hypot(self.target[2] - height, max(radial - self.radius, LD(0)))
            self.minimum_gap = min(self.minimum_gap, gap)
            r, wr = panel_nodes(0, self.radius, radial, gap, order)
            phi, wphi = panel_nodes(
                angle - self.pi,
                angle + self.pi,
                angle,
                gap / max(radial, self.radius / 8),
                order,
            )
            points = np.zeros((r.size, phi.size, 3), dtype=LD)
            points[:, :, 0] = r[:, None] * np.cos(phi)[None, :]
            points[:, :, 1] = r[:, None] * np.sin(phi)[None, :]
            points[:, :, 2] = height
            normal = np.array([0, 0, sign], dtype=LD)
            profile = radial_profile(
                order_m,
                mp.nstr(self.root, 80),
                mp.nstr(mp.mpf(self.fixture.radius_m), 80),
                tuple(str(value) for value in r),
            )
            angular = (
                np.sin(order_m * phi)
                if self.fixture.order < 0
                else np.cos(order_m * phi)
            )
            mode = (
                normalization
                * profile[:, None]
                * angular[None, :]
                * ((-1) ** self.fixture.axial_index if sign > 0 else 1)
            )
            yield (
                points.reshape(-1, 3),
                normal,
                (r[:, None] * wr[:, None] * wphi[None, :] * mode).ravel(),
            )

    def evaluate(self, order, target=None):
        target = self.target if target is None else np.asarray(target, dtype=LD)
        result = np.zeros(13, dtype=LD)
        absolute = np.zeros(13, dtype=LD)
        for points, normals, weights in self.surfaces(order):
            displacement = target - points
            squared = np.sum(displacement * displacement, axis=1)
            inverse = 1 / np.sqrt(squared)
            flux = np.sum(displacement * normals, axis=1)

            def accumulate(index, kernel, weights=weights):
                weighted = weights * kernel
                result[index] += np.sum(weighted, dtype=LD)
                absolute[index] += np.sum(np.abs(weighted), dtype=LD)

            accumulate(0, flux * inverse**3)
            normal_components = np.broadcast_to(normals, points.shape)
            for i in range(3):
                accumulate(
                    1 + i,
                    normal_components[:, i] * inverse**3
                    - 3 * flux * displacement[:, i] * inverse**5,
                )
                for j in range(3):
                    kernel = (
                        -3
                        * (
                            normal_components[:, i] * displacement[:, j]
                            + displacement[:, i] * normal_components[:, j]
                            + (flux if i == j else 0)
                        )
                        * inverse**5
                    )
                    kernel += (
                        15 * flux * displacement[:, i] * displacement[:, j] * inverse**7
                    )
                    accumulate(4 + 3 * i + j, kernel)
        local = 4 * self.pi * self.local_mode_derivatives(target)
        result += local
        absolute += np.abs(local)
        result[0] *= -1
        return self.coefficient * result, abs(self.coefficient) * absolute


def surface_reference(fixture, gravitational_constant=6.67430e-11):
    engine = SurfaceNewton(fixture, gravitational_constant)
    slices = {
        "potential": slice(0, 1),
        "acceleration": slice(1, 4),
        "gradient": slice(4, 13),
    }
    resolved = False
    for multiplier in (1, 2, 3):
        orders = [value * multiplier for value in (16, 24, 32)]
        coarse, _ = engine.evaluate(orders[0])
        middle, _ = engine.evaluate(orders[1])
        high, absolute = engine.evaluate(orders[2])
        output = {}
        resolved = True
        for name, section in slices.items():
            center_parity_zero = False
            if engine.cuboid and all(value == 0 for value in fixture.target_m):
                odd_count = sum(index % 2 for index in fixture.mode)
                center_parity_zero = (
                    (name == "potential" and odd_count > 0)
                    or (name == "acceleration" and odd_count != 1)
                    or (name == "gradient" and odd_count in (1, 3))
                )
            if center_parity_zero:
                output[name] = {
                    "values_si": tuple(0j for _ in range(section.stop - section.start)),
                    "uncertainty": OracleUncertainty(
                        0.0,
                        0.0,
                        0.0,
                        "rigorous_bound",
                        "exact centered reflection and derivative parity",
                        (("exact_parity_proof", 0.0),),
                    ),
                }
                continue
            values = high[section]
            converted = tuple(complex(value) for value in values)
            quadrature = 10 * max(
                norm_ld((high - middle)[section]), norm_ld((middle - coarse)[section])
            )
            sensitivity = 1 + engine.characteristic / engine.minimum_gap
            roundoff = 256 * np.finfo(LD).eps * sensitivity * norm_ld(absolute[section])
            root_norm = (
                10
                * LD(str(engine.root_precision_estimate))
                * norm_ld(absolute[section])
            )
            conversion = norm_ld(values - np.array(converted, dtype=np.clongdouble))
            components = tuple(
                (label, float(value))
                for label, value in (
                    ("surface_quadrature_refinement", quadrature),
                    ("longdouble_roundoff_estimate", roundoff),
                    ("root_normalization_precision", root_norm),
                    ("float_conversion", conversion),
                )
            )
            total = math.fsum(value for _, value in components)
            budget = 1e-12 * engine.scales[name] + (
                1e-9 if name == "gradient" else 1e-10
            ) * float(norm_ld(values))
            qualified = total <= 0.05 * budget
            resolved &= qualified
            uncertainty = OracleUncertainty(
                total if qualified else None,
                None,
                total if qualified else None,
                "estimate" if qualified else None,
                "direct Newton surface / kernel derivatives / quadrature refinement",
                components,
            )
            output[name] = {"values_si": converted, "uncertainty": uncertainty}
        if resolved:
            break
        if not resolved:
            # Higher spatial order cannot remove the arithmetic floor.
            arithmetic_limited = any(
                not record["uncertainty"].resolved()
                and dict(record["uncertainty"].components)[
                    "longdouble_roundoff_estimate"
                ]
                > dict(record["uncertainty"].components)[
                    "surface_quadrature_refinement"
                ]
                for record in output.values()
            )
            if arithmetic_limited:
                break
    metadata = {
        "orders": orders,
        "longdouble_epsilon": str(np.finfo(LD).eps),
        "minimum_surface_gap_m": str(engine.minimum_gap),
        "local_helmholtz_term_count": int(engine.inside),
        "scales_si": engine.scales,
        "root": mp.nstr(engine.root, 80) if not engine.cuboid else None,
        "normalization": mp.nstr(engine.normalization, 80),
    }
    if any(not record["uncertainty"].resolved() for record in output.values()):
        from .surface_high_precision import high_precision_reference

        output, fallback_metadata = high_precision_reference(
            fixture, gravitational_constant, output
        )
        metadata.update(fallback_metadata)
    return output, metadata
