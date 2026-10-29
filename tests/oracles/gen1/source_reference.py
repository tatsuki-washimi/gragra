"""Independent radial/surface/signed-center source references."""

import math

import mpmath as mp
import numpy as np

from .acceptance import OracleUncertainty
from .cavity_fixtures import CuboidFixture, CylinderFixture
from .noncavity_fixtures import G_SI
from .noncavity_reference import observable_values
from .sphere_reference import differences, gauss_nodes, vector_norm
from .surface_reference import LD, SurfaceNewton, norm_ld


def physical_scales(fixture):
    with mp.workdps(80):
        family = fixture["family"]
        if family == "temperature_grid":
            edges = [fixture[name] for name in ("x_edges_m", "y_edges_m", "z_edges_m")]
            total_mass = mp.mpf(0)
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        volume = mp.fprod(
                            mp.mpf(edge[index + 1]) - mp.mpf(edge[index])
                            for edge, index in zip(edges, (i, j, k), strict=True)
                        )
                        total_mass += abs(
                            -mp.mpf(fixture["rho0_kg_m3"])
                            / mp.mpf(fixture["T0_K"])
                            * mp.mpf(fixture["delta_temperature_K"][i][j][k])
                            * volume
                        )
            return {
                name: float(mp.mpf(G_SI) * total_mass)
                for name in ("potential", "acceleration", "gradient")
            }, mp.mpf(1)
        geometry = fixture["geometry"]
        if fixture["shape"] == "sphere":
            volume = 4 * mp.pi * mp.mpf(geometry["radius_m"]) ** 3 / 3
        elif fixture["shape"] == "cuboid":
            volume = mp.fprod(mp.mpf(value) for value in geometry["lengths_m"])
        else:
            volume = (
                mp.pi * mp.mpf(geometry["radius_m"]) ** 2 * mp.mpf(geometry["height_m"])
            )
        length = volume ** (mp.mpf(1) / 3)
        if family == "sca_wall":
            amplitude = (
                mp.mpf(G_SI)
                * mp.mpf(fixture["density_kg_m3"])
                * vector_norm(
                    tuple(mp.mpf(value) for value in fixture["displacement_m"])
                )
            )
            scales = (amplitude * length, amplitude, amplitude / length)
        else:
            amplitude = mp.mpf(G_SI) * abs(
                mp.mpf(fixture["rho0_kg_m3"])
                * mp.mpf(fixture["delta_temperature_K"])
                / mp.mpf(fixture["T0_K"])
            )
            scales = (amplitude * length**2, amplitude * length, amplitude)
        return dict(
            zip(
                ("potential", "acceleration", "gradient"),
                map(float, scales),
                strict=True,
            )
        ), length


class RadialOrGrid:
    def __init__(self, fixture, digits, order):
        self.fixture, self.digits, self.order = fixture, digits, order
        self.family = fixture["family"]
        self.cache = {}
        with mp.workdps(digits):
            self.g = mp.mpf(G_SI)
            if self.family == "temperature_grid":
                self.cells = []
                edges = [
                    fixture[name] for name in ("x_edges_m", "y_edges_m", "z_edges_m")
                ]
                for i in range(2):
                    for j in range(3):
                        for k in range(2):
                            center = tuple(
                                (mp.mpf(edge[index]) + mp.mpf(edge[index + 1])) / 2
                                for edge, index in zip(edges, (i, j, k), strict=True)
                            )
                            volume = mp.fprod(
                                mp.mpf(edge[index + 1]) - mp.mpf(edge[index])
                                for edge, index in zip(edges, (i, j, k), strict=True)
                            )
                            mass = (
                                -mp.mpf(fixture["rho0_kg_m3"])
                                / mp.mpf(fixture["T0_K"])
                                * mp.mpf(fixture["delta_temperature_K"][i][j][k])
                                * volume
                            )
                            self.cells.append((center, mass))
            else:
                self.radius = mp.mpf(fixture["geometry"]["radius_m"])
                self.density = (
                    mp.mpf(fixture["density_kg_m3"])
                    if self.family == "sca_wall"
                    else -mp.mpf(fixture["rho0_kg_m3"])
                    * mp.mpf(fixture["delta_temperature_K"])
                    / mp.mpf(fixture["T0_K"])
                )
                self.displacement = tuple(
                    mp.mpf(value) for value in fixture["displacement_m"]
                )

    def integrate(self, function, lower, upper):
        half, center = (upper - lower) / 2, (upper + lower) / 2
        return half * mp.fsum(
            weight * function(center + half * node)
            for node, weight in gauss_nodes(self.order, self.digits)
        )

    def potential(self, point):
        if self.family == "temperature_grid":
            return -self.g * mp.fsum(
                mass
                / vector_norm(tuple(x - s for x, s in zip(point, center, strict=True)))
                for center, mass in self.cells
            )
        radius = vector_norm(point)
        if self.family == "finite_temperature":
            upper = min(radius, self.radius)
            mass_moment = self.integrate(lambda s: s**2, mp.mpf(0), upper)
            shell_moment = self.integrate(lambda s: s, upper, self.radius)
            return (
                -4
                * mp.pi
                * self.g
                * self.density
                * (shell_moment + (mass_moment / radius if radius else 0))
            )
        if radius == 0:
            return mp.mpf(0)
        if radius not in self.cache:
            # Cluster near mu=1 where the nearest shell point resides.
            from .surface_high_precision import panel_nodes_mp

            gap = (self.radius - radius) ** 2 / (2 * self.radius * radius)
            nodes = panel_nodes_mp(mp.mpf(-1), mp.mpf(1), mp.mpf(1), gap, self.order)
            integral = mp.fsum(
                weight
                * cosine
                / mp.sqrt(
                    self.radius**2 + radius**2 - 2 * self.radius * radius * cosine
                )
                for cosine, weight in nodes
            )
            self.cache[radius] = 2 * mp.pi * self.radius**2 * integral
        return (
            self.g
            * self.density
            * mp.fsum(u * x for u, x in zip(self.displacement, point, strict=True))
            / radius
            * self.cache[radius]
        )


class UniformGeometry(SurfaceNewton):
    """Geometry quadrature only; zero Fourier weight is not an acoustic model."""

    def __init__(self, fixture):
        self.target = np.array(fixture["target_m"], dtype=LD)
        self.normalization = mp.mpf(1)
        self.root = mp.mpf(0)
        self.pi = LD("3.1415926535897932384626433832795028841971693993751")
        self.minimum_gap = LD(np.inf)
        self.cuboid = fixture["shape"] == "cuboid"
        if self.cuboid:
            self.lengths = np.array(fixture["geometry"]["lengths_m"], dtype=LD)
            self.fixture = CuboidFixture(
                tuple(fixture["geometry"]["lengths_m"]),
                (0, 0, 0),
                fixture["target_kind"],
                tuple(fixture["target_m"]),
            )
            self.characteristic = max(self.lengths)
        else:
            self.radius = LD(fixture["geometry"]["radius_m"])
            self.height = LD(fixture["geometry"]["height_m"])
            self.fixture = CylinderFixture(
                float(self.radius),
                float(self.height),
                0,
                0,
                0,
                fixture["target_kind"],
                tuple(fixture["target_m"]),
            )
            self.characteristic = max(self.radius, self.height)


def surface_values(fixture, order):
    geometry = UniformGeometry(fixture)
    result = np.zeros(13, dtype=LD)
    absolute = np.zeros(13, dtype=LD)
    displacement_amplitude = np.array(fixture["displacement_m"], dtype=LD)
    for source, normal, weights in geometry.surfaces(order):
        delta = geometry.target - source
        inverse = 1 / np.sqrt(np.sum(delta * delta, axis=1))
        normals = np.broadcast_to(normal, source.shape)
        flux = np.sum(delta * normals, axis=1)
        if fixture["family"] == "sca_wall":
            weights = weights * np.sum(normals * displacement_amplitude, axis=1)
            kernels = [inverse]
            kernels += [-delta[:, i] * inverse**3 for i in range(3)]
            kernels += [
                3 * delta[:, i] * delta[:, j] * inverse**5
                - (inverse**3 if i == j else 0)
                for i in range(3)
                for j in range(3)
            ]
        else:
            kernels = [flux * inverse]
            kernels += [
                normals[:, i] * inverse - flux * delta[:, i] * inverse**3
                for i in range(3)
            ]
            kernels += [
                -(
                    normals[:, i] * delta[:, j]
                    + normals[:, j] * delta[:, i]
                    + (flux if i == j else 0)
                )
                * inverse**3
                + 3 * flux * delta[:, i] * delta[:, j] * inverse**5
                for i in range(3)
                for j in range(3)
            ]
        for index, kernel in enumerate(kernels):
            weighted = weights * kernel
            result[index] += np.sum(weighted, dtype=LD)
            absolute[index] += np.sum(np.abs(weighted), dtype=LD)
    result[1:] *= -1
    with mp.workdps(80):
        coefficient = mp.mpf(G_SI) * (
            mp.mpf(fixture["density_kg_m3"])
            if fixture["family"] == "sca_wall"
            else -mp.mpf(fixture["rho0_kg_m3"])
            * mp.mpf(fixture["delta_temperature_K"])
            / mp.mpf(fixture["T0_K"])
            / 2
        )
        coefficient = LD(str(coefficient))
    return coefficient * result, abs(coefficient) * absolute, geometry


def source_reference(fixture):
    scales, length = physical_scales(fixture)
    sections = {
        "potential": slice(0, 1),
        "acceleration": slice(1, 4),
        "gradient": slice(4, 13),
    }
    output = {}
    if fixture["family"] == "temperature_grid" or fixture["shape"] == "sphere":
        high, middle, coarse = (
            RadialOrGrid(fixture, 80, order) for order in (32, 24, 16)
        )
        low = RadialOrGrid(fixture, 50, 32)
        with mp.workdps(80):
            point = tuple(mp.mpf(value) for value in fixture["target_m"])
            step = length / 50
            if fixture["family"] != "temperature_grid":
                step = min(step, abs(vector_norm(point) - high.radius) / 8)
            previous = prior = None
            for refinement in range(13):
                current = observable_values(high, point, step)
                change = differences(current, previous) if previous else None
                if change and prior:
                    derivative = {
                        name: 10 * max(change[name], prior[name]) for name in current
                    }
                    budgets = {
                        name: mp.mpf("1e-12") * scales[name]
                        + mp.mpf("1e-9" if name == "gradient" else "1e-10")
                        * vector_norm(values)
                        for name, values in current.items()
                    }
                    if all(derivative[name] <= budgets[name] / 40 for name in current):
                        break
                previous, prior = current, change
                if refinement < 12:
                    step /= 2
            mid, old = (
                observable_values(middle, point, step),
                observable_values(coarse, point, step),
            )
            q1, q0 = differences(current, mid), differences(mid, old)
            with mp.workdps(50):
                lower = observable_values(low, point, step)
            precision = differences(current, lower)
            for name, values in current.items():
                converted = tuple(complex(value) for value in values)
                rounding = vector_norm(
                    tuple(
                        value - mp.mpc(actual)
                        for value, actual in zip(values, converted, strict=True)
                    )
                )
                components = (
                    ("derivative_refinement", float(derivative[name])),
                    ("quadrature_refinement", float(10 * max(q1[name], q0[name]))),
                    ("precision_propagation", float(10 * precision[name])),
                    ("float_conversion", float(rounding)),
                )
                total = math.fsum(value for _, value in components)
                qualified = total <= float(budgets[name] / 20)
                output[name] = {
                    "values_si": converted,
                    "uncertainty": OracleUncertainty(
                        total if qualified else None,
                        None,
                        total if qualified else None,
                        "estimate" if qualified else None,
                        "Newton potential / controlled target derivatives",
                        components,
                    ),
                }
    else:
        coarse, _, _ = surface_values(fixture, 16)
        middle, _, _ = surface_values(fixture, 24)
        high, absolute, geometry = surface_values(fixture, 32)
        for name, section in sections.items():
            values = high[section]
            converted = tuple(complex(value) for value in values)
            q = 10 * max(
                norm_ld((high - middle)[section]), norm_ld((middle - coarse)[section])
            )
            rounding = (
                256
                * np.finfo(LD).eps
                * (1 + geometry.characteristic / geometry.minimum_gap)
                * norm_ld(absolute[section])
            )
            conversion = norm_ld(values - np.array(converted, dtype=np.clongdouble))
            components = tuple(
                (label, float(value))
                for label, value in (
                    ("surface_quadrature_refinement", q),
                    ("longdouble_roundoff_estimate", rounding),
                    ("float_conversion", conversion),
                )
            )
            total = math.fsum(value for _, value in components)
            budget = 1e-12 * scales[name] + (
                1e-9 if name == "gradient" else 1e-10
            ) * float(norm_ld(values))
            qualified = total <= 0.05 * budget
            output[name] = {
                "values_si": converted,
                "uncertainty": OracleUncertainty(
                    total if qualified else None,
                    None,
                    total if qualified else None,
                    "estimate" if qualified else None,
                    "uniform Newton surface / independently differentiated kernels",
                    components,
                ),
            }
    if fixture.get("target_kind") == "center":
        zeros = (
            ("acceleration",)
            if fixture["family"] == "finite_temperature"
            else ("potential", "gradient")
        )
        for name in zeros:
            count = sections[name].stop - sections[name].start
            output[name] = {
                "values_si": tuple(0j for _ in range(count)),
                "uncertainty": OracleUncertainty(
                    0.0,
                    0.0,
                    0.0,
                    "rigorous_bound",
                    "centered inversion parity, all observable components",
                    (("exact_inversion_parity", 0.0),),
                ),
            }
    return output, {
        "scales_si": scales,
        "orders": [16, 24, 32],
        "potential_route": "radial/angular/explicit_grid"
        if fixture["family"] == "temperature_grid" or fixture["shape"] == "sphere"
        else "uniform_divergence_or_sheet_surface",
        "contact_term_added": False,
    }
