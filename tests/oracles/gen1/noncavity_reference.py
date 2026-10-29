"""Independent potential integrals and controlled target differentiation."""

import math

import mpmath as mp

from .acceptance import OracleUncertainty
from .noncavity_fixtures import G_SI
from .sphere_reference import differences, gauss_nodes, vector_norm


class PotentialOracle:
    def __init__(self, fixture, digits, order):
        self.fixture, self.digits, self.order = fixture, digits, order
        self.family = fixture["family"]
        self.cache = {}
        self.tail_potential = mp.mpf(0)
        with mp.workdps(digits):
            self.g = mp.mpf(G_SI)
            if self.family == "gaussian":
                self.length = mp.mpf(fixture["sigma_m"])
                self.density = (
                    -mp.mpf(fixture["rho0_kg_m3"])
                    * mp.mpf(fixture["delta_temperature_K"])
                    / mp.mpf(fixture["T0_K"])
                )
                self.cutoff = 32 * self.length
                self.radial_cells = []
                for index in range(32):
                    lower, upper = index * self.length, (index + 1) * self.length
                    self.radial_cells.append(
                        (
                            self.integrate(
                                lambda s: s**2 * mp.exp(-(s**2) / (2 * self.length**2)),
                                lower,
                                upper,
                            ),
                            self.integrate(
                                lambda s: s * mp.exp(-(s**2) / (2 * self.length**2)),
                                lower,
                                upper,
                            ),
                        )
                    )
                self.tail_potential = (
                    4
                    * mp.pi
                    * self.g
                    * abs(self.density)
                    * self.length**2
                    * mp.exp(-512)
                )
            elif self.family == "halfspace":
                self.k = (
                    2
                    * mp.pi
                    * mp.mpf(fixture["frequency_hz"])
                    / mp.mpf(fixture["sound_speed_m_s"])
                )
                self.q = self.k * mp.sin(mp.mpf(fixture["incidence_rad"]))
                self.p = self.k * mp.cos(mp.mpf(fixture["incidence_rad"]))
                self.length = 1 / self.k
                self.density = (
                    mp.mpc(*fixture["pressure_pa"])
                    / mp.mpf(fixture["sound_speed_m_s"]) ** 2
                )
                self.step = 1 / max(self.q, abs(self.p))
                self.cutoff = max(mp.mpf(fixture["target_m"][2]), 0) + 80 / self.q
                self.halfspace_cells = []
                lower = mp.mpf(0)
                while lower < self.cutoff:
                    upper = min(lower + self.step, self.cutoff)
                    self.halfspace_cells.append(
                        (
                            lower,
                            upper,
                            self.integrate(
                                lambda s: mp.exp(self.q * s) * mp.cos(self.p * s),
                                lower,
                                upper,
                            ),
                            self.integrate(
                                lambda s: mp.exp(-self.q * s) * mp.cos(self.p * s),
                                lower,
                                upper,
                            ),
                        )
                    )
                    lower = upper
                self.tail_potential = (
                    2 * mp.pi * self.g * abs(self.density) / self.q**2 * mp.exp(-79)
                )
            elif self.family == "point_force":
                self.length = mp.mpf(fixture["vp_m_s"])
                self.density = 1 / self.length**4
            elif self.family == "temperature_plane":
                self.wavevector = tuple(
                    mp.mpf(value) for value in fixture["wavevector_m_inv"]
                )
                self.wave_squared = mp.fsum(value * value for value in self.wavevector)
                self.length = 1 / mp.sqrt(self.wave_squared)
                self.density = (
                    -mp.mpf(fixture["rho0_kg_m3"])
                    * mp.mpf(fixture["delta_temperature_K"])
                    / mp.mpf(fixture["T0_K"])
                )
            else:
                self.length = mp.mpf(1)
                self.density = mp.fsum(abs(value) for value in fixture["masses_kg"])
                if self.family == "oscillation":
                    self.density *= mp.sqrt(
                        mp.fsum(
                            abs(mp.mpc(*value)) ** 2
                            for value in fixture["displacement_m"]
                        )
                    )
            self.scales = {
                "potential": self.g * abs(self.density) * self.length**2,
                "acceleration": self.g * abs(self.density) * self.length,
                "gradient": self.g * abs(self.density),
            }

    def integrate(self, function, lower, upper):
        if lower == upper:
            return mp.mpf(0)
        half, middle = (upper - lower) / 2, (lower + upper) / 2
        return half * mp.fsum(
            weight * function(middle + half * node)
            for node, weight in gauss_nodes(self.order, self.digits)
        )

    def force(self, elapsed):
        if elapsed < 0 or elapsed >= 2:
            return mp.mpf(0)
        if elapsed <= 1:
            return 3 * elapsed**2 - 2 * elapsed**3
        local = elapsed - 1
        return 1 - 3 * local**2 + 2 * local**3

    def alternate_potential(self, point):
        """Tanh-sinh integral crosscheck of Gaussian/Fourier/history potentials."""
        if self.family == "gaussian":
            radius = vector_norm(point)
            inner = (
                mp.quad(
                    lambda s: s**2 * mp.exp(-(s**2) / (2 * self.length**2)), [0, radius]
                )
                if radius
                else 0
            )
            outer = mp.quad(
                lambda s: s * mp.exp(-(s**2) / (2 * self.length**2)),
                [radius, radius + self.length, radius + 4 * self.length, self.cutoff],
            )
            return (
                -4
                * mp.pi
                * self.g
                * self.density
                * (outer + (inner / radius if radius else 0))
            )
        if self.family == "halfspace":
            x, y, z = point
            split = max(z, 0)
            endpoints = sorted(
                {
                    mp.mpf(0),
                    split,
                    split + 1 / self.q,
                    split + 5 / self.q,
                    split + 20 / self.q,
                    self.cutoff,
                }
            )
            integral = mp.quad(
                lambda s: mp.exp(-self.q * abs(z - s)) * mp.cos(self.p * s), endpoints
            )
            azimuth = mp.mpf(self.fixture["azimuth_rad"])
            phase = self.q * (mp.cos(azimuth) * x + mp.sin(azimuth) * y)
            return (
                -2
                * mp.pi
                * self.g
                * self.density
                / self.q
                * mp.exp(mp.j * phase)
                * integral
            )
        if self.family == "point_force":
            radius = vector_norm(point)
            elapsed = mp.mpf(self.fixture["elapsed_s"])
            delay = radius / mp.mpf(self.fixture["vp_m_s"])
            endpoints = sorted(
                {
                    mp.mpf(0),
                    delay,
                    *(
                        elapsed - knot
                        for knot in (0, 1, 2)
                        if 0 < elapsed - knot < delay
                    ),
                }
            )
            return (
                -self.g
                * point[0]
                * mp.quad(lambda tau: tau * self.force(elapsed - tau), endpoints)
                / radius**3
            )
        return self.potential(point)

    def potential(self, point):
        if self.family == "gaussian":
            radius = vector_norm(point)
            if radius in self.cache:
                return self.cache[radius]
            cell = int(mp.floor(radius / self.length))
            if cell >= 32:
                raise ValueError("Gaussian oracle stencil exceeds frozen cutoff")
            inner = mp.fsum(value[0] for value in self.radial_cells[:cell])
            inner += self.integrate(
                lambda s: s**2 * mp.exp(-(s**2) / (2 * self.length**2)),
                cell * self.length,
                radius,
            )
            outer = mp.fsum(value[1] for value in self.radial_cells[cell + 1 :])
            outer += self.integrate(
                lambda s: s * mp.exp(-(s**2) / (2 * self.length**2)),
                radius,
                (cell + 1) * self.length,
            )
            result = (
                -4
                * mp.pi
                * self.g
                * self.density
                * (outer + (inner / radius if radius else 0))
            )
            self.cache[radius] = result
            return result
        if self.family == "halfspace":
            x, y, z = point
            if z in self.cache:
                integral = self.cache[z]
            elif z < 0:
                integral = mp.exp(self.q * z) * mp.fsum(
                    cell[3] for cell in self.halfspace_cells
                )
                self.cache[z] = integral
            else:
                left, right = mp.mpf(0), mp.mpf(0)
                for lower, upper, positive, negative in self.halfspace_cells:
                    if upper <= z:
                        left += positive
                    elif lower >= z:
                        right += negative
                    else:
                        left += self.integrate(
                            lambda s: mp.exp(self.q * s) * mp.cos(self.p * s), lower, z
                        )
                        right += self.integrate(
                            lambda s: mp.exp(-self.q * s) * mp.cos(self.p * s), z, upper
                        )
                integral = mp.exp(-self.q * z) * left + mp.exp(self.q * z) * right
                self.cache[z] = integral
            azimuth = mp.mpf(self.fixture["azimuth_rad"])
            phase = self.q * (mp.cos(azimuth) * x + mp.sin(azimuth) * y)
            return (
                -2
                * mp.pi
                * self.g
                * self.density
                / self.q
                * mp.exp(mp.j * phase)
                * integral
            )
        if self.family == "point_force":
            radius = vector_norm(point)
            if radius == 0:
                raise ValueError("point source rejected")
            elapsed = mp.mpf(self.fixture["elapsed_s"])
            delay = radius / mp.mpf(self.fixture["vp_m_s"])
            endpoints = sorted(
                {
                    mp.mpf(0),
                    delay,
                    *(
                        elapsed - knot
                        for knot in (0, 1, 2)
                        if 0 < elapsed - knot < delay
                    ),
                }
            )
            moment = mp.fsum(
                self.integrate(
                    lambda tau: tau * self.force(elapsed - tau), lower, upper
                )
                for lower, upper in zip(endpoints[:-1], endpoints[1:], strict=True)
            )
            return -self.g * point[0] * moment / radius**3
        if self.family == "temperature_plane":
            phase = mp.fsum(
                k * (x - mp.mpf(origin))
                for k, x, origin in zip(
                    self.wavevector, point, self.fixture["phase_origin_m"], strict=True
                )
            )
            return (
                -4
                * mp.pi
                * self.g
                * self.density
                * mp.exp(mp.j * phase)
                / self.wave_squared
            )
        positions = [
            tuple(mp.mpf(value) for value in position)
            for position in self.fixture["positions_m"]
        ]
        elapsed = mp.mpf(self.fixture["elapsed_s"])
        if self.family == "constant_speed":
            positions = [
                tuple(
                    value + elapsed * mp.mpf(velocity)
                    for value, velocity in zip(
                        position, self.fixture["velocity_m_s"], strict=True
                    )
                )
                for position in positions
            ]
        elif self.family == "rotation":
            axis = tuple(mp.mpf(value) for value in self.fixture["rotation_axis"])
            norm = vector_norm(axis)
            angle = elapsed * mp.mpf(self.fixture["angular_speed_rad_s"])
            scalar = mp.cos(angle / 2)
            vector = tuple(value / norm * mp.sin(angle / 2) for value in axis)
            # Quaternion product q*(0,p)*conj(q), independently implemented.
            transformed = []
            for position in positions:
                dot = mp.fsum(a * b for a, b in zip(vector, position, strict=True))
                cross = (
                    vector[1] * position[2] - vector[2] * position[1],
                    vector[2] * position[0] - vector[0] * position[2],
                    vector[0] * position[1] - vector[1] * position[0],
                )
                transformed.append(
                    tuple(
                        (scalar**2 - mp.fsum(v * v for v in vector)) * position[i]
                        + 2 * dot * vector[i]
                        + 2 * scalar * cross[i]
                        for i in range(3)
                    )
                )
            positions = transformed
        if self.family == "oscillation":
            displacement = tuple(
                mp.mpc(*value) for value in self.fixture["displacement_m"]
            )

            def displaced_potential(parameter):
                return -self.g * mp.fsum(
                    mp.mpf(mass)
                    / mp.sqrt(
                        mp.fsum(
                            (point[i] - position[i] - parameter * displacement[i]) ** 2
                            for i in range(3)
                        )
                    )
                    for mass, position in zip(
                        self.fixture["masses_kg"], positions, strict=True
                    )
                )

            return mp.diff(displaced_potential, mp.mpf(0))
        return -self.g * mp.fsum(
            mp.mpf(mass)
            / vector_norm(tuple(x - s for x, s in zip(point, position, strict=True)))
            for mass, position in zip(self.fixture["masses_kg"], positions, strict=True)
        )


def observable_values(oracle, point, step):
    first = tuple(mp.mpf(value) / 60 for value in (-1, 9, -45, 0, 45, -9, 1))
    second = tuple(mp.mpf(value) / 180 for value in (2, -27, 270, -490, 270, -27, 2))
    axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    def directional(direction):
        values = [
            oracle.potential(
                tuple(point[i] + offset * step * direction[i] for i in range(3))
            )
            for offset in range(-3, 4)
        ]
        return mp.fsum(
            w * value for w, value in zip(first, values, strict=True)
        ) / step, mp.fsum(
            w * value for w, value in zip(second, values, strict=True)
        ) / step**2

    axis = [directional(direction) for direction in axes]
    hessian = [[mp.mpc(0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        hessian[i][i] = axis[i][1]
        for j in range(i):
            hessian[i][j] = hessian[j][i] = (
                directional(tuple(axes[i][k] + axes[j][k] for k in range(3)))[1]
                - axis[i][1]
                - axis[j][1]
            ) / 2
    return {
        "potential": (oracle.potential(point),),
        "acceleration": tuple(-value[0] for value in axis),
        "gradient": tuple(-value for row in hessian for value in row),
    }


def noncavity_reference(fixture):
    high = PotentialOracle(fixture, 80, 32)
    middle = PotentialOracle(fixture, 80, 24)
    coarse = PotentialOracle(fixture, 80, 16)
    low = PotentialOracle(fixture, 50, 32)
    with mp.workdps(80):
        point = tuple(mp.mpf(value) for value in fixture["target_m"])
        step = high.length / 50
        if fixture["family"] == "point_force":
            radius = vector_norm(point)
            gaps = [
                abs(
                    radius
                    - mp.mpf(fixture["vp_m_s"]) * (mp.mpf(fixture["elapsed_s"]) - knot)
                )
                for knot in (0, 1, 2)
            ]
            step = min(step, radius / 12, min(gap for gap in gaps if gap > 0) / 12)
        if fixture["family"] == "halfspace":
            step = min(step, abs(point[2]) / 8)
        previous = prior = None
        resolved = False
        for refinement in range(13):
            current = observable_values(high, point, step)
            change = differences(current, previous) if previous else None
            if change and prior:
                derivative = {
                    name: 10 * max(change[name], prior[name]) for name in current
                }
                budgets = {
                    name: mp.mpf("1e-12") * high.scales[name]
                    + mp.mpf("1e-9" if name == "gradient" else "1e-10")
                    * vector_norm(values)
                    for name, values in current.items()
                }
                if all(derivative[name] <= budgets[name] / 40 for name in current):
                    resolved = True
                    break
            previous, prior = current, change
            if refinement < 12:
                step /= 2
        mid, old = (
            observable_values(middle, point, step),
            observable_values(coarse, point, step),
        )
        final_change, prior_change = differences(current, mid), differences(mid, old)
        with mp.workdps(50):
            low_values = observable_values(low, point, step)
        precision = differences(current, low_values)
        alternate_potential_difference = abs(
            high.alternate_potential(point) - current["potential"][0]
        )
        output = {}
        for name, values in current.items():
            converted = tuple(complex(value) for value in values)
            conversion = vector_norm(
                tuple(
                    value - mp.mpc(actual)
                    for value, actual in zip(values, converted, strict=True)
                )
            )
            tail = (
                high.tail_potential
                * (
                    1
                    if name == "potential"
                    else mp.sqrt(2) * high.k
                    if name == "acceleration"
                    else 2 * high.k**2
                )
                if fixture["family"] == "halfspace"
                else high.tail_potential
                if name == "potential"
                else 0
            )
            components = tuple(
                (label, float(value))
                for label, value in (
                    ("derivative_refinement", derivative[name]),
                    (
                        "quadrature_refinement",
                        10 * max(final_change[name], prior_change[name]),
                    ),
                    ("precision_propagation", 10 * precision[name]),
                    (
                        "alternate_potential_quadrature",
                        10 * alternate_potential_difference
                        if name == "potential"
                        else 0,
                    ),
                    ("explicit_truncation_tail", tail),
                    ("float_conversion", conversion),
                )
            )
            total = math.fsum(value for _, value in components)
            qualified = resolved and total <= float(budgets[name] / 20)
            output[name] = {
                "values_si": converted,
                "uncertainty": OracleUncertainty(
                    total if qualified else None,
                    None,
                    total if qualified else None,
                    "estimate" if qualified else None,
                    "independent potential / seven-point target derivatives",
                    components,
                ),
            }
        return output, {
            "orders": [16, 24, 32],
            "digits": [50, 80],
            "derivative_refinement": refinement,
            "step_m": str(step),
            "scales_si": {name: float(value) for name, value in high.scales.items()},
            "explicit_tail_method": "Gaussian shell / positive-q exponential"
            if high.tail_potential
            else "not applicable",
        }
