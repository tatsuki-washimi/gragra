"""Discovery-following high-degree matrix; no computational oracle helpers."""

from .sphere_fixtures import SphereFixture


def sphere_high_degree_fixtures():
    for degree in (20, 30, 50, 60):
        cases = (
            (0, "axis_inside", (0.0, 0.0, 0.73)),
            (0, "axis_outside", (0.0, 0.0, 1.2)),
            (1, "generic_inside", (0.27, -0.18, 0.42)),
            (-1, "generic_outside", (0.82, -0.36, 0.84)),
            (degree, "sectoral_inside", (0.76, 0.11, 0.08)),
            (-degree, "sectoral_outside", (1.14, 0.21, -0.15)),
            (0, "center", (0.0, 0.0, 0.0)),
            (degree, "sectoral_axis_outside", (0.0, 0.0, 1.2)),
            (-1, "sine_axis_inside", (0.0, 0.0, 0.73)),
        )
        for order, kind, point in cases:
            yield SphereFixture(degree, order, 1, kind, point, 1.0, 343.0, 1.0, 0.0)
