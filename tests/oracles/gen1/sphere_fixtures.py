"""Public deterministic Sphere cases; contains no oracle computational helper."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SphereFixture:
    degree: int
    order: int
    radial_index: int
    target_kind: str
    target_m: tuple[float, float, float]
    radius_m: float = 1.7
    sound_speed_m_s: float = 343.0
    pressure_real_pa: float = 1.3
    pressure_imag_pa: float = -0.7


def sphere_fixtures(radius=1.7):
    direction = (2 / 3, -1 / 3, 2 / 3)
    assert math.isclose(sum(v * v for v in direction), 1.0)
    targets = {
        "center": (0.0, 0.0, 0.0),
        "axis": (0.0, 0.0, 0.37),
        "interior": (0.23, -0.14, 0.29),
        "exterior": (1.4, 0.2, 0.3),
        "near_inside": tuple(0.999 * v for v in direction),
        "near_outside": tuple(1.001 * v for v in direction),
    }
    for degree in range(4):
        for order in range(-degree, degree + 1):
            for radial_index in range(1, 4):
                for kind, unit in targets.items():
                    yield SphereFixture(
                        degree,
                        order,
                        radial_index,
                        kind,
                        tuple(radius * v for v in unit),
                        radius,
                    )
