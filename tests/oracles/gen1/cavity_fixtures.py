"""Public Cuboid/Cylinder data; no reference computational helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CuboidFixture:
    lengths_m: tuple[float, float, float]
    mode: tuple[int, int, int]
    target_kind: str
    target_m: tuple[float, float, float]
    sound_speed_m_s: float = 343.0
    pressure_real_pa: float = 1.3
    pressure_imag_pa: float = -0.7


@dataclass(frozen=True)
class CylinderFixture:
    radius_m: float
    height_m: float
    order: int
    radial_index: int
    axial_index: int
    target_kind: str
    target_m: tuple[float, float, float]
    sound_speed_m_s: float = 343.0
    pressure_real_pa: float = 1.3
    pressure_imag_pa: float = -0.7


def cuboid_fixtures():
    targets = {
        "center": (0, 0, 0),
        "axis": (0.2, 0, 0),
        "interior": (0.17, -0.11, 0.13),
        "exterior": (1.3, 0.7, 0.4),
        "near_inside": (0.45, 0.13, -0.11),
        "near_outside": (0.55, 0.13, -0.11),
        "face_extension": (0.5, 1.2, 1.3),
        "edge_inside": (0.45, 0.45, 0.13),
        "edge_outside": (0.55, 0.55, 0.13),
        "corner_inside": (0.45, 0.45, 0.45),
        "corner_outside": (0.55, 0.55, 0.55),
    }
    for lengths in ((1.0, 1.0, 1.0), (1.0, 2.0, 3.0), (0.1, 1.0, 10.0)):
        for mode in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (2, 1, 0)):
            for kind, unit in targets.items():
                yield CuboidFixture(
                    lengths,
                    mode,
                    kind,
                    tuple(v * length for v, length in zip(unit, lengths, strict=True)),
                )


def cylinder_fixtures():
    radius = 1.3
    for aspect in (0.1, 1.0, 10.0):
        height = radius * aspect
        targets = {
            "center": (0.0, 0.0, 0.0),
            "axis": (0.0, 0.0, 0.17 * height),
            "interior": (0.21 * radius, -0.17 * radius, 0.13 * height),
            "exterior": (1.3 * radius, 0.3 * radius, 0.7 * height),
            "side_inside": (0.95 * radius, 0.0, 0.11 * height),
            "side_outside": (1.05 * radius, 0.0, 0.11 * height),
            "cap_inside": (0.23 * radius, -0.19 * radius, 0.45 * height),
            "cap_outside": (0.23 * radius, -0.19 * radius, 0.55 * height),
            "side_extension": (radius, 0.0, 0.7 * height),
            "cap_extension": (1.3 * radius, 0.0, height / 2),
            "rim_inside": (0.95 * radius, 0.0, 0.45 * height),
            "rim_outside": (1.05 * radius, 0.0, 0.55 * height),
        }
        modes = [(order, 1, axial) for order in range(-3, 4) for axial in (0, 1)]
        modes += [(0, 0, axial) for axial in (1, 2)]
        for order, radial, axial in modes:
            for kind, target in targets.items():
                yield CylinderFixture(
                    radius, height, order, radial, axial, kind, target
                )
