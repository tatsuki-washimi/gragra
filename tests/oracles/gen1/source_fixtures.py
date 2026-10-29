"""Public finite Temperature, SCA and nonsymmetric Grid science inputs."""


def source_fixtures():
    geometries = (
        ("sphere", {"radius_m": 1.1}),
        ("cuboid", {"lengths_m": (0.9, 1.3, 1.7)}),
        ("cylinder", {"radius_m": 0.8, "height_m": 1.1}),
    )
    for shape, parameters in geometries:
        if shape == "cuboid":
            lengths = parameters["lengths_m"]
            units = {
                "center": (0, 0, 0),
                "axis": (0.17, 0, 0),
                "interior": (0.17, -0.11, 0.13),
                "exterior": (1.3, 0.7, 0.4),
                "near_inside": (0.45, 0.13, -0.11),
                "near_outside": (0.55, 0.13, -0.11),
            }
            targets = {
                kind: tuple(
                    value * length for value, length in zip(unit, lengths, strict=True)
                )
                for kind, unit in units.items()
            }
        elif shape == "sphere":
            radius = parameters["radius_m"]
            units = {
                "center": (0, 0, 0),
                "axis": (0, 0, 0.37),
                "interior": (0.23, -0.14, 0.29),
                "exterior": (1.4, 0.2, 0.3),
                "near_inside": (0.57, 0, 0.76),
                "near_outside": (0.63, 0, 0.84),
            }
            targets = {
                kind: tuple(radius * value for value in unit)
                for kind, unit in units.items()
            }
        else:
            radius, height = parameters["radius_m"], parameters["height_m"]
            targets = {
                "center": (0, 0, 0),
                "axis": (0, 0, 0.17 * height),
                "interior": (0.21 * radius, -0.17 * radius, 0.13 * height),
                "exterior": (1.3 * radius, 0.3 * radius, 0.7 * height),
                "side_inside": (0.95 * radius, 0, 0.11 * height),
                "side_outside": (1.05 * radius, 0, 0.11 * height),
                "cap_inside": (0.23 * radius, -0.19 * radius, 0.45 * height),
                "cap_outside": (0.23 * radius, -0.19 * radius, 0.55 * height),
            }
        for family in ("finite_temperature", "sca_wall"):
            for kind, target in targets.items():
                yield {
                    "family": family,
                    "shape": shape,
                    "geometry": parameters,
                    "target_kind": kind,
                    "target_m": target,
                    "rho0_kg_m3": 1.2,
                    "T0_K": 293.0,
                    "delta_temperature_K": 2.5,
                    "density_kg_m3": 2600.0,
                    "displacement_m": (0.03, -0.01, 0.02),
                }
    for target in ((1.7, 0.4, -0.3), (-1.4, -0.6, 1.6), (0.1, 1.8, 0.7)):
        yield {
            "family": "temperature_grid",
            "target_m": target,
            "x_edges_m": (-1.0, -0.2, 0.7),
            "y_edges_m": (-0.8, 0.1, 0.9, 1.4),
            "z_edges_m": (-0.6, 0.3, 1.1),
            "delta_temperature_K": (
                ((0.7, -0.2), (0.4, 1.1), (-0.5, 0.3)),
                ((1.3, 0.2), (-0.7, 0.8), (0.6, -0.4)),
            ),
            "rho0_kg_m3": 1.2,
            "T0_K": 293.0,
        }
