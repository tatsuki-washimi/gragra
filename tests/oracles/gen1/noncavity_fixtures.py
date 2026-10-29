"""Public deterministic noncavity inputs; contains no reference math helper."""

import math

G_SI = 6.67430e-11
RHO0 = 1.2
T0 = 293.0
DELTA_T = 2.5


def noncavity_fixtures():
    for kind, target in {
        "center": (0.0, 0.0, 0.0),
        "near_center": (1.7e-6, 0.0, 0.0),
        "axis": (0.0, 0.0, 0.51),
        "interior": (0.391, -0.238, 0.493),
        "width": (1.7, 0.0, 0.0),
        "outer": (3.06, -4.08, 0.0),
        "far": (20.4, -27.2, 0.0),
    }.items():
        yield {
            "family": "gaussian",
            "target_kind": kind,
            "target_m": target,
            "sigma_m": 1.7,
            "rho0_kg_m3": RHO0,
            "T0_K": T0,
            "delta_temperature_K": DELTA_T,
        }
    for incidence in (0.31, 0.83, math.pi / 2):
        for height in (-1.3, -0.2, 0.2, 1.3):
            yield {
                "family": "halfspace",
                "target_m": (0.23, -0.17, height),
                "incidence_rad": incidence,
                "azimuth_rad": 0.37,
                "frequency_hz": 17.0,
                "sound_speed_m_s": 343.0,
                "pressure_pa": (1.3, -0.7),
                "phase_origin_m": (0.0, 0.0, 0.0),
            }
    for target in ((1.2, 0.3, -0.4), (2.0, 0.0, 0.0), (0.0, 1.3, 0.2)):
        for elapsed in (0.0, 0.1, 0.5, 0.9, 1.0, 1.4, 2.0, 2.3, 3.0):
            yield {
                "family": "point_force",
                "target_m": target,
                "elapsed_s": elapsed,
                "vp_m_s": 2.3,
                "direction": (1.0, 0.0, 0.0),
                "knots_s": (0.0, 1.0, 2.0),
                "coefficients_N": ((0.0, 0.0, 3.0, -2.0), (1.0, 0.0, -3.0, 2.0)),
            }
    for family in ("constant_speed", "rotation", "oscillation"):
        for target in ((1.7, 0.2, 0.4), (-1.1, 0.6, -0.3)):
            for elapsed in (0.0, 0.4, 1.3):
                yield {
                    "family": family,
                    "target_m": target,
                    "elapsed_s": elapsed,
                    "masses_kg": (2.0, 3.0),
                    "positions_m": ((-0.4, 0.1, 0.2), (0.7, -0.2, 0.3)),
                    "velocity_m_s": (0.1, -0.07, 0.03),
                    "rotation_axis": (1.0, 2.0, -1.0),
                    "angular_speed_rad_s": 0.7,
                    "displacement_m": ((0.03, -0.02), (0.01, 0.01), (-0.02, 0.005)),
                }
    for target in ((0.0, 0.0, 0.0), (0.7, -0.3, 0.4), (2.1, 0.5, -0.8)):
        yield {
            "family": "temperature_plane",
            "target_m": target,
            "wavevector_m_inv": (0.3, -0.2, 0.4),
            "phase_origin_m": (0.2, -0.1, 0.3),
            "rho0_kg_m3": RHO0,
            "T0_K": T0,
            "delta_temperature_K": DELTA_T,
        }
