"""Print target-wise C errors and independent reference convergence.

Run with the worktree src and environment site-packages on PYTHONPATH,
using python -S tests/oracles/cylinder_validation_report.py.
"""

import numpy as np
from cylinder_surface_reference import surface_reference

from gragra.analytic import cylinder_acceleration_gradient_g_free


def main():
    cases = [
        ("required-interior", 2.0, 3.0, [0.2, 0.3, 0.1]),
        ("required-near-axis", 2.0, 3.0, [1e-5, -2e-5, 0.7]),
        ("required-near-center", 2.0, 3.0, [1e-6, 0.0, 0.0]),
        ("required-exterior", 2.0, 3.0, [2.6, -1.05, 2.1]),
        ("side-extension", 2.0, 3.0, [2.0, 0.0, 3.0]),
        ("cap-extension", 2.0, 3.0, [3.0, 0.0, 1.5]),
        ("far", 2.0, 3.0, [200.0, 50.0, 100.0]),
        ("thin", 2.0, 0.02, [0.4, 0.2, 0.001]),
        ("long", 2.0, 200.0, [0.4, 0.2, 0.1]),
    ]
    for distance in (0.2, 0.02, 0.002):
        for sign in (-1, 1):
            cases.append(
                (
                    f"side {sign * distance:+g}",
                    2.0,
                    3.0,
                    [2.0 + sign * distance, 0.0, 0.4],
                )
            )
            cases.append(
                (
                    f"cap {sign * distance:+g}",
                    2.0,
                    3.0,
                    [0.4, 0.0, 1.5 + sign * distance],
                )
            )
    print("| case | a error | Gamma error | a ref delta | Gamma ref delta |")
    print("|---|---:|---:|---:|---:|")
    for label, radius, height, point in cases:
        low = surface_reference(radius, height, 1800.0, point, 1e-8)
        high = surface_reference(radius, height, 1800.0, point, 1e-11)
        acc, grad = cylinder_acceleration_gradient_g_free(
            [0.0, 0.0, 0.0], radius, height, 1800.0, [point]
        )
        scales = [
            1800 * radius + np.linalg.norm(high[0]),
            1800 + np.linalg.norm(high[1]),
        ]
        error = [
            np.linalg.norm(a - b) / scale
            for a, b, scale in zip((acc[0], grad[0]), high[:2], scales, strict=True)
        ]
        delta = [
            np.linalg.norm(a - b) / scale
            for a, b, scale in zip(low[:2], high[:2], scales, strict=True)
        ]
        print(
            f"| {label} | {error[0]:.3e} | {error[1]:.3e} "
            f"| {delta[0]:.3e} | {delta[1]:.3e} |"
        )


if __name__ == "__main__":
    main()
