"""Reproduce V's per-target potential error and reference-refinement table."""

from oracles.cylinder_potential_reference import boundary_potential, volume_potential

from gragra.analytic import cylinder_analytic_observables_g_free

CASES = [
    ("interior", 2, 3, [0.2, 0.3, 0.1]),
    ("exterior", 2, 3, [2.6, -1.05, 2.1]),
    ("axis", 2, 3, [0, 0, 0.7]),
    ("near-axis", 2, 3, [1e-5, -2e-5, 0.7]),
    ("near-axis-small", 2, 3, [1e-12, 0, 0]),
    ("side-extension", 2, 3, [2, 0, 3]),
    ("cap-extension", 2, 3, [3, 0, 1.5]),
    ("thin", 2, 0.02, [0.4, 0.2, 0.001]),
    ("long", 2, 200, [0.4, 0.2, 0.1]),
    *[
        (f"side-{sign:+d}-{gap:g}", 2, 3, [2 + sign * gap, 0, 0.4])
        for gap in (0.2, 0.02, 0.002)
        for sign in (-1, 1)
    ],
    *[
        (f"cap-{sign:+d}-{gap:g}", 2, 3, [0.4, 0, 1.5 + sign * gap])
        for gap in (0.2, 0.02, 0.002)
        for sign in (-1, 1)
    ],
    ("rim-inside", 2, 3, [2 - 1e-6, 0, 1.5 - 1e-6]),
    ("rim-outside", 2, 3, [2 + 1e-6, 0, 1.5 + 1e-6]),
    ("far-series-before", 2, 3, [150, 0, 0]),
    ("far-series-after", 2, 3, [170, 0, 0]),
]


def main():
    print("| Target | normalized Phi error | reference refinement |")
    print("|---|---:|---:|")
    for name, radius, height, point in CASES:
        low, _ = boundary_potential(radius, height, 1800, point, tolerance=1e-8)
        high, _ = boundary_potential(radius, height, 1800, point, tolerance=1e-11)
        actual = cylinder_analytic_observables_g_free(
            [0, 0, 0], radius, height, 1800, [point]
        )
        scale = 1800 * radius**2 + abs(high)
        error, refinement = (
            abs(actual.potential[0] - high) / scale,
            abs(low - high) / scale,
        )
        print(f"| {name} | {error:.3e} | {refinement:.3e} |")
        assert error <= 1e-10 and refinement <= 1e-11
    for name, radius, height, point in CASES[:2]:
        low, _ = volume_potential(radius, height, 1800, point, tolerance=1e-8)
        high, _ = volume_potential(radius, height, 1800, point, tolerance=1e-11)
        actual = cylinder_analytic_observables_g_free(
            [0, 0, 0], radius, height, 1800, [point]
        )
        scale = 1800 * radius**2 + abs(high)
        print(
            f"volume {name}: error={abs(actual.potential[0] - high) / scale:.3e}, "
            f"refinement={abs(low - high) / scale:.3e}"
        )


if __name__ == "__main__":
    main()
