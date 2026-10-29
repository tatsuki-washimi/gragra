"""Reproduce the symbolic sign and long-wavelength checks for M2 T08."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    """Verify the Newton kernel, interface jump, bulk sign, and wall limit."""
    # Assumptions: three-dimensional Euclidean space; zero softening; source != target
    # for pointwise kernels; positive k, R0, and rho0 for the P-wave limits.
    sx, sy, sz, tx, ty, tz = sp.symbols("sx sy sz tx ty tz", real=True)
    source = sp.Matrix([sx, sy, sz])
    target = sp.Matrix([tx, ty, tz])
    delta = source - target
    radius_squared = sp.expand(delta.dot(delta))
    radius = sp.sqrt(radius_squared)
    potential_kernel = -1 / radius
    acceleration_kernel = sp.Matrix(
        [-sp.diff(potential_kernel, coordinate) for coordinate in target]
    )
    expected_acceleration = delta / radius**3
    assert all(
        sp.simplify(actual - expected) == 0
        for actual, expected in zip(
            acceleration_kernel, expected_acceleration, strict=True
        )
    )

    tensor = sp.Matrix(
        3,
        3,
        lambda i, j: (
            3 * delta[i] * delta[j] / radius**5 - sp.KroneckerDelta(i, j) / radius**3
        ),
    )
    assert all(
        sp.simplify(sp.diff(acceleration_kernel[i], source[j]) + tensor[i, j]) == 0
        for i in range(3)
        for j in range(3)
    )

    signed_distance = sp.symbols("s", real=True)
    rho_minus, rho_plus, normal_displacement = sp.symbols(
        "rho_minus rho_plus xi_n", real=True
    )
    density_step = rho_minus * sp.Heaviside(-signed_distance) + rho_plus * sp.Heaviside(
        signed_distance
    )
    density_derivative = sp.diff(density_step, signed_distance)
    density_jump = rho_plus - rho_minus
    assert (
        sp.simplify(density_derivative - density_jump * sp.DiracDelta(signed_distance))
        == 0
    )
    surface_density = sp.simplify(
        -normal_displacement * density_derivative.coeff(sp.DiracDelta(signed_distance))
    )
    assert (
        sp.simplify(surface_density - (rho_minus - rho_plus) * normal_displacement) == 0
    )
    reversed_normal_density = (-rho_minus + rho_plus) * (-normal_displacement)
    assert sp.simplify(reversed_normal_density - surface_density) == 0

    theta, azimuth = sp.symbols("theta azimuth", real=True)
    unit_normal = sp.Matrix(
        [
            sp.sin(theta) * sp.cos(azimuth),
            sp.sin(theta) * sp.sin(azimuth),
            sp.cos(theta),
        ]
    )
    angular_tensor = sp.Matrix(
        3,
        3,
        lambda i, j: sp.integrate(
            sp.integrate(
                unit_normal[i] * unit_normal[j] * sp.sin(theta),
                (azimuth, 0, 2 * sp.pi),
            ),
            (theta, 0, sp.pi),
        ),
    )
    assert angular_tensor == (4 * sp.pi / 3) * sp.eye(3)

    q = sp.symbols("q", positive=True, real=True)
    angular_z = (
        2
        * sp.pi
        * sp.integrate(
            sp.sin(theta) * sp.cos(theta) * sp.exp(sp.I * q * sp.cos(theta)),
            (theta, 0, sp.pi),
        )
    )
    spherical_j0 = sp.sin(q) / q
    spherical_j1 = sp.sin(q) / q**2 - sp.cos(q) / q
    assert (
        sp.trigsimp(sp.expand_complex(angular_z - 4 * sp.pi * sp.I * spherical_j1)) == 0
    )
    assert sp.simplify(sp.diff(spherical_j0, q) + spherical_j1) == 0
    assert sp.limit(spherical_j0, q, sp.oo) == 0
    assert sp.limit(spherical_j0, q, 0, dir="+") == 1

    wavenumber, background_density, displacement = sp.symbols(
        "k rho0 xi0", positive=True, real=True
    )
    combined_factor = sp.simplify(
        (-sp.I * wavenumber * background_density * displacement)
        * (4 * sp.pi * sp.I * spherical_j1)
    )
    assert (
        sp.simplify(
            combined_factor
            - 4 * sp.pi * wavenumber * background_density * displacement * spherical_j1
        )
        == 0
    )
    bulk_limit = sp.limit(
        4 * sp.pi * background_density * displacement * spherical_j0,
        q,
        0,
        dir="+",
    )
    assert sp.simplify(bulk_limit - 4 * sp.pi * background_density * displacement) == 0

    wall_j1_over_q = sp.limit(spherical_j1 / q, q, 0, dir="+")
    wall_limit = sp.limit(
        -4
        * sp.pi
        * background_density
        * displacement
        * (spherical_j0 - 2 * spherical_j1 / q),
        q,
        0,
        dir="+",
    )
    assert sp.simplify(wall_j1_over_q - sp.Rational(1, 3)) == 0
    assert (
        sp.simplify(
            wall_limit + sp.Rational(4, 3) * sp.pi * background_density * displacement
        )
        == 0
    )

    print(
        f"PASS SymPy={sp.__version__}: acceleration sign, source derivative=-T, "
        "distributional interface jump, normal reversal, angular tensor=4*pi*I/3, "
        "bulk factor=+4*pi*rho0*xi0, wall limit=-4*pi*rho0*xi0/3"
    )


if __name__ == "__main__":
    main()
