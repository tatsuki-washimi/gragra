"""Exact CAS checks of U9 added derivations (verification dependency only)."""

import sympy as s


def main():
    x, y, z = q = s.symbols("x y z", real=True)
    ux, uy, uz = u = s.symbols("ux uy uz")
    radius, mass = s.symbols("a M", positive=True)
    distance = s.sqrt(sum(v * v for v in q))
    dot = sum(v * w for v, w in zip(q, u, strict=True))
    potential = mass * dot / distance**3
    acceleration = [
        mass * (-u[i] / distance**3 + 3 * q[i] * dot / distance**5) for i in range(3)
    ]
    count = 0

    def check(expr):
        nonlocal count
        assert s.simplify(expr) == 0, expr
        count += 1

    for i in range(3):
        check(s.diff(potential, q[i]) + acceleration[i])
        for j in range(3):
            explicit = mass * (
                3 * (dot * int(i == j) + u[i] * q[j] + q[i] * u[j]) / distance**5
                - 15 * dot * q[i] * q[j] / distance**7
            )
            check(s.diff(acceleration[i], q[j]) - explicit)
    # Cuboid corner target derivative: corner coordinates decrease with target.
    for j, coordinate in enumerate(q):
        denominator = x * x * distance**2 + y * y * z * z
        partial = [
            (-x * x * y * z / distance - distance * y * z),
            x * z * (distance - y * y / distance),
            x * y * (distance - z * z / distance),
        ][j] / denominator
        check(s.diff(s.atan(y * z / (x * distance)), coordinate) - partial)
    check(s.diff(s.log(z + distance), z) - 1 / distance)
    check(s.diff(s.log(z + distance), x) - x / (distance * (z + distance)))
    # Rationalization of the negative corner-log argument, for x²+y²>0.
    check((z + distance) * (distance - z) - x * x - y * y)
    # Angular basis derivative at fixed r,z.
    angle = s.symbols("phi", real=True)
    aa, bb, cc, dd = s.symbols("A B C D", real=True)
    er = s.Matrix([s.cos(angle), s.sin(angle), 0])
    ep = s.Matrix([-s.sin(angle), s.cos(angle), 0])
    ez = s.Matrix([0, 0, 1])
    gamma = (
        aa * er * er.T + bb * ep * ep.T + cc * ez * ez.T + dd * (er * ez.T + ez * er.T)
    )
    expected = (aa - bb) * (er * ep.T + ep * er.T) + dd * (ep * ez.T + ez * ep.T)
    for entry in gamma.diff(angle) - expected:
        check(entry)
    # Ring reduction for derivative of disk solid angle.
    r, v, c = s.symbols("r v c", real=True)
    d2 = radius**2 + r * r + v * v - 2 * radius * r * c
    check(-radius * (radius - r * c) - ((r * r + v * v - radius**2) / 2 - d2 / 2))
    # Axis recurrence derives mixed entries, retaining transverse u.
    f = s.Function("f")(z)
    phi = (
        f
        - (x * x + y * y) * s.diff(f, z, 2) / 4
        + (x * x + y * y) ** 2 * s.diff(f, z, 4) / 64
    )
    for i in range(3):
        for j in range(3):
            for k in range(3):
                check(s.diff(phi, q[i], q[j], q[k]) - s.diff(phi, q[k], q[j], q[i]))
    check((-s.diff(phi, x, x, z)).subs({x: 0, y: 0}) - s.diff(f, z, 3) / 2)
    check((-s.diff(phi, z, z, z)).subs({x: 0, y: 0}) + s.diff(f, z, 3))
    # Axis static series: Gamma_xz from radial a equals Gamma_zx from axial a.
    t = s.symbols("t", nonnegative=True)
    az = sum(
        (-1) ** n * s.diff(f, z, 2 * n) * t**n / (4**n * s.factorial(n) ** 2)
        for n in range(4)
    )
    radial_over_r = sum(
        (-1) ** (n + 1)
        * s.diff(f, z, 2 * n + 1)
        * t**n
        / (2 * 4**n * s.factorial(n) * s.factorial(n + 1))
        for n in range(3)
    )
    check(2 * s.diff(az, t) - s.diff(radial_over_r, z))
    m = s.symbols("m", positive=True)
    check(
        s.diff(s.elliptic_k(m), m)
        - (s.elliptic_e(m) / (1 - m) - s.elliptic_k(m)) / (2 * m)
    )
    check(s.diff(s.elliptic_e(m), m) - (s.elliptic_e(m) - s.elliptic_k(m)) / (2 * m))
    # Polynomial endpoint recurrence used by the axis derivative evaluator.
    polynomial = z
    for n in range(1, 13):
        following = s.expand(
            (radius**2 + z * z) * s.diff(polynomial, z) + (1 - 2 * n) * z * polynomial
        )
        check(
            s.diff(polynomial * (radius**2 + z * z) ** (s.Rational(1, 2) - n), z)
            - following * (radius**2 + z * z) ** (-s.Rational(1, 2) - n)
        )
        polynomial = following
    print(f"{count} exact identities passed; real targets off walls, a,M>0")
    print("CAS is derivation checking, not independent physical integration.")


if __name__ == "__main__":
    main()
