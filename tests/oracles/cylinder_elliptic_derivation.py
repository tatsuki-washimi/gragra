"""Reproducible CAS checks for C's original transformations.

Assumptions: R,r > 0; u real and away from an end plane for derivatives
of sign(u); no material boundary. The angular variable t=cos(phi) is real.
K,E,Pi use parameter m, characteristic n, with 0<=m<1 and 0<=n<1.
The r=R and r=0 limits are tested numerically separately.

Run directly or with ai-harness/bin/verify-symbolic --file this_file.
"""

# Names follow the source equations to make the CAS correspondence reviewable.
# ruff: noqa: N806
import sympy as s


def residuals():
    R, r = s.symbols("R r", positive=True)
    u, t = s.symbols("u t", real=True)
    q = R**2 + r**2 - 2 * R * r * t
    d = s.sqrt(q + u**2)
    # Integration by parts of cos(phi)*asinh(u/sqrt(q)).
    reduced = (-((R**2 - r**2) ** 2) / q + 2 * (R**2 + r**2) - q) / (4 * R * r * d)
    yield s.factor(reduced - R * r * (1 - t**2) / (q * d))
    # Disk potential: planar divergence of X/(sqrt(X²+u²)+|u|).
    v, w, h = s.symbols("v w h", positive=True)
    D = s.sqrt(v * v + w * w + h * h)
    yield s.simplify(s.diff(v / (D + h), v) + s.diff(w / (D + h), w) - 1 / D)
    # Analytical mixed derivative comes directly from the radial side
    # primitive BEFORE elliptic reduction, du/dz_target=-1.
    q_positive = s.symbols("q_positive", positive=True)
    yield s.simplify(
        s.diff(-R * t * s.asinh(u / s.sqrt(q_positive)), u)
        + R * t / s.sqrt(q_positive + u * u)
    )
    # Solid angle integrand is -d(cap potential)/du away from u=0.
    cap = R * (R - r * t) * (d - u) / q
    solid = R * (R - r * t) * (1 - u / d) / q
    yield s.simplify(s.diff(cap, u) + solid)
    # Five successive target derivatives of the axis endpoint primitive.
    D = s.sqrt(R * R + u * u)
    first = u / D  # The constant sign term only affects the first derivative.
    expected = [
        -(R**2) / D**3,
        -3 * R**2 * u / D**5,
        3 * R**2 * (R**2 - 4 * u**2) / D**7,
        15 * R**2 * u * (3 * R**2 - 4 * u**2) / D**9,
    ]
    for expression in expected:
        first = -s.diff(first, u)
        yield s.simplify(first - expression)
    # Potential radial Taylor recurrence yields the 3D azimuthal component.
    z = s.symbols("z", real=True)
    f = s.Function("f")(z)
    potential = f - r * r * s.diff(f, z, 2) / 4 + r**4 * s.diff(f, z, 4) / 64
    ar = -s.diff(potential, r)
    az = -s.diff(potential, z)
    yield s.simplify(s.diff(ar, z) - s.diff(az, r))
    laplace = s.diff(ar, r) + ar / r + s.diff(az, z)
    yield s.expand(laplace).coeff(r, 0)
    yield s.expand(laplace).coeff(r, 2)
    # Uniform-cylinder multipole moments: integrate solid zonal harmonics
    # independently and compare the coefficient formula used in production.
    radial, axial, half = s.symbols("radial axial half", positive=True)
    for degree in range(0, 10, 2):
        solid = s.expand(
            (radial**2 + axial**2) ** s.Rational(degree, 2)
            * s.legendre(degree, axial / s.sqrt(radial**2 + axial**2))
        )
        solid = s.simplify(solid)
        average = s.integrate(
            s.integrate(solid * radial, (radial, 0, R)), (axial, -half, half)
        ) / (R**2 * half)
        formula = sum(
            (-1) ** j
            * s.factorial(degree)
            / (s.factorial(degree - 2 * j) * s.factorial(j) ** 2 * 4**j)
            * R ** (2 * j)
            * half ** (degree - 2 * j)
            / ((j + 1) * (degree - 2 * j + 1))
            for j in range(degree // 2 + 1)
        )
        yield s.simplify(average - formula)
    # General angular polynomial: Cartesian Hessian formula, checked for
    # all employed degrees at y=0 using axial symmetry (r>0).
    distance = s.sqrt(r * r + z * z)
    for degree in range(0, 10, 2):
        t = s.symbols("t", real=True)
        poly = s.legendre(degree, t)
        P, dp, ddp = [s.diff(poly, t, n).subs(t, z / distance) for n in range(3)]
        A = (degree + 1) * P + z / distance * dp
        Ap = (degree + 2) * dp + z / distance * ddp
        az = (dp - A * z / distance) / distance ** (degree + 2)
        rz = (
            ((degree + 3) * A + z / distance * Ap) * r * z / distance**2
            - Ap * r / distance
        ) / distance ** (degree + 3)
        zz = (
            ((degree + 3) * A + z / distance * Ap) * z * z / distance**2
            - 2 * Ap * z / distance
            + ddp
            - A
        ) / distance ** (degree + 3)
        yield s.simplify(s.diff(az, r) - rz)
        yield s.simplify(s.diff(az, z) - zz)


def derived_expression():
    values = list(residuals())
    assert all(value == 0 for value in values), values
    return s.Add(*values)


def reference_expression():
    return s.Integer(0)


if __name__ == "__main__":
    derived_expression()
    print(f"{len(list(residuals()))} exact CAS identities verified")
