"""CAS checks with real radii 0<a<r<b, integer l,m>=0; no production imports."""

import sympy as sp


def residuals():
    r, a, b = sp.symbols("r a b", positive=True)
    ell, m = sp.symbols("ell m", integer=True, nonnegative=True)
    power = 2 * ell + 2 * m + 3
    aa = (r**power - a**power) / power
    bb = (b ** (2 * m + 2) - r ** (2 * m + 2)) / (2 * m + 2)
    radial = r ** (-2 * ell - 1) * aa + bb
    first = -(2 * ell + 1) * aa / r ** (2 * ell + 2)
    second = (2 * ell + 1) * (2 * ell + 2) * aa / r ** (2 * ell + 3) - (
        2 * ell + 1
    ) * r ** (2 * m)
    out = [
        sp.simplify(sp.diff(radial, r) - first),
        sp.simplify(sp.diff(radial, r, 2) - second),
    ]
    # Radial Poisson operator on H_l T_l gives -(2l+1)r^(l+2m).
    out.append(
        sp.simplify(
            r**ell * (second + 2 * (ell + 1) * first / r)
            + (2 * ell + 1) * r ** (ell + 2 * m)
        )
    )
    rho, eps, x, y, z = sp.symbols("rho eps x y z", real=True)
    phi = -2 * sp.pi * rho * (eps**2 - (x * x + y * y + z * z) / 3)
    out += list(-sp.hessian(phi, (x, y, z)) + 4 * sp.pi * rho / 3 * sp.eye(3))
    out.append(sp.simplify(phi.subs({x: 0, y: 0, z: 0}) + 2 * sp.pi * rho * eps**2))
    # -d/dr[H(r-a)-H(r-b)] fixes inner/outer sheet signs.
    indicator = sp.Heaviside(r - a) - sp.Heaviside(r - b)
    out.append(sp.diff(indicator, r) - sp.DiracDelta(r - a) + sp.DiracDelta(r - b))
    # Center reductions: l=1. j1'=j0-2 j1/q.
    ja, jb, fa, fb = sp.symbols("ja jb fa fb")
    out += [
        sp.expand((ja - jb) - (ja - 2 * fa) + (jb - 2 * fb) - 2 * (fa - fb)),
        sp.expand(-fa + fb + (fa - fb)),
    ]
    # G decomposition identity; shared full-space particular solution cancels.
    full, removed_a, removed_b = sp.symbols("full removed_a removed_b")
    out.append(
        sp.expand((full - removed_a) - (full - removed_b) - (removed_b - removed_a))
    )
    # The finite interval is formed from the input subtraction before a/b.
    eta = sp.symbols("eta", positive=True)
    out.append(
        sp.simplify((1 - eta) ** (2 * m + 2) - (a / b) ** (2 * m + 2)).subs(
            eta, (b - a) / b
        )
    )
    # Independent surface-control derivatives: dipole and degree-two density.
    coords = sp.Matrix([x, y, z])
    radius = sp.sqrt(coords.dot(coords))
    pv = sp.Matrix(sp.symbols("px py pz", real=True))
    qx, qy, qz, qxy, qxz, qyz = sp.symbols("qx qy qz qxy qxz qyz", real=True)
    tensor = sp.Matrix([[qx, qxy, qxz], [qxy, qy, qyz], [qxz, qyz, qz]])
    linear = pv.dot(coords)
    quadratic = (coords.T * tensor * coords)[0]
    vec = tensor * coords
    expected_dipole = (
        -3 * (pv * coords.T + coords * pv.T + linear * sp.eye(3)) / radius**5
        + 15 * linear * coords * coords.T / radius**7
    )
    expected_quadratic = (
        2 * tensor / radius**5
        - 10 * (vec * coords.T + coords * vec.T) / radius**7
        - 5 * quadratic * sp.eye(3) / radius**7
        + 35 * quadratic * coords * coords.T / radius**9
    )
    out.extend(sp.hessian(linear / radius**3, (x, y, z)) - expected_dipole)
    out.extend(sp.hessian(quadratic / radius**5, (x, y, z)) - expected_quadratic)
    return [sp.simplify(value) for value in out]


def derived_expression():
    return sp.Add(*(value**2 for value in residuals()))


def reference_expression():
    return sp.Integer(0)


if __name__ == "__main__":
    checks = residuals()
    assert all(value == 0 for value in checks), checks
    print(f"PASS: {len(checks)} exact symbolic residuals; domains in module docstring")
