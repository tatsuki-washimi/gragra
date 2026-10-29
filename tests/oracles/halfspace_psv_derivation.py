"""Exact algebra: elastic coefficients, Poisson matching and target jets.

Domain: q>=0, rho>=0, kP,kS>0, s>=0, p real>=0 or positive imaginary;
Den!=0. Exact q=0 uses the continuous-q gauge (tested numerically too).
SymPy is verification-only, never imported by gragra.
"""

import sympy as sp


def verify():
    q, p, s, d, rho = sp.symbols("q p s d rho")
    a, c = sp.symbols("A C")
    den = d * d + 4 * q * q * p * s
    b = ((4 * q * q * p * s - d * d) * a - 4 * q * s * d * c) / den
    reflected_sv = (4 * q * p * d * a + (4 * q * q * p * s - d * d) * c) / den
    assert sp.cancel(-d * (a + b) + 2 * q * s * (reflected_sv - c)) == 0
    assert sp.cancel(2 * q * p * (b - a) + d * (c + reflected_sv)) == 0
    # Propagating-mode depth flux is rho*omega^3*kz*|potential|²/2.
    # Algebraically real p,s,d,q suffices for real P or SV incident input.
    for ai, ci, incoming in [(1, 0, p), (0, 1, s)]:
        br = b.subs({a: ai, c: ci})
        dr = reflected_sv.subs({a: ai, c: ci})
        assert sp.cancel(p * br**2 + s * dr**2 - incoming) == 0
    b, reflected_sv = sp.symbols("B D")
    outside = -2 * sp.pi * rho * (a + b + sp.I * (c + reflected_sv))
    inside = 2 * sp.pi * rho * (a + b - sp.I * (c + reflected_sv))
    particular = -4 * sp.pi * rho * (a + b)
    particular_prime = -4 * sp.pi * sp.I * rho * p * (a - b)
    sigma = sp.I * rho * (p * (a - b) + q * (c + reflected_sv))
    assert sp.expand(outside - inside - particular) == 0
    assert (
        sp.expand(-q * outside - q * inside - particular_prime - 4 * sp.pi * sigma) == 0
    )
    x, y, z, qx, qy = sp.symbols("x y z qx qy", real=True)
    f = sp.Function("F")(z)
    phase = sp.exp(sp.I * (qx * x + qy * y))
    phi = phase * f
    accel = sp.Matrix([-sp.diff(phi, coordinate) for coordinate in (x, y, z)])
    expected = phase * sp.Matrix([-sp.I * qx * f, -sp.I * qy * f, -sp.diff(f, z)])
    assert sp.simplify(accel - expected) == sp.zeros(3, 1)
    hessian = accel.jacobian([x, y, z]) / phase
    expected_hessian = sp.Matrix(
        [
            [qx * qx * f, qx * qy * f, -sp.I * qx * sp.diff(f, z)],
            [qx * qy * f, qy * qy * f, -sp.I * qy * sp.diff(f, z)],
            [-sp.I * qx * sp.diff(f, z), -sp.I * qy * sp.diff(f, z), -sp.diff(f, z, 2)],
        ]
    )
    assert sp.simplify(hessian - expected_hessian) == sp.zeros(3, 3)


if __name__ == "__main__":
    verify()
    print("P/SV traction, flux, Poisson matching and target jets: exact CAS pass")
