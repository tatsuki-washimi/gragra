"""CAS checks of the adopted +i eigenfunctions, not a gravity oracle."""

import sympy as sp


def main():
    k, q, s = sp.symbols("k q s", positive=True)
    z = sp.symbols("z", real=True)
    eta = sp.symbols("eta")
    zeta = 2 * q / (1 + s * s)
    amplitude = eta / (k * (q - zeta))
    horizontal = (
        sp.I * k * amplitude * (sp.exp(k * q * z) - s * zeta * sp.exp(k * s * z))
    )
    vertical = k * amplitude * (q * sp.exp(k * q * z) - zeta * sp.exp(k * s * z))
    divergence = sp.I * k * horizontal + sp.diff(vertical, z)
    assert sp.simplify(vertical.subs(z, 0) - eta) == 0
    assert (
        sp.simplify(divergence - k * k * amplitude * (q * q - 1) * sp.exp(k * q * z))
        == 0
    )
    shear = sp.diff(horizontal, z) + sp.I * k * vertical
    assert sp.simplify(shear.subs(z, 0)) == 0
    # vp^2/vs^2=(1-s^2)/(1-q^2); secular: 4qs=(1+s^2)^2.
    normal = ((1 - s * s) / (1 - q * q) - 2) * divergence + 2 * sp.diff(vertical, z)
    residual = sp.factor(normal.subs(z, 0) / (k * k * amplitude))
    assert sp.simplify(residual - ((1 + s * s) ** 2 - 4 * q * s) / (1 + s * s)) == 0
    assert sp.simplify((zeta * zeta - q / s).subs(q, (1 + s * s) ** 2 / (4 * s))) == 0
    # Horizontal surface displacement / eta is imaginary for real q,s.
    assert sp.simplify(sp.re((horizontal / eta).subs(z, 0))) == 0
    print("PASS: surface normalization, divergence, shear traction, normal traction")
    print("PASS: secular normalization and quadrature phase; assumptions k,q,s>0")
    print("NOT an independent gravity integral or proof of source meaning")


if __name__ == "__main__":
    main()
