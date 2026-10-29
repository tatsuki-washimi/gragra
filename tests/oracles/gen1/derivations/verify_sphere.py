"""Sphere exact checks + independent high precision radial normalization checks.
Assume real normalized Y, R,k>0, positive derivative root; no production imports.
"""

import mpmath as mp
import sympy as s


def zero(v):
    assert s.simplify(s.expand_func(v)) == 0, v


def main():
    r, r_symbol, k, j_symbol = s.symbols("r R k J", positive=True)
    degree = s.symbols("l", integer=True, nonnegative=True)
    j = s.Function("j")(r)
    inner = -j + (degree + 1) / (2 * degree + 1) * j_symbol * (r / r_symbol) ** degree
    outer = -degree / (2 * degree + 1) * j_symbol * (r_symbol / r) ** (degree + 1)
    boundary = {j.subs(r, r_symbol): j_symbol, s.diff(j, r).subs(r, r_symbol): 0}
    zero((inner - outer).subs(r, r_symbol).subs(boundary))
    zero((s.diff(inner, r) - s.diff(outer, r)).subs(r, r_symbol).subs(boundary))

    def op(f):
        return s.diff(f, r, 2) + 2 / r * s.diff(f, r) - degree * (degree + 1) / r**2 * f

    ode = {
        s.diff(j, r, 2): -2 / r * s.diff(j, r)
        + (degree * (degree + 1) / r**2 - k**2) * j
    }
    zero(op(inner).subs(ode) - k**2 * j)
    zero(op(outer))
    normprimitive = (
        r**3
        / 2
        * (
            (1 - degree * (degree + 1) / (k * k * r * r)) * j * j
            + s.diff(j, r) ** 2 / (k * k)
            + j * s.diff(j, r) / (k * k * r)
        )
    )
    zero((s.diff(normprimitive, r) - r * r * j * j).subs(ode))
    print(
        "Sphere arbitrary l: Poisson, exterior harmonic, "
        "two interface identities EXACT PASS"
    )
    # Real harmonic m= +/-1 at l=1, Condon Shortley convention.
    th, ph = s.symbols("th ph", real=True)
    yp = s.expand_func(s.Ynm(1, 1, th, ph))
    zero(s.sqrt(2) * (-1) * s.re(yp) - s.sqrt(3 / (4 * s.pi)) * s.sin(th) * s.cos(ph))
    zero(s.sqrt(2) * (-1) * s.im(yp) - s.sqrt(3 / (4 * s.pi)) * s.sin(th) * s.sin(ph))
    # Generic radial-solid-harmonic identity; H arbitrary, regular center.
    x, y, z = s.symbols("x y z", real=True)
    v = s.Matrix([x, y, z])
    u = v.dot(v)
    h_symbol = s.Function("H")(x, y, z)
    f = s.Function("f")(u)
    q_symbol = f * h_symbol
    grad_h = s.Matrix([s.diff(h_symbol, a) for a in v])
    hess_h = s.hessian(h_symbol, list(v))
    fp = s.Subs(
        s.Derivative(s.Function("f")(s.Symbol("u")), s.Symbol("u")), s.Symbol("u"), u
    )
    fpp = s.Subs(
        s.Derivative(s.Function("f")(s.Symbol("u")), s.Symbol("u"), 2), s.Symbol("u"), u
    )
    for i in range(3):
        zero(s.diff(q_symbol, v[i]) - 2 * fp * v[i] * h_symbol - f * grad_h[i])
        for jdx in range(3):
            zero(
                s.diff(q_symbol, v[i], v[jdx])
                - (4 * fpp * v[i] * v[jdx] + 2 * fp * s.KroneckerDelta(i, jdx))
                * h_symbol
                - 2 * fp * (v[i] * grad_h[jdx] + v[jdx] * grad_h[i])
                - f * hess_h[i, jdx]
            )
    print("Sphere real m +/-1 and regular Cartesian derivative identities EXACT PASS")
    mp.mp.dps = 45
    for ell in range(5):

        def jl(a, ell=ell):
            return mp.sqrt(mp.pi / (2 * a)) * mp.besselj(ell + mp.mpf(".5"), a)

        def derivative(a):
            return mp.diff(jl, a)

        # bracket sign changes, exclude root zero.
        roots = []
        a = mp.mpf(".01")
        fa = derivative(a)
        while len(roots) < 3:
            b = a + mp.mpf(".1")
            fb = derivative(b)
            if fa * fb < 0:
                roots.append(mp.findroot(derivative, (a, b)))
            a, fa = b, fb
        for alpha in roots:
            integral = mp.quad(
                lambda t, alpha=alpha: t * t * jl(alpha * t) ** 2, [0, 1]
            )
            closed = jl(alpha) ** 2 / 2 * (1 - ell * (ell + 1) / alpha**2)
            assert abs(integral - closed) < mp.mpf("1e-40")
    print(
        "Sphere l=0..4 roots 1..3: independent 45dps "
        "normalization CHECK PASS (numerical)"
    )


if __name__ == "__main__":
    main()
