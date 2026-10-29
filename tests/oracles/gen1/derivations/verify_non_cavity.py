"""Exact Gen1 derivations, assumptions: r,sigma,c,k,q>0; real coordinates.
No production imports. G and amplitudes symbolic, cgs/SI conversion absent.
"""

import sympy as s


def zero(x):
    assert s.simplify(x) == 0, x


def main():
    length, mass, time = s.symbols("length mass time", positive=True)
    gravity_dimension = length**3 / (mass * time**2)
    gaussian_mass_dimension = (mass / length**3) * length**3
    zero(gravity_dimension * gaussian_mass_dimension / length - length**2 / time**2)
    zero(gravity_dimension * gaussian_mass_dimension / length**2 - length / time**2)
    zero(gravity_dimension * gaussian_mass_dimension / length**3 - 1 / time**2)
    moment_dimension = mass * length / time**2 * time**2
    zero(gravity_dimension * moment_dimension / length**2 - length**2 / time**2)
    zero(gravity_dimension * moment_dimension / length**3 - length / time**2)
    zero(gravity_dimension * moment_dimension / length**4 - 1 / time**2)
    print("Gaussian and point-force dimensions EXACT PASS")
    r, sigma, g_symbol, rho = s.symbols("r sigma G rho", positive=True)
    m_symbol = (2 * s.pi) ** s.Rational(3, 2) * rho * sigma**3
    e = s.exp(-(r**2) / (2 * sigma**2))
    h_symbol = s.erf(r / (s.sqrt(2) * sigma)) - s.sqrt(2 / s.pi) * r / sigma * e
    phi = -g_symbol * m_symbol * s.erf(r / (s.sqrt(2) * sigma)) / r
    ar = -g_symbol * m_symbol * h_symbol / r**2
    rr = 2 * g_symbol * m_symbol * h_symbol / r**3 - 4 * s.pi * g_symbol * rho * e
    tt = -g_symbol * m_symbol * h_symbol / r**3
    zero(ar + s.diff(phi, r))
    zero(rr - s.diff(ar, r))
    zero(tt - ar / r)
    zero(rr + 2 * tt + 4 * s.pi * g_symbol * rho * e)
    zero(s.limit(phi, r, 0) + 4 * s.pi * g_symbol * rho * sigma**2)
    zero(s.limit(ar, r, 0))
    zero(s.limit(rr, r, 0) + 4 * s.pi * g_symbol * rho / 3)
    zero(s.limit(tt, r, 0) + 4 * s.pi * g_symbol * rho / 3)
    zero(s.limit(r * phi, r, s.oo) + g_symbol * m_symbol)
    zero(s.limit(r**2 * ar, r, s.oo) + g_symbol * m_symbol)
    zero(s.limit(r**3 * rr, r, s.oo) - 2 * g_symbol * m_symbol)
    zero(s.limit(r**3 * tt, r, s.oo) + g_symbol * m_symbol)
    print("Gaussian: derivative, Poisson, center and far limits EXACT PASS")
    z, q, p = s.symbols("z q p", real=True, positive=True)
    ellair = s.exp(-q * z) / 2 - s.cos(p * z)
    ellground = -s.exp(q * z) / 2
    zero(s.diff(ellair, z, 2) - q**2 * ellair - (q**2 + p**2) * s.cos(p * z))
    zero(s.diff(ellground, z, 2) - q**2 * ellground)
    zero(ellair.subs(z, 0) - ellground.subs(z, 0))
    zero(s.diff(ellair, z).subs(z, 0) - s.diff(ellground, z).subs(z, 0))
    print(
        "Halfspace: both sides Poisson and potential/normal derivative "
        "continuity EXACT PASS"
    )
    # F(u)=u^2 for u>=0 is C1 at onset. Postfront moment and prompt are explicit.
    t, c, tau = s.symbols("t c tau", positive=True)
    jpost_symbol = s.integrate(tau * (t - tau) ** 2, (tau, 0, r / c))
    zero(s.diff(jpost_symbol, r) - r * (t - r / c) ** 2 / c**2)
    jprompt_symbol = s.integrate(tau * (t - tau) ** 2, (tau, 0, t))
    zero(jprompt_symbol - t**4 / 12)
    zero(jpost_symbol.subs(r, c * t) - jprompt_symbol)
    print(
        "Point force: smooth causal F=t^2 delay moment and front continuity EXACT PASS"
    )
    # First displacement derivative of a point mass, valid r>0.
    x, y, z, dx, dy, dz, eps = s.symbols("x y z dx dy dz eps", real=True)
    coords = s.Matrix([x, y, z])
    d = s.Matrix([dx, dy, dz])
    rad = s.sqrt(coords.dot(coords))
    shifted = coords - eps * d
    potential = -g_symbol / s.sqrt(shifted.dot(shifted))
    phichange = -g_symbol * coords.dot(d) / rad**3
    zero(s.diff(potential, eps).subs(eps, 0) - phichange)
    avec = g_symbol * (d / rad**3 - 3 * coords.dot(d) * coords / rad**5)
    for j in range(3):
        zero(avec[j] + s.diff(phichange, coords[j]))
    gchange = g_symbol * (
        -3 * (d * coords.T + coords * d.T + coords.dot(d) * s.eye(3)) / rad**5
        + 15 * coords.dot(d) * coords * coords.T / rad**7
    )
    for i in range(3):
        for j in range(3):
            zero(s.diff(avec[i], coords[j]) - gchange[i, j])
    zero(s.trace(gchange))
    print(
        "Objects: linear oscillation potential/acceleration/gradient/trace EXACT PASS"
    )

    # Ground-side Rayleigh stochastic tensor from the deterministic literature
    # potential. C_R denotes the common gravity/displacement transfer
    # 2*pi*G*rho*gamma*xi_z. The exact azimuth average is independent of
    # the overall amplitude and height factor.
    theta = s.symbols("theta", real=True)
    k_r, c_r = s.symbols("k_R C_R", positive=True, real=True)
    x_r, y_r, z_r = s.symbols("x_R y_R z_R", real=True)
    nx = s.cos(theta)
    ny = s.sin(theta)
    phase_r = s.exp(s.I * k_r * (nx * x_r + ny * y_r))
    phi_r = -(c_r / k_r) * s.exp(-k_r * z_r) * phase_r
    a_r = s.Matrix([-s.diff(phi_r, coord) for coord in (x_r, y_r, z_r)])
    expected_r = (
        c_r
        * s.exp(-k_r * z_r)
        * phase_r
        * s.Matrix([s.I * nx, s.I * ny, -1])
    )
    for j in range(3):
        zero(a_r[j] - expected_r[j])

    hvec = s.Matrix([s.I * nx, s.I * ny, -1])
    isotropic_csd = s.Matrix(
        3,
        3,
        lambda i, j: s.simplify(
            s.integrate(
                hvec[i] * s.conjugate(hvec[j]),
                (theta, 0, 2 * s.pi),
            )
            / (2 * s.pi)
        ),
    )
    expected_csd = s.diag(s.Rational(1, 2), s.Rational(1, 2), 1)
    for i in range(3):
        for j in range(3):
            zero(isotropic_csd[i, j] - expected_csd[i, j])
    print(
        "Rayleigh stochastic: ground-side acceleration transfer and exact "
        "isotropic 3x3 CSD average EXACT PASS"
    )

    # Isotropic P-wave full-space displacement tensor. For an isotropic
    # distribution of P propagation directions, <n_i n_j> = delta_ij/3.
    polar, azimuth = s.symbols("polar azimuth", real=True)
    nvec = s.Matrix(
        [
            s.sin(polar) * s.cos(azimuth),
            s.sin(polar) * s.sin(azimuth),
            s.cos(polar),
        ]
    )
    p_direction_average = s.Matrix(
        3,
        3,
        lambda i, j: s.simplify(
            s.integrate(
                s.integrate(
                    nvec[i] * nvec[j] * s.sin(polar),
                    (azimuth, 0, 2 * s.pi),
                ),
                (polar, 0, s.pi),
            )
            / (4 * s.pi)
        ),
    )
    expected_p_average = s.eye(3) / 3
    for i in range(3):
        for j in range(3):
            zero(p_direction_average[i, j] - expected_p_average[i, j])
    print(
        "P-wave stochastic: exact isotropic direction tensor "
        "<n_i n_j>=delta_ij/3 EXACT PASS"
    )

    # Spherical-cavity body-wave coupling: exact small-ka limit of j1(x)/x
    # and the resulting isotropic P/S mixture coefficient.
    x_mode = s.symbols("x_mode", positive=True, real=True)
    j1_over_x = s.sin(x_mode) / x_mode**3 - s.cos(x_mode) / x_mode**2
    zero(s.limit(j1_over_x, x_mode, 0) - s.Rational(1, 3))

    x_p, x_s, eta_p = s.symbols("x_P x_S eta_P", positive=True, real=True)
    f_p = s.sin(x_p) / x_p**3 - s.cos(x_p) / x_p**2
    f_s = s.sin(x_s) / x_s**3 - s.cos(x_s) / x_s**2
    mixed = 4 * f_p**2 * eta_p + f_s**2 * (1 - eta_p)
    mixed_small = s.limit(s.limit(mixed, x_p, 0), x_s, 0)
    zero(mixed_small - (1 + 3 * eta_p) / 9)
    print(
        "Spherical-cavity stochastic P/S: j1(x)/x limit and "
        "small-cavity mixed PSD coefficient EXACT PASS"
    )

    # Surface infrasound with rigid reflection. Relative to total surface
    # pressure, the air-side surface acceleration transfer for an incident
    # direction (polar angle from the surface normal) is proportional to
    # sin(theta) * (i cos(phi), i sin(phi), 1). Isotropic incidence is a
    # uniform solid-angle distribution over the upper hemisphere.
    inc_polar, inc_azimuth = s.symbols("inc_polar inc_azimuth", real=True)
    acoustic_hvec = s.Matrix(
        [
            s.I * s.sin(inc_polar) * s.cos(inc_azimuth),
            s.I * s.sin(inc_polar) * s.sin(inc_azimuth),
            s.sin(inc_polar),
        ]
    )
    acoustic_csd = s.Matrix(
        3,
        3,
        lambda i, j: s.simplify(
            s.integrate(
                s.integrate(
                    acoustic_hvec[i]
                    * s.conjugate(acoustic_hvec[j])
                    * s.sin(inc_polar),
                    (inc_azimuth, 0, 2 * s.pi),
                ),
                (inc_polar, 0, s.pi / 2),
            )
            / (2 * s.pi)
        ),
    )
    expected_acoustic_csd = s.diag(
        s.Rational(1, 3), s.Rational(1, 3), s.Rational(2, 3)
    )
    for i in range(3):
        for j in range(3):
            zero(acoustic_csd[i, j] - expected_acoustic_csd[i, j])
    print(
        "Acoustic stochastic: exact isotropic-hemisphere surface "
        "3x3 CSD average EXACT PASS"
    )

    # Acoustic cavern spherical-volume kernel used by the centered
    # half-spherical-cavern stochastic reference. Align k with z. The angular
    # Newton integral gives 4*pi*i times the radial integral below.
    k_cav, r_cav, radius_cav = s.symbols(
        "k_cav r_cav R_cav", positive=True, real=True
    )
    x_cav = k_cav * r_cav
    radial_integrand = (
        s.sin(x_cav) - x_cav * s.cos(x_cav)
    ) / x_cav**2
    radial = s.integrate(radial_integrand, (r_cav, 0, radius_cav))
    expected_radial = (
        1 - s.sin(k_cav * radius_cav) / (k_cav * radius_cav)
    ) / k_cav
    zero(radial - expected_radial)

    # Isotropic plane-wave directions then give one third of the squared
    # scalar transfer in every Cartesian component.
    zero(p_direction_average[0, 0] - s.Rational(1, 3))
    zero(p_direction_average[1, 1] - s.Rational(1, 3))
    zero(p_direction_average[2, 2] - s.Rational(1, 3))
    c_sound, f_sound = s.symbols("c_sound f_sound", positive=True, real=True)
    k_sound = 2 * s.pi * f_sound / c_sound
    zero(4 * s.pi / (c_sound**2 * k_sound) - 2 / (c_sound * f_sound))
    x_small = s.symbols("x_small", positive=True, real=True)
    cavern_geometry = s.Rational(1, 3) * (1 - s.sin(x_small) / x_small) ** 2
    zero(
        s.limit(cavern_geometry / x_small**4, x_small, 0)
        - s.Rational(1, 108)
    )
    print(
        "Acoustic cavern stochastic: spherical-volume Newton kernel "
        "1-sinc(kR), 2G/(c_s f) prefactor reduction, isotropic 1/3 tensor, "
        "and small-kR x^4/108 PSD factor EXACT PASS"
    )

    # Uniform-advection temperature NN: the radial streamline integrals in
    # Creighton's exact one-sided single-test-mass result are positive
    # combinations of modified Bessel K functions. Verify them by
    # differentiation of the definite-integral tails.
    x_temp = s.symbols("x_temp", positive=True, real=True)
    k0 = s.besselk(0, x_temp)
    k1 = s.besselk(1, x_temp)
    k2 = s.besselk(2, x_temp)
    tail_k0 = x_temp**2 * (k1**2 - k0**2) / 2
    tail_k1 = x_temp**2 * (k0 * k2 - k1**2) / 2
    zero(s.diff(tail_k0, x_temp) + x_temp * k0**2)
    zero(s.diff(tail_k1, x_temp) + x_temp * k1**2)

    p_temp = s.symbols("p_temp", real=True)
    zero(
        s.cos(s.pi * (p_temp - 1) / 2)
        - s.sin(s.pi * p_temp / 2)
    )

    # Dimensional check for the projected acceleration PSD:
    # c_T^2 has K^2 / length^p when D_T(r)=c_T^2 r^p.
    temp = s.symbols("temperature", positive=True)
    p_dim = s.symbols("p_dim", real=True)
    ct2_dim = temp**2 / length**p_dim
    v_dim = length / time
    omega_dim = 1 / time
    accel_psd_dim = (
        (gravity_dimension * (mass / length**3) * length / temp) ** 2
        * ct2_dim
        * v_dim**p_dim
        * omega_dim ** (-(p_dim + 1))
    )
    zero(accel_psd_dim - length**2 / time**3)
    print(
        "Temperature stochastic: uniform-flow Bessel tail identities, "
        "trigonometric coefficient, and acceleration-PSD dimensions EXACT PASS"
    )


if __name__ == "__main__":
    main()
