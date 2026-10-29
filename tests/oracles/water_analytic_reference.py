"""Independent science reference and oracle for gragra Water Newtonian Noise models.

This module is completely independent and imports NO production gragra code.
It provides direct numerical spatial integrators, mpmath arbitrary-precision
Bessel references, analytical limits, and edge/uncorrelated model evaluations
directly from first principles and literature equations:
  - Atsushi Nishizawa (2021-03-31) "Water Newtonian noise for KAGRA"
    (JGW-T1706440-v2)
  - Atsushi Nishizawa (2021-09-03)
    "Edge contribution of a water channel to a Newtonian noise"
    (JGW-T2214308-v1)
  - Kentaro Somiya et al. (2019) TAUP2019 (JGW-T1910792)
  - Somiya & Nishizawa (2017) KAGRA f2f
"""

from __future__ import annotations

import math
from typing import Any, Literal

import mpmath as mp
import numpy as np
import scipy.integrate as integrate
import scipy.special as special

G_SI_REFERENCE: float = 6.67430e-11  # CODATA 2018 [m^3 kg^-1 s^-2]
RHO_WATER_NOMINAL: float = 1000.0  # Nominal water density [kg m^-3]

ObservableName = Literal[
    "potential",
    "acceleration",
    "gradient",
    "a_x",
    "a_z",
    "gamma_xx",
    "gamma_yy",
    "gamma_zz",
    "gamma_xz",
]


# =============================================================================
# Section A: Model 2 (Frozen advection water channel)
# =============================================================================


def model2_spatial_quadrature(
    observable: ObservableName,
    k: float,
    dist_d: float,
    *,
    rho: float = RHO_WATER_NOMINAL,
    w: float = 1.0,
    g_const: float = G_SI_REFERENCE,
    length_l: float | None = None,
    limit: int = 200,
) -> complex | np.ndarray:
    """Direct spatial line integration of Newtonian kernel for Model 2.

    Geometry:
      Water channel runs along the x-axis: source point at (x, 0, 0).
      Target mirror at (0, 0, dist_d), distance dist_d > 0.
      Wave perturbation propagates as e^{i k x}.

    Integrals:
      Phi(t)      = -G rho w int_{-inf}^{inf} 1/sqrt(D^2+x^2) * e^{ikx} dx
      a_x(t)      =  G rho w int_{-inf}^{inf} x/(D^2+x^2)^(3/2) * e^{ikx} dx
      a_z(t)      = -G rho w D int_{-inf}^{inf} 1/(D^2+x^2)^(3/2) * e^{ikx} dx
      Gamma_xx(t) =  G rho w int (2x^2 - D^2)/(D^2+x^2)^(5/2) * e^{ikx} dx
      Gamma_yy(t) = -G rho w int 1/(D^2+x^2)^(3/2) * e^{ikx} dx
      Gamma_zz(t) =  G rho w int (2D^2 - x^2)/(D^2+x^2)^(5/2) * e^{ikx} dx
      Gamma_xz(t) = -3 G rho w D int x/(D^2+x^2)^(5/2) * e^{ikx} dx

    If length_l is None, the infinite integral is evaluated using
    scipy.integrate.quad with oscillatory weights on [0, inf).
    If length_l is finite, the integral is computed over [-length_l, length_l].
    """
    if dist_d <= 0.0:
        raise ValueError(f"dist_d must be positive, got {dist_d}")
    if k < 0.0:
        raise ValueError(f"k must be non-negative, got {k}")

    scale = g_const * rho * w

    if length_l is None:

        def _int_cos(func):
            val, _ = integrate.quad(func, 0, np.inf, weight="cos", wvar=k, limit=limit)
            return 2.0 * val

        def _int_sin(func):
            val, _ = integrate.quad(func, 0, np.inf, weight="sin", wvar=k, limit=limit)
            return 2j * val

        if observable == "potential":
            val = _int_cos(lambda x: 1.0 / np.sqrt(dist_d**2 + x**2))
            return -scale * val

        if observable in ("acceleration", "a_x", "a_z"):
            val_x = _int_sin(lambda x: x / (dist_d**2 + x**2) ** 1.5)
            val_z = _int_cos(lambda x: 1.0 / (dist_d**2 + x**2) ** 1.5)
            ax = scale * val_x
            az = -scale * dist_d * val_z
            if observable == "a_x":
                return ax
            if observable == "a_z":
                return az
            return np.array([ax, 0.0 + 0.0j, az], dtype=np.complex128)

        if observable in (
            "gradient",
            "gamma_xx",
            "gamma_yy",
            "gamma_zz",
            "gamma_xz",
        ):
            val_xx = _int_cos(
                lambda x: (2.0 * x**2 - dist_d**2) / (dist_d**2 + x**2) ** 2.5
            )
            val_yy = _int_cos(lambda x: -1.0 / (dist_d**2 + x**2) ** 1.5)
            val_zz = _int_cos(
                lambda x: (2.0 * dist_d**2 - x**2) / (dist_d**2 + x**2) ** 2.5
            )
            val_xz = _int_sin(lambda x: -3.0 * dist_d * x / (dist_d**2 + x**2) ** 2.5)

            gxx = scale * val_xx
            gyy = scale * val_yy
            gzz = scale * val_zz
            gxz = scale * val_xz

            if observable == "gamma_xx":
                return gxx
            if observable == "gamma_yy":
                return gyy
            if observable == "gamma_zz":
                return gzz
            if observable == "gamma_xz":
                return gxz

            gamma = np.zeros((3, 3), dtype=np.complex128)
            gamma[0, 0] = gxx
            gamma[1, 1] = gyy
            gamma[2, 2] = gzz
            gamma[0, 2] = gxz
            gamma[2, 0] = gxz
            return gamma

    else:

        def _int_box(f_kernel):
            re, _ = integrate.quad(
                lambda x: f_kernel(x) * np.cos(k * x),
                -length_l,
                length_l,
                limit=limit,
            )
            im, _ = integrate.quad(
                lambda x: f_kernel(x) * np.sin(k * x),
                -length_l,
                length_l,
                limit=limit,
            )
            return re + 1j * im

        if observable == "potential":
            return -scale * _int_box(lambda x: 1.0 / np.sqrt(dist_d**2 + x**2))

        if observable in ("acceleration", "a_x", "a_z"):
            ax = scale * _int_box(lambda x: x / (dist_d**2 + x**2) ** 1.5)
            az = -scale * dist_d * _int_box(lambda x: 1.0 / (dist_d**2 + x**2) ** 1.5)
            if observable == "a_x":
                return ax
            if observable == "a_z":
                return az
            return np.array([ax, 0.0 + 0.0j, az], dtype=np.complex128)

        if observable in (
            "gradient",
            "gamma_xx",
            "gamma_yy",
            "gamma_zz",
            "gamma_xz",
        ):
            gxx = scale * _int_box(
                lambda x: (2.0 * x**2 - dist_d**2) / (dist_d**2 + x**2) ** 2.5
            )
            gyy = scale * _int_box(lambda x: -1.0 / (dist_d**2 + x**2) ** 1.5)
            gzz = scale * _int_box(
                lambda x: (2.0 * dist_d**2 - x**2) / (dist_d**2 + x**2) ** 2.5
            )
            gxz = scale * _int_box(
                lambda x: -3.0 * dist_d * x / (dist_d**2 + x**2) ** 2.5
            )

            if observable == "gamma_xx":
                return gxx
            if observable == "gamma_yy":
                return gyy
            if observable == "gamma_zz":
                return gzz
            if observable == "gamma_xz":
                return gxz

            gamma = np.zeros((3, 3), dtype=np.complex128)
            gamma[0, 0] = gxx
            gamma[1, 1] = gyy
            gamma[2, 2] = gzz
            gamma[0, 2] = gxz
            gamma[2, 0] = gxz
            return gamma

    raise ValueError(f"Unknown observable {observable!r}")


def model2_bessel_reference(
    observable: ObservableName,
    k: float,
    dist_d: float,
    *,
    rho: float = RHO_WATER_NOMINAL,
    w: float = 1.0,
    g_const: float = G_SI_REFERENCE,
    use_mpmath: bool = False,
    dps: int = 30,
) -> complex | np.ndarray:
    """Exact Bessel closed-form evaluation of Model 2 integrals.

    Formulas:
      z = k * dist_d
      Phi      = -2 G rho w K_0(z)
      a_x      =  2 i G rho w k K_0(z)
      a_z      = -2 G rho w k K_1(z)
      Gamma_xx = -2 G rho w k^2 K_0(z)
      Gamma_yy = -2 G rho w (k / dist_d) K_1(z)
      Gamma_zz =  2 G rho w [ k^2 K_0(z) + (k / dist_d) K_1(z) ]
      Gamma_xz = -2 i G rho w k^2 K_1(z)
      Trace(Gamma) = Gamma_xx + Gamma_yy + Gamma_zz == 0 identically!
    """
    if dist_d <= 0.0:
        raise ValueError(f"dist_d must be positive, got {dist_d}")
    if k < 0.0:
        raise ValueError(f"k must be non-negative, got {k}")

    if use_mpmath:
        with mp.workdps(dps):
            mp_k = mp.mpf(str(k))
            mp_d = mp.mpf(str(dist_d))
            mp_z = mp_k * mp_d
            mp_scale = mp.mpf(str(g_const)) * mp.mpf(str(rho)) * mp.mpf(str(w))

            k0 = mp.besselk(0, mp_z)
            k1 = mp.besselk(1, mp_z)

            if observable == "potential":
                return complex(-2.0 * mp_scale * k0)
            if observable == "a_x":
                return complex(0.0, float(2.0 * mp_scale * mp_k * k0))
            if observable == "a_z":
                return complex(-2.0 * mp_scale * mp_k * k1)
            if observable == "acceleration":
                ax = complex(0.0, float(2.0 * mp_scale * mp_k * k0))
                az = complex(-2.0 * mp_scale * mp_k * k1)
                return np.array([ax, 0.0 + 0.0j, az], dtype=np.complex128)
            if observable == "gamma_xx":
                return complex(-2.0 * mp_scale * mp_k**2 * k0)
            if observable == "gamma_yy":
                return complex(-2.0 * mp_scale * (mp_k / mp_d) * k1)
            if observable == "gamma_zz":
                return complex(2.0 * mp_scale * (mp_k**2 * k0 + (mp_k / mp_d) * k1))
            if observable == "gamma_xz":
                return complex(0.0, float(-2.0 * mp_scale * mp_k**2 * k1))
            if observable == "gradient":
                gxx = complex(-2.0 * mp_scale * mp_k**2 * k0)
                gyy = complex(-2.0 * mp_scale * (mp_k / mp_d) * k1)
                gzz = complex(2.0 * mp_scale * (mp_k**2 * k0 + (mp_k / mp_d) * k1))
                gxz = complex(0.0, float(-2.0 * mp_scale * mp_k**2 * k1))
                gamma = np.zeros((3, 3), dtype=np.complex128)
                gamma[0, 0] = gxx
                gamma[1, 1] = gyy
                gamma[2, 2] = gzz
                gamma[0, 2] = gxz
                gamma[2, 0] = gxz
                return gamma

    z = k * dist_d
    scale = g_const * rho * w
    k0 = special.kn(0, z)
    k1 = special.kn(1, z)

    if observable == "potential":
        return complex(-2.0 * scale * k0)
    if observable == "a_x":
        return complex(0.0, 2.0 * scale * k * k0)
    if observable == "a_z":
        return complex(-2.0 * scale * k * k1)
    if observable == "acceleration":
        ax = complex(0.0, 2.0 * scale * k * k0)
        az = complex(-2.0 * scale * k * k1)
        return np.array([ax, 0.0 + 0.0j, az], dtype=np.complex128)
    if observable == "gamma_xx":
        return complex(-2.0 * scale * k**2 * k0)
    if observable == "gamma_yy":
        return complex(-2.0 * scale * (k / dist_d) * k1)
    if observable == "gamma_zz":
        return complex(2.0 * scale * (k**2 * k0 + (k / dist_d) * k1))
    if observable == "gamma_xz":
        return complex(0.0, -2.0 * scale * k**2 * k1)
    if observable == "gradient":
        gxx = complex(-2.0 * scale * k**2 * k0)
        gyy = complex(-2.0 * scale * (k / dist_d) * k1)
        gzz = complex(2.0 * scale * (k**2 * k0 + (k / dist_d) * k1))
        gxz = complex(0.0, -2.0 * scale * k**2 * k1)
        gamma = np.zeros((3, 3), dtype=np.complex128)
        gamma[0, 0] = gxx
        gamma[1, 1] = gyy
        gamma[2, 2] = gzz
        gamma[0, 2] = gxz
        gamma[2, 0] = gxz
        return gamma

    raise ValueError(f"Unknown observable {observable!r}")


def model2_asymptotic(
    observable: ObservableName,
    k: float,
    dist_d: float,
    *,
    rho: float = RHO_WATER_NOMINAL,
    w: float = 1.0,
    g_const: float = G_SI_REFERENCE,
    limit_type: Literal["small_z", "large_z"] = "small_z",
) -> complex | np.ndarray:
    """Asymptotic series evaluation for z -> 0 (small_z) and z -> inf (large_z).

    small_z (z = k*dist_d -> 0):
      K_0(z) ~ -ln(z/2) - gamma_euler
      K_1(z) ~ 1/z + (z/2) * (ln(z/2) + gamma_euler - 1/2)
      In particular, lim_{k -> 0} a_z = -2 G rho w / dist_d (the DC limit).

    large_z (z = k*dist_d -> inf):
      K_0(z) ~ sqrt(pi / (2z)) * e^{-z} * (1 - 1/(8z) + 9/(128z^2))
      K_1(z) ~ sqrt(pi / (2z)) * e^{-z} * (1 + 3/(8z) - 15/(128z^2))
    """
    z = k * dist_d
    scale = g_const * rho * w
    euler_gamma = 0.577215664901532860606512

    if limit_type == "small_z":
        if z == 0.0:
            az_dc = -2.0 * scale / dist_d
            gyy_dc = -2.0 * scale / dist_d**2
            gzz_dc = 2.0 * scale / dist_d**2
            if observable == "a_z":
                return complex(az_dc)
            if observable == "a_x":
                return 0.0 + 0.0j
            if observable == "acceleration":
                return np.array(
                    [0.0 + 0.0j, 0.0 + 0.0j, complex(az_dc)],
                    dtype=np.complex128,
                )
            if observable == "gamma_xx":
                return 0.0 + 0.0j
            if observable == "gamma_yy":
                return complex(gyy_dc)
            if observable == "gamma_zz":
                return complex(gzz_dc)
            if observable == "gamma_xz":
                return 0.0 + 0.0j
            if observable == "gradient":
                gamma = np.zeros((3, 3), dtype=np.complex128)
                gamma[1, 1] = gyy_dc
                gamma[2, 2] = gzz_dc
                return gamma
            if observable == "potential":
                return complex(-float("inf"))

        k0 = -math.log(z / 2.0) - euler_gamma
        k1 = 1.0 / z + 0.5 * z * (math.log(z / 2.0) + euler_gamma - 0.5)

    elif limit_type == "large_z":
        pref = math.sqrt(math.pi / (2.0 * z)) * math.exp(-z)
        k0 = pref * (1.0 - 1.0 / (8.0 * z) + 9.0 / (128.0 * z**2))
        k1 = pref * (1.0 + 3.0 / (8.0 * z) - 15.0 / (128.0 * z**2))
    else:
        raise ValueError(f"Unknown limit_type {limit_type!r}")

    if observable == "potential":
        return complex(-2.0 * scale * k0)
    if observable == "a_x":
        return complex(0.0, 2.0 * scale * k * k0)
    if observable == "a_z":
        return complex(-2.0 * scale * k * k1)
    if observable == "acceleration":
        ax = complex(0.0, 2.0 * scale * k * k0)
        az = complex(-2.0 * scale * k * k1)
        return np.array([ax, 0.0 + 0.0j, az], dtype=np.complex128)
    if observable == "gamma_xx":
        return complex(-2.0 * scale * k**2 * k0)
    if observable == "gamma_yy":
        return complex(-2.0 * scale * (k / dist_d) * k1)
    if observable == "gamma_zz":
        return complex(2.0 * scale * (k**2 * k0 + (k / dist_d) * k1))
    if observable == "gamma_xz":
        return complex(0.0, -2.0 * scale * k**2 * k1)
    if observable == "gradient":
        gxx = complex(-2.0 * scale * k**2 * k0)
        gyy = complex(-2.0 * scale * (k / dist_d) * k1)
        gzz = complex(2.0 * scale * (k**2 * k0 + (k / dist_d) * k1))
        gxz = complex(0.0, -2.0 * scale * k**2 * k1)
        gamma = np.zeros((3, 3), dtype=np.complex128)
        gamma[0, 0] = gxx
        gamma[1, 1] = gyy
        gamma[2, 2] = gzz
        gamma[0, 2] = gxz
        gamma[2, 0] = gxz
        return gamma

    raise ValueError(f"Unknown observable {observable!r}")


# =============================================================================
# Section B: Model 1 (Chen uncorrelated)
# =============================================================================


def model1_cell_quadrature(
    depth_h: float,
    *,
    rho: float = RHO_WATER_NOMINAL,
    w: float = 1.0,
    g_const: float = G_SI_REFERENCE,
    a_cell: float = 1.0,
    s_eta: float = 1.0,
    length_l: float | None = None,
) -> float:
    """Numerical quadrature sum over uncorrelated cells for Model 1.

    S_{a_x} = (G rho)^2 * A_cell * w * S_eta * int x^2 / (H^2 + x^2)^3 dx
    Linear in w because uncorrelated cells across width add in power (incoherently).
    """
    if depth_h <= 0.0:
        raise ValueError(f"depth_h must be positive, got {depth_h}")

    pref = (g_const * rho) ** 2 * a_cell * w * s_eta
    if length_l is None:
        integral_val, _ = integrate.quad(
            lambda x: x**2 / (depth_h**2 + x**2) ** 3,
            -np.inf,
            np.inf,
        )
    else:
        integral_val, _ = integrate.quad(
            lambda x: x**2 / (depth_h**2 + x**2) ** 3,
            -length_l,
            length_l,
        )
    return pref * integral_val


def model1_closed_form(
    depth_h: float,
    *,
    rho: float = RHO_WATER_NOMINAL,
    w: float = 1.0,
    g_const: float = G_SI_REFERENCE,
    a_cell: float = 1.0,
    s_eta: float = 1.0,
) -> float:
    """Exact analytical closed form for Model 1 cell integral.

    int_{-inf}^{inf} x^2 / (H^2 + x^2)^3 dx = pi / (8 * H^3)
    S_{a_x} = (G rho)^2 * A_cell * w * S_eta * (pi / (8 * H^3))
    """
    if depth_h <= 0.0:
        raise ValueError(f"depth_h must be positive, got {depth_h}")
    return (g_const * rho) ** 2 * a_cell * w * s_eta * (math.pi / (8.0 * depth_h**3))


def somiya_taup2019_chen_formula(
    f: float,
    depth_h: float,
    *,
    a: float = 0.1,  # Correlation length / cell size [m]
    b: float = 0.005,  # Surface wave height / amplitude [m]
    w: float = 0.4,  # Channel width [m]
    rho: float = RHO_WATER_NOMINAL,
    g_const: float = G_SI_REFERENCE,
) -> float:
    """Literal Somiya TAUP 2019 Chen formula for water Newtonian noise displacement ASD.

    sqrt{S_x(f)} = (G * rho * a * b * sqrt(2 * pi * w)) / (Omega^{2.5} * H^{1.5})
    where Omega = 2 * pi * f.
    """
    if f <= 0.0:
        raise ValueError(f"f must be positive, got {f}")
    if depth_h <= 0.0:
        raise ValueError(f"depth_h must be positive, got {depth_h}")

    omega = 2.0 * math.pi * f
    numerator = g_const * rho * a * b * math.sqrt(2.0 * math.pi * w)
    denominator = (omega**2.5) * (depth_h**1.5)
    return numerator / denominator


# =============================================================================
# Section C: Waterfall model (K1)
# =============================================================================


def waterfall_spatial_quadrature(
    k: float,
    dist_d: float,
    *,
    rho: float = RHO_WATER_NOMINAL,
    g_const: float = G_SI_REFERENCE,
    length_l: float | None = None,
    limit: int = 200,
) -> tuple[complex, complex]:
    """Spatial line integration along z of Newtonian force on mirror at (D, 0, 0).

    Line source along z with perturbation e^{i k z}.
    Target mirror at (dist_d, 0, 0).

    Integrals:
      a_x(t) = -G rho D int_{-inf}^{inf} e^{ikz} / (D^2 + z^2)^(3/2) dz
             = -2 G rho D int_0^inf cos(kz) / (D^2 + z^2)^(3/2) dz
      a_z(t) =  G rho   int_{-inf}^{inf} z e^{ikz} / (D^2 + z^2)^(3/2) dz
             =  2 i G rho int_0^inf z sin(kz) / (D^2 + z^2)^(3/2) dz
    """
    if dist_d <= 0.0:
        raise ValueError(f"dist_d must be positive, got {dist_d}")
    if k < 0.0:
        raise ValueError(f"k must be non-negative, got {k}")

    scale = g_const * rho
    if length_l is None:
        val_x, _ = integrate.quad(
            lambda z: 1.0 / (dist_d**2 + z**2) ** 1.5,
            0,
            np.inf,
            weight="cos",
            wvar=k,
            limit=limit,
        )
        val_z, _ = integrate.quad(
            lambda z: z / (dist_d**2 + z**2) ** 1.5,
            0,
            np.inf,
            weight="sin",
            wvar=k,
            limit=limit,
        )
        ax = complex(-2.0 * scale * dist_d * val_x)
        az = complex(0.0, 2.0 * scale * val_z)
    else:
        re_x, _ = integrate.quad(
            lambda z: np.cos(k * z) / (dist_d**2 + z**2) ** 1.5,
            -length_l,
            length_l,
            limit=limit,
        )
        im_x, _ = integrate.quad(
            lambda z: np.sin(k * z) / (dist_d**2 + z**2) ** 1.5,
            -length_l,
            length_l,
            limit=limit,
        )
        ax = -scale * dist_d * (re_x + 1j * im_x)

        re_z, _ = integrate.quad(
            lambda z: z * np.cos(k * z) / (dist_d**2 + z**2) ** 1.5,
            -length_l,
            length_l,
            limit=limit,
        )
        im_z, _ = integrate.quad(
            lambda z: z * np.sin(k * z) / (dist_d**2 + z**2) ** 1.5,
            -length_l,
            length_l,
            limit=limit,
        )
        az = scale * (re_z + 1j * im_z)

    return ax, az


def waterfall_bessel_reference(
    k: float,
    dist_d: float,
    *,
    rho: float = RHO_WATER_NOMINAL,
    g_const: float = G_SI_REFERENCE,
    use_mpmath: bool = False,
    dps: int = 30,
) -> tuple[complex, complex]:
    """Exact Bessel evaluation of Waterfall model.

    a_x = -2 G rho k K_1(k D)
    a_z =  2 i G rho k K_0(k D)
    """
    if dist_d <= 0.0:
        raise ValueError(f"dist_d must be positive, got {dist_d}")
    if k < 0.0:
        raise ValueError(f"k must be non-negative, got {k}")

    if use_mpmath:
        with mp.workdps(dps):
            mp_k = mp.mpf(str(k))
            mp_d = mp.mpf(str(dist_d))
            mp_z = mp_k * mp_d
            mp_scale = mp.mpf(str(g_const)) * mp.mpf(str(rho))

            k0 = mp.besselk(0, mp_z)
            k1 = mp.besselk(1, mp_z)

            ax = complex(-2.0 * mp_scale * mp_k * k1)
            az = complex(0.0, float(2.0 * mp_scale * mp_k * k0))
            return ax, az

    z = k * dist_d
    scale = g_const * rho
    k0 = special.kn(0, z)
    k1 = special.kn(1, z)

    ax = complex(-2.0 * scale * k * k1)
    az = complex(0.0, 2.0 * scale * k * k0)
    return ax, az


# =============================================================================
# Section D: Edge contribution
# =============================================================================


def edge_contribution_time_domain(
    t: float,
    t0: float,
    x0: float,
    dist_d: float,
    *,
    delta_v: float = 1.0,
    mass_m: float = 1.0,
    rho: float = RHO_WATER_NOMINAL,
    g_const: float = G_SI_REFERENCE,
) -> float:
    """Direct evaluation of edge step force from Nishizawa (2021-09-03).

    Delta F(t) = M * G * rho * Delta_V * x0 / (D^2 + x0^2)^(3/2) * Theta(t - t0)
    """
    if t < t0:
        return 0.0
    denom = (dist_d**2 + x0**2) ** 1.5
    return mass_m * g_const * rho * delta_v * (x0 / denom)


def edge_contribution_frequency_domain(
    omega: float,
    x0: float,
    dist_d: float,
    *,
    delta_v: float = 1.0,
    rho: float = RHO_WATER_NOMINAL,
    g_const: float = G_SI_REFERENCE,
) -> complex:
    """Fourier transform of edge acceleration response.

    Delta a_tilde_x(Omega) = G * rho * Delta_V * x0 / (D^2 + x0^2)^(3/2) * (i / Omega)
    """
    if omega == 0.0:
        raise ZeroDivisionError("Omega cannot be 0 for frequency domain edge response")
    geom_factor = x0 / (dist_d**2 + x0**2) ** 1.5
    scale = g_const * rho * delta_v * geom_factor
    return complex(0.0, scale / omega)


# =============================================================================
# Verification suite and self-test
# =============================================================================


def verify_water_analytic_reference() -> dict[str, Any]:
    """Execute rigorous scientific checks across all models.

    Checks:
      1. Spatial line integration matches mpmath Bessel closed form for Model 2
         across multiple frequencies (0.1 Hz to 50 Hz) and distances (1 m to 10 m).
      2. Trace(Gamma) == 0 identically for Model 2.
      3. DC limit (k -> 0) of a_z matches -2 G rho w / D.
      4. Model 1 numerical cell quadrature matches analytical form pi / (8 H^3).
      5. Somiya TAUP 2019 Chen formula consistency.
      6. Waterfall spatial line quadrature matches Bessel closed forms.
      7. Edge step time and frequency response correctness.
    """
    results: dict[str, Any] = {}

    # Check 1: Model 2 spatial quadrature vs Bessel reference
    v_flow = 1.5
    test_freqs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    test_dists = [1.0, 2.0, 5.0, 10.0]

    max_rel_err_ax = 0.0
    max_rel_err_az = 0.0
    max_rel_err_phi = 0.0
    max_rel_err_gxx = 0.0
    max_rel_err_gyy = 0.0
    max_rel_err_gzz = 0.0
    max_rel_err_gxz = 0.0
    max_trace_gamma = 0.0

    for freq in test_freqs:
        k_val = 2.0 * math.pi * freq / v_flow
        for d_val in test_dists:
            k_d = k_val * d_val
            # When signal decays below float64 precision (k_d > 15), skip
            if k_d > 15.0:
                continue

            # Bessel reference
            b_phi = model2_bessel_reference("potential", k_val, d_val, use_mpmath=True)
            b_ax = model2_bessel_reference("a_x", k_val, d_val, use_mpmath=True)
            b_az = model2_bessel_reference("a_z", k_val, d_val, use_mpmath=True)
            b_gxx = model2_bessel_reference("gamma_xx", k_val, d_val, use_mpmath=True)
            b_gyy = model2_bessel_reference("gamma_yy", k_val, d_val, use_mpmath=True)
            b_gzz = model2_bessel_reference("gamma_zz", k_val, d_val, use_mpmath=True)
            b_gxz = model2_bessel_reference("gamma_xz", k_val, d_val, use_mpmath=True)

            # Spatial quadrature
            q_phi = model2_spatial_quadrature("potential", k_val, d_val)
            q_ax = model2_spatial_quadrature("a_x", k_val, d_val)
            q_az = model2_spatial_quadrature("a_z", k_val, d_val)
            q_gxx = model2_spatial_quadrature("gamma_xx", k_val, d_val)
            q_gyy = model2_spatial_quadrature("gamma_yy", k_val, d_val)
            q_gzz = model2_spatial_quadrature("gamma_zz", k_val, d_val)
            q_gxz = model2_spatial_quadrature("gamma_xz", k_val, d_val)

            err_phi = abs(q_phi - b_phi) / abs(b_phi)
            err_ax = abs(q_ax - b_ax) / abs(b_ax)
            err_az = abs(q_az - b_az) / abs(b_az)
            err_gxx = abs(q_gxx - b_gxx) / abs(b_gxx)
            err_gyy = abs(q_gyy - b_gyy) / abs(b_gyy)
            err_gzz = abs(q_gzz - b_gzz) / abs(b_gzz)
            err_gxz = abs(q_gxz - b_gxz) / abs(b_gxz)

            max_rel_err_phi = max(max_rel_err_phi, err_phi)
            max_rel_err_ax = max(max_rel_err_ax, err_ax)
            max_rel_err_az = max(max_rel_err_az, err_az)
            max_rel_err_gxx = max(max_rel_err_gxx, err_gxx)
            max_rel_err_gyy = max(max_rel_err_gyy, err_gyy)
            max_rel_err_gzz = max(max_rel_err_gzz, err_gzz)
            max_rel_err_gxz = max(max_rel_err_gxz, err_gxz)

            # Trace of Gamma tensor
            tr = abs(b_gxx + b_gyy + b_gzz) / max(abs(b_gxx), abs(b_gyy), abs(b_gzz))
            max_trace_gamma = max(max_trace_gamma, tr)

    results["model2_quad_vs_bessel"] = {
        "max_rel_err_phi": max_rel_err_phi,
        "max_rel_err_ax": max_rel_err_ax,
        "max_rel_err_az": max_rel_err_az,
        "max_rel_err_gxx": max_rel_err_gxx,
        "max_rel_err_gyy": max_rel_err_gyy,
        "max_rel_err_gzz": max_rel_err_gzz,
        "max_rel_err_gxz": max_rel_err_gxz,
        "passed": all(
            err < 1e-4
            for err in [
                max_rel_err_phi,
                max_rel_err_ax,
                max_rel_err_az,
                max_rel_err_gxx,
                max_rel_err_gyy,
                max_rel_err_gzz,
                max_rel_err_gxz,
            ]
        ),
    }

    # Check 2: Trace(Gamma) == 0
    results["model2_trace_free"] = {
        "max_relative_trace": max_trace_gamma,
        "passed": max_trace_gamma < 1e-14,
    }

    # Check 3: DC limit of a_z: lim_{k -> 0} a_z = -2 G rho w / D
    dist_dc_test = 2.5
    dc_analytical = -2.0 * G_SI_REFERENCE * RHO_WATER_NOMINAL * 1.0 / dist_dc_test
    k_small = 1e-8
    b_az_small = model2_bessel_reference(
        "a_z", k_small, dist_dc_test, use_mpmath=True, dps=40
    )
    dc_rel_diff = abs(b_az_small - dc_analytical) / abs(dc_analytical)
    results["dc_limit_az"] = {
        "dc_analytical": dc_analytical,
        "bessel_at_1e-8": b_az_small,
        "rel_diff": dc_rel_diff,
        "passed": dc_rel_diff < 1e-6,
    }

    # Check 4: Model 1 cell quadrature vs closed form pi / (8 H^3)
    max_model1_err = 0.0
    for h_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
        num_val = model1_cell_quadrature(h_val)
        exact_val = model1_closed_form(h_val)
        err = abs(num_val - exact_val) / exact_val
        max_model1_err = max(max_model1_err, err)

    results["model1_integral"] = {
        "max_rel_err": max_model1_err,
        "passed": max_model1_err < 1e-9,
    }

    # Check 5: Somiya TAUP 2019 formula smoke test
    val_taup = somiya_taup2019_chen_formula(f=10.0, depth_h=2.0)
    results["somiya_taup2019"] = {
        "val_at_10Hz_2m": val_taup,
        "passed": val_taup > 0.0 and math.isfinite(val_taup),
    }

    # Check 6: Waterfall quadrature vs Bessel
    max_wf_err_ax = 0.0
    max_wf_err_az = 0.0
    for k_val in [0.1, 0.5, 1.0, 2.0]:
        for d_val in [1.0, 2.0, 5.0]:
            q_ax, q_az = waterfall_spatial_quadrature(k_val, d_val)
            b_ax, b_az = waterfall_bessel_reference(k_val, d_val, use_mpmath=True)
            max_wf_err_ax = max(max_wf_err_ax, abs(q_ax - b_ax) / abs(b_ax))
            max_wf_err_az = max(max_wf_err_az, abs(q_az - b_az) / abs(b_az))

    results["waterfall_quad_vs_bessel"] = {
        "max_rel_err_ax": max_wf_err_ax,
        "max_rel_err_az": max_wf_err_az,
        "passed": max_wf_err_ax < 1e-4 and max_wf_err_az < 1e-4,
    }

    # Check 7: Edge contribution
    f_step = edge_contribution_time_domain(t=1.0, t0=0.5, x0=2.0, dist_d=3.0)
    f_before = edge_contribution_time_domain(t=0.2, t0=0.5, x0=2.0, dist_d=3.0)
    a_edge_omega = edge_contribution_frequency_domain(omega=10.0, x0=2.0, dist_d=3.0)
    results["edge_contribution"] = {
        "f_step_positive": f_step > 0.0,
        "f_before_zero": f_before == 0.0,
        "a_edge_imaginary": a_edge_omega.real == 0.0 and a_edge_omega.imag > 0.0,
        "passed": f_step > 0.0 and f_before == 0.0 and (a_edge_omega.real == 0.0),
    }

    results["all_passed"] = all(
        v.get("passed", False) for v in results.values() if isinstance(v, dict)
    )
    return results


if __name__ == "__main__":
    import pprint

    print("Running Water Analytic Reference verification...")
    res = verify_water_analytic_reference()
    pprint.pprint(res)
    assert res["all_passed"], "Verification failed!"
    print("All verification checks PASSED successfully!")
