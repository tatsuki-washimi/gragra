"""Analytic gravity of traction-free incident/reflected P/SV half-space waves.

The medium occupies z < 0; the surface itself is excluded. All spatial
derivatives are analytic and all outputs are G-free. See
``docs/design/halfspace-general-psv.md`` for the potential convention,
traction derivation, evanescent branch, and degenerate-domain exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from gragra._arrays import as_xyz_vector
from gragra.analytic.bodywave_literature import (
    _assemble_observables,
    _horizontal_phase,
    _reject_non_real_dtype,
    _validate_amplitude,
    _validate_azimuth,
    _validate_frequency,
    _validate_incidence_angle,
    _validate_non_negative_scalar,
    _validate_targets_and_split,
)
from gragra.analytic.values import GFreeGravityValues
from gragra.waves.rayleigh import _validate_speeds


@dataclass(frozen=True)
class HalfspacePSVReflection:
    """Dimensionless reflection data in the Helmholtz potential convention.

    ``u = grad(phi) + curl(psi * e_transverse)``; upward incident SV has
    displacement direction ``cos(theta)*h - sin(theta)*e_z``. The two
    potential ratios divide the reflected scalar potentials by the incident
    scalar potential (P: phi; SV: psi), not by displacement in meters.
    ``p_over_ks`` has nonnegative imaginary part; reflected P uses ``kz=-p``.
    Propagating reflected displacement ratios are also supplied, with
    polarizations ``(q*h-p*e_z)/kP`` and ``(s*h+q*e_z)/kS`` respectively.
    For evanescent P that polarization is complex, not a unit vector under
    the Hermitian norm, so its coefficient is not a displacement magnitude.
    ``incident_plus_reflected_potential_ratio`` is the stable evaluation of
    one plus the same-mode potential ratio, used to avoid cancellation.
    """

    q_over_ks: float
    p_over_ks: complex
    s_over_ks: float
    reflected_p_potential_ratio: complex
    reflected_sv_potential_ratio: complex
    reflected_p_displacement_ratio: complex
    reflected_sv_displacement_ratio: complex
    incident_plus_reflected_potential_ratio: complex


def halfspace_psv_reflection(
    *,
    wave_type: Literal["P", "SV"],
    v_p_m_s: float,
    v_s_m_s: float,
    incidence_angle_rad: float,
) -> HalfspacePSVReflection:
    """Return lossless free-surface P/SV reflection and conversion ratios.

    Angles use the incident mode's speed, measured from +z, in [0, pi/2).
    Normal and ordinary critical incidence are supported analytically.
    Exact grazing is excluded. At zero Lame lambda and exact SV critical
    incidence the traction system loses rank: a horizontal P mode is
    undetermined. This isolated degenerate configuration raises ValueError.
    """
    if wave_type not in ("P", "SV"):
        raise ValueError("wave_type must be 'P' or 'SV'")
    vp, vs = _validate_speeds(v_p_m_s, v_s_m_s)
    if not vp * vp > (4.0 / 3.0) * vs * vs:
        raise ValueError("requires v_p_m_s**2 > (4/3) * v_s_m_s**2")
    theta = _validate_incidence_angle("incidence_angle_rad", incidence_angle_rad)
    ratio = vs / vp
    q = (ratio if wave_type == "P" else 1.0) * np.sin(theta)
    # Exact critical is defined by its input angle, avoiding a rounded
    # negative radicand at the branch point. No finite near-critical band
    # is projected onto critical incidence.
    critical = wave_type == "SV" and theta == np.arcsin(ratio)
    p = (
        complex(ratio * np.cos(theta))
        if wave_type == "P"
        else (0j if critical else np.sqrt(complex((ratio - q) * (ratio + q))))
    )
    s = np.cos(theta) if wave_type == "SV" else np.sqrt((1 - q) * (1 + q))
    d = 1 - 2 * q * q
    if critical and abs(d) <= 8 * np.finfo(float).eps:
        raise ValueError(
            "degenerate zero-lambda SV critical incidence has nonunique reflection"
        )
    den = d * d + 4 * q * q * p * s
    same = (4 * q * q * p * s - d * d) / den
    if wave_type == "P":
        b, converted = same, 4 * q * p * d / den
        rp, rs = b, converted / ratio
    else:
        b, converted = -4 * q * s * d / den, same
        rp, rs = -ratio * b, -converted
    return HalfspacePSVReflection(
        float(q),
        complex(p),
        float(s),
        complex(b),
        complex(converted),
        complex(rp),
        complex(rs),
        complex(8 * q * q * p * s / den),
    )


def halfspace_psv_observables_g_free(
    targets_m,
    *,
    wave_type: Literal["P", "SV"],
    density_kg_m3: float,
    v_p_m_s: float,
    v_s_m_s: float,
    frequency_hz,
    incidence_angle_rad: float,
    azimuth_rad: float,
    incident_displacement_m,
    phase_origin_m=(0.0, 0.0, 0.0),
) -> GFreeGravityValues:
    """Ground/underground potential, acceleration and gradient, G-free.

    Incident P polarization is sin(theta)*h+cos(theta)*e_z; incident SV
    polarization is cos(theta)*h-sin(theta)*e_z. Complex displacement is
    referenced at the fixed phase origin, using exp(i*k.x-i*omega*t).
    Reflection coefficients refer all legs to the surface once. The surface
    remains z=0 for every origin. Converted evanescent P decays as z -> -inf.

    ``targets_m`` is (M,3), z != 0. Positive ``frequency_hz`` is scalar or
    (K,); displacement is scalar or matching (K,). Outputs have shapes
    (M,), (M,3), (M,3,3), or (K,M), (K,M,3), (K,M,3,3), respectively.
    Units are kg/m, kg/m^2, kg/m^3; arrays are read-only complex128.
    Stable isotropic speeds and nonnegative density are required. At q=0
    the continuous-q gauge sets the upper constant potential to zero.
    The lower oscillating potential is retained. Exact grazing and the
    isolated zero-lambda SV critical configuration are excluded.
    """
    targets, z = _validate_targets_and_split(targets_m, allow_underground=True)
    reflection = halfspace_psv_reflection(
        wave_type=wave_type,
        v_p_m_s=v_p_m_s,
        v_s_m_s=v_s_m_s,
        incidence_angle_rad=incidence_angle_rad,
    )
    rho = _validate_non_negative_scalar("density_kg_m3", density_kg_m3)
    azimuth = _validate_azimuth("azimuth_rad", azimuth_rad)
    freq = _validate_frequency("frequency_hz", frequency_hz)
    amp = _validate_amplitude(
        "incident_displacement_m",
        incident_displacement_m,
        k_len=freq.size if freq.ndim else None,
    )
    _reject_non_real_dtype("phase_origin_m", phase_origin_m)
    origin = as_xyz_vector("phase_origin_m", phase_origin_m)
    ks = 2 * np.pi * freq / v_s_m_s
    if freq.ndim:
        ks = ks[:, None]
    if amp.ndim:
        amp = amp[:, None]
    kp = ks * (v_s_m_s / v_p_m_s)
    q, p = ks * reflection.q_over_ks, ks * reflection.p_over_ks
    s = ks * reflection.s_over_ks
    inc_kz = p if wave_type == "P" else s
    amp = amp * np.exp(-1j * inc_kz * origin[2])
    # A,B denote P scalar potentials; C,D denote SV scalar potentials.
    # Multiplication by i*k converts each potential to displacement.
    incident = amp / (1j * kp) if wave_type == "P" else -amp / (1j * ks)
    a = incident if wave_type == "P" else 0.0
    b = incident * reflection.reflected_p_potential_ratio
    d = incident * reflection.reflected_sv_potential_ratio
    same_sum = incident * reflection.incident_plus_reflected_potential_ratio
    total_p, total_sv = (same_sum, d) if wave_type == "P" else (b, same_sum)
    above_coefficient = -2 * np.pi * rho * (total_p + 1j * total_sv)
    below_coefficient = 2 * np.pi * rho * (total_p - 1j * total_sv)
    qx, qy, phase = _horizontal_phase(targets, origin, azimuth, q)
    above, below = z > 0, z < 0
    fields = [np.empty(phase.shape, dtype=np.complex128) for _ in range(3)]
    for order, field in enumerate(fields):
        field[..., above] = above_coefficient * (-q) ** order * np.exp(-q * z[above])
        # Do not evaluate the downward evanescent wave at above-ground
        # targets, or an overflow would occur before selecting regions.
        field[..., below] = below_coefficient * q**order * np.exp(q * z[below])
        if wave_type == "P":
            field[..., below] -= (
                4 * np.pi * rho * a * (1j * p) ** order * np.exp(1j * p * z[below])
            )
        field[..., below] -= (
            4 * np.pi * rho * b * (-1j * p) ** order * np.exp(-1j * p * z[below])
        )
        field *= phase
    if not all(np.isfinite(field).all() for field in fields):
        raise ValueError("non-finite half-space result; rescale extreme inputs")
    return _assemble_observables(phase, *fields, qx, qy)
