"""SI physical observable entrypoints for analytical water Newtonian noise models."""

from __future__ import annotations

import numpy as np

from gragra.analytic.water import (
    _validate_positive_scalar,
    water_channel_advection_dc_limit_g_free,
    water_channel_advection_observables_g_free,
    water_channel_advection_transfer_g_free,
    water_channel_uncorrelated_cell_coupling_g_free,
    water_edge_step_transfer_g_free,
    water_waterfall_transfer_g_free,
)
from gragra.constants import G_SI
from gragra.observables.analytic import GravityObservables, to_gravity_observables


def _validate_spectral_density_input(
    name: str, value: float | np.ndarray
) -> np.ndarray:
    """Validate that spectral density input is non-negative and finite."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative (>= 0)")
    return arr


def water_channel_advection_observables(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    width_m: float,
    density_kg_m3: float = 1000.0,
    eta_m: complex | float | np.ndarray = 1.0,
) -> GravityObservables:
    """Compute physical SI gravity observables for advected water channel (Model 2).

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0).
    velocity_mps : float
        Flow velocity v in m/s (non-zero).
    distance_m : float
        Perpendicular distance D from channel to mirror in meters (D > 0).
    width_m : float
        Channel width w in meters (w > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density rho in kg/m^3 (rho > 0).
    eta_m : complex, float or array_like, default=1.0
        Surface displacement amplitude eta in meters (finite).

    Returns
    -------
    GravityObservables
        Potential [m^2/s^2], acceleration [m/s^2], and gradient [s^-2] in SI units.
    """
    g_free_values = water_channel_advection_observables_g_free(
        frequency_hz=frequency_hz,
        velocity_mps=velocity_mps,
        distance_m=distance_m,
        width_m=width_m,
        density_kg_m3=density_kg_m3,
        eta_m=eta_m,
    )
    return to_gravity_observables(g_free_values)


def water_channel_advection_dc_limit(
    distance_m: float,
    width_m: float,
    density_kg_m3: float = 1000.0,
    eta_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact finite physical DC limit (f -> 0) of acceleration and gradient.

    Parameters
    ----------
    distance_m : float
        Distance D in meters (D > 0).
    width_m : float
        Channel width w in meters (w > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density in kg/m^3 (rho > 0).
    eta_m : float, default=1.0
        DC water surface displacement in meters (finite).

    Returns
    -------
    acc_dc_si : np.ndarray, shape (3,)
        Acceleration in SI units [m / s^2].
        a_x = 0, a_y = 0, a_z = -2 * G_SI * rho * w * eta / D.
    grad_dc_si : np.ndarray, shape (3, 3)
        Gravity gradient tensor in SI units [s^-2].
        Gamma_yy = -2 * G_SI * rho * w * eta / D^2,
        Gamma_zz = +2 * G_SI * rho * w * eta / D^2, others zero.
    """
    acc_g_free, grad_g_free = water_channel_advection_dc_limit_g_free(
        distance_m=distance_m,
        width_m=width_m,
        density_kg_m3=density_kg_m3,
        eta_m=eta_m,
    )
    acc_si = G_SI * acc_g_free
    grad_si = G_SI * grad_g_free
    acc_si.setflags(write=False)
    grad_si.setflags(write=False)
    return acc_si, grad_si


def water_channel_advection_asd(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    width_m: float,
    surface_asd_m_per_rt_hz: float | np.ndarray,
    density_kg_m3: float = 1000.0,
    component: str = "x",
) -> np.ndarray:
    """Compute acceleration amplitude spectral density (ASD) for advected water channel.

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0).
    velocity_mps : float
        Flow velocity v in m/s (non-zero).
    distance_m : float
        Distance D in meters (D > 0).
    width_m : float
        Width w in meters (w > 0).
    surface_asd_m_per_rt_hz : float or array_like
        Water surface displacement ASD sqrt(S_eta(f)) in m / Hz^0.5 (non-negative).
    density_kg_m3 : float, default=1000.0
        Fluid density in kg/m^3 (rho > 0).
    component : {"x", "z", "all"}, default="x"
        Acceleration component to return: along-flow ("x"), perpendicular ("z"),
        or both as shape (..., 2) or (..., 3) ("all").

    Returns
    -------
    np.ndarray
        Acceleration ASD in m / (s^2 * Hz^0.5).
    """
    s_eta = _validate_spectral_density_input(
        "surface_asd_m_per_rt_hz", surface_asd_m_per_rt_hz
    )
    _, acc_trans, _ = water_channel_advection_transfer_g_free(
        frequency_hz=frequency_hz,
        velocity_mps=velocity_mps,
        distance_m=distance_m,
        width_m=width_m,
        density_kg_m3=density_kg_m3,
        allow_dc=False,
    )

    if component == "x":
        acc_x_mag = np.abs(acc_trans[..., 0])
        return G_SI * acc_x_mag * s_eta
    if component == "z":
        acc_z_mag = np.abs(acc_trans[..., 2])
        return G_SI * acc_z_mag * s_eta
    if component == "all":
        acc_mag = np.abs(acc_trans)
        return G_SI * acc_mag * np.expand_dims(s_eta, axis=-1)
    raise ValueError(f"Unknown component '{component}', must be 'x', 'z', or 'all'")


def water_channel_advection_strain_asd(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    width_m: float,
    surface_asd_m_per_rt_hz: float | np.ndarray,
    arm_length_m: float = 3000.0,
    density_kg_m3: float = 1000.0,
) -> np.ndarray:
    """Compute GW detector strain ASD for advected water channel (along-flow x).

    h(f) = a_x(f) / (Omega^2 * L) = a_x(f) / ((2*pi*f)^2 * L).

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0).
    velocity_mps : float
        Flow velocity v in m/s (non-zero).
    distance_m : float
        Distance D in meters (D > 0).
    width_m : float
        Width w in meters (w > 0).
    surface_asd_m_per_rt_hz : float or array_like
        Surface displacement ASD in m / Hz^0.5 (non-negative).
    arm_length_m : float, default=3000.0
        Interferometer arm length L in meters (L > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density in kg/m^3 (rho > 0).

    Returns
    -------
    np.ndarray
        Strain ASD in 1 / Hz^0.5.
    """
    l_arm = _validate_positive_scalar("arm_length_m", arm_length_m)
    f_arr = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(f_arr <= 0.0):
        raise ValueError("frequency_hz must be strictly positive (> 0) for strain ASD")

    acc_asd = water_channel_advection_asd(
        frequency_hz=f_arr,
        velocity_mps=velocity_mps,
        distance_m=distance_m,
        width_m=width_m,
        surface_asd_m_per_rt_hz=surface_asd_m_per_rt_hz,
        density_kg_m3=density_kg_m3,
        component="x",
    )
    omega = 2.0 * np.pi * f_arr
    return acc_asd / ((omega**2) * l_arm)


def water_channel_kolmogorov_surface_asd(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    largest_turb_scale_m: float,
    turb_velocity_fluc_mps: float,
) -> np.ndarray:
    """Compute Kolmogorov water surface height ASD sqrt(S_b(f)).

    Nishizawa (2021-03-31) Eq. (10):
      sqrt(S_b(Omega)) = delta_v_ell / (2*pi) * f^(-5/6) * (ell / v)^(2/3)

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0).
    velocity_mps : float
        Average flow velocity v in m/s (v > 0).
    largest_turb_scale_m : float
        Largest scale of turbulence ell in meters (ell > 0).
    turb_velocity_fluc_mps : float
        Velocity fluctuation delta_v_ell at largest scale in m/s (delta_v_ell > 0).

    Returns
    -------
    np.ndarray
        Surface height ASD in m / Hz^0.5.
    """
    f = np.asarray(frequency_hz, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("frequency_hz must contain only finite values")
    if np.any(f <= 0.0):
        raise ValueError("frequency_hz must be strictly positive (> 0)")
    v = _validate_positive_scalar("velocity_mps", velocity_mps)
    ell = _validate_positive_scalar("largest_turb_scale_m", largest_turb_scale_m)
    dv = _validate_positive_scalar("turb_velocity_fluc_mps", turb_velocity_fluc_mps)

    return (dv / (2.0 * np.pi)) * (f ** (-5.0 / 6.0)) * ((ell / v) ** (2.0 / 3.0))


def water_waterfall_asd(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    area_asd_m2_per_rt_hz: float | np.ndarray,
    density_kg_m3: float = 1000.0,
) -> np.ndarray:
    """Compute horizontal acceleration ASD for vertical waterfall.

    Parameters
    ----------
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0).
    velocity_mps : float
        Fall velocity v in m/s (non-zero).
    distance_m : float
        Horizontal distance D in meters (D > 0).
    area_asd_m2_per_rt_hz : float or array_like
        Cross-sectional area perturbation ASD sqrt(S_A(f)) in m^2 / Hz^0.5
        (non-negative).
    density_kg_m3 : float, default=1000.0
        Fluid density in kg/m^3 (rho > 0).

    Returns
    -------
    np.ndarray
        Horizontal acceleration ASD in m / (s^2 * Hz^0.5).
    """
    s_a = _validate_spectral_density_input(
        "area_asd_m2_per_rt_hz", area_asd_m2_per_rt_hz
    )
    _, acc_trans, _ = water_waterfall_transfer_g_free(
        frequency_hz=frequency_hz,
        velocity_mps=velocity_mps,
        distance_m=distance_m,
        density_kg_m3=density_kg_m3,
        allow_dc=False,
    )
    acc_x_mag = np.abs(acc_trans[..., 0])
    return G_SI * acc_x_mag * s_a


def water_waterfall_strain_asd(
    frequency_hz: float | list | tuple | np.ndarray,
    velocity_mps: float,
    distance_m: float,
    area_asd_m2_per_rt_hz: float | np.ndarray,
    arm_length_m: float = 3000.0,
    density_kg_m3: float = 1000.0,
) -> np.ndarray:
    """Compute GW detector strain ASD for vertical waterfall.

    Nishizawa (2021) Eq. (18).
    """
    l_arm = _validate_positive_scalar("arm_length_m", arm_length_m)
    f_arr = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(f_arr <= 0.0):
        raise ValueError("frequency_hz must be strictly positive (> 0)")

    acc_asd = water_waterfall_asd(
        frequency_hz=f_arr,
        velocity_mps=velocity_mps,
        distance_m=distance_m,
        area_asd_m2_per_rt_hz=area_asd_m2_per_rt_hz,
        density_kg_m3=density_kg_m3,
    )
    omega = 2.0 * np.pi * f_arr
    return acc_asd / ((omega**2) * l_arm)


def water_channel_uncorrelated_asd(
    distance_m: float,
    width_m: float,
    correlation_area_m2: float,
    surface_asd_m_per_rt_hz: float | np.ndarray,
    density_kg_m3: float = 1000.0,
) -> np.ndarray:
    """Compute acceleration ASD for Model 1 (narrow-channel uncorrelated surface).

    ASD_a(f) = G_SI * (C_a / G) * ASD_eta(f)
             = G_SI * rho * sqrt(A_cell * w) * sqrt(pi / (8 * H^3)) * ASD_eta(f)

    Parameters
    ----------
    distance_m : float
        Distance H from channel to mirror in meters (H > 0).
    width_m : float
        Channel width w in meters (w > 0).
    correlation_area_m2 : float
        Correlation cell area A_cell in m^2 (A_cell > 0).
    surface_asd_m_per_rt_hz : float or array_like
        Surface displacement ASD in m / Hz^0.5 (non-negative).
    density_kg_m3 : float, default=1000.0
        Fluid density in kg/m^3 (rho > 0).

    Returns
    -------
    np.ndarray
        Acceleration ASD in m / (s^2 * Hz^0.5).
    """
    c_g_free = water_channel_uncorrelated_cell_coupling_g_free(
        distance_m=distance_m,
        width_m=width_m,
        correlation_area_m2=correlation_area_m2,
        density_kg_m3=density_kg_m3,
    )
    s_eta = _validate_spectral_density_input(
        "surface_asd_m_per_rt_hz", surface_asd_m_per_rt_hz
    )
    return G_SI * c_g_free * s_eta


def water_edge_step_fourier_amplitude(
    x0_m: float,
    distance_m: float,
    volume_m3: float,
    frequency_hz: float | list | tuple | np.ndarray,
    density_kg_m3: float = 1000.0,
) -> np.ndarray:
    """Compute single-event acceleration Fourier amplitude for edge step.

    Nishizawa (2021) Eq. (6).

    Physical Meaning and Units:
      This is the Fourier transform magnitude |Delta a_x(Omega)| of a SINGLE
      deterministic Heaviside volume step event Delta V at coordinate x0:
        |Delta a_x(Omega)| = (G_SI * rho * Delta V * |x0| /
                              (D^2 + x0^2)^(3/2) * (1 / (2*pi*f))).
      The physical unit is acceleration * time = m / (s^2 * Hz) = m / s.
      Note: This is NOT a stationary ASD (which would have units
      m / (s^2 * Hz^0.5)). A stationary noise ASD requires a separate,
      physically defined stationary event model; this API describes one
      deterministic step only.

    Parameters
    ----------
    x0_m : float
        Horizontal distance along channel to edge in meters (finite).
    distance_m : float
        Perpendicular distance D from channel to mirror in meters (D > 0).
    volume_m3 : float
        Displaced volume Delta V in m^3 (Delta V > 0).
    frequency_hz : float or array_like
        Frequency f in Hz (f > 0).
    density_kg_m3 : float, default=1000.0
        Fluid density rho in kg/m^3 (rho > 0).

    Returns
    -------
    np.ndarray
        Single-event Fourier amplitude |Delta a_x(Omega)| in m / (s^2 * Hz) [m / s].
    """
    acc_t_g_free = water_edge_step_transfer_g_free(
        x0_m=x0_m,
        distance_m=distance_m,
        volume_m3=volume_m3,
        frequency_hz=frequency_hz,
        density_kg_m3=density_kg_m3,
    )
    return G_SI * np.abs(acc_t_g_free)


def water_edge_step_strain_fourier_amplitude(
    x0_m: float,
    distance_m: float,
    volume_m3: float,
    frequency_hz: float | list | tuple | np.ndarray,
    arm_length_m: float = 3000.0,
    density_kg_m3: float = 1000.0,
) -> np.ndarray:
    """Compute single-event strain Fourier spectral amplitude for channel edge step.

    Nishizawa (2021) Eq. (7).
    |h(Omega)| = |Delta a_x(Omega)| / (Omega^2 * L_arm).
    Physical unit: 1 / Hz [s].
    """
    l_arm = _validate_positive_scalar("arm_length_m", arm_length_m)
    f = np.asarray(frequency_hz, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("frequency_hz must contain only finite values")
    if np.any(f <= 0.0):
        raise ValueError("frequency_hz must be strictly positive (> 0)")

    acc_amp = water_edge_step_fourier_amplitude(
        x0_m=x0_m,
        distance_m=distance_m,
        volume_m3=volume_m3,
        frequency_hz=f,
        density_kg_m3=density_kg_m3,
    )
    omega = 2.0 * np.pi * f
    return acc_amp / ((omega**2) * l_arm)
