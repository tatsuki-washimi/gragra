"""G-free acceleration of the underground Rayleigh leading SCA model.

The wall displacement is frozen at the geometric center. The background
contains both rock bulk and ground surface. This is not a scattered-wave or
finite-cavity solution. See docs/design/u11-rayleigh-small-cavity.md.
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.analytic.cylinder_sca import cylinder_sca_wall_coupling_tensor_g_free
from gragra.analytic.rayleigh_literature import (
    rayleigh_halfspace_allspace_observables_g_free,
)
from gragra.analytic.sca_tensor import (
    cuboid_sca_wall_coupling_tensor_g_free,
    sca_wall_acceleration_from_tensor_g_free,
    sphere_sca_wall_coupling_tensor_g_free,
)
from gragra.waves.rayleigh import _rayleigh_parameters, rayleigh_displacement


def _numeric(value, name, *, complex_ok=False):
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind not in ("iufc" if complex_ok else "iuf"):
        raise TypeError(
            f"{name} must contain numeric {'complex' if complex_ok else 'real'} values"
        )
    result = np.array(raw, dtype=np.complex128 if complex_ok else np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name, *, zero=False):
    result = _numeric(value, name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    value = float(result)
    if value < 0 if zero else value <= 0:
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return value


def _vector(value, name):
    result = _numeric(value, name)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return result


def _readonly(value, *, complex_ok=False):
    result = np.array(
        value, dtype=np.complex128 if complex_ok else np.float64, copy=True
    )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite derived value; inputs exceed numerical domain")
    result.setflags(write=False)
    return result


def rayleigh_small_cavity_acceleration_g_free(
    targets_m,
    *,
    cavity_shape,
    center_m,
    density_kg_m3,
    v_p_m_s,
    v_s_m_s,
    frequency_hz,
    vertical_surface_amplitude_m,
    radius_m=None,
    height_m=None,
    half_lengths_m=None,
    rotation=None,
    direction=(1.0, 0.0, 0.0),
    phase_origin_m=(0.0, 0.0, 0.0),
    max_sca_parameter=0.1,
    return_diagnostics=False,
):
    """Return background_halfspace, cavity_wall_sca, total_leading_sca.

    Values are read-only complex128 acceleration arrays in kg/m²: (M,3)
    for scalar frequency or (K,M,3) for a nonempty frequency vector.
    eta is vertical surface displacement at the fixed horizontal origin,
    with phase exp(+i k d.(s-origin)-2 pi i f t). The origin has z=0.

    Sphere requires radius; cuboid half_lengths; cylinder radius/full height.
    Rotation is proper local-to-world, including when supplied for a sphere
    (where it has no physical effect). All targets and the entire cavity
    must have positive interior/underground margins. eps<=0.1 is the default
    declared SCA domain, not an error guarantee. Zero amplitudes/density do
    not bypass validation. Diagnostics include center displacement in m.
    """
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise TypeError("return_diagnostics must be boolean")
    if not isinstance(cavity_shape, str) or cavity_shape not in (
        "sphere",
        "cuboid",
        "cylinder",
    ):
        raise ValueError("cavity_shape must be sphere, cuboid or cylinder")
    targets = _numeric(targets_m, "targets_m")
    if targets.ndim != 2 or targets.shape[1:] != (3,) or len(targets) == 0:
        raise ValueError("targets_m must have nonempty shape (M,3)")
    center = _vector(center_m, "center_m")
    origin = _vector(phase_origin_m, "phase_origin_m")
    if origin[2] != 0:
        raise ValueError("phase_origin_m must have z=0")
    d = _vector(direction, "direction")
    if d[2] != 0 or np.max(np.abs(d)) == 0:
        raise ValueError("direction must be nonzero and horizontal")
    d /= np.max(np.abs(d))
    d /= np.linalg.norm(d)
    rho = _scalar(density_kg_m3, "density_kg_m3", zero=True)
    vp = _scalar(v_p_m_s, "v_p_m_s")
    vs = _scalar(v_s_m_s, "v_s_m_s")
    if not vs < vp or (vs / vp) ** 2 >= 0.75:
        raise ValueError("speeds require stable isotropic material: (vs/vp)^2 < 3/4")
    threshold = _scalar(max_sca_parameter, "max_sca_parameter")
    freq = _numeric(frequency_hz, "frequency_hz")
    if freq.ndim not in (0, 1) or freq.size == 0 or np.any(freq <= 0):
        raise ValueError("frequency_hz must be positive scalar or nonempty (K,)")
    eta = _numeric(
        vertical_surface_amplitude_m, "vertical_surface_amplitude_m", complex_ok=True
    )
    if eta.ndim != 0 and not (freq.ndim == 1 and eta.shape == freq.shape):
        raise ValueError("amplitude must be scalar or match frequency (K,)")
    rot = np.eye(3) if rotation is None else _numeric(rotation, "rotation")
    if (
        rot.shape != (3, 3)
        or not np.allclose(rot.T @ rot, np.eye(3), rtol=0, atol=1e-12)
        or not np.isclose(np.linalg.det(rot), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("rotation must be a proper orthogonal (3,3) matrix")
    required = {
        "sphere": (True, False, False),
        "cuboid": (False, False, True),
        "cylinder": (True, True, False),
    }[cavity_shape]
    if tuple(v is not None for v in (radius_m, height_m, half_lengths_m)) != required:
        raise ValueError("missing or extra shape dimensions")
    radius = _scalar(radius_m, "radius_m") if radius_m is not None else None
    height = _scalar(height_m, "height_m") if height_m is not None else None
    half = (
        _vector(half_lengths_m, "half_lengths_m")
        if half_lengths_m is not None
        else None
    )
    if half is not None and np.any(half <= 0):
        raise ValueError("half_lengths_m must be positive")
    local = (targets - center) @ rot
    if cavity_shape == "sphere":
        length = radius
        top = center[2] + radius
        clearance = radius - np.hypot.reduce(targets - center, axis=1)
    elif cavity_shape == "cuboid":
        length = float(np.hypot.reduce(half))
        top = center[2] + np.abs(rot[2]) @ half
        clearance = np.min(half - np.abs(local), axis=1)
    else:
        length = float(np.hypot(radius, height / 2))
        top = center[2] + abs(rot[2, 2]) * height / 2 + radius * np.hypot(*rot[2, :2])
        clearance = np.minimum(
            radius - np.hypot(local[:, 0], local[:, 1]),
            height / 2 - np.abs(local[:, 2]),
        )
    margin = max(
        1e-10 * max(1.0, length),
        64 * np.finfo(float).eps * max(np.max(np.abs(center)), np.max(np.abs(targets))),
    )
    if not np.isfinite(top) or top >= -margin:
        raise ValueError("entire cavity must be underground with a positive margin")
    bad = np.flatnonzero(~np.isfinite(clearance) | (clearance <= margin))
    if bad.size:
        raise ValueError(
            f"targets must be strictly interior with margin; target {bad[0]}"
        )
    speed, q, s = _rayleigh_parameters(vp, vs)
    ks = np.atleast_1d(freq) * (2 * np.pi / speed)
    eps = length * np.maximum.reduce([ks, ks * q, ks * s])
    bad = np.flatnonzero(~np.isfinite(eps) | (eps > threshold) | (ks <= 0))
    if bad.size:
        raise ValueError(
            f"SCA domain exceeded at sample {bad[0]}: "
            f"eps={eps[bad[0]]}, limit={threshold}"
        )
    if cavity_shape == "sphere":
        tensor = sphere_sca_wall_coupling_tensor_g_free(center, radius, targets)
    elif cavity_shape == "cuboid":
        tensor = cuboid_sca_wall_coupling_tensor_g_free(
            center, 2 * half, targets, rotation_matrix=rot
        )
    else:
        tensor = cylinder_sca_wall_coupling_tensor_g_free(
            center, radius, height, targets, rotation_matrix=rot
        )
    uc = np.stack(
        [
            rayleigh_displacement(
                center[None, :], d, float(k), vp, vs, amplitude_m=1.0
            )[0]
            * a
            * np.exp(-1j * k * np.dot(d[:2], origin[:2]))
            for k, a in zip(ks, np.broadcast_to(eta, ks.shape), strict=True)
        ]
    )
    wall = np.stack(
        [sca_wall_acceleration_from_tensor_g_free(tensor, rho, u) for u in uc]
    )
    background = rayleigh_halfspace_allspace_observables_g_free(
        targets,
        density_kg_m3=rho,
        v_p_m_s=vp,
        v_s_m_s=vs,
        frequency_hz=freq,
        vertical_surface_amplitude_m=eta,
        direction=d,
        phase_origin_m=origin,
    ).acceleration
    if freq.ndim == 0:
        wall = wall[0]
        uc = uc[0]
    parts = {
        "background_halfspace": _readonly(background, complex_ok=True),
        "cavity_wall_sca": _readonly(wall, complex_ok=True),
        "total_leading_sca": _readonly(background + wall, complex_ok=True),
    }
    if not return_diagnostics:
        return parts
    return parts, {
        "approximation": "background halfspace plus center-frozen leading SCA wall",
        "L_cav_m": length,
        "eps": _readonly(eps[0] if freq.ndim == 0 else eps),
        "max_sca_parameter": threshold,
        "u_c": _readonly(uc, complex_ok=True),
        "u_c_units": "m",
        "center_m": _readonly(center),
        "ground_clearance_m": float(-top),
        "interior_clearance_m": _readonly(clearance),
        "boundary_margin_m": margin,
        "phase_convention": "exp(+i k d.(s-phase_origin)-2 pi i f t); phase_origin z=0",
    }
