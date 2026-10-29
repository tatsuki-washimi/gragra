"""G-free analytic solid-harmonic series for a fixed spherical cavity.

Prescribed exp(+i k.d.(s-phase_origin)) P/S displacement; no elastic scattering.
Potential uses the full-space plane-wave particular-solution gauge. See the
derivation and convergence contract in docs/design/phase12a_analytic_benchmarks.md.
All target derivatives are analytic Cartesian product-rule recurrences.
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.analytic._bessel import spherical_j0, spherical_j1
from gragra.analytic.values import GFreeGravityValues


def _spherical_bessel_sequence(q: float, degree: int) -> np.ndarray:
    """Miller downward recurrence, normalized at the better conditioned j0/j1."""
    if q < 0.1:
        result = np.empty(degree + 1)
        leading = 1.0
        for n in range(degree + 1):
            term, total = 1.0, 1.0
            # Twelve terms bound the omitted correction well below float64
            # roundoff throughout q<0.1, including subnormal leading powers.
            for m in range(1, 12):
                term *= -q * q / (2 * m * (2 * n + 2 * m + 1))
                total += term
            result[n] = leading * total
            leading *= q / (2 * n + 3)
        return result
    last = max(degree + 40, int(q) + 60)
    work = np.zeros(last + 2)
    work[last] = 1.0
    for n in range(last, 0, -1):
        if abs(work[n]) > 1e150:
            work[n:] *= 1e-150
        work[n - 1] = (2 * n + 1) / q * work[n] - work[n + 1]
    j0, j1 = float(spherical_j0(q)), float(spherical_j1(q))
    index, value = (0, j0) if abs(j0) >= abs(j1) else (1, j1)
    return work[: degree + 1] / work[index] * value


def _jet(value=0.0, gradient=None, hessian=None):
    result = np.zeros(13)
    result[0] = value
    if gradient is not None:
        result[1:4] = gradient
    if hessian is not None:
        result[4:] = np.asarray(hessian).ravel()
    return result


def _multiply(a, b):
    cross = np.outer(a[1:4], b[1:4])
    return _jet(
        a[0] * b[0],
        a[0] * b[1:4] + b[0] * a[1:4],
        a[0] * b[4:].reshape(3, 3) + b[0] * a[4:].reshape(3, 3) + cross + cross.T,
    )


def _norm(value):
    """Euclidean/Frobenius norm without squaring dimensional magnitudes."""
    scale = np.max(abs(value))
    return 0.0 if scale == 0 else scale * np.linalg.norm(value / scale)


def _vector(value, name):
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    vector = np.asarray(raw, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite (3,) vector")
    return vector


def spherical_cavity_observables_g_free(
    targets_m,
    *,
    radius_m: float,
    density_kg_m3: float,
    wavenumber_rad_m: float | np.ndarray,
    displacement_m: complex | np.ndarray,
    propagation_direction,
    polarization,
    wave_type: str,
    center_m=(0.0, 0.0, 0.0),
    phase_origin_m=(0.0, 0.0, 0.0),
    rtol: float = 1e-12,
    atol: tuple[float, float, float] | None = None,
    max_degree: int = 128,
) -> dict[str, GFreeGravityValues]:
    """Return bulk, wall, total potential/acceleration/gradient at (M,3) targets.

    Positive scalar or (K,) wavenumber, scalar radius; nonnegative density;
    finite complex
    displacement; normalized real parallel (P) or transverse (S) polarization.
    The exact wall is excluded (eight floating-point epsilons on r/a).
    ``atol`` has G-free units (kg/m, kg/m², kg/m³). Its default is 1e-13
    times 4πρ|ξ|(a,1,1/a). Each target, term and observable must pass four
    consecutive degree tests after degree ka+4, with a factor-eight margin.
    This is an a posteriori truncation estimate, not a rigorous tail bound.
    Reaching ``max_degree`` without it raises ValueError naming the target.
    No spatial quadrature or finite differences are used.
    """
    for name, value, complex_ok in (
        ("radius_m", radius_m, False),
        ("density_kg_m3", density_kg_m3, False),
        ("wavenumber_rad_m", wavenumber_rad_m, False),
        ("displacement_m", displacement_m, True),
    ):
        _check_no_bool(value, name)
        raw_parameter = np.asarray(value)
        if raw_parameter.dtype.kind not in ("iufc" if complex_ok else "iuf"):
            raise TypeError(f"{name} has invalid numeric type")
    wave_numbers = np.asarray(wavenumber_rad_m)
    displacements = np.asarray(displacement_m)
    if wave_numbers.ndim == 1:
        if not len(wave_numbers) or displacements.shape not in ((), wave_numbers.shape):
            raise ValueError(
                "wavenumber must be nonempty (K,); displacement scalar or (K,)"
            )
        amplitudes = np.broadcast_to(displacements, wave_numbers.shape)
        samples = [
            spherical_cavity_observables_g_free(
                targets_m,
                radius_m=radius_m,
                density_kg_m3=density_kg_m3,
                wavenumber_rad_m=number,
                displacement_m=disp,
                propagation_direction=propagation_direction,
                polarization=polarization,
                wave_type=wave_type,
                center_m=center_m,
                phase_origin_m=phase_origin_m,
                rtol=rtol,
                atol=atol,
                max_degree=max_degree,
            )
            for number, disp in zip(wave_numbers, amplitudes, strict=True)
        ]
        return {
            part: GFreeGravityValues(
                *(
                    np.stack([getattr(sample[part], name) for sample in samples])
                    for name in ("potential", "acceleration", "gradient")
                )
            )
            for part in ("bulk", "wall", "total")
        }
    if wave_numbers.ndim != 0 or displacements.ndim != 0:
        raise ValueError("scalar wavenumber requires scalar displacement")
    radius, density, k = float(radius_m), float(density_kg_m3), float(wavenumber_rad_m)
    if (
        not np.isfinite([radius, density, k]).all()
        or radius <= 0
        or k <= 0
        or density < 0
    ):
        raise ValueError(
            "radius and wavenumber must be positive; density nonnegative; all finite"
        )
    if (
        isinstance(max_degree, (bool, np.bool_))
        or not isinstance(max_degree, (int, np.integer))
        or max_degree < 0
    ):
        raise ValueError("max_degree must be a nonnegative integer")
    _check_no_bool(rtol, "rtol")
    raw_rtol = np.asarray(rtol)
    if raw_rtol.ndim != 0 or raw_rtol.dtype.kind not in "iuf":
        raise TypeError("rtol must be a real scalar")
    rtol = float(raw_rtol)
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative")
    displacement = complex(displacement_m)
    if not np.isfinite(displacement):
        raise ValueError("displacement_m must be finite")
    d = _vector(propagation_direction, "propagation_direction")
    e = _vector(polarization, "polarization")
    if np.max(abs(d)) == 0 or np.max(abs(e)) == 0:
        raise ValueError("direction and polarization must be nonzero")
    d, e = d / np.max(abs(d)), e / np.max(abs(e))
    d, e = d / np.linalg.norm(d), e / np.linalg.norm(e)
    p = float(d @ e)
    if wave_type == "P" and abs(abs(p) - 1) <= 1e-12:
        pass
    elif wave_type == "S" and abs(p) <= 1e-12:
        p = 0.0
    else:
        raise ValueError(
            "wave_type/polarization must specify parallel P or transverse S"
        )
    center, origin = (
        _vector(center_m, "center_m"),
        _vector(phase_origin_m, "phase_origin_m"),
    )
    _check_no_bool(targets_m, "targets_m")
    raw = np.asarray(targets_m)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise TypeError("targets_m must be real numeric")
    targets = np.asarray(raw, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
        raise ValueError("targets_m must have finite shape (M,3)")
    y = (targets - center) / radius
    radii = np.linalg.norm(y, axis=1)
    if np.any(abs(radii - 1) <= 8 * np.finfo(float).eps):
        raise ValueError("target lies on the spherical cavity wall")
    q = k * radius
    amplitude = (
        4 * np.pi * density * displacement * np.exp(1j * k * (d @ (center - origin)))
    )
    scales = (
        4 * np.pi * density * abs(displacement) * np.array([radius, 1.0, 1 / radius])
    )
    if atol is None:
        tolerances = scales * 1e-13
    else:
        _check_no_bool(atol, "atol")
        raw_atol = np.asarray(atol)
        if raw_atol.dtype.kind not in "iuf":
            raise TypeError("atol must contain real values")
        if raw_atol.shape != (3,):
            raise ValueError("atol must contain three unit-bearing tolerances")
        tolerances = np.asarray(raw_atol, dtype=float)
    if not np.isfinite(tolerances).all() or np.any(tolerances < 0):
        raise ValueError(
            "atol must contain three finite nonnegative unit-bearing tolerances"
        )
    if (
        not np.isfinite(q)
        or q <= 0
        or not np.isfinite(amplitude)
        or not np.isfinite(y).all()
        or not np.isfinite(radii).all()
    ):
        raise ValueError("derived coordinates or wave parameters are non-finite")
    if q > max_degree and amplitude != 0:
        raise ValueError("series convergence requires max_degree above ka")
    fields = {
        name: np.zeros((len(targets), 13), complex)
        for name in ("bulk", "wall", "total")
    }
    if amplitude == 0 or len(targets) == 0:
        return {
            name: GFreeGravityValues(f[:, 0], f[:, 1:4], f[:, 4:].reshape(-1, 3, 3))
            for name, f in fields.items()
        }
    bessel = _spherical_bessel_sequence(q, max_degree + 1)
    slices = (slice(0, 1), slice(1, 4), slice(4, 13))
    units = np.r_[radius, np.full(3, -1.0), np.full(9, -1 / radius)]
    for index, (point, r) in enumerate(zip(y, radii, strict=True)):
        if amplitude == 0:
            continue
        exterior = r > 1
        scale = r if exterior else 1.0
        v = point / scale
        z = _jet(d @ v, d / scale)
        tangent = _jet((e - p * d) @ v, (e - p * d) / scale)
        square = _jet(v @ v, 2 * v / scale, 2 * np.eye(3) / scale**2)
        prev, prev2, deriv, deriv2 = _jet(1.0), _jet(), _jet(), _jet()
        sums = {name: np.zeros(13, complex) for name in fields}
        if exterior and p != 0:
            full = 1j * p / q * np.exp(1j * q * (d @ point))
            sums["bulk"] = (
                amplitude
                * units
                * np.r_[
                    full, 1j * q * d * full, (-q * q * np.outer(d, d) * full).ravel()
                ]
            )
        consecutive = 0
        for degree in range(max_degree + 1):
            if degree == 0:
                harmonic, angular = prev, deriv
            else:
                factor = (2 * degree - 1) / degree
                harmonic = factor * _multiply(z, prev) - (
                    degree - 1
                ) / degree * _multiply(square, prev2)
                angular = factor * (_multiply(tangent, prev) + _multiply(z, deriv)) - (
                    degree - 1
                ) / degree * _multiply(square, deriv2)
                prev2, prev, deriv2, deriv = prev, harmonic, deriv, angular
            if exterior:
                power = -2 * degree - 1
                radial = _jet(
                    r ** (-degree - 1),
                    r ** (-degree - 1) * power * point / r**2,
                    r ** (-degree - 1)
                    * (
                        power * np.eye(3) / r**2
                        + power * (power - 2) * np.outer(point, point) / r**4
                    ),
                )
                harmonic, angular = (
                    _multiply(harmonic, radial),
                    _multiply(angular, radial),
                )
            lower = np.cos(q) / q if degree == 0 else bessel[degree - 1]
            bulk = (
                -1j * p * (1j**degree) * bessel[degree + 1]
                if exterior
                else 1j * p * (1j**degree) * lower
            ) * harmonic
            derivative = (
                -bessel[1]
                if degree == 0
                else degree / q * bessel[degree] - bessel[degree + 1]
            )
            wall = 1j ** (degree - 1) * (
                p * derivative * harmonic + bessel[degree] / q * angular
            )
            terms = {"bulk": amplitude * units * bulk, "wall": amplitude * units * wall}
            terms["total"] = terms["bulk"] + terms["wall"]
            sums["bulk"] += terms["bulk"]
            sums["wall"] += terms["wall"]
            sums["total"] = sums["bulk"] + sums["wall"]
            if not all(np.isfinite(s).all() for s in sums.values()):
                raise ValueError(f"non-finite series at target {index}")
            small = all(
                _norm(terms[part][sl])
                <= (tolerances[j] + rtol * _norm(sums[part][sl])) / 8
                for part in sums
                for j, sl in enumerate(slices)
            )
            consecutive = consecutive + 1 if small and degree > q + 4 else 0
            if consecutive >= 4:
                for part in fields:
                    fields[part][index] = sums[part]
                break
        else:
            raise ValueError(
                f"series did not converge at target {index} by max_degree={max_degree}"
            )
    return {
        name: GFreeGravityValues(f[:, 0], f[:, 1:4], f[:, 4:].reshape(-1, 3, 3))
        for name, f in fields.items()
    }
