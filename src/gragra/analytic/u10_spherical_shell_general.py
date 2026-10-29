"""G-free gravity of a finite shell with a prescribed P/S plane wave.

The phasor is U*p*exp(+i*k*d.(s-phase_origin)-2*pi*i*f*t); there is no
elastic boundary scattering. Potential vanishes at infinity. See the U10
design for analytic finite-interval coefficients and convergence limitations.
"""

from __future__ import annotations

import numpy as np

from gragra._arrays import _check_no_bool
from gragra.analytic._u10_shell_series import degree_terms, norm
from gragra.analytic.values import GFreeGravityValues

_PARTS = ("bulk", "inner_wall", "outer_boundary", "total")
_BLOCKS = (slice(0, 1), slice(1, 4), slice(4, 13))
_NAMES = ("potential", "acceleration", "gradient")


class ShellConvergenceError(ValueError):
    """A query failed its requested final-field budget; diagnostics are per call."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _numeric(value, name, *, complex_allowed=False):
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind not in ("iufc" if complex_allowed else "iuf"):
        raise TypeError(f"{name} has invalid numeric dtype")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite")
    return raw


def _scalar(value, name):
    raw = _numeric(value, name)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return np.longdouble(raw)


def _vector(value, name, *, normalize=False):
    raw = _numeric(value, name)
    if raw.shape != (3,):
        raise ValueError(f"{name} must be (3,)")
    result = np.array(raw, dtype=np.longdouble, copy=True)
    if normalize:
        scale = np.max(abs(result))
        if scale == 0:
            raise ValueError(f"{name} must be nonzero")
        result /= scale
        result /= norm(result)
    return result


def _tolerance(value, name):
    raw = _numeric(value, name)
    if raw.shape != (3,) or np.any(raw < 0):
        raise ValueError(f"{name} must be three nonnegative values")
    return np.asarray(raw, dtype=np.longdouble)


def spherical_shell_observables_g_free(
    targets_m,
    *,
    inner_radius_m,
    outer_radius_m,
    density_kg_m3,
    wavenumber_rad_m,
    displacement_m,
    propagation_direction,
    polarization,
    wave_type,
    center_m=(0.0, 0.0, 0.0),
    phase_origin_m=(0.0, 0.0, 0.0),
    rtol=(1e-10, 1e-10, 1e-9),
    atol=None,
    max_degree=128,
    return_diagnostics=False,
):
    """Return bulk/inner_wall/outer_boundary/total GFreeGravityValues.

    k is positive scalar or nonempty (K,); U is scalar or matching (K,).
    Targets are nonempty (M,3), excluding the two exact walls. Complex128
    output shapes are (M,), (M,3), (M,3,3), prefixed by K for series inputs.
    Directions are real, scale-safely normalized and parallel/antiparallel (P)
    or transverse (S). Inputs are not mutated; output copies are read-only.

    atol has three G-free units (kg/m, kg/m², kg/m³). The default is 1e-13
    times 4*pi*rho*abs(U)*(b,1,1/b), with shell thickness factor for bulk and
    total only at r/a<=0.5 or r/b>=2. Diagnostics retain all evaluated degree
    estimates per sample/target. ShellConvergenceError carries failed and
    pending query states. Tail and roundoff estimates are not rigorous bounds.
    No spatial quadrature, finite differences or optional dependency is used.
    """
    a = _scalar(inner_radius_m, "inner_radius_m")
    b = _scalar(outer_radius_m, "outer_radius_m")
    rho = _scalar(density_kg_m3, "density_kg_m3")
    if not 0 < a < b or rho < 0:
        raise ValueError("require 0 < inner radius < outer radius and density >= 0")
    kraw = _numeric(wavenumber_rad_m, "wavenumber_rad_m")
    uraw = _numeric(displacement_m, "displacement_m", complex_allowed=True)
    if kraw.ndim not in (0, 1) or kraw.size == 0 or np.any(kraw <= 0):
        raise ValueError("wavenumber must be positive scalar or nonempty (K,)")
    if uraw.ndim != 0 and (kraw.ndim != 1 or uraw.shape != kraw.shape):
        raise ValueError("displacement must be scalar or match a (K,) wavenumber")
    ks = np.atleast_1d(kraw).astype(np.longdouble)
    us = np.broadcast_to(uraw, ks.shape).astype(np.clongdouble)
    raw = _numeric(targets_m, "targets_m")
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] == 0:
        raise ValueError("targets_m must be nonempty (M,3)")
    center = _vector(center_m, "center_m")
    origin = _vector(phase_origin_m, "phase_origin_m")
    points = (np.asarray(raw, dtype=np.longdouble) - center) / b
    radii = np.array([norm(point) for point in points])
    # Subtract the input radii before dividing. Forming 1-a/b would discard
    # significant bits of the finite interval for an extremely thin shell.
    thickness = (b - a) / b
    inner = a / b
    if not np.isfinite(points).all() or not np.isfinite(radii).all():
        raise ValueError("derived coordinates are non-finite")
    # Guard floating representation of an exact wall as well as literal equality.
    eps = np.finfo(float).eps
    if np.any(abs(radii - inner) <= 8 * eps * inner) or np.any(
        abs(radii - 1) <= 8 * eps
    ):
        raise ValueError("target lies on a shell boundary")
    d = _vector(propagation_direction, "propagation_direction", normalize=True)
    p = _vector(polarization, "polarization", normalize=True)
    projection = d @ p
    if wave_type == "P" and abs(abs(projection) - 1) <= 1e-12:
        pass
    elif wave_type == "S" and abs(projection) <= 1e-12:
        projection = np.longdouble(0)
    else:
        raise ValueError("require parallel P or transverse S polarization")
    relative = _tolerance(rtol, "rtol")
    absolute = None if atol is None else _tolerance(atol, "atol")
    if (
        isinstance(max_degree, (bool, np.bool_))
        or not isinstance(max_degree, (int, np.integer))
        or max_degree < 0
    ):
        raise ValueError("max_degree must be a nonnegative integer")
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise TypeError("return_diagnostics must be bool")
    diagnostics = {
        "status": "RUNNING",
        "internal_epsilon": float(np.finfo(np.longdouble).eps),
        "shell_thickness_fraction": float(thickness),
        "normalized_radius_rounding": float(abs((1 - inner) - thickness)),
        "normalized_radius_rounding_fraction_of_thickness": float(
            abs((1 - inner) - thickness) / thickness
        ),
        "max_degree": int(max_degree),
        "radial_term_cap": 256,
        "units": ["kg/m", "kg/m^2", "kg/m^3"],
        "queries": [
            {"sample": s, "target": t, "status": "PENDING"}
            for s in range(len(ks))
            for t in range(len(points))
        ],
    }
    output = np.zeros((4, len(ks), len(points), 13), dtype=np.clongdouble)
    units = np.r_[
        b, np.full(3, -1, dtype=np.longdouble), np.full(9, -1 / b, dtype=np.longdouble)
    ]
    pi = np.arccos(np.longdouble(-1))
    for sample, (k, u) in enumerate(zip(ks, us, strict=True)):
        amplitude = (
            4 * pi * rho * u * np.exp(np.clongdouble(1j) * k * (d @ (center - origin)))
        )
        scale = 4 * pi * rho * abs(u) * np.array([b, 1, 1 / b], dtype=np.longdouble)
        if (
            not np.isfinite(amplitude)
            or not np.isfinite(scale).all()
            or not np.isfinite(k * b)
        ):
            raise ValueError("derived amplitude, scale or wavenumber is non-finite")
        for target, (point, radius) in enumerate(zip(points, radii, strict=True)):
            record = diagnostics["queries"][sample * len(points) + target]
            record.update(
                status="EVALUATING",
                region="inner"
                if radius < inner
                else "outer"
                if radius > 1
                else "material",
                history=[],
            )
            floors = np.broadcast_to(
                scale * 1e-13 if absolute is None else absolute, (4, 3)
            ).copy()
            if absolute is None and (radius <= inner / 2 or radius >= 2):
                floors[[0, 3]] *= thickness
            signal_scale = scale.copy()
            if radius <= inner / 2 or radius >= 2:
                signal_scale *= thickness
            record["atol"] = floors.astype(float).tolist()
            if amplitude == 0:
                record.update(status="CONVERGED_ZERO", degree=0)
                continue
            sums = np.zeros((4, 13), dtype=np.clongdouble)
            magnitudes = np.zeros((4, 13), dtype=np.longdouble)
            consecutive = 0
            try:
                for degree, terms, absolute_terms, radial_count in degree_terms(
                    point, inner, thickness, k * b, d, p, projection, max_degree
                ):
                    terms = amplitude * units * terms
                    sums += terms
                    magnitudes += abs(amplitude * units) * absolute_terms
                    if not np.isfinite(sums).all() or not np.isfinite(magnitudes).all():
                        raise ArithmeticError("non-finite series")
                    roundoff = 64 * np.finfo(np.longdouble).eps * magnitudes + np.finfo(
                        float
                    ).eps * abs(sums)
                    small = True
                    blocks = []
                    for part in range(4):
                        for observable, sl in enumerate(_BLOCKS):
                            budget = floors[part, observable] + relative[
                                observable
                            ] * norm(sums[part, sl])
                            component_budget = floors[part, observable] + relative[
                                observable
                            ] * abs(sums[part, sl])
                            estimate = 16 * norm(terms[part, sl]) + norm(
                                roundoff[part, sl]
                            )
                            component_estimate = (
                                16 * abs(terms[part, sl]) + roundoff[part, sl]
                            )
                            ok = estimate <= budget and np.all(
                                component_estimate <= component_budget
                            )
                            small = small and bool(ok)
                            blocks.append(
                                {
                                    "part": _PARTS[part],
                                    "observable": _NAMES[observable],
                                    "term_norm": float(norm(terms[part, sl])),
                                    "budget": float(budget),
                                    "roundoff_estimate": float(
                                        norm(roundoff[part, sl])
                                    ),
                                    "component_estimate": component_estimate.astype(
                                        float
                                    ).tolist(),
                                    "component_budget": component_budget.astype(
                                        float
                                    ).tolist(),
                                }
                            )
                    record["history"].append(
                        {
                            "degree": degree,
                            "radial_terms": radial_count,
                            "blocks": blocks,
                        }
                    )
                    consecutive = consecutive + 1 if small and degree > k * b + 4 else 0
                    if consecutive >= 8:
                        output[:, sample, target] = sums
                        record.update(status="CONVERGED", degree=degree)
                        record["wall_cancellation_ratio"] = {
                            name: float(
                                (norm(sums[1, sl]) + norm(sums[2, sl]))
                                / (signal_scale[j] + norm(sums[3, sl]))
                            )
                            for j, (name, sl) in enumerate(
                                zip(_NAMES, _BLOCKS, strict=True)
                            )
                        }
                        break
                else:
                    raise ArithmeticError(
                        "degree or numerical precision budget exhausted"
                    )
            except (ArithmeticError, OverflowError, FloatingPointError) as exc:
                record.update(
                    status="NOT_CONVERGED",
                    reason=str(exc),
                    partial_real=sums.real.astype(float).tolist(),
                    partial_imag=sums.imag.astype(float).tolist(),
                )
                diagnostics.update(
                    status="NOT_CONVERGED", failed_index=[sample, target]
                )
                raise ShellConvergenceError(
                    f"shell failed: sample {sample}, target {target}: {exc}",
                    diagnostics,
                ) from exc
    diagnostics["status"] = "CONVERGED"
    arrays = output[:, 0] if kraw.ndim == 0 else output
    parts = {
        part: GFreeGravityValues(
            values[..., 0],
            values[..., 1:4],
            values[..., 4:].reshape(*values.shape[:-1], 3, 3),
        )
        for part, values in zip(_PARTS, arrays, strict=True)
    }
    return (parts, diagnostics) if return_diagnostics else parts
