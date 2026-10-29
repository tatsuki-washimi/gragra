"""Independent Newton-integral references for the finite water-surface fixture.

This module deliberately imports no :mod:`gragra` code.  The adaptive sheet
reference uses SciPy only when called; the signed-volume reference uses NumPy
Gauss--Legendre rules over the complete integration intervals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

G_SI_REFERENCE = 6.67430e-11
L_M = 1.0
DENSITY_JUMP_KG_M3 = 998.8
OBSERVABLES = ("potential", "acceleration", "gradient")
SHAPES = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}


class ReferenceEvaluationLimitError(RuntimeError):
    """Raised when an adaptive reference exceeds its predeclared kernel cap."""


@dataclass
class _Counter:
    cap: int
    count: int = 0

    def tick(self) -> None:
        self.count += 1
        if self.count > self.cap:
            raise ReferenceEvaluationLimitError(
                f"Newton kernel evaluation cap exceeded: {self.count}>{self.cap}"
            )


def _shape_function(x_over_l, y_over_l):
    return np.cos(np.pi * x_over_l / 2.0) * np.cos(np.pi * y_over_l / 2.0)


def _kernel(observable: str, sources_m: np.ndarray, target_m: np.ndarray):
    """Separately written, unsoftened Newton kernels with source-target delta."""
    delta = np.asarray(sources_m, dtype=np.float64) - np.asarray(
        target_m, dtype=np.float64
    )
    r2 = np.sum(delta * delta, axis=-1)
    radius = np.sqrt(r2)
    if observable == "potential":
        return -1.0 / radius
    if observable == "acceleration":
        return delta / (r2 * radius)[..., None]
    if observable == "gradient":
        outer = delta[..., :, None] * delta[..., None, :]
        return (
            3.0 * outer / (r2 * r2 * radius)[..., None, None]
            - np.eye(3) / (r2 * radius)[..., None, None]
        )
    raise ValueError(f"unsupported observable {observable!r}")


def _transfer_scale(observable: str) -> float:
    length_factor = {
        "potential": L_M,
        "acceleration": 1.0,
        "gradient": 1.0 / L_M,
    }[observable]
    return G_SI_REFERENCE * DENSITY_JUMP_KG_M3 * length_factor


def block_norm(value, observable: str) -> float:
    array = np.asarray(value)
    if observable == "potential":
        return float(np.abs(array))
    return float(np.linalg.norm(array))


def adaptive_surface_reference(
    target_m,
    observable: str,
    *,
    tolerance: float,
    limit: int = 200,
    evaluation_cap: int = 2_000_000,
) -> dict:
    """Integrate one per-metre sheet response using nested dimensionless quad."""
    from scipy.integrate import IntegrationWarning, quad

    target = np.asarray(target_m, dtype=np.float64)
    if target.shape != (3,):
        raise ValueError("target_m must have shape (3,)")
    shape = SHAPES[observable]
    component_count = int(np.prod(shape)) if shape else 1
    values = np.empty(component_count, dtype=np.float64)
    diagnostics = np.empty(component_count, dtype=np.float64)
    counter = _Counter(evaluation_cap)
    warning_messages: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        for component in range(component_count):
            inner_errors: list[float] = []

            def outer_integrand(
                u, component_index=component, component_inner_errors=inner_errors
            ):
                def inner_integrand(v):
                    counter.tick()
                    source_over_l = np.array((u, v, 0.0))
                    raw = np.asarray(
                        _kernel(observable, source_over_l, target / L_M)
                    ).reshape(-1)
                    return _shape_function(u, v) * raw[component_index]

                inner_value, inner_error = quad(
                    inner_integrand,
                    -1.0,
                    1.0,
                    epsabs=tolerance,
                    epsrel=tolerance,
                    limit=limit,
                )
                component_inner_errors.append(float(inner_error))
                return inner_value

            value, outer_error = quad(
                outer_integrand,
                -1.0,
                1.0,
                epsabs=tolerance,
                epsrel=tolerance,
                limit=limit,
            )
            values[component] = value
            diagnostics[component] = float(outer_error) + 2.0 * max(
                inner_errors, default=0.0
            )
        warning_messages.extend(str(item.message) for item in caught)

    physical_factor = _transfer_scale(observable)
    result = (physical_factor * values).reshape(shape)
    component_error = (physical_factor * diagnostics).reshape(shape)
    return {
        "value": result,
        "component_error": component_error,
        "aggregate_error": block_norm(component_error, observable),
        "evaluations": counter.count,
        "warnings": warning_messages,
        "tolerance": tolerance,
        "limit": limit,
        "evaluation_cap": evaluation_cap,
    }


def adaptive_surface_reference_pair(target_m, observable: str) -> dict:
    """Return the 1e-10/1e-12 reference refinement and final uncertainty."""
    initial = adaptive_surface_reference(target_m, observable, tolerance=1e-10)
    final = adaptive_surface_reference(target_m, observable, tolerance=1e-12)
    refinement = block_norm(final["value"] - initial["value"], observable)
    uncertainty = max(refinement, final["aggregate_error"])
    status = (
        "CONVERGED"
        if not initial["warnings"]
        and not final["warnings"]
        and initial["evaluations"] <= initial["evaluation_cap"]
        and final["evaluations"] <= final["evaluation_cap"]
        else "REFERENCE_NOT_CONVERGED"
    )
    return {
        "value": final["value"],
        "initial": initial,
        "final": final,
        "refinement_difference": refinement,
        "uncertainty": uncertainty,
        "status": status,
    }


def _whole_interval_rule(order: int, lower: float, upper: float):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    return midpoint + half * nodes, half * weights


def signed_volume_observables(
    alpha_m: float,
    targets_m,
    *,
    m_order: int,
    k_order: int,
) -> dict[str, np.ndarray]:
    """Integrate the signed replacement layer without gragra production code."""
    x, wx = _whole_interval_rule(m_order, -L_M, L_M)
    y, wy = _whole_interval_rule(m_order, -L_M, L_M)
    t, wt = _whole_interval_rule(k_order, 0.0, 1.0)
    xx, yy, tt = np.meshgrid(x, y, t, indexing="ij")
    wxx, wyy, wtt = np.meshgrid(wx, wy, wt, indexing="ij")
    shape = _shape_function(xx / L_M, yy / L_M)
    zz = alpha_m * shape * tt
    sources = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    signed_volume = (wxx * wyy * wtt * alpha_m * shape).reshape(-1)
    mass_weights = DENSITY_JUMP_KG_M3 * signed_volume

    result: dict[str, list[np.ndarray]] = {name: [] for name in OBSERVABLES}
    for target in np.asarray(targets_m, dtype=np.float64):
        for observable in OBSERVABLES:
            kernel = _kernel(observable, sources, target)
            reshape = (len(mass_weights),) + (1,) * len(SHAPES[observable])
            integrated = G_SI_REFERENCE * np.sum(
                kernel * mass_weights.reshape(reshape), axis=0
            )
            result[observable].append(integrated)
    return {name: np.asarray(rows) for name, rows in result.items()}


def central_volume_derivative(
    alpha_m: float,
    targets_m,
    *,
    m_order: int,
    k_order: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return central and same-node symmetric-layer derivatives in Q/m."""
    if alpha_m <= 0.0:
        raise ValueError("alpha_m must be positive")
    plus = signed_volume_observables(
        alpha_m, targets_m, m_order=m_order, k_order=k_order
    )
    minus = signed_volume_observables(
        -alpha_m, targets_m, m_order=m_order, k_order=k_order
    )

    # Evaluate the algebraically equivalent symmetric layer with the identical
    # mapped nodes and weights, avoiding subtraction of Q(+alpha) and Q(-alpha).
    x, wx = _whole_interval_rule(m_order, -L_M, L_M)
    y, wy = _whole_interval_rule(m_order, -L_M, L_M)
    t, wt = _whole_interval_rule(k_order, 0.0, 1.0)
    xx, yy, tt = np.meshgrid(x, y, t, indexing="ij")
    wxx, wyy, wtt = np.meshgrid(wx, wy, wt, indexing="ij")
    shape = _shape_function(xx / L_M, yy / L_M)
    z_abs = alpha_m * shape * tt
    xy = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    z_flat = z_abs.reshape(-1)
    sources_plus = np.column_stack((xy, z_flat))
    sources_minus = np.column_stack((xy, -z_flat))
    derivative_weights = (DENSITY_JUMP_KG_M3 * wxx * wyy * wtt * shape).reshape(-1)

    output: dict[str, dict[str, np.ndarray]] = {}
    targets = np.asarray(targets_m, dtype=np.float64)
    for observable in OBSERVABLES:
        direct_rows = []
        for target in targets:
            symmetric_kernel = 0.5 * (
                _kernel(observable, sources_plus, target)
                + _kernel(observable, sources_minus, target)
            )
            reshape = (len(derivative_weights),) + (1,) * len(SHAPES[observable])
            direct_rows.append(
                G_SI_REFERENCE
                * np.sum(symmetric_kernel * derivative_weights.reshape(reshape), axis=0)
            )
        central = (plus[observable] - minus[observable]) / (2.0 * alpha_m)
        output[observable] = {
            "plus": plus[observable],
            "minus": minus[observable],
            "derivative": central,
            "symmetric_derivative": np.asarray(direct_rows),
        }
    return output
