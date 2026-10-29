#!/usr/bin/env python3
"""Validate a finite smooth water surface and common CSD propagation.

The deterministic fixture validates the existing gragra surface-density path.
It is not either literature water model; both model-specific inputs remain
blocked until their original normalization and covariance definitions exist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from gragra.constants import G_SI
from gragra.fields import FixedMeshSurfaceDensitySeries
from gragra.observables.series import surface_density_series_observable

L_M = 1.0
ETA_REF_M = 0.02
DENSITY_JUMP_KG_M3 = 1000.0 - 1.2
TARGETS_M = np.array(
    (
        (0.2, 0.1, 2.5),
        (-0.2, 0.1, 2.5),
        (0.2, -0.1, 2.5),
        (-0.2, -0.1, 2.5),
    ),
    dtype=np.float64,
)
TARGETS_M.setflags(write=False)
OBSERVABLES = ("potential", "acceleration", "gradient")
COMPONENT_SHAPES = {"potential": (), "acceleration": (3,), "gradient": (3, 3)}
EPSILON = {"potential": 1e-6, "acceleration": 1e-6, "gradient": 1e-5}
N_SEQUENCE = (16, 32, 64, 128, 256)
ALPHAS_M = (0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625)
GRADIENT_COMPONENT_ORDER = (
    "xx",
    "xy",
    "xz",
    "yx",
    "yy",
    "yz",
    "zx",
    "zy",
    "zz",
)
OBSERVABLE_UNITS = {
    "potential": "m^2/s^2",
    "acceleration": "m/s^2",
    "gradient": "1/s^2",
}
TRANSFER_UNITS = {
    "potential": "m/s^2",
    "acceleration": "1/s^2",
    "gradient": "1/(m*s^2)",
}
OUTPUT_CSD_UNITS = {
    "potential": "m^4/s^4/Hz",
    "acceleration": "m^2/s^4/Hz",
    "gradient": "1/s^4/Hz",
}
SOURCE_MODEL_STATUS = {
    "status": "SOURCE_INPUT_BLOCKED",
    "model_arrays_generated": False,
    "missing_inputs": [
        "official_revision_identity",
        "model1_covariance",
        "model1_physical_cell_prescription",
        "complete_fourier_psd_convention",
        "corrected_downstream_spectrum",
        "Chen_2016_original",
        "Nishizawa_DocDB_6440",
        "Somiya_DocDB_10792",
        "edge_effect_DocDB_14308",
    ],
    "candidate_note": (
        "Nishizawa water-v5 explicitly marks the 2pi correction complete only "
        "through Eq. 8; later spectrum normalization is not adopted."
    ),
}


def _shape_function(x_m, y_m):
    return np.cos(np.pi * x_m / (2.0 * L_M)) * np.cos(np.pi * y_m / (2.0 * L_M))


def composite_gl2_surface(n_cells: int):
    """Return z=0 composite two-point GL nodes and area weights."""
    if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells <= 0:
        raise ValueError("n_cells must be a positive integer")
    local_nodes = np.array((-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)))
    edges = np.linspace(-L_M, L_M, n_cells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_width = L_M / n_cells
    axis_nodes = (centers[:, None] + half_width * local_nodes).reshape(-1)
    xx, yy = np.meshgrid(axis_nodes, axis_nodes, indexing="ij")
    positions = np.column_stack((xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)))
    areas = np.full(xx.size, half_width**2, dtype=np.float64)
    normals = np.zeros_like(positions)
    normals[:, 2] = 1.0
    source_ids = tuple(f"gl2-{index:06d}" for index in range(xx.size))
    return positions, areas, normals, source_ids


def make_surface_series(
    n_cells: int,
    amplitude_m: complex,
    *,
    density_jump_kg_m3: float = DENSITY_JUMP_KG_M3,
    frequency_hz: float | None = None,
    t_ref_s: float | None = None,
    phase_origin_m: np.ndarray | None = None,
) -> FixedMeshSurfaceDensitySeries:
    """Build one physical-height sample without using its sample axis as a basis."""
    positions, areas, normals, source_ids = composite_gl2_surface(n_cells)
    density = (
        density_jump_kg_m3
        * amplitude_m
        * _shape_function(positions[:, 0], positions[:, 1])
    )
    if frequency_hz is None:
        if np.iscomplexobj(amplitude_m):
            raise ValueError("complex amplitude requires frequency_hz")
        axis_values = np.array((0.0,))
        axis_kind = "time"
        kwargs = {}
    else:
        axis_values = np.array((frequency_hz,), dtype=np.float64)
        axis_kind = "frequency"
        kwargs = {
            "t_ref_s": t_ref_s,
            "phase_origin_m": phase_origin_m,
        }
    return FixedMeshSurfaceDensitySeries(
        positions,
        areas,
        normals,
        np.asarray(density).reshape(1, -1),
        axis_values,
        axis_kind,
        {
            "fixture": "finite_smooth_water_surface",
            "quadrature": "composite_gl2",
            "n_cells_per_axis": n_cells,
            "normal": "+z_water_to_air",
        },
        "local_cartesian",
        np.zeros(3),
        source_ids,
        {
            "density_jump_kg_m3": density_jump_kg_m3,
            "amplitude_m": float(np.real(amplitude_m))
            if not np.iscomplexobj(amplitude_m)
            else (float(np.real(amplitude_m)), float(np.imag(amplitude_m))),
        },
        **kwargs,
    )


def evaluate_surface(
    n_cells: int,
    amplitude_m: complex,
    *,
    targets_m: np.ndarray = TARGETS_M,
    density_jump_kg_m3: float = DENSITY_JUMP_KG_M3,
    frequency_hz: float | None = None,
    t_ref_s: float | None = None,
    phase_origin_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate all three SI observables through the existing series path."""
    series = make_surface_series(
        n_cells,
        amplitude_m,
        density_jump_kg_m3=density_jump_kg_m3,
        frequency_hz=frequency_hz,
        t_ref_s=t_ref_s,
        phase_origin_m=phase_origin_m,
    )
    output = {}
    for observable in OBSERVABLES:
        value = surface_density_series_observable(
            series,
            observable,
            targets_m,
            softening_m=0.0,
            chunk_size=8192,
            backend="numpy",
        )[0]
        value.setflags(write=False)
        output[observable] = value
    return output


def propagate_csd(transfer, input_csd) -> np.ndarray:
    """Propagate one component-by-basis transfer block as H S H-dagger."""
    h = np.asarray(transfer, dtype=np.complex128)
    spectrum = np.asarray(input_csd, dtype=np.complex128)
    if h.ndim != 2:
        raise ValueError("transfer must have shape (components, bases)")
    if spectrum.shape != (h.shape[1], h.shape[1]):
        raise ValueError("input_csd must have shape (bases, bases)")
    result = h @ spectrum @ np.conjugate(h.T)
    result.setflags(write=False)
    return result


def csd_transfer_error_bound(transfer_reference, e_h: float, input_csd) -> float:
    """Return the predeclared H-error contribution to the output CSD 2-norm."""
    h_reference = np.asarray(transfer_reference, dtype=np.complex128)
    spectrum = np.asarray(input_csd, dtype=np.complex128)
    if h_reference.ndim != 2:
        raise ValueError("transfer_reference must have shape (components, bases)")
    if spectrum.shape != (h_reference.shape[1], h_reference.shape[1]):
        raise ValueError("input_csd must have shape (bases, bases)")
    if not np.isfinite(e_h) or e_h < 0.0:
        raise ValueError("e_h must be finite and non-negative")
    return float(
        (2.0 * np.linalg.norm(h_reference, ord=2) * e_h + e_h**2)
        * np.linalg.norm(spectrum, ord=2)
    )


def phasor_samples(q_hat, frequency_hz: float, times_s, *, t_ref_s: float):
    """Evaluate Re[q_hat exp(-2 pi i f (t-t_ref))] with component axis last."""
    q = np.asarray(q_hat, dtype=np.complex128)
    times = np.asarray(times_s, dtype=np.float64)
    phase = np.exp(-2.0j * np.pi * frequency_hz * (times - t_ref_s))
    return np.real(phase[:, None] * q[None, :])


def _norm(value, observable: str) -> float:
    array = np.asarray(value)
    if observable == "potential":
        return float(np.abs(array))
    return float(np.linalg.norm(array))


def _scale(observable: str, *, transfer: bool) -> float:
    length_factor = {
        "potential": L_M,
        "acceleration": 1.0,
        "gradient": 1.0 / L_M,
    }[observable]
    scale = G_SI * abs(DENSITY_JUMP_KG_M3) * length_factor
    return scale if transfer else ETA_REF_M * scale


def _list(value):
    return np.asarray(value).tolist()


def _array_record(value) -> dict:
    array = np.asarray(value)
    record = {"shape": list(array.shape), "real": np.real(array).tolist()}
    if np.iscomplexobj(array):
        record["imag"] = np.imag(array).tolist()
    return record


@lru_cache(maxsize=1)
def _oracle_module():
    """Load the validation-only oracle without making ``tests`` a package."""
    path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "oracles"
        / "water_surface_reference.py"
    )
    spec = importlib.util.spec_from_file_location("_water_surface_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load independent water oracle from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _surface_references():
    adaptive_surface_reference_pair = _oracle_module().adaptive_surface_reference_pair
    references = []
    for target in TARGETS_M:
        references.append(
            {
                observable: adaptive_surface_reference_pair(target, observable)
                for observable in OBSERVABLES
            }
        )
    return references


def _surface_accuracy(references):
    production = {}
    evaluated_n = []
    adopted_n = None
    for n_cells in N_SEQUENCE:
        if n_cells > 64 and adopted_n is not None:
            break
        production[n_cells] = evaluate_surface(n_cells, ETA_REF_M)
        evaluated_n.append(n_cells)
        if n_cells < 64:
            continue
        all_pass = True
        for target_index in range(len(TARGETS_M)):
            for observable in OBSERVABLES:
                reference = references[target_index][observable]
                q_ref = ETA_REF_M * reference["value"]
                q = production[n_cells][observable][target_index]
                budget = EPSILON[observable] * (
                    _scale(observable, transfer=False) + _norm(q_ref, observable)
                )
                component_budget = EPSILON[observable] * (
                    _scale(observable, transfer=False) + np.abs(q_ref)
                )
                uncertainty = ETA_REF_M * reference["uncertainty"]
                all_pass &= (
                    reference["status"] == "CONVERGED"
                    and _norm(q - q_ref, observable) <= budget
                    and np.all(np.abs(q - q_ref) <= component_budget)
                    and uncertainty <= budget / 10.0
                )
        if all_pass:
            adopted_n = n_cells
    if adopted_n is None:
        adopted_n = evaluated_n[-1]

    target_records = []
    for target_index, target in enumerate(TARGETS_M):
        observable_records = {}
        for observable in OBSERVABLES:
            reference = references[target_index][observable]
            q_ref = ETA_REF_M * reference["value"]
            budget = EPSILON[observable] * (
                _scale(observable, transfer=False) + _norm(q_ref, observable)
            )
            component_budget = EPSILON[observable] * (
                _scale(observable, transfer=False) + np.abs(q_ref)
            )
            trends = []
            for n_cells in evaluated_n:
                production_value = production[n_cells][observable][target_index]
                error = production_value - q_ref
                trends.append(
                    {
                        "n_cells": n_cells,
                        "production_value": _list(production_value),
                        "norm_error": _norm(error, observable),
                        "component_abs_error": _list(np.abs(error).reshape(-1)),
                    }
                )
            final_value = production[adopted_n][observable][target_index]
            final_error = final_value - q_ref
            reference_uncertainty = ETA_REF_M * reference["uncertainty"]
            failure_causes = []
            if (
                reference["status"] != "CONVERGED"
                or reference_uncertainty > budget / 10.0
            ):
                failure_causes.append("REFERENCE_NOT_CONVERGED")
            if _norm(final_error, observable) > budget:
                failure_causes.append("PRODUCTION_BLOCK_NORM_NOT_CONVERGED")
            if np.any(np.abs(final_error) > component_budget):
                failure_causes.append("PRODUCTION_COMPONENT_NOT_CONVERGED")
            observable_records[observable] = {
                "status": "VERIFIED" if not failure_causes else "FAILED",
                "failure_causes": failure_causes,
                "reference_status": reference["status"],
                "reference_value": _list(q_ref),
                "production_value": _list(final_value),
                "reference_initial_tolerance": 1e-10,
                "reference_final_tolerance": 1e-12,
                "reference_refinement_difference": ETA_REF_M
                * reference["refinement_difference"],
                "reference_adaptive_error": ETA_REF_M
                * reference["final"]["aggregate_error"],
                "reference_component_error": _list(
                    ETA_REF_M
                    * np.asarray(reference["final"]["component_error"]).reshape(-1)
                ),
                "reference_uncertainty": reference_uncertainty,
                "reference_evaluations": reference["final"]["evaluations"],
                "reference_warnings": reference["initial"]["warnings"]
                + reference["final"]["warnings"],
                "norm_error": _norm(final_error, observable),
                "norm_budget": budget,
                "norm_pass": _norm(final_error, observable) <= budget,
                "component_abs_error": _list(np.abs(final_error).reshape(-1)),
                "component_budget": _list(np.asarray(component_budget).reshape(-1)),
                "component_pass": _list(
                    (np.abs(final_error) <= component_budget).reshape(-1)
                ),
                "trend": trends,
            }
        target_records.append(
            {"target_m": target.tolist(), "observables": observable_records}
        )
    passed = all(
        block["reference_status"] == "CONVERGED"
        and block["norm_pass"]
        and all(block["component_pass"])
        and block["reference_uncertainty"] <= block["norm_budget"] / 10.0
        for target in target_records
        for block in target["observables"].values()
    )
    return {
        "status": "VERIFIED" if passed else "CONVERGENCE_NOT_MET",
        "evaluated_n": evaluated_n,
        "adopted_n": adopted_n,
        "maximum_n": 256,
        "targets": target_records,
    }


def _classify_volume_block(
    *,
    reference_ok: bool,
    u_m_ok: bool,
    u_k_ok: bool,
    u_vol_ok: bool,
    roundoff_ok: bool,
    total_error: float,
    budget: float,
) -> dict:
    """Classify numerical prerequisites before assessing linearization."""
    failure_causes = []
    if not reference_ok:
        failure_causes.append("REFERENCE_NOT_CONVERGED")
    if not u_m_ok:
        failure_causes.append("QUADRATURE_U_M_NOT_CONVERGED")
    if not u_k_ok:
        failure_causes.append("QUADRATURE_U_K_NOT_CONVERGED")
    if not u_vol_ok:
        failure_causes.append("QUADRATURE_U_VOL_NOT_CONVERGED")
    if not roundoff_ok:
        failure_causes.append("ROUNDOFF_NOT_RESOLVED")

    numerical_prerequisites_ok = reference_ok and u_vol_ok and roundoff_ok
    if not numerical_prerequisites_ok:
        linearization_status = "NOT_ASSESSED"
    elif total_error > budget:
        linearization_status = "NOT_RESOLVED"
        failure_causes.append("LINEARIZATION_NOT_RESOLVED")
    else:
        linearization_status = "WITHIN_TOTAL_BUDGET"
    return {
        "status": "VERIFIED" if not failure_causes else "FAILED",
        "failure_causes": failure_causes,
        "u_m_status": "VERIFIED" if u_m_ok else "NOT_CONVERGED",
        "u_k_status": "VERIFIED" if u_k_ok else "NOT_CONVERGED",
        "u_vol_status": "VERIFIED" if u_vol_ok else "NOT_CONVERGED",
        "roundoff_status": "VERIFIED" if roundoff_ok else "NOT_RESOLVED",
        "linearization_status": linearization_status,
    }


def _select_amplitude_record(alpha_records: list[dict]) -> tuple[dict, str]:
    """Select the first passing alpha, or retain the last with neutral status."""
    if not alpha_records:
        raise ValueError("alpha_records must not be empty")
    adopted = next(
        (
            record
            for record in alpha_records
            if all(
                block["pass"]
                for target in record["targets"]
                for block in target["observables"].values()
            )
        ),
        None,
    )
    if adopted is not None:
        return adopted, "VERIFIED"
    return alpha_records[-1], "NOT_VERIFIED"


def _volume_amplitude(references):
    central_volume_derivative = _oracle_module().central_volume_derivative
    cache = {}

    def calculation(alpha, m_order, k_order):
        key = (alpha, m_order, k_order)
        if key not in cache:
            cache[key] = central_volume_derivative(
                alpha,
                TARGETS_M,
                m_order=m_order,
                k_order=k_order,
            )
        return cache[key]

    adopted_m, adopted_k = 32, 16
    # Escalation is based on the predeclared U_vol budget over every retained
    # amplitude, target and observable; tolerances are never relaxed.
    while True:
        need_m = False
        need_k = False
        for alpha in ALPHAS_M:
            main = calculation(alpha, adopted_m, adopted_k)
            back_m = calculation(alpha, adopted_m // 2, adopted_k)
            back_k = calculation(alpha, adopted_m, adopted_k // 2)
            for target_index in range(len(TARGETS_M)):
                for observable in OBSERVABLES:
                    t_ref = references[target_index][observable]["value"]
                    budget = EPSILON[observable] * (
                        _scale(observable, transfer=True) + _norm(t_ref, observable)
                    )
                    derivative = main[observable]["derivative"][target_index]
                    u_m = _norm(
                        derivative - back_m[observable]["derivative"][target_index],
                        observable,
                    )
                    u_k = _norm(
                        derivative - back_k[observable]["derivative"][target_index],
                        observable,
                    )
                    if u_m + u_k > budget / 10.0:
                        if u_m >= u_k:
                            if adopted_m < 64:
                                need_m = True
                            elif adopted_k < 32:
                                need_k = True
                        else:
                            if adopted_k < 32:
                                need_k = True
                            elif adopted_m < 64:
                                need_m = True
        changed = False
        if need_m and adopted_m < 64:
            adopted_m = 64
            changed = True
        if need_k and adopted_k < 32:
            adopted_k = 32
            changed = True
        if not changed:
            break

    alpha_records = []
    for alpha in ALPHAS_M:
        main = calculation(alpha, adopted_m, adopted_k)
        back_m = calculation(alpha, adopted_m // 2, adopted_k)
        back_k = calculation(alpha, adopted_m, adopted_k // 2)
        target_records = []
        for target_index, target in enumerate(TARGETS_M):
            observable_records = {}
            for observable in OBSERVABLES:
                t_ref = references[target_index][observable]["value"]
                derivative = main[observable]["derivative"][target_index]
                symmetric = main[observable]["symmetric_derivative"][target_index]
                q_plus = main[observable]["plus"][target_index]
                q_minus = main[observable]["minus"][target_index]
                u_m = _norm(
                    derivative - back_m[observable]["derivative"][target_index],
                    observable,
                )
                u_k = _norm(
                    derivative - back_k[observable]["derivative"][target_index],
                    observable,
                )
                u_vol = u_m + u_k
                arithmetic_floor = (
                    128.0
                    * np.finfo(np.float64).eps
                    * (_norm(q_plus, observable) + _norm(q_minus, observable))
                    / (2.0 * alpha)
                )
                same_node_difference = _norm(derivative - symmetric, observable)
                u_round = max(arithmetic_floor, same_node_difference)
                u_ref = references[target_index][observable]["uncertainty"]
                residual = _norm(derivative - t_ref, observable)
                budget = EPSILON[observable] * (
                    _scale(observable, transfer=True) + _norm(t_ref, observable)
                )
                total_error = residual + u_ref + u_vol + u_round
                reference_ok = (
                    references[target_index][observable]["status"] == "CONVERGED"
                    and u_ref <= budget / 10.0
                )
                u_m_ok = u_m <= budget / 10.0
                u_k_ok = u_k <= budget / 10.0
                u_vol_ok = u_vol <= budget / 10.0
                roundoff_ok = u_round <= budget / 10.0
                classification = _classify_volume_block(
                    reference_ok=reference_ok,
                    u_m_ok=u_m_ok,
                    u_k_ok=u_k_ok,
                    u_vol_ok=u_vol_ok,
                    roundoff_ok=roundoff_ok,
                    total_error=total_error,
                    budget=budget,
                )
                observable_records[observable] = {
                    **classification,
                    "residual": residual,
                    "budget": budget,
                    "u_ref": u_ref,
                    "u_m": u_m,
                    "u_k": u_k,
                    "u_vol": u_vol,
                    "u_round_arithmetic": arithmetic_floor,
                    "u_round_same_node": same_node_difference,
                    "u_round": u_round,
                    "total_error": total_error,
                    "reference_status": (
                        references[target_index][observable]["status"]
                    ),
                    "pass": not classification["failure_causes"],
                    "derivative": _list(derivative),
                    "symmetric_derivative": _list(symmetric),
                    "q_plus": _list(q_plus),
                    "q_minus": _list(q_minus),
                    "richardson": None,
                }
            target_records.append(
                {"target_m": target.tolist(), "observables": observable_records}
            )
        alpha_records.append({"alpha_m": alpha, "targets": target_records})

    for index in range(len(alpha_records) - 1):
        current = alpha_records[index]
        finer = alpha_records[index + 1]
        for target_index in range(len(TARGETS_M)):
            for observable in OBSERVABLES:
                coarse_value = np.asarray(
                    current["targets"][target_index]["observables"][observable][
                        "derivative"
                    ]
                )
                fine_value = np.asarray(
                    finer["targets"][target_index]["observables"][observable][
                        "derivative"
                    ]
                )
                richardson = (4.0 * fine_value - coarse_value) / 3.0
                current["targets"][target_index]["observables"][observable][
                    "richardson"
                ] = _list(richardson)

    order_records = []
    for index in range(len(alpha_records) - 1):
        coarse = alpha_records[index]
        fine = alpha_records[index + 1]
        for target_index in range(len(TARGETS_M)):
            for observable in OBSERVABLES:
                coarse_block = coarse["targets"][target_index]["observables"][
                    observable
                ]
                fine_block = fine["targets"][target_index]["observables"][observable]
                coarse_floor = (
                    coarse_block["u_ref"]
                    + coarse_block["u_vol"]
                    + coarse_block["u_round"]
                )
                fine_floor = (
                    fine_block["u_ref"] + fine_block["u_vol"] + fine_block["u_round"]
                )
                resolvable = (
                    coarse_block["residual"] > 10.0 * coarse_floor
                    and fine_block["residual"] > 10.0 * fine_floor
                )
                observed_order = None
                if resolvable:
                    observed_order = float(
                        np.log(coarse_block["residual"] / fine_block["residual"])
                        / np.log(coarse["alpha_m"] / fine["alpha_m"])
                    )
                order_records.append(
                    {
                        "target_index": target_index,
                        "observable": observable,
                        "alpha_pair_m": [coarse["alpha_m"], fine["alpha_m"]],
                        "coarse_residual": coarse_block["residual"],
                        "fine_residual": fine_block["residual"],
                        "coarse_numerical_floor": coarse_floor,
                        "fine_numerical_floor": fine_floor,
                        "status": "MEASURED" if resolvable else "AT_NUMERICAL_FLOOR",
                        "observed_order": observed_order,
                    }
                )

    adopted_record, status = _select_amplitude_record(alpha_records)
    return {
        "status": status,
        "evaluated_alphas_m": list(ALPHAS_M),
        "adopted_alpha_m": adopted_record["alpha_m"],
        "adopted_m": adopted_m,
        "adopted_k": adopted_k,
        "maximum_m": 64,
        "maximum_k": 32,
        "targets": adopted_record["targets"],
        "alpha_records": alpha_records,
        "order_diagnostics": {
            "required_for_acceptance": False,
            "floor_separation_factor": 10.0,
            "measured_count": sum(
                item["status"] == "MEASURED" for item in order_records
            ),
            "excluded_count": sum(
                item["status"] != "MEASURED" for item in order_records
            ),
            "records": order_records,
        },
        "order_claim": (
            "Richardson values are diagnostic; no second-order claim is made "
            "for zero coefficients or values at the numerical floor."
        ),
    }


def _common_spectra(references):
    from scipy.signal import csd

    h_real = np.array(((1.0, 2.0), (3.0, -1.0)))
    h_complex = np.array(((1.0, 1.0j), (1.0, -1.0j)))
    independent = np.diag((2.0, 3.0))
    q_hat = np.array((1.0, 1.0j))
    line_csd = np.outer(q_hat, np.conjugate(q_hat)) / 2.0
    single_basis_input_csd = np.array(((1.0,),))
    fixture_pass = all(
        (
            np.array_equal(
                propagate_csd(h_real, independent),
                np.array(((14.0, 0.0), (0.0, 21.0))),
            ),
            np.array_equal(
                propagate_csd(h_real, line_csd),
                np.array(((2.5, 0.5 + 3.5j), (0.5 - 3.5j, 5.0))),
            ),
            np.array_equal(
                propagate_csd(h_complex, independent),
                np.array(((5.0, -1.0), (-1.0, 5.0))),
            ),
            np.array_equal(
                propagate_csd(h_complex, line_csd),
                np.array(((0.0, 0.0), (0.0, 2.0))),
            ),
        )
    )

    sample_count = 256
    times = np.arange(sample_count) / sample_count
    samples = phasor_samples(q_hat, 7.0, times, t_ref_s=0.0)
    positive_bin = np.fft.rfft(samples, axis=0)[7] / sample_count
    fft_pass = np.allclose(positive_bin, np.conjugate(q_hat) / 2.0, atol=2e-15)
    frequencies, cross = csd(
        samples[:, 0],
        samples[:, 1],
        fs=float(sample_count),
        window="boxcar",
        nperseg=sample_count,
        noverlap=0,
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
    )
    scipy_csd_pass = frequencies[7] == 7.0 and np.allclose(cross[7], -0.5j, atol=2e-15)

    csd_blocks = {}
    all_hermitian = True
    all_psd = True
    for observable in OBSERVABLES:
        rows = []
        min_eigenvalues = []
        hermitian_residuals = []
        numerical_ranks = []
        rank_tolerances = []
        for target_index in range(len(TARGETS_M)):
            transfer = np.asarray(
                references[target_index][observable]["value"]
            ).reshape(-1, 1)
            output = propagate_csd(transfer, single_basis_input_csd)
            rows.append(output)
            hermitian_residuals.append(
                float(np.linalg.norm(output - np.conjugate(output.T)))
            )
            min_eigenvalues.append(float(np.min(np.linalg.eigvalsh(output))))
            singular_values = np.linalg.svd(output, compute_uv=False)
            rank_tolerance = float(
                max(output.shape) * np.finfo(float).eps * singular_values[0]
            )
            rank_tolerances.append(rank_tolerance)
            numerical_ranks.append(
                int(np.count_nonzero(singular_values > rank_tolerance))
            )
        stacked = np.asarray(rows)
        roundoff = (
            128.0
            * np.finfo(float).eps
            * max(float(np.linalg.norm(stacked)), np.finfo(float).tiny)
        )
        hermitian = max(hermitian_residuals) <= roundoff
        psd = min(min_eigenvalues) >= -roundoff
        all_hermitian &= hermitian
        all_psd &= psd
        csd_blocks[observable] = {
            "shape": list(stacked.shape),
            "values": _array_record(stacked),
            "hermitian": hermitian,
            "psd_within_roundoff": psd,
            "minimum_eigenvalue": min(min_eigenvalues),
            "roundoff_width": roundoff,
            "component_order": {
                "potential": ["scalar"],
                "acceleration": ["x", "y", "z"],
                "gradient": list(GRADIENT_COMPONENT_ORDER),
            }[observable],
            "output_units": OUTPUT_CSD_UNITS[observable],
            "numerical_rank_by_target": numerical_ranks,
            "rank_tolerance_by_target": rank_tolerances,
            "fixture_rank_limit": 1,
            "no_jitter_clipping_or_projection": True,
        }
    status = (
        "VERIFIED"
        if fixture_pass and fft_pass and scipy_csd_pass and all_hermitian and all_psd
        else "FAILED"
    )
    return {
        "status": status,
        "fourier_convention": "Re[q_hat exp(-2pi i f(t-t_ref_s))]",
        "input_csd_index_order": "E[eta_i conj(eta_j)]",
        "input_csd": {
            "value": single_basis_input_csd.tolist(),
            "eta_units": "m",
            "units": "m^2/Hz",
            "spectral_measure": "one_sided_Hz_density",
        },
        "literal_line_fixture": {
            "definition": "q_hat q_hat^dagger / 2",
            "spectral_measure": "bin_integrated_line_power",
            "is_hz_density": False,
        },
        "scipy_csd_fixture": {
            "scaling": "spectrum",
            "frequency_bin_width_hz": float(frequencies[1] - frequencies[0]),
            "spectral_measure": "bin_integrated_line_power",
            "is_hz_density": False,
        },
        "transfer_source": "independent_continuous_reference",
        "transfer_validation_scope": (
            "This transfer is not the production quadrature H and does not "
            "validate a literature source model."
        ),
        "structural_rank_limits": {
            "fixture_single_basis": 1,
            "general_gradient_stf_single_point_support": 5,
        },
        "fft_positive_bin_matches_conjugate_phasor": bool(fft_pass),
        "scipy_csd_imaginary_sign_matches": bool(scipy_csd_pass),
        "literal_propagation_fixtures_pass": bool(fixture_pass),
        "single_basis_output_csd": csd_blocks,
        "scope": "deterministic single smooth basis only; not a literature-model H",
    }


def _units_record() -> dict:
    return {
        "observable_q": OBSERVABLE_UNITS,
        "eta": "m",
        "transfer_q_per_eta": TRANSFER_UNITS,
        "input_csd": "m^2/Hz",
        "output_csd": OUTPUT_CSD_UNITS,
        "csd_definition": "Q_i*conj(Q_j)/Hz, separately within each observable block",
        "gradient_component_order": list(GRADIENT_COMPONENT_ORDER),
    }


def _fixture_record() -> dict:
    return {
        "surface": "z=0, [-1,1] m x [-1,1] m",
        "shape": "cos(pi*x/(2L))*cos(pi*y/(2L))",
        "l_m": L_M,
        "eta_ref_m": ETA_REF_M,
        "density_water_kg_m3": 1000.0,
        "density_air_kg_m3": 1.2,
        "density_jump_kg_m3": DENSITY_JUMP_KG_M3,
        "targets_m": TARGETS_M.tolist(),
        "potential_gauge": "finite patch, zero at infinity",
        "softening_m": 0.0,
    }


def _scope_limit() -> str:
    return (
        "The smooth finite fixture validates this gravity path only; it does not "
        "validate untested model bases, frequencies, or channel lengths."
    )


def run_deterministic_validation() -> dict:
    """Run every currently executable water-surface acceptance gate."""
    try:
        references = _surface_references()
    except _oracle_module().ReferenceEvaluationLimitError as error:
        return {
            "schema_version": 2,
            "classification": {
                "deterministic_fixture": "REFERENCE_NOT_CONVERGED",
                "common_csd": "NOT_RUN_REFERENCE_NOT_CONVERGED",
                "literature_models": "SOURCE_INPUT_BLOCKED",
            },
            "units": _units_record(),
            "fixture": _fixture_record(),
            "surface_accuracy": {
                "status": "REFERENCE_NOT_CONVERGED",
                "failure_causes": ["REFERENCE_EVALUATION_CAP_EXCEEDED"],
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            "amplitude_convergence": {"status": "NOT_RUN_REFERENCE_NOT_CONVERGED"},
            "common_spectra": {"status": "NOT_RUN_REFERENCE_NOT_CONVERGED"},
            "source_models": SOURCE_MODEL_STATUS,
            "overall_executable_status": "FAILED",
            "scope_limit": _scope_limit(),
        }
    surface = _surface_accuracy(references)
    amplitude = _volume_amplitude(references)
    spectra = _common_spectra(references)
    return {
        "schema_version": 2,
        "classification": {
            "deterministic_fixture": "Observed numerical validation",
            "common_csd": "Observed convention validation",
            "literature_models": "SOURCE_INPUT_BLOCKED",
        },
        "units": _units_record(),
        "fixture": _fixture_record(),
        "surface_accuracy": surface,
        "amplitude_convergence": amplitude,
        "common_spectra": spectra,
        "source_models": SOURCE_MODEL_STATUS,
        "overall_executable_status": (
            "VERIFIED"
            if surface["status"]
            == amplitude["status"]
            == spectra["status"]
            == "VERIFIED"
            else "FAILED"
        ),
        "scope_limit": _scope_limit(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _environment_record() -> dict:
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "float64": {
            "eps": np.finfo(np.float64).eps,
            "max": np.finfo(np.float64).max,
            "nmant": np.finfo(np.float64).nmant,
        },
    }


def write_validation(output_dir: Path) -> tuple[Path, Path]:
    """Write raw result and a reproduction manifest outside the repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_deterministic_validation()
    result_path = output_dir / "water-surface-results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    )

    repository = Path(__file__).resolve().parents[1]
    allowed_paths = (
        "scripts/research_water_surface_models.py",
        "tests/test_water_surface_models.py",
        "tests/oracles/water_surface_reference.py",
        "docs/design/water-surface-models.md",
        "docs/validation/water-surface-models.md",
    )
    code_hashes = {
        name: _sha256(repository / name)
        for name in allowed_paths
        if (repository / name).exists()
    }
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 2,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "repository": str(repository),
        "seed_head": head,
        "approved_plan_sha256": (
            "56ab29acf7a46a38838c853ed0cc4f867f287a9e521d7d01246cba89f98ab984"
        ),
        "environment": _environment_record(),
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "code_sha256": code_hashes,
    }
    manifest_path = output_dir / "water-surface-reproduction-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return result_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result, manifest = write_validation(args.output_dir)
    print(result)
    print(manifest)


if __name__ == "__main__":
    main()
