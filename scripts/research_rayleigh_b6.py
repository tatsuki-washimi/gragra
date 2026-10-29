#!/usr/bin/env python3
"""Run the saved-field Rayleigh B6 three-way comparison.

This is a bounded research script, not a general streaming or solver adapter.
It keeps element-wise GLL quadrature contributions and applies the
post-predictor snapshot time convention only through ``snapshot_physical_times``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The installed editable package may provide an import redirector.  B6 must
# use this science worktree's implementation explicitly.
_SCIENCE_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(_SCIENCE_ROOT / "src"))

from gragra.adapters.specfem.reader import read_specfem_gll_hdf5
from gragra.constants import G_SI
from gragra.observables.dipole import displacement_transfer_acceleration
from gragra.research.rayleigh_timefit import (
    complex_relative_error,
    fit_complex_amplitude,
    snapshot_physical_times,
)
from gragra.waves.rayleigh import (
    _depth_profiles,
    _rayleigh_parameters,
    rayleigh_displacement,
)


VP = 2800.0
VS = 1500.0
RHO = 2300.0
FREQUENCY_HZ = 2.5
PHASE_ORIGIN_X_M = 0.0
PHASE_ORIGIN_Y_M = 0.0
AMPLITUDE_M = 1.0e-9
C_R = 1390.8255605658555
LAMBDA_R = C_R / FREQUENCY_HZ
OMEGA = 2.0 * np.pi * FREQUENCY_HZ
T_INIT_S = 10.0
DT_S = 0.005
TARGETS = np.array(
    [
        [3000.0, 3000.0, 0.25 * LAMBDA_R],
        [3000.0, 3000.0, 0.50 * LAMBDA_R],
        [3000.0, 3000.0, 0.75 * LAMBDA_R],
    ],
    dtype=np.float64,
)
TARGET_LABELS = ("0.25_lambda", "0.5_lambda", "0.75_lambda")


def _complex_vector(values: np.ndarray) -> list[list[float]]:
    """Encode a complex vector as explicit real/imaginary pairs for JSON."""
    values = np.asarray(values, dtype=np.complex128).reshape(-1)
    return [[float(value.real), float(value.imag)] for value in values]


def _analytic_rayleigh_gravity() -> np.ndarray:
    """Return C, the infinite half-space Rayleigh gravity phasor above z=0.

    For a target height H>0 and a source field proportional to exp(i k x),
    the horizontal Fourier transform of the dipole kernel gives

      Cx = 2 pi G rho k exp(-k H) (Ix + i Iz)
      Cz = 2 pi G rho k exp(-k H) (i Ix - Iz),

    where Ix and Iz are the depth integrals of the Rayleigh displacement
    components multiplied by exp(-k h).  The two exponential integrals are
    evaluated in closed form using the same q, s, and normalization as
    ``rayleigh_displacement``.  This reference is intentionally local to the
    B6 research calculation and is not added to the public API.
    """
    k = 2.0 * np.pi / LAMBDA_R
    _, q, s = _rayleigh_parameters(VP, VS)
    f_x0, f_z0 = _depth_profiles(np.float64(0.0), q, s)
    coef_x = 2.0 * q * s / (1.0 + s * s)
    coef_z = 2.0 / (1.0 + s * s)
    # Same normalization as rayleigh_displacement: u_z(0)=AMPLITUDE_M.
    alpha = AMPLITUDE_M / (q * k * f_z0)
    ix = alpha * 1j * (1.0 / (1.0 + q) - coef_x / (1.0 + s))
    iz = alpha * q * (1.0 / (1.0 + q) - coef_z / (1.0 + s))
    out = np.empty((len(TARGETS), 3), dtype=np.complex128)
    height = TARGETS[:, 2]
    factor = 2.0 * np.pi * G_SI * RHO * k * np.exp(-k * height)
    out[:, 0] = factor * (ix + 1j * iz)
    out[:, 1] = 0.0
    out[:, 2] = factor * (1j * ix - iz)
    # The depth integrals above are evaluated at the phase origin.  C is
    # reported at the requested target x, so retain the same exp(i k x)
    # spatial phase used by rayleigh_displacement.
    out *= np.exp(1j * k * (TARGETS[:, 0] - PHASE_ORIGIN_X_M))[:, None]
    return out


def _fit_block(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Fit one source block with the shared Rayleigh timefit."""
    return fit_complex_amplitude(values, times_s, OMEGA, t_ref_s=T_INIT_S)


def _integrate_blocks(
    coordinates: np.ndarray,
    mass_kg: np.ndarray,
    displacement_complex: np.ndarray,
    *,
    block_sources: int = 256000,
) -> np.ndarray:
    """Apply the existing dipole observable blockwise, preserving GLL rows."""
    sources = coordinates.reshape(-1, 3)
    masses = mass_kg.reshape(-1)
    displacements = displacement_complex.reshape(-1, 3)
    total: np.ndarray | None = None
    for start in range(0, sources.shape[0], block_sources):
        stop = min(start + block_sources, sources.shape[0])
        contribution = displacement_transfer_acceleration(
            sources[start:stop],
            masses[start:stop],
            displacements[start:stop],
            TARGETS,
            chunk_size=65536,
        )
        total = contribution if total is None else total + contribution
    assert total is not None
    return np.asarray(total, dtype=np.complex128)


def _evaluate_window(dataset, mask: np.ndarray, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit A and B source blocks, then integrate each with the existing API."""
    assert dataset.displacement_m is not None
    n_elements, ngll, _ = dataset.coordinates_m.shape
    a_total = np.zeros((len(TARGETS), 3), dtype=np.complex128)
    b_total = np.zeros((len(TARGETS), 3), dtype=np.complex128)
    k = 2.0 * np.pi / LAMBDA_R
    for start in range(0, n_elements, 2048):
        stop = min(start + 2048, n_elements)
        coords = dataset.coordinates_m[start:stop]
        mass = dataset.density_kg_m3[start:stop] * dataset.quadrature_volume_m3[start:stop]
        sem = dataset.displacement_m[mask, start:stop]
        c_sem = _fit_block(sem.reshape(sem.shape[0], -1, 3), times_s)
        u0 = rayleigh_displacement(
            coords.reshape(-1, 3),
            (1.0, 0.0, 0.0),
            k,
            VP,
            VS,
            amplitude_m=AMPLITUDE_M,
        ).reshape(stop - start, ngll, 3)
        analytic_samples = np.real(
            u0[None, ...] * np.exp(-1j * OMEGA * (times_s - T_INIT_S)[:, None, None, None])
        )
        c_analytic = _fit_block(
            analytic_samples.reshape(analytic_samples.shape[0], -1, 3), times_s
        )
        a_total += _integrate_blocks(coords, mass, c_sem)
        b_total += _integrate_blocks(coords, mass, c_analytic)
    return a_total, b_total


def _element_residuals(
    dataset, mask: np.ndarray, times_s: np.ndarray
) -> np.ndarray:
    """Return per-element complex gravity contributions for A-B."""
    assert dataset.displacement_m is not None
    n_elements, ngll, _ = dataset.coordinates_m.shape
    residual = np.zeros((n_elements, len(TARGETS), 3), dtype=np.complex128)
    k = 2.0 * np.pi / LAMBDA_R
    for start in range(0, n_elements, 2048):
        stop = min(start + 2048, n_elements)
        coords = dataset.coordinates_m[start:stop]
        mass = dataset.density_kg_m3[start:stop] * dataset.quadrature_volume_m3[start:stop]
        sem = dataset.displacement_m[mask, start:stop]
        c_sem = _fit_block(sem.reshape(sem.shape[0], -1, 3), times_s)
        u0 = rayleigh_displacement(
            coords.reshape(-1, 3),
            (1.0, 0.0, 0.0),
            k,
            VP,
            VS,
            amplitude_m=AMPLITUDE_M,
        ).reshape(stop - start, ngll, 3)
        analytic_samples = np.real(
            u0[None, ...] * np.exp(-1j * OMEGA * (times_s - T_INIT_S)[:, None, None, None])
        )
        c_analytic = _fit_block(
            analytic_samples.reshape(analytic_samples.shape[0], -1, 3), times_s
        )
        c_sem = c_sem.reshape(stop - start, ngll, 3)
        c_analytic = c_analytic.reshape(stop - start, ngll, 3)
        for local in range(stop - start):
            residual[start + local] = displacement_transfer_acceleration(
                coords[local],
                mass[local],
                c_sem[local] - c_analytic[local],
                TARGETS,
                chunk_size=65536,
            )
    return residual


def _subset_timefit_check(dataset, times_s: np.ndarray) -> float:
    """Check time-domain gravity fit against fit-then-gravity once."""
    assert dataset.displacement_m is not None
    n_elem = min(16, dataset.element_id.size)
    coords = dataset.coordinates_m[:n_elem].reshape(-1, 3)
    mass = (
        dataset.density_kg_m3[:n_elem] * dataset.quadrature_volume_m3[:n_elem]
    ).reshape(-1)
    time_gravity = np.stack(
        [
            displacement_transfer_acceleration(
                coords, mass, sample[:n_elem].reshape(-1, 3), TARGETS, chunk_size=65536
            )
            for sample in dataset.displacement_m
        ],
        axis=0,
    )
    fitted_gravity = fit_complex_amplitude(
        time_gravity.real, times_s, OMEGA, t_ref_s=T_INIT_S
    )
    fitted_displacement = fit_complex_amplitude(
        dataset.displacement_m[:, :n_elem].reshape(len(times_s), -1, 3),
        times_s,
        OMEGA,
        t_ref_s=T_INIT_S,
    )
    gravity_after_fit = _integrate_blocks(
        dataset.coordinates_m[:n_elem],
        dataset.density_kg_m3[:n_elem] * dataset.quadrature_volume_m3[:n_elem],
        fitted_displacement,
    )
    return complex_relative_error(fitted_gravity, gravity_after_fit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    dataset = read_specfem_gll_hdf5(args.hdf5, profile_density_range=None)
    times_s = snapshot_physical_times(
        dataset.time_step, t_init_s=T_INIT_S, dt_s=DT_S
    )
    front_mask = (times_s >= T_INIT_S) & (times_s <= T_INIT_S + 0.8 + 1e-12)
    analytic = _analytic_rayleigh_gravity()
    results = []
    saved = {
        "time_h5": dataset.time_s,
        "time_corrected": times_s,
        "time_step": dataset.time_step,
        "targets_xyz_m": TARGETS,
    }
    a_windows = []
    b_windows = []
    residual_windows = []
    for label, mask in (("front_0_0.8s", front_mask), ("full_1.2s", np.ones(len(times_s), dtype=bool))):
        a, b = _evaluate_window(dataset, mask, times_s[mask])
        residual = _element_residuals(dataset, mask, times_s[mask])
        a_windows.append(a)
        b_windows.append(b)
        residual_windows.append(residual)
        for i, target_label in enumerate(TARGET_LABELS):
            c = analytic[i]
            results.append({
                "window": label,
                "target": target_label,
                "target_xyz_m": TARGETS[i].tolist(),
                "A_solver_finite_volume": _complex_vector(a[i]),
                "B_analytic_same_gll_finite_volume": _complex_vector(b[i]),
                "C_analytic_infinite_halfspace": _complex_vector(c),
                "A_minus_C_over_C": complex_relative_error(a[i], c),
                "A_minus_B_over_C": float(np.linalg.norm(a[i] - b[i]) / max(np.linalg.norm(c), 1e-30)),
                "B_minus_C_over_C": complex_relative_error(b[i], c),
                "component_diagnostics": [
                    {
                        "component": component,
                        "A_over_C_amplitude_ratio": float(abs(a[i, j]) / abs(c[j])) if abs(c[j]) > 1e-30 else None,
                        "A_minus_C_phase_deg": float(np.angle(a[i, j] / c[j], deg=True)) if abs(c[j]) > 1e-30 else None,
                        "B_over_C_amplitude_ratio": float(abs(b[i, j]) / abs(c[j])) if abs(c[j]) > 1e-30 else None,
                        "B_minus_C_phase_deg": float(np.angle(b[i, j] / c[j], deg=True)) if abs(c[j]) > 1e-30 else None,
                        "A_minus_C_absolute": float(abs(a[i, j] - c[j])),
                        "B_minus_C_absolute": float(abs(b[i, j] - c[j])),
                    }
                    for j, component in enumerate(("x", "y", "z"))
                ],
            })
    subset_check = _subset_timefit_check(dataset, times_s)
    saved["A_complex"] = np.asarray(a_windows)
    saved["B_complex"] = np.asarray(b_windows)
    saved["C_complex"] = np.broadcast_to(analytic, (2, *analytic.shape))
    saved["element_id"] = dataset.element_id
    saved["element_residual_A_minus_B"] = np.asarray(residual_windows)

    element_bounds = np.stack(
        [
            dataset.coordinates_m.min(axis=1),
            dataset.coordinates_m.max(axis=1),
        ],
        axis=1,
    )
    saved["element_bounds_m"] = element_bounds
    boundary_atol_m = 1.0e-5
    region_results = []
    side_bottom_results = []
    for d in (500.0, 1000.0):
        inner_mask = (
            (element_bounds[:, 0, 0] >= d - boundary_atol_m)
            & (element_bounds[:, 1, 0] <= 6000.0 - d + boundary_atol_m)
            & (element_bounds[:, 0, 1] >= d - boundary_atol_m)
            & (element_bounds[:, 1, 1] <= 6000.0 - d + boundary_atol_m)
            & (element_bounds[:, 0, 2] >= -3000.0 + d - boundary_atol_m)
            & (element_bounds[:, 1, 2] <= boundary_atol_m)
        )
        for window_index, window_name in enumerate(("front_0_0.8s", "full_1.2s")):
            inner = residual_windows[window_index][inner_mask].sum(axis=0)
            shell = residual_windows[window_index][~inner_mask].sum(axis=0)
            total = a_windows[window_index] - b_windows[window_index]
            region_results.append(
                {
                    "d_m": d,
                    "window": window_name,
                    "inner_element_count": int(inner_mask.sum()),
                    "shell_element_count": int((~inner_mask).sum()),
                    "boundary_classification_atol_m": boundary_atol_m,
                    "inner_complex": [_complex_vector(value) for value in inner],
                    "shell_complex": [_complex_vector(value) for value in shell],
                    "total_A_minus_B_complex": _complex_vector(total),
                    "inner_plus_shell_complex": _complex_vector(inner + shell),
                    "sum_check_absolute_m_s2": float(np.max(np.abs(inner + shell - total))),
                    "inner_norm_m_s2": np.linalg.norm(inner, axis=1).tolist(),
                    "shell_norm_m_s2": np.linalg.norm(shell, axis=1).tolist(),
                    "total_norm_m_s2": np.linalg.norm(total, axis=1).tolist(),
                    "inner_norm_over_C": [float(np.linalg.norm(inner[i]) / max(np.linalg.norm(analytic[i]), 1e-30)) for i in range(len(TARGETS))],
                    "shell_norm_over_C": [float(np.linalg.norm(shell[i]) / max(np.linalg.norm(analytic[i]), 1e-30)) for i in range(len(TARGETS))],
                }
            )
            side_mask = (
                (element_bounds[:, 0, 0] < d - boundary_atol_m)
                | (element_bounds[:, 1, 0] > 6000.0 - d + boundary_atol_m)
                | (element_bounds[:, 0, 1] < d - boundary_atol_m)
                | (element_bounds[:, 1, 1] > 6000.0 - d + boundary_atol_m)
            ) & ~inner_mask
            bottom_mask = ~inner_mask & ~side_mask
            side = residual_windows[window_index][side_mask].sum(axis=0)
            bottom = residual_windows[window_index][bottom_mask].sum(axis=0)
            shell = residual_windows[window_index][~inner_mask].sum(axis=0)
            side_bottom_results.append(
                {
                    "d_m": d,
                    "window": window_name,
                    "corner_assignment": "side (side tested before bottom)",
                    "side_element_count": int(side_mask.sum()),
                    "bottom_element_count": int(bottom_mask.sum()),
                    "side_complex": [_complex_vector(value) for value in side],
                    "bottom_complex": [_complex_vector(value) for value in bottom],
                    "shell_complex": [_complex_vector(value) for value in shell],
                    "side_plus_bottom_check_absolute_m_s2": float(
                        np.max(np.abs(side + bottom - shell))
                    ),
                    "side_norm_m_s2": np.linalg.norm(side, axis=1).tolist(),
                    "bottom_norm_m_s2": np.linalg.norm(bottom, axis=1).tolist(),
                    "side_norm_over_C": [float(np.linalg.norm(side[i]) / max(np.linalg.norm(analytic[i]), 1e-30)) for i in range(len(TARGETS))],
                    "bottom_norm_over_C": [float(np.linalg.norm(bottom[i]) / max(np.linalg.norm(analytic[i]), 1e-30)) for i in range(len(TARGETS))],
                }
            )
    out = {
        "science_worktree": str(_SCIENCE_ROOT),
        "import_sources": {
            "reader": str(Path(sys.modules["gragra.adapters.specfem.reader"].__file__).resolve()),
            "rayleigh": str(Path(sys.modules["gragra.waves.rayleigh"].__file__).resolve()),
            "timefit": str(Path(sys.modules["gragra.research.rayleigh_timefit"].__file__).resolve()),
        },
        "source_hdf5": str(args.hdf5),
        "time_rule": "t_physical = t_init + (time_step + 1)*DT",
        "t_init_s": T_INIT_S,
        "dt_s": DT_S,
        "time_h5_first_last_s": [float(dataset.time_s[0]), float(dataset.time_s[-1])],
        "time_corrected_first_last_s": [float(times_s[0]), float(times_s[-1])],
        "material": {"vp_m_s": VP, "vs_m_s": VS, "rho_kg_m3": RHO},
        "rayleigh": {"f_hz": FREQUENCY_HZ, "c_R_m_s": C_R, "lambda_R_m": LAMBDA_R, "amplitude_m": AMPLITUDE_M},
        "targets_xyz_m": TARGETS.tolist(),
        "windows": {"front_0_0.8s_samples": int(front_mask.sum()), "full_1.2s_samples": int(len(times_s))},
        "subset_time_gravity_vs_displacement_fit_relative_error": subset_check,
        "element_residual_definition": "integral_element(xi_SEM_fit - xi_R_fit)",
        "region_results": region_results,
        "side_bottom_results": side_bottom_results,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(args.output_npz, **saved)
    print(json.dumps({"output_json": str(args.output_json), "output_npz": str(args.output_npz), "subset_check": subset_check}, indent=2))


if __name__ == "__main__":
    main()
