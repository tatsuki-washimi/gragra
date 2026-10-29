#!/usr/bin/env python3
"""Compare the saved normal 4-rank and 8-rank Rayleigh B6 fields."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(_ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "research_rayleigh_b6", _ROOT / "scripts" / "research_rayleigh_b6.py"
)
assert _SPEC and _SPEC.loader
_B6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_B6)


def _vector(values: np.ndarray) -> list[list[float]]:
    return [[float(v.real), float(v.imag)] for v in np.asarray(values).reshape(-1)]


def _from_pairs(values: list[list[float]]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[:, 0] + 1j * np.asarray(values, dtype=np.float64)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eight_hdf5", type=Path)
    parser.add_argument("--four-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    dataset = _B6.read_specfem_gll_hdf5(args.eight_hdf5, profile_density_range=None)
    times = _B6.snapshot_physical_times(
        dataset.time_step, t_init_s=_B6.T_INIT_S, dt_s=_B6.DT_S
    )
    windows = (
        ("front_0_0.8s", (times >= _B6.T_INIT_S) & (times <= _B6.T_INIT_S + 0.8 + 1e-12)),
        ("full_1.2s", np.ones(times.shape, dtype=bool)),
    )
    four_rows = json.loads(args.four_json.read_text())["results"]
    four = {
        (row["window"], row["target"]): _from_pairs(row["A_solver_finite_volume"])
        for row in four_rows
    }
    analytic = _B6._analytic_rayleigh_gravity()
    a8_all = []
    b8_all = []
    results = []
    for label, mask in windows:
        a8, b8 = _B6._evaluate_window(dataset, mask, times[mask])
        a8_all.append(a8)
        b8_all.append(b8)
        for i, target in enumerate(_B6.TARGET_LABELS):
            a4 = four[(label, target)]
            c = analytic[i]
            amp_ratio = []
            phase_change = []
            for component in range(3):
                if abs(c[component]) <= 1e-30:
                    amp_ratio.append(None)
                    phase_change.append(None)
                else:
                    amp_ratio.append(float(abs(a8[i, component]) / abs(a4[component])))
                    phase_change.append(float(np.angle(a8[i, component] / a4[component], deg=True)))
            results.append(
                {
                    "window": label,
                    "target": target,
                    "A4_complex": _vector(a4),
                    "A8_complex": _vector(a8[i]),
                    "B4_available_from_reference": True,
                    "B8_complex": _vector(b8[i]),
                    "C_complex": _vector(c),
                    "E_MPI_A8_minus_A4_over_C": float(np.linalg.norm(a8[i] - a4) / max(np.linalg.norm(c), 1e-30)),
                    "E_ref_8_A8_minus_C_over_C": _B6.complex_relative_error(a8[i], c),
                    "E_B_B8_minus_B4_over_C": None,
                    "A8_over_A4_component_amplitude_ratio": amp_ratio,
                    "A8_minus_A4_component_phase_deg": phase_change,
                }
            )
    # Compute B4 once from the same analytic finite-domain definition using the
    # already stored 4-rank HDF5 result if supplied by the caller's JSON.
    # The existing 4-rank result JSON stores B as well, so replace the marker.
    four_b = {
        (row["window"], row["target"]): _from_pairs(row["B_analytic_same_gll_finite_volume"])
        for row in four_rows
    }
    for row in results:
        b4 = four_b[(row["window"], row["target"])]
        b8 = np.asarray(row["B8_complex"])[:, 0] + 1j * np.asarray(row["B8_complex"])[:, 1]
        c = np.asarray(row["C_complex"])[:, 0] + 1j * np.asarray(row["C_complex"])[:, 1]
        row["E_B_B8_minus_B4_over_C"] = float(np.linalg.norm(b8 - b4) / max(np.linalg.norm(c), 1e-30))

    output = {
        "science_worktree": str(_ROOT),
        "source_8rank_hdf5": str(args.eight_hdf5),
        "source_4rank_json": str(args.four_json),
        "time_rule": "t_physical = t_init + (time_step + 1)*DT",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(
        args.output_npz,
        time_step=dataset.time_step,
        time_corrected=times,
        targets_xyz_m=_B6.TARGETS,
        A8=np.asarray(a8_all),
        B8=np.asarray(b8_all),
        C=np.broadcast_to(analytic, (2, *analytic.shape)),
    )
    print(json.dumps({"output_json": str(args.output_json), "output_npz": str(args.output_npz)}))


if __name__ == "__main__":
    main()
