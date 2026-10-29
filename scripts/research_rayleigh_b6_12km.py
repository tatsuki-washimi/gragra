#!/usr/bin/env python3
"""Compare the saved 6 km and 12 km Rayleigh B6 fields.

The numerical kernels and time-fit implementation are reused from
``research_rayleigh_b6.py``.  This script only adds the physical-coordinate
overlap/added-region accounting for the one 12 km domain case.
"""

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

_B6_SPEC = importlib.util.spec_from_file_location(
    "research_rayleigh_b6", _ROOT / "scripts" / "research_rayleigh_b6.py"
)
assert _B6_SPEC and _B6_SPEC.loader
_B6 = importlib.util.module_from_spec(_B6_SPEC)
_B6_SPEC.loader.exec_module(_B6)


def _complex_vector(values: np.ndarray) -> list[list[float]]:
    return [[float(v.real), float(v.imag)] for v in np.asarray(values).reshape(-1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--old-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    dataset = _B6.read_specfem_gll_hdf5(args.hdf5, profile_density_range=None)
    times_s = _B6.snapshot_physical_times(
        dataset.time_step, t_init_s=_B6.T_INIT_S, dt_s=_B6.DT_S
    )
    front_mask = (times_s >= _B6.T_INIT_S) & (
        times_s <= _B6.T_INIT_S + 0.8 + 1e-12
    )
    windows = (
        ("front_0_0.8s", front_mask),
        ("full_1.2s", np.ones(len(times_s), dtype=bool)),
    )
    analytic = _B6._analytic_rayleigh_gravity()
    a_windows = []
    b_windows = []
    residual_windows = []
    for _, mask in windows:
        a, b = _B6._evaluate_window(dataset, mask, times_s[mask])
        residual = _B6._element_residuals(dataset, mask, times_s[mask])
        a_windows.append(a)
        b_windows.append(b)
        residual_windows.append(residual)

    old = np.load(args.old_npz)
    old_a = old["A_complex"]
    old_b = old["B_complex"]
    old_total = old_a - old_b

    bounds = np.stack(
        [dataset.coordinates_m.min(axis=1), dataset.coordinates_m.max(axis=1)],
        axis=1,
    )
    atol = 1e-5
    overlap = (
        (bounds[:, 0, 0] >= -atol)
        & (bounds[:, 1, 0] <= 6000.0 + atol)
        & (bounds[:, 0, 1] >= -atol)
        & (bounds[:, 1, 1] <= 6000.0 + atol)
        & (bounds[:, 0, 2] >= -3000.0 - atol)
        & (bounds[:, 1, 2] <= atol)
    )
    region_results = []
    comparisons = []
    for wi, (window_name, _) in enumerate(windows):
        overlap_value = residual_windows[wi][overlap].sum(axis=0)
        added_value = residual_windows[wi][~overlap].sum(axis=0)
        total = a_windows[wi] - b_windows[wi]
        region_results.append(
            {
                "window": window_name,
                "overlap_element_count": int(overlap.sum()),
                "added_element_count": int((~overlap).sum()),
                "overlap_complex": [_complex_vector(v) for v in overlap_value],
                "added_complex": [_complex_vector(v) for v in added_value],
                "total_A_minus_B_complex": [_complex_vector(v) for v in total],
                "overlap_plus_added_complex": [_complex_vector(v) for v in overlap_value + added_value],
                "sum_check_absolute_m_s2": float(np.max(np.abs(overlap_value + added_value - total))),
                "overlap_norm_m_s2": np.linalg.norm(overlap_value, axis=1).tolist(),
                "added_norm_m_s2": np.linalg.norm(added_value, axis=1).tolist(),
                "total_norm_m_s2": np.linalg.norm(total, axis=1).tolist(),
                "overlap_norm_over_C": [float(np.linalg.norm(overlap_value[i]) / max(np.linalg.norm(analytic[i]), 1e-30)) for i in range(len(_B6.TARGETS))],
                "added_norm_over_C": [float(np.linalg.norm(added_value[i]) / max(np.linalg.norm(analytic[i]), 1e-30)) for i in range(len(_B6.TARGETS))],
            }
        )
        comparisons.append(
            {
                "window": window_name,
                "old_6km_total_A_minus_B_complex": [_complex_vector(v) for v in old_total[wi]],
                "new_12km_overlap_A_minus_B_complex": [_complex_vector(v) for v in overlap_value],
                "difference_new_overlap_minus_old_absolute_m_s2": np.abs(overlap_value - old_total[wi]).tolist(),
                "difference_relative_to_old_total": [float(np.linalg.norm(overlap_value[i] - old_total[wi, i]) / max(np.linalg.norm(old_total[wi, i]), 1e-30)) for i in range(len(_B6.TARGETS))],
            }
        )

    results = []
    for wi, (window_name, _) in enumerate(windows):
        a = a_windows[wi]
        b = b_windows[wi]
        for i, target in enumerate(_B6.TARGET_LABELS):
            results.append(
                {
                    "window": window_name,
                    "target": target,
                    "target_xyz_m": _B6.TARGETS[i].tolist(),
                    "A_solver_finite_volume": _complex_vector(a[i]),
                    "B_analytic_same_gll_finite_volume": _complex_vector(b[i]),
                    "C_analytic_infinite_halfspace": _complex_vector(analytic[i]),
                    "A_minus_C_over_C": _B6.complex_relative_error(a[i], analytic[i]),
                    "A_minus_B_over_C": float(np.linalg.norm(a[i] - b[i]) / max(np.linalg.norm(analytic[i]), 1e-30)),
                    "B_minus_C_over_C": _B6.complex_relative_error(b[i], analytic[i]),
                }
            )

    output = {
        "science_worktree": str(_ROOT),
        "import_sources": {
            "reader": str(Path(sys.modules["gragra.adapters.specfem.reader"].__file__).resolve()),
            "rayleigh": str(Path(sys.modules["gragra.waves.rayleigh"].__file__).resolve()),
            "timefit": str((_ROOT / "src/gragra/research/rayleigh_timefit.py").resolve()),
        },
        "source_hdf5": str(args.hdf5),
        "old_6km_npz": str(args.old_npz),
        "time_rule": "t_physical = t_init + (time_step + 1)*DT",
        "time_h5_first_last_s": [float(dataset.time_s[0]), float(dataset.time_s[-1])],
        "time_corrected_first_last_s": [float(times_s[0]), float(times_s[-1])],
        "overlap_definition": "[0,6000] x [0,6000] x [-3000,0] by physical element bounds",
        "overlap_classification_atol_m": atol,
        "results": results,
        "region_results": region_results,
        "old_overlap_comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(
        args.output_npz,
        time_h5=dataset.time_s,
        time_corrected=times_s,
        time_step=dataset.time_step,
        targets_xyz_m=_B6.TARGETS,
        A_complex=np.asarray(a_windows),
        B_complex=np.asarray(b_windows),
        C_complex=np.broadcast_to(analytic, (2, *analytic.shape)),
        element_id=dataset.element_id,
        element_bounds_m=bounds,
        overlap_mask=overlap,
        element_residual_A_minus_B=np.asarray(residual_windows),
    )
    print(json.dumps({"output_json": str(args.output_json), "output_npz": str(args.output_npz)}))


if __name__ == "__main__":
    main()
