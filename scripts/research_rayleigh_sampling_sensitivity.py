#!/usr/bin/env python3
"""Sampling sensitivity of the saved, normal 12 km Rayleigh run."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    dataset = _B6.read_specfem_gll_hdf5(args.hdf5, profile_density_range=None)
    times = _B6.snapshot_physical_times(
        dataset.time_step, t_init_s=_B6.T_INIT_S, dt_s=_B6.DT_S
    )
    windows = (
        ("front_0_0.8s", (times >= _B6.T_INIT_S) & (times <= _B6.T_INIT_S + 0.8 + 1e-12)),
        ("full_1.2s", np.ones(times.shape, dtype=bool)),
    )
    every1 = np.ones(times.shape, dtype=bool)
    every2 = (dataset.time_step % 2) == 0
    analytic = _B6._analytic_rayleigh_gravity()
    results = []
    a1_all = []
    a2_all = []
    for label, window in windows:
        a1, _ = _B6._evaluate_window(dataset, window & every1, times[window & every1])
        a2, _ = _B6._evaluate_window(dataset, window & every2, times[window & every2])
        a1_all.append(a1)
        a2_all.append(a2)
        for i, target_label in enumerate(_B6.TARGET_LABELS):
            amp_ratio = []
            phase_delta_deg = []
            for component in range(3):
                if abs(analytic[i, component]) <= 1e-30:
                    amp_ratio.append(None)
                    phase_delta_deg.append(None)
                else:
                    amp_ratio.append(float(abs(a2[i, component]) / abs(a1[i, component])))
                    phase_delta_deg.append(
                        float(np.angle(a2[i, component] / a1[i, component], deg=True))
                    )
            results.append(
                {
                    "window": label,
                    "target": target_label,
                    "A_every1_complex": _vector(a1[i]),
                    "A_every2_complex": _vector(a2[i]),
                    "C_complex": _vector(analytic[i]),
                    "A_every2_minus_every1_over_C": float(
                        np.linalg.norm(a2[i] - a1[i]) / max(np.linalg.norm(analytic[i]), 1e-30)
                    ),
                    "A_every2_minus_C_over_C": _B6.complex_relative_error(a2[i], analytic[i]),
                    "component_amplitude_ratio_every2_over_every1": amp_ratio,
                    "component_phase_change_deg_every2_minus_every1": phase_delta_deg,
                }
            )

    output = {
        "science_worktree": str(_ROOT),
        "source_hdf5": str(args.hdf5),
        "time_rule": "t_physical = t_init + (time_step + 1)*DT",
        "every1_time_step_count": int(every1.sum()),
        "every2_time_step_count": int(every2.sum()),
        "every2_selection": "original time_step % 2 == 0; no renumbering",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(
        args.output_npz,
        time_step=dataset.time_step,
        time_corrected=times,
        every2_mask=every2,
        targets_xyz_m=_B6.TARGETS,
        A_every1=np.asarray(a1_all),
        A_every2=np.asarray(a2_all),
        C=np.broadcast_to(analytic, (2, *analytic.shape)),
    )
    print(json.dumps({"output_json": str(args.output_json), "output_npz": str(args.output_npz)}))


if __name__ == "__main__":
    main()
