#!/usr/bin/env python3
"""Compare the completed half-DT Rayleigh field at common physical times."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
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


def _pairs(values: np.ndarray) -> list[list[float]]:
    return [[float(v.real), float(v.imag)] for v in np.asarray(values).reshape(-1)]


def _from_pairs(values: list[list[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[:, 0] + 1j * array[:, 1]


class _SelectedDisplacement:
    """Expose selected HDF5 time rows with the array indexing B6 uses."""

    def __init__(self, dataset, selected: np.ndarray) -> None:
        self._dataset = dataset
        self._selected = np.asarray(selected, dtype=np.int64)
        self.shape = (self._selected.size, *dataset.shape[1:])

    def __getitem__(self, key):
        time_key, element_key = key
        if isinstance(time_key, np.ndarray) and time_key.dtype == bool:
            time_key = self._selected[time_key]
        elif isinstance(time_key, (int, np.integer)):
            time_key = self._selected[int(time_key)]
        else:
            time_key = self._selected[time_key]
        return self._dataset[time_key, element_key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("half_hdf5", type=Path)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    h5 = h5py.File(args.half_hdf5, "r")
    mesh = h5["mesh"]
    field = h5["field"]
    full_time_step = np.asarray(h5["time_step"], dtype=np.int64)
    selected = np.flatnonzero(full_time_step % 2 == 1)
    half = SimpleNamespace(
        time_step=full_time_step[selected],
        time_s=np.asarray(h5["time"], dtype=np.float64)[selected],
        coordinates_m=np.asarray(mesh["coordinates"], dtype=np.float64),
        quadrature_volume_m3=np.asarray(mesh["quadrature_volume"], dtype=np.float64),
        density_kg_m3=np.asarray(h5["material"]["density"], dtype=np.float64),
        displacement_m=_SelectedDisplacement(field["displacement"], selected),
    )
    times = _B6.snapshot_physical_times(
        half.time_step, t_init_s=_B6.T_INIT_S, dt_s=_B6.DT_S / 2.0
    )
    windows = (
        ("front_0_0.8s", (times >= _B6.T_INIT_S) & (times <= _B6.T_INIT_S + 0.8 + 1e-12)),
        ("full_1.2s", np.ones(times.shape, dtype=bool)),
    )
    baseline = json.loads(args.baseline_json.read_text())["results"]
    a4 = {
        (row["window"], row["target"]): _from_pairs(row["A_solver_finite_volume"])
        for row in baseline
    }
    analytic = _B6._analytic_rayleigh_gravity()
    rows = []
    a_half_windows = []
    b_half_windows = []
    for label, mask in windows:
        a_half, b_half = _B6._evaluate_window(half, mask, times[mask])
        a_half_windows.append(a_half)
        b_half_windows.append(b_half)
        for i, target in enumerate(_B6.TARGET_LABELS):
            base = a4[(label, target)]
            c = analytic[i]
            amp_ratio = []
            phase_change = []
            for component in range(3):
                if abs(c[component]) <= 1e-30 or abs(base[component]) <= 1e-30:
                    amp_ratio.append(None)
                    phase_change.append(None)
                else:
                    amp_ratio.append(float(abs(a_half[i, component]) / abs(base[component])))
                    phase_change.append(float(np.angle(a_half[i, component] / base[component], deg=True)))
            rows.append(
                {
                    "window": label,
                    "target": target,
                    "A_baseline_complex": _pairs(base),
                    "A_dt_half_complex": _pairs(a_half[i]),
                    "B_dt_half_complex": _pairs(b_half[i]),
                    "C_complex": _pairs(c),
                    "E_DT_Ahalf_minus_Abaseline_over_C": float(
                        np.linalg.norm(a_half[i] - base) / max(np.linalg.norm(c), 1e-30)
                    ),
                    "E_ref_Ahalf_minus_C_over_C": _B6.complex_relative_error(a_half[i], c),
                    "Ahalf_over_Abaseline_component_amplitude_ratio": amp_ratio,
                    "Ahalf_minus_Abaseline_component_phase_deg": phase_change,
                }
            )

    output = {
        "science_worktree": str(_ROOT),
        "source_half_dt_hdf5": str(args.half_hdf5),
        "source_baseline_json": str(args.baseline_json),
        "baseline_dt_s": _B6.DT_S,
        "half_dt_s": _B6.DT_S / 2.0,
        "time_rule": "t_physical = t_init + (time_step + 1)*DT",
        "common_time_mapping": "baseline time_step j -> half-DT time_step 2*j+1; native steps 2,4,...,480",
        "selected_time_step_first_last": [int(half.time_step[0]), int(half.time_step[-1])],
        "selected_physical_time_first_last_s": [float(times[0]), float(times[-1])],
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(
        args.output_npz,
        time_step=half.time_step,
        time_physical_s=times,
        A_dt_half=np.asarray(a_half_windows),
        B_dt_half=np.asarray(b_half_windows),
        C=np.broadcast_to(analytic, (2, *analytic.shape)),
    )
    h5.close()
    print(json.dumps({"output_json": str(args.output_json), "output_npz": str(args.output_npz)}))


if __name__ == "__main__":
    main()
