#!/usr/bin/env python3
"""Compare 200 m and 250 m saved fields at common physical times."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("research_rayleigh_b6", ROOT / "scripts" / "research_rayleigh_b6.py")
assert spec and spec.loader
B6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B6)


class SelectedDisplacement:
    def __init__(self, dataset, selected: np.ndarray) -> None:
        self.dataset = dataset
        self.selected = np.asarray(selected, dtype=np.int64)
        self.shape = (self.selected.size, *dataset.shape[1:])

    def __getitem__(self, key):
        time_key, element_key = key
        if isinstance(time_key, (int, np.integer)):
            time_key = self.selected[int(time_key)]
        elif isinstance(time_key, np.ndarray) and time_key.dtype == bool:
            time_key = self.selected[time_key]
        else:
            time_key = self.selected[time_key]
        return self.dataset[time_key, element_key]


def load_dataset(path: Path, *, select_odd: bool) -> tuple[h5py.File, SimpleNamespace]:
    h5 = h5py.File(path, "r")
    step = np.asarray(h5["time_step"], dtype=np.int64)
    selected = np.flatnonzero(step % 2 == 1) if select_odd else np.arange(step.size)
    if not np.array_equal(step[selected], np.arange(1, 480, 2)):
        raise ValueError(f"{path}: common DT-half steps are not 1,3,...,479")
    data = SimpleNamespace(
        time_step=step[selected],
        coordinates_m=np.asarray(h5["mesh/coordinates"], dtype=np.float64),
        quadrature_volume_m3=np.asarray(h5["mesh/quadrature_volume"], dtype=np.float64),
        density_kg_m3=np.asarray(h5["material/density"], dtype=np.float64),
        displacement_m=SelectedDisplacement(h5["field/displacement"], selected),
        element_id=np.asarray(h5["mesh/element_id"], dtype=np.int64),
    )
    if data.displacement_m.shape[0] != 240:
        raise ValueError(f"{path}: expected 240 common snapshots")
    return h5, data


def pairs(values: np.ndarray) -> list[list[float]]:
    return [[float(v.real), float(v.imag)] for v in np.asarray(values).reshape(-1)]


def component_diagnostics(
    value: np.ndarray, reference: np.ndarray, zero_reference: np.ndarray
) -> tuple[list[float | None], list[float | None]]:
    ratios: list[float | None] = []
    phases: list[float | None] = []
    for got, ref, analytic in zip(value, reference, zero_reference):
        if abs(analytic) <= 1e-30:
            ratios.append(None)
            phases.append(None)
        else:
            ratios.append(float(abs(got) / abs(ref)))
            phases.append(float(np.angle(got / ref, deg=True)))
    return ratios, phases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh200_hdf5", type=Path)
    parser.add_argument("--baseline-hdf5", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    h5_200, d200 = load_dataset(args.mesh200_hdf5, select_odd=False)
    h5_250, d250 = load_dataset(args.baseline_hdf5, select_odd=True)
    if not np.array_equal(d200.time_step, d250.time_step):
        raise ValueError("common normalized time_step arrays differ")
    times = B6.snapshot_physical_times(d200.time_step, t_init_s=B6.T_INIT_S, dt_s=B6.DT_S / 2.0)
    windows = (
        ("front_0_0.8s", (times >= B6.T_INIT_S) & (times <= B6.T_INIT_S + 0.8 + 1e-12)),
        ("full_1.2s", np.ones(times.shape, dtype=bool)),
    )
    analytic = B6._analytic_rayleigh_gravity()
    rows = []
    arrays = []
    for label, mask in windows:
        a200, b200 = B6._evaluate_window(d200, mask, times[mask])
        a250, b250 = B6._evaluate_window(d250, mask, times[mask])
        arrays.append((a200, b200, a250, b250))
        for i, target in enumerate(B6.TARGET_LABELS):
            c = analytic[i]
            e_mesh = float(np.linalg.norm(a200[i] - a250[i]) / max(np.linalg.norm(c), 1e-30))
            e_ref = B6.complex_relative_error(a200[i], c)
            e_quad = float(np.linalg.norm(b200[i] - b250[i]) / max(np.linalg.norm(c), 1e-30))
            amp, phase = component_diagnostics(a200[i], a250[i], c)
            rows.append({
                "window": label,
                "target": target,
                "A_200_complex": pairs(a200[i]),
                "A_250_complex": pairs(a250[i]),
                "B_200_complex": pairs(b200[i]),
                "B_250_complex": pairs(b250[i]),
                "C_complex": pairs(c),
                "E_mesh_A200_minus_A250_over_C": e_mesh,
                "E_ref200_A200_minus_C_over_C": e_ref,
                "E_quadrature_B200_minus_B250_over_C": e_quad,
                "A200_over_A250_component_amplitude_ratio": amp,
                "A200_minus_A250_component_phase_deg": phase,
            })
    result = {
        "science_worktree": str(ROOT),
        "source_200m_hdf5": str(args.mesh200_hdf5),
        "source_250m_hdf5": str(args.baseline_hdf5),
        "time_rule": "t_physical = t_init + (time_step + 1)*DT",
        "common_time_mapping": "DT-half normalized steps 1,3,...,479; native steps 2,4,...,480",
        "mesh_200m": "60x60x15 elements, 200 m, 8 ranks, 4x2",
        "mesh_250m": "48x48x12 elements, 250 m, 8 ranks, 4x2",
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output_npz,
        time_step=d200.time_step,
        time_physical_s=times,
        A_200=np.asarray([a for a, _, _, _ in arrays]),
        B_200=np.asarray([b for _, b, _, _ in arrays]),
        A_250=np.asarray([a for _, _, a, _ in arrays]),
        B_250=np.asarray([b for _, _, _, b in arrays]),
        C=np.broadcast_to(analytic, (2, *analytic.shape)),
    )
    h5_200.close()
    h5_250.close()
    print(json.dumps({"output_json": str(args.output_json), "output_npz": str(args.output_npz)}))


if __name__ == "__main__":
    main()
