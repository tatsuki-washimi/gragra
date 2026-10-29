"""Run the accepted G acoustic-density bridge in a native GWexpy runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import shlex
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.research_wave6_environment_acoustic import (
    FREQUENCY_HZ,
    _sha256,
    acoustic_density_phasor,
    midpoint_box_sources,
)
from scripts.research_wave6_environment_acoustic_series import _acceptance_hashes


def _extend(package_name: str, directory: Path) -> None:
    package = importlib.import_module(package_name)
    value = str(directory.resolve())
    if value not in package.__path__:
        package.__path__.append(value)


def run_bridge(
    *, output: Path, worktree: Path, g_root: Path, g_acceptance: Path
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_path = Path(__import__("gragra").__file__).resolve()
    expected_source = worktree.resolve() / "src/gragra/__init__.py"
    if source_path != expected_source:
        raise RuntimeError(
            f"gragra imported from {source_path}, expected {expected_source}"
        )
    accepted = _acceptance_hashes(g_acceptance)
    required = (
        "src/gragra/fields/_series.py",
        "src/gragra/fields/density_series.py",
        "src/gragra/adapters/_gwexpy_series.py",
        "src/gragra/adapters/gwexpy_density_series.py",
    )
    dependencies = {}
    for relative in required:
        path = g_root.resolve() / relative
        observed = _sha256(path)
        if observed != accepted.get(relative):
            raise ValueError(f"accepted G hash mismatch for {relative}")
        dependencies[relative] = {"path": str(path), "sha256": observed}
    _extend("gragra.fields", g_root / "src/gragra/fields")
    _extend("gragra.adapters", g_root / "src/gragra/adapters")
    density_module = importlib.import_module("gragra.fields.density_series")
    bridge_module = importlib.import_module("gragra.adapters.gwexpy_density_series")
    gwexpy = importlib.import_module("gwexpy")
    positions, volumes = midpoint_box_sources(
        horizontal_half_width_wavelengths=2,
        height_wavelengths=1,
        cells_per_wavelength=8,
    )
    selection = np.linspace(0, len(positions) - 1, 16, dtype=int)
    positions = positions[selection]
    volumes = volumes[selection]
    density = acoustic_density_phasor(positions)
    series = density_module.DensityPhasors(
        positions,
        volumes,
        density[None, :],
        np.array([FREQUENCY_HZ]),
        0.0,
        np.zeros(3),
        "acoustic-finite-air-box",
        np.zeros(3),
        tuple(f"cell-{index}" for index in selection),
    )
    started = time.monotonic()
    bundle = bridge_module.to_gwexpy_density_phasors(series)
    restored = bridge_module.from_gwexpy_density_phasors(bundle)
    exact = {
        "positions_m": bool(np.array_equal(restored.positions_m, series.positions_m)),
        "volumes_m3": bool(np.array_equal(restored.volumes_m3, series.volumes_m3)),
        "density_kg_m3": bool(
            np.array_equal(restored.density_kg_m3, series.density_kg_m3)
        ),
        "frequency_hz": bool(
            np.array_equal(restored.frequency_hz, series.frequency_hz)
        ),
        "phase_origin_m": bool(
            np.array_equal(restored.phase_origin_m, series.phase_origin_m)
        ),
        "source_ids": restored.source_ids == series.source_ids,
    }
    if not all(exact.values()):
        raise AssertionError(f"GWexpy density phasor round trip changed data: {exact}")
    result = {
        "status": "complete",
        "classification": "observed native GWexpy density-phasor round trip",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "gragra": str(source_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "gwexpy": getattr(gwexpy, "__version__", "unknown"),
            "argv": list(sys.argv),
            "argv_shell": shlex.join(sys.argv),
            "elapsed_s": time.monotonic() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
        },
        "G_parent_acceptance": {
            "path": str(g_acceptance.resolve()),
            "sha256": _sha256(g_acceptance),
        },
        "accepted_source_files": dependencies,
        "sample": {
            "frequency_hz": FREQUENCY_HZ,
            "source_count": len(positions),
            "density_dtype": str(series.density_kg_m3.dtype),
            "roundtrip_exact": exact,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--g-root", type=Path, required=True)
    parser.add_argument("--g-acceptance", type=Path, required=True)
    args = parser.parse_args()
    run_bridge(
        output=args.output,
        worktree=args.worktree,
        g_root=args.g_root,
        g_acceptance=args.g_acceptance,
    )


if __name__ == "__main__":
    main()
