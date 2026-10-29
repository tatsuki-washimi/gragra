#!/usr/bin/env python3
"""Stage-4 diagnostics for the saved 6 km Rayleigh front/full window-fit gap.

``docs/research/wave4-6-I.md`` (stage 3) found that the front/full harmonic-fit
design-matrix condition number (~1), used to argue the A-C window-sensitivity
gap is not a least-squares ill-conditioning artifact, has no computation in
this repository -- it is quoted from an external review artifact. This module
reproduces that number from the repository's own fit code (diagnostic 1),
measures the actual solver fit residual on the saved case for the front and
full windows (diagnostic 2), and applies the identical fit pipeline to a
synthetic analytic Rayleigh waveform sampled on the same GLL geometry and
times (diagnostic 3). Diagnostic 3 isolates the fit pipeline's own numerical
residual; any residual gap left over in diagnostic 2 once that pipeline
residual is subtracted out is attributable to the solver displacement field
itself (or to window choice), not to the fit pipeline. Note that all three
diagnostics operate on the raw displacement array -- none of them apply mass,
GLL quadrature volume, or a gravity kernel, so a residual-to-signal ratio
reported here is a displacement-fit-quality figure, not a gravity-domain
error fraction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    item
    for item in sys.meta_path
    if type(item).__name__ != "ScikitBuildRedirectingFinder"
]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import gragra  # noqa: E402
from gragra.adapters.specfem.streaming import SpecfemGLLStreamReader  # noqa: E402
from gragra.research.rayleigh_timefit import fit_complex_amplitude  # noqa: E402
from gragra.waves.rayleigh import rayleigh_displacement  # noqa: E402
from scripts import research_wave5_rayleigh_crosspaths as study  # noqa: E402

if not Path(gragra.__file__).resolve().is_relative_to(ROOT / "src"):
    raise RuntimeError(
        "research script imported gragra outside its integration worktree"
    )

DEFAULT_SOURCE_BLOCK_ELEMENTS = 512
DEFAULT_MEMORY_BUDGET_BYTES = 2 * 1024**3


def design_matrix_condition_numbers() -> dict[str, float]:
    """Reproduce the front/full harmonic-fit design-matrix condition numbers.

    Uses only the saved case's fixed time contract (``rayleigh_windows``); no
    HDF5 input is required since the design matrix depends solely on times.
    """
    steps = np.arange(study.SAVED_SAMPLE_COUNT, dtype=np.int64)
    raw_times = study.T_REF_S + steps.astype(np.float64) * study.DT_S
    windows = study.rayleigh_windows(raw_times, steps)
    result = {}
    for window_name in ("front", "full"):
        physical_times = np.asarray(
            windows[window_name]["physical_times_s"], dtype=np.float64
        )
        phase = study.OMEGA_RAD_S * (physical_times - study.T_REF_S)
        design = np.stack((np.cos(phase), np.sin(phase)), axis=1)
        result[window_name] = float(np.linalg.cond(design))
    return result


def _reconstructed(
    fitted: np.ndarray, physical_times: np.ndarray, t_ref_s: float
) -> np.ndarray:
    phase = study.OMEGA_RAD_S * (physical_times - t_ref_s)
    return np.real(fitted[None, :, :] * np.exp(-1j * phase)[:, None, None])


def window_fit_residual_rms(
    hdf5_path: str | Path,
    *,
    source: str,
    element_start: int = 0,
    element_stop: int | None = None,
    source_block_elements: int = DEFAULT_SOURCE_BLOCK_ELEMENTS,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> dict[str, dict[str, Any]]:
    """Compare front/full time-domain fit residual for solver or analytic input.

    ``source="solver"`` fits the saved solver displacement directly
    (diagnostic 2). ``source="analytic"`` synthesizes a time series from
    :func:`rayleigh_displacement` on the same GLL positions and times, then
    applies the identical fit (diagnostic 3): its residual isolates the fit
    pipeline's own numerical error from any solver-field or window effect.
    """
    if source not in ("solver", "analytic"):
        raise ValueError("source must be 'solver' or 'analytic'")
    element_slice = study._selected_element_slice(element_start, element_stop)
    slab_elements = (
        128 if element_slice is None else min(128, element_slice.stop - element_slice.start)
    )
    block_sources = source_block_elements * study.GLL_POINTS_PER_ELEMENT
    with SpecfemGLLStreamReader(
        hdf5_path,
        profile_density_range=(study.RHO_KG_M3, study.RHO_KG_M3),
        element_slice=element_slice,
        slab_elements=slab_elements,
        memory_budget_bytes=memory_budget_bytes,
    ) as reader:
        provider = reader.provider
        layout = reader._layout
        if layout is None:  # pragma: no cover - guarded by reader.provider
            raise RuntimeError("validated SPECFEM layout is unavailable")
        windows = study.rayleigh_windows(layout.time_s, layout.time_step)
        result: dict[str, dict[str, Any]] = {}
        for window_name in ("front", "full"):
            contract = windows[window_name]
            sample_indices = contract["input_sample_indices"]
            sample_slice = slice(sample_indices[0], sample_indices[-1] + 1)
            physical_times = np.asarray(contract["physical_times_s"], dtype=np.float64)
            sq_residual = 0.0
            sq_signal = 0.0
            element_count = 0
            for block in provider.blocks(sample_slice, block_sources):
                if source == "solver":
                    series = np.asarray(block.displacement_m, dtype=np.float64)
                else:
                    phasor = rayleigh_displacement(
                        block.positions_m,
                        study.DIRECTION,
                        study.WAVENUMBER_RAD_M,
                        study.VP_M_S,
                        study.VS_M_S,
                        amplitude_m=study.SURFACE_VERTICAL_AMPLITUDE_M,
                    )
                    series = _reconstructed(
                        np.asarray(phasor, dtype=np.complex128),
                        physical_times,
                        study.T_REF_S,
                    )
                fitted = fit_complex_amplitude(
                    series, physical_times, study.OMEGA_RAD_S, t_ref_s=study.T_REF_S
                )
                reconstructed = _reconstructed(fitted, physical_times, study.T_REF_S)
                residual = series - reconstructed
                sq_residual += float(np.sum(residual**2))
                sq_signal += float(np.sum(series**2))
                element_count += len(block.source_ids)
            total_scalar_count = len(sample_indices) * element_count * 3
            residual_rms = (sq_residual / total_scalar_count) ** 0.5
            signal_rms = (sq_signal / total_scalar_count) ** 0.5
            result[window_name] = {
                "residual_rms": residual_rms,
                "signal_rms": signal_rms,
                "residual_to_signal_ratio": (
                    residual_rms / signal_rms if signal_rms > 0.0 else float("nan")
                ),
                "sample_count": len(sample_indices),
                "source_count": provider.source_count,
            }
        return result


def _write_json(path: Path, payload: dict[str, Any], *, exclusive: bool) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(
    hdf5_path: str | Path,
    output_path: str | Path,
    *,
    source_block_elements: int = DEFAULT_SOURCE_BLOCK_ELEMENTS,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> Path:
    """Write a create-new diagnostic result, retaining ``.partial`` on failure."""
    output = Path(output_path).expanduser().resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("research output must be outside the repository")
    partial = Path(f"{output}.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    started_utc = datetime.now(UTC).isoformat()
    _write_json(
        partial,
        {"status": "incomplete", "started_utc": started_utc},
        exclusive=True,
    )
    try:
        condition_numbers = design_matrix_condition_numbers()
        solver_residual = window_fit_residual_rms(
            hdf5_path,
            source="solver",
            source_block_elements=source_block_elements,
            memory_budget_bytes=memory_budget_bytes,
        )
        analytic_residual = window_fit_residual_rms(
            hdf5_path,
            source="analytic",
            source_block_elements=source_block_elements,
            memory_budget_bytes=memory_budget_bytes,
        )
        payload = {
            "status": "complete",
            "started_utc": started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "study": (
                "Stage-4 diagnostics for the saved 6 km Rayleigh window-fit gap"
            ),
            "input_file": str(Path(hdf5_path).expanduser().resolve()),
            "diagnostic_1_design_matrix_condition_number": condition_numbers,
            "diagnostic_2_solver_fit_residual": solver_residual,
            "diagnostic_3_analytic_fit_residual": analytic_residual,
        }
        _write_json(partial, payload, exclusive=False)
        reopened = json.loads(partial.read_text(encoding="utf-8"))
        if reopened != payload:
            raise RuntimeError("serialized result differs on reopened readback")
        try:
            os.link(partial, output)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {output}") from error
        partial.unlink()
        return output
    except Exception as error:
        _write_json(
            partial,
            {
                "status": "failed",
                "started_utc": started_utc,
                "failed_utc": datetime.now(UTC).isoformat(),
                "failure": {"type": type(error).__name__, "message": str(error)},
            },
            exclusive=False,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-block-elements", type=int, default=DEFAULT_SOURCE_BLOCK_ELEMENTS
    )
    parser.add_argument(
        "--memory-budget-bytes", type=int, default=DEFAULT_MEMORY_BUDGET_BYTES
    )
    args = parser.parse_args()
    print(
        run(
            args.hdf5,
            args.output,
            source_block_elements=args.source_block_elements,
            memory_budget_bytes=args.memory_budget_bytes,
        )
    )


if __name__ == "__main__":
    main()
