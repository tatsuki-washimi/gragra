#!/usr/bin/env python3
"""Export the completed DT-half Rayleigh run with explicit provenance."""

from pathlib import Path
import sys

import h5py
import numpy as np

SCIENCE_ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(SCIENCE_ROOT / "src"))

from gragra.adapters.specfem import export


SOLVER_ROOT = Path("/home/washimi/work/specfem3d-rayleigh-init-20260907")
CASE = SOLVER_ROOT / "research_cases/rayleigh_b5_250m_12km_8rank_dt_half"
OUTPUT = Path("/tmp/gragra-rayleigh-b5-250m-12km-8rank-dt-half.h5")


def _repair_fixed_utf8_metadata(path: Path) -> None:
    """Match the science reader's fixed-length UTF-8 metadata contract."""
    with h5py.File(path, "r+") as h5:
        metadata = h5["metadata"]
        attrs = {
            key: value for key, value in metadata.attrs.items() if isinstance(value, str)
        }
        for key, value in attrs.items():
            del metadata.attrs[key]
            metadata.attrs.create(
                key,
                np.asarray(value, dtype=f"S{max(1, len(value))}"),
            )
        datasets = list(metadata.keys())
        values = {
            key: metadata[key][()].decode()
            if isinstance(metadata[key][()], bytes)
            else str(metadata[key][()])
            for key in datasets
        }
        for key in datasets:
            del metadata[key]
            metadata.create_dataset(
                key,
                data=values[key],
                dtype=h5py.string_dtype(
                    encoding="utf-8", length=max(1, len(values[key]))
                ),
            )


def main() -> None:
    identity = {
        "build_kind": "source",
        "specfem_git_commit": "7ddf59ed6a5394b90a8e1a1e7a945023f0c1285f",
        "source_tree_clean": False,
        "source_tree_hash": (
            "sha256:c0e4f764abdfe0ce582ba0c633f4e483d76ddd421816838821af4a006ae0889a"
        ),
        "build_configuration": {},
    }
    cadence = {
        "output_interval_steps": 1,
        "f_min_analysis": 2.4,
        "f_max_analysis": 2.5,
        "minimum_cycles": 1,
        "sampling_oversampling_factor": 4.0,
        "antialias_filter_applied": False,
    }
    config = export.ExportConfig(
        specfem_dir=CASE,
        output_path=OUTPUT,
        output_sampling_kind="time_sampled",
        ngll_shape=(5, 5, 5),
        exporter_code_ref="science-worktree-623796a",
        solver_version="research-7ddf59ed",
        float_precision="single",
        f_resolved_estimate=4.8,
        t0=-10.0,
        t0_source="12km dt-half output_solver.txt",
        cadence=cadence,
        cadence_provenance="confirmed",
        build_identity=identity,
    )
    output = export.export_specfem_gll_hdf5(config)
    _repair_fixed_utf8_metadata(output)
    print(output)


if __name__ == "__main__":
    main()
