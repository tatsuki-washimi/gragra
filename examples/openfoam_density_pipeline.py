"""Evaluate two bounded-profile OpenFOAM VTK density snapshots.

This example accepts only the initial Foundation OpenFOAM 14 legacy-ASCII VTK
profile. It compares discrete native-cell point-source sums; it does not
certify a CFD solution or a continuum-error bound.
"""

from __future__ import annotations

import argparse
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

from gragra.adapters.gwexpy_observable import (  # noqa: E402
    from_gwexpy_observable,
    to_gwexpy_observable,
)
from gragra.adapters.openfoam_density import load_openfoam_density  # noqa: E402
from gragra.observables import (  # noqa: E402
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)

_EXPORT = (
    "foamToVTK -case /path/to/your/openfoam/case/"
    "W6/reproduction-v2/case -time 0,.125 -useTimeName -ascii -noPointValues "
    "-polyhedra none -fields '(rho delta_rho C Vc)'"
)
_TARGETS_M = np.array([[3.0, 0.7, -0.4], [-2.0, 1.0, 2.0]])


def _provenance(export_command: str):
    return {
        "distribution": "OpenFOAM Foundation",
        "version": "14",
        "patch": "20260724",
        "solver": "manufactured-no-solver",
        "compressibility": "not-applicable",
        "region": "region0",
        "reconstructed": True,
        "exporter": "foamToVTK",
        "export_command": export_command,
    }


def _load(path: str | Path, time_s: float, time_name: str, export_command: str):
    snap = load_openfoam_density(
        path,
        field_name="delta_rho",
        time_s=time_s,
        time_name=time_name,
        coordinate_frame="case",
        coordinate_origin_m=np.zeros(3),
        density_unit="kg/m^3",
        coordinate_unit="m",
        volume_unit="m^3",
        time_unit="s",
        provenance=_provenance(export_command),
    )
    # Check the exact recorded command for each input, not just a nonempty
    # producer string.
    assert snap.provenance["export_command"] == export_command
    return snap


def _evaluate(snapshot):
    source = snapshot.as_volume_element_source().as_weighted_source()
    return {
        "potential": point_potential(source, _TARGETS_M),
        "acceleration": point_acceleration(source, _TARGETS_M),
        "gradient": point_gravity_gradient(source, _TARGETS_M),
    }


def _w3_metadata(*snapshots):
    first = snapshots[0]
    return {
        "coordinate_frame": first.coordinate_frame,
        "coordinate_origin_m": first.coordinate_origin_m,
        "target_ids": ["target-0", "target-1"],
        "source_metadata": {
            "adapter": "openfoam_density",
            "cell_ids": tuple(int(cell_id) for cell_id in first.cell_ids),
            "snapshots": tuple(
                {
                    "input_path": snapshot.input_path,
                    "input_sha256": snapshot.provenance["input_sha256"],
                    "time_name": snapshot.time_name,
                    "export_command": snapshot.provenance["export_command"],
                }
                for snapshot in snapshots
            ),
        },
        "quadrature_metadata": {"kind": "native-C-Vc-point-source"},
        "material_metadata": {
            "field_name": first.field_name,
            "density_unit": first.density_unit,
        },
    }


def run(first_path, second_path, *, export_command: str = _EXPORT):
    """Stream two inputs and return time-first arrays plus compact provenance."""
    first = _load(first_path, 0.0, "0", export_command)
    first_values = _evaluate(first)

    second = _load(second_path, 0.125, "0.125", export_command)
    first.assert_compatible_successor(second)
    second_values = _evaluate(second)
    metadata = _w3_metadata(first, second)

    arrays = {
        name: np.stack((first_values[name], second_values[name]))
        for name in first_values
    }
    axis_values = np.array([first.time_s, second.time_s])
    for observable, values in arrays.items():
        bundle = to_gwexpy_observable(
            values,
            _TARGETS_M,
            observable=observable,
            layout="time_first",
            axis_values=axis_values,
            metadata=metadata,
        )
        restored = from_gwexpy_observable(bundle)
        assert np.array_equal(restored.values, values)
        restored_metadata = restored.companion["metadata"]
        assert set(restored_metadata) == {
            "coordinate_frame",
            "coordinate_origin_m",
            "target_ids",
            "t_ref_s",
            "phase_origin_m",
            "fourier_convention",
            "name",
            "channel",
            "source_metadata",
            "quadrature_metadata",
            "material_metadata",
        }
        assert restored_metadata["coordinate_frame"] == metadata["coordinate_frame"]
        assert np.array_equal(
            restored_metadata["coordinate_origin_m"], metadata["coordinate_origin_m"]
        )
        assert restored_metadata["target_ids"] == tuple(metadata["target_ids"])
        assert restored_metadata["t_ref_s"] is None
        assert restored_metadata["phase_origin_m"] is None
        assert restored_metadata["fourier_convention"] is None
        assert restored_metadata["name"] is None
        assert restored_metadata["channel"] is None
        for key in ("source_metadata", "quadrature_metadata", "material_metadata"):
            assert restored_metadata[key] == metadata[key]

    return {
        "time_s": axis_values,
        **arrays,
        "provenance": {
            "coordinate_frame": first.coordinate_frame,
            "coordinate_origin_m": first.coordinate_origin_m.copy(),
            "export_command": export_command,
            "field_name": first.field_name,
            "source": "native C/Vc point sources",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_vtk", type=Path)
    parser.add_argument("second_vtk", type=Path)
    parser.add_argument("--export-command", default=_EXPORT)
    args = parser.parse_args()
    result = run(args.first_vtk, args.second_vtk, export_command=args.export_command)
    import gragra
    import gragra.adapters.openfoam_density as openfoam_density

    assert gragra.__file__.startswith(str(_ROOT / "src"))
    assert openfoam_density.__file__.startswith(str(_ROOT / "src"))
    print(
        json.dumps(
            {
                "time_s": result["time_s"].tolist(),
                "shapes": {
                    key: list(result[key].shape)
                    for key in ("potential", "acceleration", "gradient")
                },
                "provenance": result["provenance"],
                "origins": {
                    "gragra": gragra.__file__,
                    "openfoam_density": openfoam_density.__file__,
                },
            },
            default=lambda value: (
                value.tolist() if isinstance(value, np.ndarray) else value
            ),
        )
    )


if __name__ == "__main__":
    main()
