#!/usr/bin/env python3
"""Wave 6-A: external mesh -> three observables -> result exchange integration.

This script wires together already-existing, unmodified building blocks to
walk one representative case end to end:

    ``VolumeMesh`` (read from a real meshio file)
      -> ``cell_density_from_group_tags`` / ``cell_density_nodes_from_group_tags``
      -> ``piecewise_tetra_p1_density_sources``
      -> potential / acceleration / gradient

and then cross-checks that walk three ways (see module docstrings of the
individual comparison functions below):

    (a) connectivity: a hand-built ``VolumeMesh`` vs. the same mesh read back
        from a real gmsh file, bit-identical at every pipeline stage;
    (b) evaluation/exchange: direct ``displacement_transfer_*`` calls vs. the
        ``gragra.observables.series`` dispatch entry point vs. an
        HDF5 (``gragra.adapters.field_series_hdf5``) round trip vs. a GWexpy
        (``gragra.adapters.gwexpy_displacement_series``) round trip;
    (c) physics representation: the existing
        ``compare_tetra_p1_density_representations`` mass-displacement vs.
        bulk+interface comparison, reported (never summed) per the existing
        finite_body_p1 contract.

No new gragra core code is introduced.  The host/cavity box dimensions,
background density, prescribed P-wave definition, and near-cavity target
points are the existing Proposed values from ``docs/research/wave4-6-F.md``,
reused verbatim via the existing ``scripts.research_wave4_finite_body`` and
``scripts.research_wave6_finite_body_cavity`` modules.

The mesh resolution used here (``REPRESENTATIVE_H_M = 100.0``, coarser than
any resolution in the Wave 4-6 sweeps) is a **new, test-only** choice: because
the inner cavity bounds are always inserted as axis breakpoints
(``build_cartesian_six_tetra_mesh``), this nominal ``h`` collapses the host
box to exactly the 3x3x3 = 27 boxes implied by the host and cavity bounds
(162 tetrahedra), which keeps this integration example fast while still
using the same box geometry as the production sweeps. It is not a claim
about the h=1,2,4 m physical accuracy sweeps in ``wave4-6-F.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gragra.adapters.field_series_hdf5 import (
    read_fixed_mesh_series_hdf5,
    write_fixed_mesh_series_hdf5,
)
from gragra.adapters.finite_body_mesh import (
    cell_density_nodes_from_group_tags,
    to_unstructured_displacement_field,
)
from gragra.adapters.gwexpy_displacement_series import (
    from_gwexpy_displacement_series,
    to_gwexpy_displacement_series,
)
from gragra.adapters.meshio_adapter import VolumeMesh, from_meshio_volume
from gragra.constants import G_SI
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.finite_body_p1 import (
    TetraP1DensityLinearizedSources,
    compare_tetra_p1_density_representations,
    piecewise_tetra_p1_density_sources,
)
from gragra.observables.dipole import (
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
)
from gragra.observables.series import fixed_mesh_displacement_series_observable
from gragra.observables.streaming import collect_observable_blocks, stream_observable
from scripts.research_wave4_finite_body import (
    CAVITY_TARGETS_M,
    HOST_BOUNDS_M,
    HOST_DENSITY_KG_M3,
    INNER_BOUNDS_M,
    OBSERVABLES,
    OUTSIDE_TARGETS_M,
    build_cartesian_six_tetra_mesh,
)
from scripts.research_wave6_finite_body_cavity import (
    AMPLITUDE_M,
    VP_M_S,
    prescribed_p_wave_displacement,
)

HOST_GROUP_TAG = 100
CAVITY_GROUP_TAG = 200
REPRESENTATIVE_H_M = 100.0
WAVELENGTHS_M = (20.0, 40.0, 80.0)
TARGETS_M = np.concatenate([CAVITY_TARGETS_M, OUTSIDE_TARGETS_M], axis=0)

_DIRECT_DISPLACEMENT_TRANSFER = {
    "potential": displacement_transfer_potential,
    "acceleration": displacement_transfer_acceleration,
    "gradient": displacement_transfer_gravity_gradient,
}


def density_by_tag() -> dict[int, float]:
    """The existing wave4-6-F host/cavity densities, keyed by this script's tags."""
    return {HOST_GROUP_TAG: HOST_DENSITY_KG_M3, CAVITY_GROUP_TAG: 0.0}


def build_host_cavity_volume_mesh(h_m: float = REPRESENTATIVE_H_M) -> VolumeMesh:
    """Build the wave4-6-F host+cavity box geometry as an in-memory ``VolumeMesh``.

    Reuses ``build_cartesian_six_tetra_mesh`` (unmodified) for the geometry and
    just re-expresses its per-tetra density as ``VolumeMesh`` region tags, so
    the group-tag adapter path (``cell_density_from_group_tags`` and friends)
    can be exercised.
    """
    mesh = build_cartesian_six_tetra_mesh(
        HOST_BOUNDS_M,
        h_m,
        inner_bounds_m=INNER_BOUNDS_M,
        host_density_kg_m3=HOST_DENSITY_KG_M3,
        inner_density_kg_m3=0.0,
    )
    tags = np.where(
        mesh.cell_density_kg_m3 == HOST_DENSITY_KG_M3, HOST_GROUP_TAG, CAVITY_GROUP_TAG
    ).astype(np.int64)
    return VolumeMesh(
        mesh.vertices_m,
        (("tetra", mesh.tetra),),
        cell_groups=(tags,),
        group_names={"host": HOST_GROUP_TAG, "cavity": CAVITY_GROUP_TAG},
    )


def write_host_cavity_gmsh_file(volume_mesh: VolumeMesh, path: str | Path) -> Path:
    """Write ``volume_mesh`` as a real ASCII gmsh 2.2 file meshio can read back.

    ASCII gmsh 2.2 (rather than the meshio-default 4.1 writer) is used because
    the 4.1 writer silently drops ``gmsh:physical`` cell tags for a mesh with
    only geometrical (no explicit entity) tag data; 2.2 preserves them exactly.
    """
    if len(volume_mesh.cell_blocks) != 1 or volume_mesh.cell_blocks[0][0] != "tetra":
        raise ValueError("volume_mesh must contain exactly one tetra cell block")
    if volume_mesh.cell_groups is None:
        raise ValueError("volume_mesh must carry cell group tags")
    import meshio

    gmsh_mesh = meshio.Mesh(
        points=volume_mesh.vertices_m,
        cells=[("tetra", volume_mesh.cell_blocks[0][1])],
        cell_data={"gmsh:physical": [volume_mesh.cell_groups[0]]},
        field_data={
            "host": np.array([HOST_GROUP_TAG, 3]),
            "cavity": np.array([CAVITY_GROUP_TAG, 3]),
        },
    )
    resolved = Path(path)
    meshio.write(resolved, gmsh_mesh, file_format="gmsh22", binary=False)
    return resolved


def build_mass_displacement_sources(
    volume_mesh: VolumeMesh, *, wavelength_m: float, amplitude_m: float = AMPLITUDE_M
) -> TetraP1DensityLinearizedSources:
    """Run VolumeMesh -> region tags -> P1 density sources for one wavelength."""
    displacement_nodes = prescribed_p_wave_displacement(
        volume_mesh.vertices_m, wavelength_m=wavelength_m, amplitude_m=amplitude_m
    )
    field = to_unstructured_displacement_field(volume_mesh, displacement_nodes)
    nodes = cell_density_nodes_from_group_tags(volume_mesh, density_by_tag())
    return piecewise_tetra_p1_density_sources(field, nodes)


def compare_connectivity(
    volume_mesh: VolumeMesh, file_mesh: VolumeMesh, *, wavelength_m: float
) -> dict[str, bool]:
    """(a) Connectivity comparison: hand-built VolumeMesh vs. its file round trip.

    Every stage of the pipeline (region density, per-node density, and the
    full P1 mass-displacement/bulk/interface source bundle) must be
    bit-identical whether the mesh came from memory or from a real file.
    """
    sources_a = build_mass_displacement_sources(volume_mesh, wavelength_m=wavelength_m)
    sources_b = build_mass_displacement_sources(file_mesh, wavelength_m=wavelength_m)
    return {
        "vertices_m": np.array_equal(volume_mesh.vertices_m, file_mesh.vertices_m),
        "tetra": np.array_equal(
            volume_mesh.cell_blocks[0][1], file_mesh.cell_blocks[0][1]
        ),
        "cell_groups": np.array_equal(
            volume_mesh.cell_groups[0], file_mesh.cell_groups[0]
        ),
        "mass_kg": np.array_equal(sources_a.mass_kg, sources_b.mass_kg),
        "mass_positions_m": np.array_equal(
            sources_a.mass_positions_m, sources_b.mass_positions_m
        ),
        "mass_displacement_m": np.array_equal(
            sources_a.mass_displacement_m, sources_b.mass_displacement_m
        ),
        "bulk_density_kg_m3": np.array_equal(
            np.asarray(sources_a.bulk_source.density_kg_m3),
            np.asarray(sources_b.bulk_source.density_kg_m3),
        ),
        "interface_weights_kg": np.array_equal(
            sources_a.interface_source.weights_kg, sources_b.interface_source.weights_kg
        ),
    }


def mass_displacement_series(
    sources: TetraP1DensityLinearizedSources,
    *,
    wavelength_m: float,
    vp_m_s: float = VP_M_S,
) -> FixedMeshDisplacementSeries:
    """Wrap the mass-displacement representation as a one-sample frequency series.

    ``piecewise_tetra_p1_density_sources`` produces one complex phasor
    snapshot (no time/frequency axis of its own); ``frequency_hz = vp / lambda``
    is the physically meaningful frequency of the prescribed P-wave phasor at
    this wavelength, used only as this series type's required single sample
    axis value. ``t_ref_s=0.0`` / ``phase_origin_m=0`` are the same zero-phase
    reference convention already used elsewhere (e.g. ``test_observable_series.py``).
    """
    n = len(sources.mass_positions_m)
    return FixedMeshDisplacementSeries(
        sources.mass_positions_m,
        sources.mass_kg,
        sources.mass_displacement_m[None, :, :],
        np.array([vp_m_s / wavelength_m]),
        "frequency",
        "wave6_a_host_cavity_mesh",
        np.zeros(3),
        tuple(f"mass_quadrature_point_{i}" for i in range(n)),
        t_ref_s=0.0,
        phase_origin_m=np.zeros(3),
    )


def evaluate_exchange_paths(
    sources: TetraP1DensityLinearizedSources,
    series: FixedMeshDisplacementSeries,
    provider: Any,
    recovered_series: FixedMeshDisplacementSeries,
    targets_m: np.ndarray,
    *,
    observable: str,
) -> dict[str, np.ndarray]:
    """(b) Evaluate the same mass-displacement representation four ways.

    "direct" and "series" call the exact same underlying
    ``direct_point_displacement_sum`` math (the latter via
    ``fixed_mesh_displacement_series_observable``); "hdf5" and "gwexpy" go
    through a full write/read (resp. native-field) round trip first. All four
    are expected to be bit-identical for this deterministic NumPy backend.
    """
    direct = _DIRECT_DISPLACEMENT_TRANSFER[observable](
        sources.mass_positions_m,
        sources.mass_kg,
        sources.mass_displacement_m,
        targets_m,
    )
    via_series = fixed_mesh_displacement_series_observable(
        series, observable, targets_m
    )[0]
    blocks = stream_observable(provider, targets_m, observable=observable)
    via_hdf5 = collect_observable_blocks(
        blocks, sample_count=1, target_count=len(targets_m), axis_kind="frequency"
    )[:, 0]
    via_gwexpy = fixed_mesh_displacement_series_observable(
        recovered_series, observable, targets_m
    )[0]
    return {
        "direct": direct,
        "series": via_series,
        "hdf5": via_hdf5,
        "gwexpy": via_gwexpy,
    }


def run_wave6_a_mesh_exchange(work_dir: str | Path) -> dict[str, Any]:
    """Run the full (a)/(b)/(c) walkthrough once and return a structured result.

    ``work_dir`` holds the real mesh file and the per-wavelength HDF5 series
    files; it is the caller's responsibility (a temp directory in tests, or
    an explicit ``--output``-adjacent directory from ``main``).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    volume_mesh = build_host_cavity_volume_mesh()
    mesh_path = write_host_cavity_gmsh_file(volume_mesh, work_dir / "host_cavity.msh")
    file_mesh = from_meshio_volume(mesh_path)

    connectivity = compare_connectivity(
        volume_mesh, file_mesh, wavelength_m=WAVELENGTHS_M[0]
    )

    cases: dict[str, Any] = {}
    for wavelength_m in WAVELENGTHS_M:
        sources = build_mass_displacement_sources(file_mesh, wavelength_m=wavelength_m)
        series = mass_displacement_series(sources, wavelength_m=wavelength_m)

        hdf5_path = work_dir / f"mass_displacement_series_{wavelength_m:g}m.h5"
        write_fixed_mesh_series_hdf5(hdf5_path, series)
        provider = read_fixed_mesh_series_hdf5(hdf5_path)
        recovered_series = from_gwexpy_displacement_series(
            to_gwexpy_displacement_series(series)
        )

        exchange: dict[str, Any] = {}
        representation: dict[str, Any] = {}
        for observable in OBSERVABLES:
            paths = evaluate_exchange_paths(
                sources,
                series,
                provider,
                recovered_series,
                TARGETS_M,
                observable=observable,
            )
            exchange[observable] = {
                "direct_vs_series": bool(
                    np.array_equal(paths["direct"], paths["series"])
                ),
                "direct_vs_hdf5": bool(np.array_equal(paths["direct"], paths["hdf5"])),
                "direct_vs_gwexpy": bool(
                    np.array_equal(paths["direct"], paths["gwexpy"])
                ),
            }

            report = compare_tetra_p1_density_representations(
                sources, TARGETS_M, observable=observable
            )
            representation[observable] = {
                "max_abs_mass_displacement_g_free": float(
                    np.max(np.abs(report.mass_displacement_response))
                ),
                "max_abs_bulk_interface_g_free": float(
                    np.max(np.abs(report.bulk_interface_response))
                ),
                "max_abs_mass_displacement_physical": float(
                    np.max(np.abs(report.mass_displacement_response)) * G_SI
                ),
                "max_abs_bulk_interface_physical": float(
                    np.max(np.abs(report.bulk_interface_response)) * G_SI
                ),
                "max_rel_residual": float(np.max(report.rel_residual)),
            }

        cases[f"{wavelength_m:g}"] = {
            "wavelength_m": wavelength_m,
            "frequency_hz": VP_M_S / wavelength_m,
            "exchange": exchange,
            "representation": representation,
        }

    return {
        "mesh": {
            "n_tetra": len(file_mesh.cell_blocks[0][1]),
            "n_vertices": len(file_mesh.vertices_m),
            "host_tetra_count": int(np.sum(file_mesh.cell_groups[0] == HOST_GROUP_TAG)),
            "cavity_tetra_count": int(
                np.sum(file_mesh.cell_groups[0] == CAVITY_GROUP_TAG)
            ),
            "nominal_h_m": REPRESENTATIVE_H_M,
        },
        "connectivity": connectivity,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to write the JSON summary."
    )
    args = parser.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result = run_wave6_a_mesh_exchange(tmp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
