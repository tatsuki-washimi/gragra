"""Mutation tests for scientific sensitivity of the SPECFEM Tier A checks.

The tests deliberately mutate exported HDF5 fields to document which
scientific checks detect a change and which geometric cases are outside their
scope. They exercise numerical sensitivity only.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from _tier_a_checks import (
    check_a1_gll_ordering,
    check_a2_quadrature_volume,
    check_a3_material_mass,
    check_a4_masks_scalars_provenance,
)

from gragra.adapters.specfem import schema


def _h5py():
    return pytest.importorskip("h5py")


def _specfem_export():
    pytest.importorskip(
        "gragra.adapters.specfem.export",
        reason="SPECFEM exporter is unavailable",
    )
    from gragra.adapters.specfem import export

    return export


def _synthetic():
    return pytest.importorskip(
        "_specfem_synthetic",
        reason="SPECFEM synthetic fixture is unavailable",
    )


def _build_cg_continuous_fixture(tmp_path: Path):
    """Build the same genuinely-CG-continuous time_sampled fixture as
    ``tests/test_specfem_tier_a.py``'s
    ``test_tier_a4_check_cg_continuity_displacement_smoke`` (duplicated here
    rather than imported, per this file's new-file-only write scope --
    see module docstring). Returns the exported HDF5 path.
    """
    export = _specfem_export()
    synthetic = _synthetic()

    tree_root = tmp_path / "gate_mutation_fixture_tree"
    info = synthetic.write_synthetic_specfem_tree(tree_root, partition_variant="a")

    par_text = info.par_file_path.read_text()
    old = "SAVE_DISPLACEMENT               = .false."
    new = "SAVE_DISPLACEMENT               = .true."
    assert old in par_text, f"expected to find {old!r} in the synthetic Par_file"
    info.par_file_path.write_text(par_text.replace(old, new))

    its = [1, 2, 3]
    synthetic.write_rank_displacement_snapshot_files_cg_continuous(info, its=its)

    h5_path = tmp_path / "gate_mutation_fixture.h5"
    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=h5_path,
        output_sampling_kind="time_sampled",
        ngll_shape=info.ngll_shape,
        t0=10.0,
        t0_source="test_fixture_injected",
        cadence={
            "f_max_analysis": 5.0,
            "f_min_analysis": 0.1,
            "minimum_cycles": 3.0,
            "sampling_oversampling_factor": 4.0,
            "antialias_filter_applied": False,
        },
        build_identity={
            "build_kind": "package",
            "package_version": "gragra-synthetic-fixture",
            "package_digest": "sha256:" + "de" * 32,
            "build_configuration": {},
        },
        exporter_code_ref="gragra-synthetic-fixture",
        f_resolved_estimate=1.0,
    )
    export.export_specfem_gll_hdf5(config)
    return h5_path


def _flat_coordinate_groups(coords_flat: np.ndarray) -> dict[bytes, list[int]]:
    groups: dict[bytes, list[int]] = {}
    for idx in range(coords_flat.shape[0]):
        groups.setdefault(coords_flat[idx].tobytes(), []).append(idx)
    return groups


# ===========================================================================
# Items 2 & 4: A4 CG-continuity's structural blind spots, as executable
# mutation checks for the interior-point and shared-coordinate cases.
# ===========================================================================


def test_a4_cg_continuity_blind_to_interior_only_displacement_corruption(tmp_path):
    """Locks 穴4: a GLL point that shares no coordinate with any other
    point (the 21.6%-of-125-per-element "interior" points, ``NGLL=5``) is
    outside CG-continuity's grouping mechanism entirely -- corrupting it
    must not make the scientific check fail.
    """
    h5py = _h5py()
    h5_path = _build_cg_continuous_fixture(tmp_path)

    with h5py.File(h5_path, "r") as f:
        coords_flat = f[schema.PATH_MESH_COORDINATES][()].reshape(-1, 3)
    groups = _flat_coordinate_groups(coords_flat)
    interior_only = [idxs[0] for idxs in groups.values() if len(idxs) == 1]
    assert interior_only, (
        "fixture premise violated: expected >=1 point with no shared "
        "coordinate group (an interior GLL point) -- this test would "
        "otherwise be vacuous"
    )
    target_flat_idx = interior_only[0]

    with h5py.File(h5_path, "r+") as f:
        displacement = f[schema.PATH_FIELD_DISPLACEMENT]
        original = displacement[:, ...].reshape(displacement.shape[0], -1, 3).copy()
        mutated = original.copy()
        mutated[:, target_flat_idx, :] += 1.0e6  # far outside any real tolerance
        displacement[...] = mutated.reshape(displacement.shape)

    assert not np.allclose(original[:, target_flat_idx, :], 1.0e6), (
        "mutation sanity check failed to change the target point meaningfully"
    )

    # The actual lock: A4 (including CG-continuity) still PASSES despite
    # the corruption, because the corrupted point belongs to no
    # shared-coordinate group.
    check_a4_masks_scalars_provenance(h5_path, profile_density_range=(1900.0, 3100.0))


def test_a4_cg_continuity_detects_shared_group_displacement_corruption(tmp_path):
    """Contrast case for the previous test: corrupting only one member of
    a genuinely shared-coordinate group (E0/E1's shared face) *is* caught
    -- establishes the precise boundary of what 穴4 means (CG-continuity
    is not blind to everything, only to points outside any shared group).
    """
    h5py = _h5py()
    h5_path = _build_cg_continuous_fixture(tmp_path)

    with h5py.File(h5_path, "r") as f:
        coords_flat = f[schema.PATH_MESH_COORDINATES][()].reshape(-1, 3)
    groups = _flat_coordinate_groups(coords_flat)
    shared = [idxs for idxs in groups.values() if len(idxs) > 1]
    assert shared, (
        "fixture premise violated: expected >=1 shared-coordinate group "
        "(E0/E1's shared face) -- this test would otherwise be vacuous"
    )
    idx_a = shared[0][0]  # leave idx_b (shared[0][1]) untouched -- the point

    with h5py.File(h5_path, "r+") as f:
        displacement = f[schema.PATH_FIELD_DISPLACEMENT]
        mutated = displacement[:, ...].reshape(displacement.shape[0], -1, 3).copy()
        mutated[:, idx_a, :] += 1.0e6  # only idx_a, not its shared partner
        displacement[...] = mutated.reshape(displacement.shape)

    with pytest.raises(ValueError, match="CG continuity"):
        check_a4_masks_scalars_provenance(
            h5_path, profile_density_range=(1900.0, 3100.0)
        )


def test_a4_check_blind_to_displacement_index_permutation_between_interior_points(
    tmp_path,
):
    """Locks 穴1 for the one currently-implemented member of the nominal
    current check set that reads ``/field/displacement`` at all (A4):
    swapping (permuting) two interior-only points' displacement values --
    a genuine D1 "index置換" mutation, not just a corruption -- is not
    detected, because neither point participates in any CG-continuity
    group.
    """
    h5py = _h5py()
    h5_path = _build_cg_continuous_fixture(tmp_path)

    with h5py.File(h5_path, "r") as f:
        coords_flat = f[schema.PATH_MESH_COORDINATES][()].reshape(-1, 3)
    groups = _flat_coordinate_groups(coords_flat)
    interior_only = [idxs[0] for idxs in groups.values() if len(idxs) == 1]
    assert len(interior_only) >= 2, (
        "fixture premise violated: expected >=2 interior-only points to permute between"
    )
    idx_a, idx_b = interior_only[0], interior_only[1]

    with h5py.File(h5_path, "r+") as f:
        displacement = f[schema.PATH_FIELD_DISPLACEMENT]
        mutated = displacement[:, ...].reshape(displacement.shape[0], -1, 3).copy()
        assert not np.allclose(mutated[:, idx_a, :], mutated[:, idx_b, :]), (
            "fixture premise violated: idx_a and idx_b must carry distinct "
            "values for a permutation to be a meaningful mutation"
        )
        mutated[:, [idx_a, idx_b], :] = mutated[:, [idx_b, idx_a], :]
        displacement[...] = mutated.reshape(displacement.shape)

    # The actual lock: A4 still PASSES despite the index swap.
    check_a4_masks_scalars_provenance(h5_path, profile_density_range=(1900.0, 3100.0))


def test_tier_a1_a2_a3_never_read_displacement_field():
    """Static/structural complement to the dynamic locks above: A1/A2/A3
    are provably insensitive to *any* displacement-array mutation
    (including index permutation) because their source code never
    references ``/field/displacement`` at all. A future change that made
    one of these functions read displacement data would fail this test
    rather than silently invalidate the documented check.
    """
    checks = (check_a1_gll_ordering, check_a2_quadrature_volume, check_a3_material_mass)
    for fn in checks:
        source = inspect.getsource(fn)
        assert schema.PATH_FIELD_DISPLACEMENT not in source, (
            f"{fn.__name__} now references {schema.PATH_FIELD_DISPLACEMENT!r}; "
            "the A1/A2/A3 displacement-independence check must be re-derived "
            "before this test may be changed"
        )


# ===========================================================================
# Item 5: profile_density_range's fail-closed contract (required keyword,
# no silent-skip default).
# ===========================================================================


def test_check_a4_requires_explicit_profile_density_range():
    """``profile_density_range`` has no ``None`` default so that a caller
    cannot silently skip A4's rule 3
    by forgetting to pass it. Omitting the keyword must raise
    ``TypeError`` at the call boundary, not silently evaluate to "not
    applicable".

    This is a pure call-signature contract: Python raises ``TypeError``
    for a missing required keyword-only argument during call binding,
    before the function body (and therefore any HDF5 I/O) ever runs, so
    this test deliberately does not build a real fixture or require
    h5py -- unlike the sibling test below, it must also pass in the CI
    matrix's pure-python jobs (2026-08-05, CI-drift remediation: this
    test previously required h5py for no functional reason and failed on
    every job that does not install the field-io extra).
    """
    with pytest.raises(TypeError):
        check_a4_masks_scalars_provenance("unused.h5")  # type: ignore[call-arg]


def test_check_a4_explicit_none_profile_density_range_opts_out(tmp_path):
    """Explicitly passing ``profile_density_range=None`` remains the
    correct way to opt out of rule 3 -- unlike the signature contract
    above, this exercises the actual A4 check end-to-end and so requires
    a real exported fixture (h5py).
    """
    _h5py()
    h5_path = _build_cg_continuous_fixture(tmp_path)
    check_a4_masks_scalars_provenance(h5_path, profile_density_range=None)
