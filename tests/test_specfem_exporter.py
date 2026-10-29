"""Tests for the SPECFEM3D Cartesian GLL exporter:
``gragra.adapters.specfem.writer`` + ``export`` + ``cli``.

Covers the documented input profile and numerical file layout:

- Normal-path static export: HDF5 structure (dataset existence/dtype,
  ``/metadata`` attributes).
- Normal-path smoke (``time_sampled``) export: ``/field/displacement``,
  ``/time``, ``/time_step``.
- Exporter-level partition independence (see
  ``test_partition_independence_element_payload_matches``'s docstring for
  why this is an exporter-level property only).
- Documented-profile input violations (each condition -> ``ValueError``).
- Writer-level missing-required-provenance-key -> ``ValueError``.
- ``h5py``-missing ``ImportError`` hint (monkeypatch, per
  ``tests/test_hdf5_field_io.py``'s convention).
- Core purity: ``import gragra`` and ``import gragra.adapters.specfem`` must
  not load ``h5py`` (subprocess, per ``tests/test_hdf5_field_io.py``'s
  ``test_import_core_does_not_load_h5py`` convention, extended to the new
  subpackage).

The A-series scientific checks (A1-A4 against a static export) live in
``tests/test_specfem_tier_a.py`` and is not duplicated here.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from _specfem_synthetic import (
    write_displacement_snapshot_files,
    write_rank_displacement_snapshot_files,
    write_synthetic_specfem_tree,
)

from gragra.adapters.specfem import export, schema, writer


def _h5py():
    """Skip (h5py extra not installed) -- matches tests/test_hdf5_field_io.py."""
    return pytest.importorskip("h5py")


# ===========================================================================
# Shared fixture-building helpers
# ===========================================================================


def _synthetic_build_identity() -> dict[str, object]:
    """A minimal, honest stand-in for provenance.collect_build_identity()'s
    result -- the synthetic fixture tree has no real .git checkout, so this
    is supplied explicitly (build_kind='package') instead of auto-collected.
    """
    return {
        "build_kind": "package",
        "package_version": "gragra-test-fixture",
        "package_digest": "sha256:" + "ab" * 32,
        "build_configuration": {},
    }


def _base_config_kwargs() -> dict[str, object]:
    return {
        "build_identity": _synthetic_build_identity(),
        "exporter_code_ref": "gragra-test-fixture",
        "f_resolved_estimate": 1.0,
    }


def _export_static(tmp_path, name: str, *, partition_variant: str = "a"):
    """Build a synthetic tree + export it in static mode. Returns
    (h5_path, SyntheticMeshInfo)."""
    info = write_synthetic_specfem_tree(
        tmp_path / f"{name}_tree", partition_variant=partition_variant
    )
    out = tmp_path / f"{name}.h5"
    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=out,
        output_sampling_kind="static",
        ngll_shape=info.ngll_shape,
        **_base_config_kwargs(),
    )
    export.export_specfem_gll_hdf5(config)
    return out, info


# ===========================================================================
# Normal path -- static export
# ===========================================================================


def test_static_export_dataset_structure(tmp_path):
    h5py = _h5py()
    out, info = _export_static(tmp_path, "static_ok")

    with h5py.File(out, "r") as f:
        n_element = info.n_elements_total
        ngll = info.ngll

        assert f[schema.PATH_MESH_COORDINATES].shape == (n_element, ngll, 3)
        assert f[schema.PATH_MESH_COORDINATES].dtype == np.float64
        assert f[schema.PATH_MESH_QUADRATURE_VOLUME].shape == (n_element, ngll)
        assert f[schema.PATH_MATERIAL_DENSITY].shape == (n_element, ngll)

        element_id = f[schema.PATH_MESH_ELEMENT_ID][()]
        np.testing.assert_array_equal(element_id, np.arange(n_element))
        assert f[schema.PATH_MESH_ELEMENT_ID].dtype == np.int64
        assert f[schema.PATH_MESH_PARTITION_ID].dtype == np.int32
        assert f[schema.PATH_MESH_DOMAIN_ID].dtype == np.int32
        assert f[schema.PATH_MESH_MATERIAL_ID].dtype == np.int32
        assert f[schema.PATH_MESH_IS_PML].dtype == np.bool_
        assert not f[schema.PATH_MESH_IS_PML][()].any()

        # static mode: no dynamic-field datasets.
        assert schema.PATH_FIELD_DISPLACEMENT not in f
        assert schema.PATH_TIME not in f
        assert schema.PATH_TIME_STEP not in f

        meta = f[schema.METADATA_GROUP]
        assert meta.attrs["schema_version"] == schema.SCHEMA_VERSION
        assert meta.attrs["mesh_type"] == schema.MESH_TYPE
        assert meta.attrs["layout"] == schema.LAYOUT_SINGLE_FILE
        assert meta.attrs["output_sampling_kind"] == "static"
        assert list(meta.attrs["ngll_shape"]) == list(info.ngll_shape)
        assert int(meta.attrs["n_elements_total"]) == n_element
        assert int(meta.attrs["nproc"]) == info.nproc
        assert meta.attrs["element_id_method"] == schema.ADOPTED_ELEMENT_ID_METHOD
        # static output must not carry time-axis provenance keys (note:
        # cadence_provenance itself is unconditionally set by
        # provenance.assemble_provenance regardless of output_sampling_kind
        # -- its presence alone is harmless; only the actual cadence/time
        # keys below are output-mode conditional).
        assert "t0" not in meta.attrs
        assert "dt_s" not in meta.attrs
        assert "output_interval_steps" not in meta.attrs


# ===========================================================================
# Normal path -- smoke (time_sampled) export
# ===========================================================================


def _patch_par_file(info, old: str, new: str) -> None:
    text = info.par_file_path.read_text()
    assert old in text, f"expected to find {old!r} in the synthetic Par_file"
    info.par_file_path.write_text(text.replace(old, new))


#: Shared cadence dict for the smoke (time_sampled) tests below -- values are
#: arbitrary but internally consistent scientific inputs (this exporter
#: cannot fabricate them; see ExportConfig.cadence's docstring).
_SMOKE_CADENCE: dict[str, object] = {
    "f_max_analysis": 5.0,
    "f_min_analysis": 0.1,
    "minimum_cycles": 3.0,
    "sampling_oversampling_factor": 4.0,
    "antialias_filter_applied": False,
}


def _enable_save_displacement(info) -> None:
    """The documented profile requires SAVE_DISPLACEMENT=.true. for time_sampled
    output; the synthetic fixture's default Par_file models a
    no-displacement-output MVP static run (.false.)."""
    _patch_par_file(
        info,
        "SAVE_DISPLACEMENT               = .false.",
        "SAVE_DISPLACEMENT               = .true.",
    )


def test_smoke_export_time_sampled_field_time_time_step(tmp_path):
    """Primary layout, real pilot-verified 2026-08-03: per-rank
    movie volume snapshot files (``proc{rank:06d}_displ_{X,Y,Z}_it######.
    bin``) -- see ``specfem_bin.read_rank_displacement_snapshots`` /
    ``detect_displacement_layout``. Unlike a shape/dtype-only check, this
    compares the exported ``/field/displacement`` against the fixture's
    per-element ground truth, so it actually exercises the exporter's
    element_id-based re-combination (each rank's rank-local snapshot data
    scattered to its correct GLOBAL row), not just that *some* array of the
    right shape came out.
    """
    h5py = _h5py()
    info = write_synthetic_specfem_tree(tmp_path / "smoke_tree", partition_variant="a")
    _enable_save_displacement(info)

    its = [1, 2, 3]
    expected = write_rank_displacement_snapshot_files(info, its=its)

    out = tmp_path / "smoke.h5"
    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=out,
        output_sampling_kind="time_sampled",
        ngll_shape=info.ngll_shape,
        t0=10.0,
        t0_source="test_fixture_injected",
        cadence=dict(_SMOKE_CADENCE),
        **_base_config_kwargs(),
    )
    export.export_specfem_gll_hdf5(config)

    with h5py.File(out, "r") as f:
        n_element = info.n_elements_total
        ngll = info.ngll
        assert f[schema.PATH_FIELD_DISPLACEMENT].shape == (len(its), n_element, ngll, 3)
        np.testing.assert_array_equal(f[schema.PATH_FIELD_DISPLACEMENT][()], expected)

        time_step = f[schema.PATH_TIME_STEP][()]
        np.testing.assert_array_equal(time_step, np.array(its, dtype=np.int64) - 1)

        dt_s = 0.05  # Par_file DT in the synthetic fixture
        expected_time = time_step.astype(np.float64) * dt_s - 10.0
        np.testing.assert_allclose(f[schema.PATH_TIME][()], expected_time)

        meta = f[schema.METADATA_GROUP]
        assert meta.attrs["cadence_provenance"] == "placeholder"
        assert int(meta.attrs["output_interval_steps"]) == 1  # NTSTEP_BETWEEN_FRAMES
        assert float(meta.attrs["t0"]) == 10.0
        assert meta.attrs["t0_source"] == "test_fixture_injected"
        assert meta.attrs["time_step_origin"] == "zero_based_normalized"
        assert int(meta.attrs["native_step_origin"]) == 1
        # Auxiliary (non-schema-required) provenance key recording which of
        # the two real on-disk layouts this export was auto-detected from.
        assert meta.attrs["displacement_layout"] == "per_rank"


def test_smoke_export_time_sampled_whole_mesh_fallback(tmp_path):
    """Fallback layout: a single whole-mesh ``displ_{X,Y,Z}_it######.bin``
    file set (specified by ``docs/design/specfem_upstream_gravity_audit.md``
    §movie volume binary の書式 -- but not reproduced by the pilot build).
    Exercised because a different SPECFEM3D build/
    version may write this layout instead of the per-rank one; see
    ``specfem_bin.detect_displacement_layout``.
    """
    h5py = _h5py()
    info = write_synthetic_specfem_tree(
        tmp_path / "smoke_whole_mesh_tree", partition_variant="a"
    )
    _enable_save_displacement(info)

    its = [1, 2, 3]
    expected = write_displacement_snapshot_files(
        info.local_path, nspec=info.n_elements_total, ngll=info.ngll, its=its
    )

    out = tmp_path / "smoke_whole_mesh.h5"
    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=out,
        output_sampling_kind="time_sampled",
        ngll_shape=info.ngll_shape,
        t0=10.0,
        t0_source="test_fixture_injected",
        cadence=dict(_SMOKE_CADENCE),
        **_base_config_kwargs(),
    )
    export.export_specfem_gll_hdf5(config)

    with h5py.File(out, "r") as f:
        n_element = info.n_elements_total
        ngll = info.ngll
        assert f[schema.PATH_FIELD_DISPLACEMENT].shape == (len(its), n_element, ngll, 3)
        np.testing.assert_array_equal(f[schema.PATH_FIELD_DISPLACEMENT][()], expected)

        time_step = f[schema.PATH_TIME_STEP][()]
        np.testing.assert_array_equal(time_step, np.array(its, dtype=np.int64) - 1)

        assert f[schema.METADATA_GROUP].attrs["displacement_layout"] == "whole_mesh"


def test_smoke_export_per_rank_it_set_mismatch_raises(tmp_path):
    """Per-rank layout: if one rank's recovered ``it`` set disagrees with
    rank 0's (e.g. a rank's snapshot for one iteration is entirely absent),
    the exporter must raise -- this cross-rank consistency cannot be
    detected by a single rank's own read."""
    info = write_synthetic_specfem_tree(
        tmp_path / "smoke_rank_it_mismatch_tree", partition_variant="a"
    )
    _enable_save_displacement(info)

    its = [1, 2, 3]
    write_rank_displacement_snapshot_files(info, its=its)
    # Rank 1 (partition "a": elements 3,4,5) loses its it=000003 snapshot
    # entirely (all 3 components) -- rank 1's recovered it set becomes
    # [1, 2], disagreeing with rank 0's [1, 2, 3].
    for comp in ("X", "Y", "Z"):
        (info.local_path / f"proc{1:06d}_displ_{comp}_it{3:06d}.bin").unlink()

    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=tmp_path / "out.h5",
        output_sampling_kind="time_sampled",
        ngll_shape=info.ngll_shape,
        t0=10.0,
        t0_source="test_fixture_injected",
        cadence=dict(_SMOKE_CADENCE),
        **_base_config_kwargs(),
    )
    with pytest.raises(ValueError, match="disagrees with rank 0's"):
        export.export_specfem_gll_hdf5(config)


def test_smoke_export_per_rank_file_missing_raises(tmp_path):
    """Per-rank layout: if a rank's displacement snapshot files are entirely
    absent (e.g. the movie volume write failed/was skipped for that rank),
    the exporter must raise rather than silently exporting a partial/zeroed
    mesh."""
    info = write_synthetic_specfem_tree(
        tmp_path / "smoke_rank_file_missing_tree", partition_variant="a"
    )
    _enable_save_displacement(info)

    its = [1, 2, 3]
    write_rank_displacement_snapshot_files(info, its=its)
    # Remove every one of rank 1's displacement files.
    for p in info.local_path.glob(f"proc{1:06d}_displ_*_it*.bin"):
        p.unlink()

    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=tmp_path / "out.h5",
        output_sampling_kind="time_sampled",
        ngll_shape=info.ngll_shape,
        t0=10.0,
        t0_source="test_fixture_injected",
        cadence=dict(_SMOKE_CADENCE),
        **_base_config_kwargs(),
    )
    with pytest.raises(ValueError, match=r"no proc000001_displ"):
        export.export_specfem_gll_hdf5(config)


# ===========================================================================
# Exporter-level partition independence (NOT a B7/S2 completion claim)
# ===========================================================================


def test_partition_independence_element_payload_matches(tmp_path):
    """Sanity-checks that exporting the *same* 6-element global mesh under
    two different rank/partition assignments ("a" and "b" -- see
    ``tests/_specfem_synthetic.py``'s ``_PARTITION_VARIANTS``) produces
    identical element_id-ascending ``/mesh``/``/material`` payloads.

    IMPORTANT: this is an **exporter-unit-level** property check. It only
    confirms this exporter's own
    ``element_id`` reconstruction + canonical reordering is partition-
    independent for this synthetic mesh -- it says nothing about the
    reader/dipole-integration path, which has separate tests.
    """
    h5py = _h5py()
    out_a, _ = _export_static(tmp_path, "variant_a", partition_variant="a")
    out_b, _ = _export_static(tmp_path, "variant_b", partition_variant="b")

    with h5py.File(out_a, "r") as fa, h5py.File(out_b, "r") as fb:
        for path in (
            schema.PATH_MESH_COORDINATES,
            schema.PATH_MESH_QUADRATURE_VOLUME,
            schema.PATH_MESH_ELEMENT_ID,
            schema.PATH_MESH_DOMAIN_ID,
            schema.PATH_MESH_MATERIAL_ID,
            schema.PATH_MESH_IS_PML,
            schema.PATH_MATERIAL_DENSITY,
        ):
            np.testing.assert_array_equal(fa[path][()], fb[path][()])
        # partition_id legitimately differs between variants (different rank
        # assignment for the same elements) -- deliberately not compared.


# ===========================================================================
# Documented-profile input checks
# ===========================================================================


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("NGNOD                           = 8", "NGNOD                           = 27"),
        (
            "ATTENUATION                     = .false.",
            "ATTENUATION                     = .true.",
        ),
        (
            "PML_INSTEAD_OF_FREE_SURFACE     = .false.",
            "PML_INSTEAD_OF_FREE_SURFACE     = .true.",
        ),
        (
            "SUPPRESS_UTM_PROJECTION         = .true.",
            "SUPPRESS_UTM_PROJECTION         = .false.",
        ),
        (
            "SAVE_MESH_FILES                 = .true.",
            "SAVE_MESH_FILES                 = .false.",
        ),
        ("SIMULATION_TYPE                 = 1", "SIMULATION_TYPE                 = 3"),
    ],
    ids=[
        "ngnod_not_8",
        "attenuation_true",
        "pml_instead_of_free_surface_true",
        "suppress_utm_projection_false",
        "save_mesh_files_false",
        "simulation_type_not_1",
    ],
)
def test_profile_check_rejects_violations(tmp_path, old, new):
    info = write_synthetic_specfem_tree(
        tmp_path / "violation_tree", partition_variant="a"
    )
    _patch_par_file(info, old, new)

    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=tmp_path / "out.h5",
        output_sampling_kind="static",
        ngll_shape=info.ngll_shape,
        **_base_config_kwargs(),
    )
    with pytest.raises(ValueError):
        export.export_specfem_gll_hdf5(config)


def test_profile_check_rejects_missing_save_displacement_for_time_sampled(tmp_path):
    info = write_synthetic_specfem_tree(
        tmp_path / "ts_violation_tree", partition_variant="a"
    )
    # Default fixture Par_file already has SAVE_DISPLACEMENT = .false..
    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=tmp_path / "out.h5",
        output_sampling_kind="time_sampled",
        ngll_shape=info.ngll_shape,
        t0=10.0,
        t0_source="x",
        cadence={
            "f_max_analysis": 5.0,
            "f_min_analysis": 0.1,
            "minimum_cycles": 3.0,
            "sampling_oversampling_factor": 4.0,
            "antialias_filter_applied": False,
        },
        **_base_config_kwargs(),
    )
    with pytest.raises(ValueError, match="SAVE_DISPLACEMENT"):
        export.export_specfem_gll_hdf5(config)


# ===========================================================================
# Writer-level missing-required-provenance-key -> ValueError
# ===========================================================================


def _minimal_writer_arrays(n_element: int = 1, ngll_shape=(2, 2, 2)):
    ngll = schema.ngll_from_shape(ngll_shape)
    return {
        "ngll_shape": ngll_shape,
        "coordinates": np.zeros((n_element, ngll, 3)),
        "quadrature_volume": np.ones((n_element, ngll)),
        "element_id": np.arange(n_element, dtype=np.int64),
        "partition_id": np.zeros(n_element, dtype=np.int32),
        "domain_id": np.full(n_element, schema.IDOMAIN_ELASTIC, dtype=np.int32),
        "material_id": np.ones(n_element, dtype=np.int32),
        "is_pml": np.zeros(n_element, dtype=bool),
        "density": np.full((n_element, ngll), 1000.0),
    }


def _minimal_static_provenance() -> dict[str, object]:
    """A hand-assembled provenance dict satisfying every key
    ``schema.required_keys("static")`` demands (writer.py injects
    ngll/ngll_shape/n_elements_total/element_id_min/max/the 4 total scalars
    itself, so those are deliberately omitted here)."""
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "mesh_type": schema.MESH_TYPE,
        "layout": schema.LAYOUT_SINGLE_FILE,
        "solver_kind": "specfem3d_cartesian",
        "solver_version": "unknown",
        "build_kind": "package",
        "package_version": "x",
        "package_digest": "sha256:" + "0" * 64,
        "build_configuration": {},
        "executables": [{"role": "exporter"}],
        "resolution_method": "ldd",
        "par_file_hash": "sha256:" + "0" * 64,
        "par_file_text": "text",
        "mesh_input_hash": "sha256:" + "0" * 64,
        "mesh_input_manifest": "manifest",
        "nproc": 1,
        "output_sampling_kind": "static",
        "float_precision": "single",
        "coordinate_convention": "ENU",
        "field_units": {"displacement": "m"},
        "element_id_method": schema.ADOPTED_ELEMENT_ID_METHOD,
        "density_source": "proc_rho_bin_derived",
        "f_resolved_estimate": 1.0,
        "pml_exclusion_applied": True,
        "pml_mask_hash": "sha256:" + "0" * 64,
        "suppress_utm_projection": True,
        "endianness": "<",
        "record_marker_bytes": 4,
        "integer_kind": 4,
    }


def test_writer_succeeds_with_complete_minimal_provenance(tmp_path):
    _h5py()
    out = writer.write_gll_hdf5(
        tmp_path / "minimal.h5",
        output_sampling_kind="static",
        provenance=_minimal_static_provenance(),
        **_minimal_writer_arrays(),
    )
    assert out.is_file()


def test_writer_raises_on_missing_required_provenance_key(tmp_path):
    _h5py()
    incomplete = _minimal_static_provenance()
    del incomplete["mesh_type"]
    with pytest.raises(ValueError, match="mesh_type"):
        writer.write_gll_hdf5(
            tmp_path / "incomplete.h5",
            output_sampling_kind="static",
            provenance=incomplete,
            **_minimal_writer_arrays(),
        )


# ===========================================================================
# ImportError hint (h5py missing)
# ===========================================================================


def test_writer_raises_importerror_when_h5py_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match='pip install "gragra\\[field-io\\]"'):
        writer.write_gll_hdf5(
            "dummy.h5",
            output_sampling_kind="static",
            provenance=_minimal_static_provenance(),
            **_minimal_writer_arrays(),
        )


# ===========================================================================
# Core purity
# ===========================================================================


def test_import_core_and_specfem_does_not_load_h5py():
    """Importing core gragra, and gragra.adapters.specfem specifically, must
    not load h5py (extends tests/test_hdf5_field_io.py's
    test_import_core_does_not_load_h5py to the new subpackage)."""
    code = (
        "import sys\n"
        "import gragra\n"
        "import gragra.adapters.specfem\n"
        "assert 'h5py' not in sys.modules, f'h5py leaked: {sys.modules.keys()}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
