"""Tests for gragra.adapters.specfem.provenance.

Covers: Par_file parsing (comments/blank-line removal, bool/number helper
conversion, hash reproducibility), the file-manifest primitive (byte-order
sorting, LF joining, symlink resolution, mesh_input_hash composition),
solver_build_identity (direct .git/HEAD reading -- plain ref, detached HEAD,
packed-refs fallback, worktree gitdir-file pointer -- and the dirty-tree
source_tree_hash's exclusion of run-output directories), runtime_binary_
manifest (ldd-output parsing, script-exporter identity, manifest assembly),
the t0/f_resolved_estimate extraction contracts (including the "must not
match a confusable string" negative case), the semd crosscheck, pml_mask_hash
canonicalization, and assemble_provenance's aggregation behavior.

No test in this file exercises the real ``subprocess``
default path -- every ``run_git``/``run_ldd`` call site is given an injected
stub, and a dedicated test (``test_no_subprocess_calls_when_runners_injected``)
monkeypatches ``subprocess.run`` to explode if it is ever reached, as a
backstop against a future regression that drops the injection.
"""

import hashlib

import numpy as np
import pytest

from gragra.adapters.specfem import provenance, schema

#: A syntactically valid (40 lowercase hex chars), but obviously fake,
#: commit sha used across the solver_build_identity tests.
COMMIT_SHA = "1234567890abcdef1234567890abcdef12345678"


def _write_fake_git_repo(root, commit_sha: str) -> None:
    """A ``.git`` tree with a symbolic HEAD (the common, non-detached case)."""
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text(commit_sha + "\n")


# ===========================================================================
# 1. Par_file parsing
# ===========================================================================

_SAMPLE_PAR_FILE = """\
#-----------------------------------------------------------
# Simulation input parameters
#-----------------------------------------------------------

SIMULATION_TYPE                 = 1
SAVE_FORWARD                    = .false.
SUPPRESS_UTM_PROJECTION         = .true.

NPROC                           = 4
NSTEP                           = 5000
DT                              = 0.05

LOCAL_PATH                      = OUTPUT_FILES/DATABASES_MPI   # trailing comment
SOME_EXPONENT                   = 1.0d-3
"""


def test_parse_par_file_values_comments_and_blanks(tmp_path):
    path = tmp_path / "Par_file"
    path.write_text(_SAMPLE_PAR_FILE)

    result = provenance.parse_par_file(path)
    values = result["values"]

    assert values["SIMULATION_TYPE"] == "1"
    assert values["SAVE_FORWARD"] == ".false."
    assert values["SUPPRESS_UTM_PROJECTION"] == ".true."
    assert values["NPROC"] == "4"
    assert values["LOCAL_PATH"] == "OUTPUT_FILES/DATABASES_MPI"
    assert "trailing comment" not in values["LOCAL_PATH"]
    assert "#" not in values["LOCAL_PATH"]
    assert result["text"] == _SAMPLE_PAR_FILE

    expected_hash = (
        f"sha256:{hashlib.sha256(_SAMPLE_PAR_FILE.encode('utf-8')).hexdigest()}"
    )
    assert result["par_file_hash"] == expected_hash

    # Reproducibility: parsing again yields the same hash.
    assert provenance.parse_par_file(path)["par_file_hash"] == expected_hash


def test_parse_par_file_bool_and_number_helpers(tmp_path):
    path = tmp_path / "Par_file"
    path.write_text(_SAMPLE_PAR_FILE)
    values = provenance.parse_par_file(path)["values"]

    assert provenance.par_file_value_as_bool(values["SAVE_FORWARD"]) is False
    assert provenance.par_file_value_as_bool(values["SUPPRESS_UTM_PROJECTION"]) is True

    nproc = provenance.par_file_value_as_number(values["NPROC"])
    assert nproc == 4
    assert isinstance(nproc, int)

    assert provenance.par_file_value_as_number(values["DT"]) == pytest.approx(0.05)
    assert provenance.par_file_value_as_number(
        values["SOME_EXPONENT"]
    ) == pytest.approx(1.0e-3)


def test_par_file_value_as_bool_rejects_invalid():
    with pytest.raises(ValueError, match="not a Fortran logical literal"):
        provenance.par_file_value_as_bool("maybe")


def test_par_file_value_as_number_rejects_invalid():
    with pytest.raises(ValueError, match="not a numeric Par_file value"):
        provenance.par_file_value_as_number("abc")


def test_parse_par_file_malformed_line_raises(tmp_path):
    path = tmp_path / "Par_file"
    path.write_text("GARBAGE LINE WITHOUT EQUALS\n")
    with pytest.raises(ValueError, match="expected 'KEY = value'"):
        provenance.parse_par_file(path)


# ===========================================================================
# 2. File manifest / mesh_input_hash
# ===========================================================================


def test_build_file_manifest_sorted_and_reproducible(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c.txt").write_bytes(b"hello c")
    (tmp_path / "a.txt").write_bytes(b"hello a")

    manifest_text, manifest_hash = provenance.build_file_manifest(
        tmp_path, ["b/c.txt", "a.txt"]
    )

    lines = manifest_text.splitlines()
    assert len(lines) == 2
    # Byte-order of the relative path strings: "a.txt" < "b/c.txt".
    assert lines[0].endswith("  a.txt")
    assert lines[1].endswith("  b/c.txt")
    for line in lines:
        assert line.startswith("sha256:")
    assert manifest_text.endswith("\n")

    expected_hash = (
        f"sha256:{hashlib.sha256(manifest_text.encode('utf-8')).hexdigest()}"
    )
    assert manifest_hash == expected_hash

    # Reproducibility.
    manifest_text2, manifest_hash2 = provenance.build_file_manifest(
        tmp_path, ["b/c.txt", "a.txt"]
    )
    assert manifest_text2 == manifest_text
    assert manifest_hash2 == manifest_hash


def test_build_file_manifest_follows_symlinks(tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"real content")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    manifest_text, _ = provenance.build_file_manifest(tmp_path, ["link.txt"])
    expected_line = f"sha256:{hashlib.sha256(b'real content').hexdigest()}  link.txt"
    assert manifest_text.strip() == expected_line


def test_build_file_manifest_rejects_empty_files(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        provenance.build_file_manifest(tmp_path, [])


def test_build_file_manifest_rejects_duplicate_relpaths(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x")
    with pytest.raises(ValueError, match="duplicate"):
        provenance.build_file_manifest(tmp_path, ["a.txt", "a.txt"])


def test_build_file_manifest_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="is not a file"):
        provenance.build_file_manifest(tmp_path, ["missing.txt"])


def test_compute_mesh_input_hash_includes_par_file(tmp_path):
    (tmp_path / "DATA").mkdir()
    (tmp_path / "DATA" / "Par_file").write_text("NPROC = 4\n")
    (tmp_path / "MESH-default").mkdir()
    (tmp_path / "MESH-default" / "nodes_coords_file").write_bytes(b"nodes")

    manifest_text, manifest_hash = provenance.compute_mesh_input_hash(
        tmp_path, ["MESH-default/nodes_coords_file"]
    )
    assert "DATA/Par_file" in manifest_text
    assert "MESH-default/nodes_coords_file" in manifest_text

    expected_text, expected_hash = provenance.build_file_manifest(
        tmp_path, ["DATA/Par_file", "MESH-default/nodes_coords_file"]
    )
    assert manifest_text == expected_text
    assert manifest_hash == expected_hash


def test_compute_mesh_input_hash_rejects_empty_mesh_files(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        provenance.compute_mesh_input_hash(tmp_path, [])


# ===========================================================================
# 3. solver_build_identity
# ===========================================================================


def _clean_run_git(_argv, _cwd):
    return ""


def test_collect_build_identity_clean_tree_has_no_source_tree_hash(tmp_path):
    root = tmp_path / "solver"
    root.mkdir()
    _write_fake_git_repo(root, COMMIT_SHA)

    identity = provenance.collect_build_identity(root, run_git=_clean_run_git)

    assert identity["build_kind"] == "source"
    assert identity["specfem_git_commit"] == COMMIT_SHA
    assert identity["source_tree_clean"] is True
    assert "source_tree_hash" not in identity
    assert identity["build_configuration"] == {
        "config.log": None,
        "config.status": None,
    }


def test_collect_build_identity_dirty_hash_excludes_run_output(tmp_path):
    root = tmp_path / "solver"
    root.mkdir()
    _write_fake_git_repo(root, COMMIT_SHA)
    (root / "src").mkdir()
    (root / "src" / "foo.f90").write_text("program foo\nend program foo\n")
    (root / "setup").mkdir()
    (root / "setup" / "constants.h.in").write_text("NGLLX = 5\n")
    (root / "OUTPUT_FILES").mkdir()

    def _dirty_run_git(argv, cwd):
        assert argv == ["git", "status", "--porcelain"]
        assert cwd == root
        return " M src/foo.f90\n"

    identity_before = provenance.collect_build_identity(root, run_git=_dirty_run_git)
    assert identity_before["source_tree_clean"] is False
    assert "source_tree_hash" in identity_before

    # Adding a file under a run-output directory must not change the hash.
    (root / "OUTPUT_FILES" / "output_solver.txt").write_text(
        "start time: -10.0 seconds\n"
    )
    identity_after = provenance.collect_build_identity(root, run_git=_dirty_run_git)
    assert identity_after["source_tree_hash"] == identity_before["source_tree_hash"]

    # Sanity: changing an actual build-input file DOES change the hash (the
    # exclusion rule only excludes run-output dirs, not everything).
    (root / "src" / "foo.f90").write_text(
        "program foo\n  print *, 1\nend program foo\n"
    )
    identity_changed = provenance.collect_build_identity(root, run_git=_dirty_run_git)
    assert identity_changed["source_tree_hash"] != identity_before["source_tree_hash"]


def test_collect_build_identity_dirty_with_no_build_inputs_raises(tmp_path):
    root = tmp_path / "solver"
    root.mkdir()
    _write_fake_git_repo(root, COMMIT_SHA)

    def _dirty_run_git(_argv, _cwd):
        return " M OUTPUT_FILES/foo.txt\n"

    with pytest.raises(ValueError, match="no build-input files"):
        provenance.collect_build_identity(root, run_git=_dirty_run_git)


def test_collect_build_identity_build_configuration_hashes_present_files(tmp_path):
    root = tmp_path / "solver"
    root.mkdir()
    _write_fake_git_repo(root, COMMIT_SHA)
    (root / "config.log").write_text("configure output\n")

    identity = provenance.collect_build_identity(root, run_git=_clean_run_git)
    assert identity["build_configuration"]["config.log"] is not None
    assert identity["build_configuration"]["config.log"].startswith("sha256:")
    assert identity["build_configuration"]["config.status"] is None


def test_collect_build_identity_detached_head(tmp_path):
    root = tmp_path / "solver"
    git_dir = root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text(COMMIT_SHA + "\n")

    identity = provenance.collect_build_identity(root, run_git=_clean_run_git)
    assert identity["specfem_git_commit"] == COMMIT_SHA


def test_collect_build_identity_packed_refs_fallback(tmp_path):
    root = tmp_path / "solver"
    git_dir = root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text(f"{COMMIT_SHA} refs/heads/main\n")

    identity = provenance.collect_build_identity(root, run_git=_clean_run_git)
    assert identity["specfem_git_commit"] == COMMIT_SHA


def test_collect_build_identity_worktree_gitdir_file(tmp_path):
    real_git_dir = tmp_path / "real_git"
    real_git_dir.mkdir()
    (real_git_dir / "HEAD").write_text(COMMIT_SHA + "\n")

    root = tmp_path / "worktree"
    root.mkdir()
    (root / ".git").write_text(f"gitdir: {real_git_dir}\n")

    identity = provenance.collect_build_identity(root, run_git=_clean_run_git)
    assert identity["specfem_git_commit"] == COMMIT_SHA


def test_collect_build_identity_missing_git_raises(tmp_path):
    root = tmp_path / "solver"
    root.mkdir()
    with pytest.raises(ValueError, match="no .git"):
        provenance.collect_build_identity(root, run_git=_clean_run_git)


def test_no_subprocess_calls_when_runners_injected(tmp_path, monkeypatch):
    """Backstop: even if a caller forgets to check, the default subprocess
    path must never be reached when a runner is injected (the
    "subprocess は一切実行しない(全 runner 注入)")."""

    def _explode(*_args, **_kwargs):
        raise AssertionError(
            "subprocess.run must not be called when a runner is injected"
        )

    monkeypatch.setattr(provenance.subprocess, "run", _explode)

    root = tmp_path / "solver"
    root.mkdir()
    _write_fake_git_repo(root, COMMIT_SHA)
    provenance.collect_build_identity(root, run_git=_clean_run_git)

    exe = tmp_path / "fake_solver"
    exe.write_bytes(b"\x7fELFfake")
    provenance.capture_executable(exe, "solver", run_ldd=_clean_run_git)


# ===========================================================================
# 4. runtime_binary_manifest
# ===========================================================================


def test_capture_executable_basic(tmp_path):
    exe_path = tmp_path / "fake_solver"
    exe_bytes = b"\x7fELFfakebytesforsolverexecutable"
    exe_path.write_bytes(exe_bytes)
    expected_sha = f"sha256:{hashlib.sha256(exe_bytes).hexdigest()}"

    libc_path = tmp_path / "libc.so.6"
    libc_path.write_bytes(b"fake libc content")
    ld_path = tmp_path / "ld-linux-x86-64.so.2"
    ld_path.write_bytes(b"fake loader content")

    ldd_text = (
        "\tlinux-vdso.so.1 (0x00007ffd93981000)\n"
        f"\tlibc.so.6 => {libc_path} (0x00007f2c1a000000)\n"
        "\tlibfoo.so => not found\n"
        f"\t{ld_path} (0x00007f2c1a200000)\n"
    )

    def _ldd_stub(argv, _cwd):
        assert argv[0] == "ldd"
        assert argv[1] == str(exe_path.resolve())
        return ldd_text

    entry = provenance.capture_executable(exe_path, "solver", run_ldd=_ldd_stub)

    assert entry["role"] == "solver"
    assert entry["resolved_path"] == str(exe_path.resolve())
    assert entry["sha256"] == expected_sha
    assert entry["version"] is None
    assert "version_unavailable_reason" in entry

    libs = {lib["name"]: lib for lib in entry["linked_libraries"]}
    assert len(libs) == 4

    assert libs["linux-vdso.so.1"]["path"] is None
    assert libs["linux-vdso.so.1"]["sha256"] is None

    assert libs["libc.so.6"]["path"] == str(libc_path)
    assert libs["libc.so.6"]["sha256"] == (
        f"sha256:{hashlib.sha256(b'fake libc content').hexdigest()}"
    )

    assert libs["libfoo.so"]["path"] is None
    assert libs["libfoo.so"]["sha256"] is None

    ld_key = str(ld_path)
    assert libs[ld_key]["path"] == ld_key
    assert libs[ld_key]["sha256"] == (
        f"sha256:{hashlib.sha256(b'fake loader content').hexdigest()}"
    )


def test_capture_executable_missing_file_raises(tmp_path):
    def _ldd_stub(_argv, _cwd):
        return ""

    with pytest.raises(ValueError, match="executable not found"):
        provenance.capture_executable(
            tmp_path / "does_not_exist", "solver", run_ldd=_ldd_stub
        )


def test_collect_exporter_identity_defaults():
    identity = provenance.collect_exporter_identity(exporter_code_ref="deadbeef")
    assert identity["role"] == "exporter"
    assert identity["resolved_path"] is None
    assert identity["sha256"] is None
    assert identity["exporter_code_ref"] == "deadbeef"
    assert identity["numpy_version"] == np.__version__
    assert isinstance(identity["python_version"], str)
    assert isinstance(identity["gragra_version"], str)


def test_collect_exporter_identity_overrides():
    identity = provenance.collect_exporter_identity(
        exporter_code_ref="deadbeef",
        gragra_version="9.9.9",
        python_version="3.11.0",
        numpy_version="2.0.0",
    )
    assert identity["gragra_version"] == "9.9.9"
    assert identity["python_version"] == "3.11.0"
    assert identity["numpy_version"] == "2.0.0"


def _minimal_runtime_manifest(tmp_path):
    exe_path = tmp_path / "fake_solver_for_manifest"
    exe_path.write_bytes(b"fake solver bytes")

    def _ldd_stub(_argv, _cwd):
        return ""

    solver_entry = provenance.capture_executable(exe_path, "solver", run_ldd=_ldd_stub)
    exporter_entry = provenance.collect_exporter_identity(exporter_code_ref="deadbeef")
    return provenance.build_runtime_manifest([solver_entry, exporter_entry])


def test_build_runtime_manifest_basic(tmp_path):
    manifest = _minimal_runtime_manifest(tmp_path)
    assert manifest["resolution_method"] == "ldd"
    roles = {e["role"] for e in manifest["executables"]}
    assert roles == {"solver", "exporter"}


def test_build_runtime_manifest_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        provenance.build_runtime_manifest([])


def test_runtime_manifest_json_matches_canonical_json(tmp_path):
    manifest = _minimal_runtime_manifest(tmp_path)
    text = provenance.runtime_manifest_json(manifest)
    assert text == schema.canonical_json(manifest)


# ===========================================================================
# 5. t0 / f_resolved_estimate
# ===========================================================================


def test_parse_t0_normal(tmp_path):
    path = tmp_path / "output_solver.txt"
    path.write_text(
        "some preamble\n start time:  -10.0000000      seconds\nmore text\n"
    )

    t0, source = provenance.parse_t0(path)
    assert t0 == pytest.approx(10.0)
    assert source == provenance.T0_SOURCE_OUTPUT_SOLVER_TXT_START_TIME


def test_parse_t0_zero_matches_raises(tmp_path):
    path = tmp_path / "output_solver.txt"
    path.write_text("no relevant line here\n")
    with pytest.raises(ValueError, match="expected exactly 1"):
        provenance.parse_t0(path)


def test_parse_t0_two_matches_raises(tmp_path):
    path = tmp_path / "output_solver.txt"
    path.write_text(
        " start time:  -10.0000000      seconds\n"
        " start time:  -5.0000000      seconds\n"
    )
    with pytest.raises(ValueError, match="expected exactly 1"):
        provenance.parse_t0(path)


def test_parse_t0_does_not_match_set_new_simulation_start_time(tmp_path):
    """USER_T0 > 0 emits a confusable "start time" string that ends in a
    bare "s", not "seconds" -- it must not be picked up (specfem_import.md
    L336-339)."""
    path = tmp_path / "output_solver.txt"
    path.write_text(
        "set new simulation start time: 5.0 s\n start time:  -10.0000000      seconds\n"
    )
    t0, _source = provenance.parse_t0(path)
    assert t0 == pytest.approx(10.0)


def test_parse_t0_confusable_string_alone_yields_zero_matches(tmp_path):
    path = tmp_path / "output_solver.txt"
    path.write_text("set new simulation start time: 5.0 s\n")
    with pytest.raises(ValueError, match="expected exactly 1"):
        provenance.parse_t0(path)


def _make_semd_text(dt, t0, n_rows, *, perturb_row=None, perturb_amount=0.0):
    """Synthesize a ``*.semd`` file simulating float32 storage + fixed-
    decimal ASCII rounding (real semd files are list-directed Fortran
    output, not bit-exact against the analytic ``n*dt - t0`` formula)."""
    lines = []
    for n in range(n_rows):
        expected = n * dt - t0
        value = float(np.float32(expected))  # simulate float32 storage
        if perturb_row is not None and n == perturb_row:
            value += perturb_amount
        lines.append(f"  {value:.7f}       0.00000000    ")
    return "\n".join(lines) + "\n"


def test_crosscheck_t0_semd_within_atol(tmp_path):
    dt, t0 = 0.05, 10.0
    semd_path = tmp_path / "test.semd"
    semd_path.write_text(_make_semd_text(dt, t0, 20))

    max_dev = provenance.crosscheck_t0_semd(semd_path, dt, t0)
    assert 0.0 <= max_dev < 2e-5


def test_crosscheck_t0_semd_exceeds_atol_raises(tmp_path):
    dt, t0 = 0.05, 10.0
    semd_path = tmp_path / "test.semd"
    semd_path.write_text(
        _make_semd_text(dt, t0, 20, perturb_row=5, perturb_amount=1e-3)
    )

    with pytest.raises(ValueError, match="exceeds atol"):
        provenance.crosscheck_t0_semd(semd_path, dt, t0)


def test_crosscheck_t0_semd_rejects_nonpositive_dt(tmp_path):
    semd_path = tmp_path / "test.semd"
    semd_path.write_text("0.0 0.0\n")
    with pytest.raises(ValueError, match="dt must be positive"):
        provenance.crosscheck_t0_semd(semd_path, 0.0, 10.0)


def test_crosscheck_t0_semd_empty_file_raises(tmp_path):
    semd_path = tmp_path / "empty.semd"
    semd_path.write_text("\n\n")
    with pytest.raises(ValueError, match="no data rows"):
        provenance.crosscheck_t0_semd(semd_path, 0.05, 10.0)


def test_parse_f_resolved_estimate_normal(tmp_path):
    path = tmp_path / "output_generate_databases.txt"
    path.write_text(
        "*** Max/min ratio =    1.00746357    \n"
        "\n"
        "*** Minimum period resolved =    3.12500000     (s)\n"
        "***          (as frequency) =   0.319999993     (Hz)\n"
    )
    assert provenance.parse_f_resolved_estimate(path) == pytest.approx(1.0 / 3.125)


def test_parse_f_resolved_estimate_duplicate_raises(tmp_path):
    path = tmp_path / "output_generate_databases.txt"
    path.write_text(
        "*** Minimum period resolved =    3.12500000     (s)\n"
        "*** Minimum period resolved =    4.0     (s)\n"
    )
    with pytest.raises(ValueError, match="expected exactly 1"):
        provenance.parse_f_resolved_estimate(path)


def test_parse_f_resolved_estimate_zero_matches_raises(tmp_path):
    path = tmp_path / "output_generate_databases.txt"
    path.write_text("nothing relevant\n")
    with pytest.raises(ValueError, match="expected exactly 1"):
        provenance.parse_f_resolved_estimate(path)


# ===========================================================================
# 6. pml_mask_hash
# ===========================================================================


def test_pml_mask_hash_deterministic():
    mask = np.zeros(10, dtype=bool)
    assert provenance.pml_mask_hash(mask) == provenance.pml_mask_hash(mask.copy())
    assert provenance.pml_mask_hash(mask).startswith("sha256:")


def test_pml_mask_hash_differs_for_different_masks():
    all_false = np.zeros(5, dtype=bool)
    mixed = np.array([False, True, False, True, False])
    assert provenance.pml_mask_hash(all_false) != provenance.pml_mask_hash(mixed)


def test_pml_mask_hash_accepts_int_0_1_array_equivalently():
    bool_mask = np.array([True, False, True])
    int_mask = np.array([1, 0, 1])
    assert provenance.pml_mask_hash(bool_mask) == provenance.pml_mask_hash(int_mask)


def test_pml_mask_hash_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        provenance.pml_mask_hash(np.zeros((2, 2), dtype=bool))


def test_pml_mask_hash_rejects_non_bool_values():
    with pytest.raises(ValueError, match="not cleanly representable"):
        provenance.pml_mask_hash(np.array([0, 1, 2]))


# ===========================================================================
# 7. assemble_provenance
# ===========================================================================


def _minimal_build_identity():
    return {
        "build_kind": "source",
        "specfem_git_commit": COMMIT_SHA,
        "source_tree_clean": True,
        "build_configuration": {"config.log": None, "config.status": None},
    }


def test_assemble_provenance_static_mode(tmp_path):
    par_file_path = tmp_path / "Par_file"
    par_file_path.write_text("NPROC = 4\n")
    par_file = provenance.parse_par_file(par_file_path)

    result = provenance.assemble_provenance(
        output_sampling_kind="static",
        par_file=par_file,
        mesh_input_hash="sha256:aaaa",
        mesh_input_manifest="sha256:bbbb  DATA/Par_file\n",
        build_identity=_minimal_build_identity(),
        runtime_manifest=_minimal_runtime_manifest(tmp_path),
        pml_mask_hash="sha256:cccc",
        element_id_method="pre_partition_global_index",
        element_id_min=0,
        element_id_max=20735,
        n_elements_total=20736,
        f_resolved_estimate=0.32,
        total_quadrature_volume_before_exclusion=1.0,
        total_quadrature_volume_after_exclusion=1.0,
        total_mass_before_exclusion=2000.0,
        total_mass_after_exclusion=2000.0,
        nproc=4,
    )

    assert result["schema_version"] == schema.SCHEMA_VERSION
    assert result["mesh_type"] == schema.MESH_TYPE
    assert result["layout"] == schema.LAYOUT_SINGLE_FILE
    assert result["output_sampling_kind"] == "static"
    assert result["coordinate_convention"] == "ENU"
    assert result["mesh_input_hash"] == "sha256:aaaa"
    assert result["par_file_hash"] == par_file["par_file_hash"]
    assert result["par_file_text"] == par_file["text"]
    assert result["pml_mask_hash"] == "sha256:cccc"
    assert result["pml_exclusion_applied"] is True
    assert result["cadence_provenance"] == "placeholder"
    assert result["build_kind"] == "source"
    assert result["specfem_git_commit"] == COMMIT_SHA
    assert result["resolution_method"] == "ldd"
    assert len(result["executables"]) == 2
    assert result["nproc"] == 4
    assert result["element_id_method"] == "pre_partition_global_index"
    assert result["n_elements_total"] == 20736

    # Time-sampled-only fields must not leak into a static-mode result when
    # the caller never supplied them.
    for key in (
        "t0",
        "t0_source",
        "t0_crosscheck",
        "time_step_origin",
        "native_step_origin",
    ):
        assert key not in result


def test_assemble_provenance_time_sampled_mode(tmp_path):
    par_file_path = tmp_path / "Par_file"
    par_file_path.write_text("NPROC = 4\n")
    par_file = provenance.parse_par_file(par_file_path)

    cadence = {
        "output_interval_steps": 1,
        "f_max_analysis": 10.0,
        "f_min_analysis": 0.1,
        "minimum_cycles": 5.0,
        "sampling_oversampling_factor": 8.0,
        "antialias_filter_applied": False,
    }

    result = provenance.assemble_provenance(
        output_sampling_kind="time_sampled",
        par_file=par_file,
        mesh_input_hash="sha256:aaaa",
        mesh_input_manifest="sha256:bbbb  DATA/Par_file\n",
        build_identity=_minimal_build_identity(),
        runtime_manifest=_minimal_runtime_manifest(tmp_path),
        pml_mask_hash="sha256:cccc",
        t0=10.0,
        t0_source=provenance.T0_SOURCE_OUTPUT_SOLVER_TXT_START_TIME,
        t0_crosscheck=provenance.T0_CROSSCHECK_SEMD_COLUMN1,
        t0_crosscheck_max_deviation=6e-6,
        time_step_origin="zero_based_normalized",
        native_step_origin=1,
        cadence=cadence,
        upstream_constants={"NGLLX": 5},
        dt_s=0.05,
        nstep=5000,
    )

    assert result["t0"] == pytest.approx(10.0)
    assert result["t0_source"] == "output_solver_txt_start_time"
    assert result["t0_crosscheck"] == "semd_column1"
    assert result["t0_crosscheck_max_deviation"] == pytest.approx(6e-6)
    assert result["time_step_origin"] == "zero_based_normalized"
    assert result["native_step_origin"] == 1
    assert result["output_interval_steps"] == 1
    assert result["f_max_analysis"] == pytest.approx(10.0)
    assert result["upstream_constants"] == {"NGLLX": 5}
    assert result["dt_s"] == pytest.approx(0.05)
    assert result["nstep"] == 5000
    assert result["cadence_provenance"] == "placeholder"


def test_assemble_provenance_rejects_invalid_output_sampling_kind(tmp_path):
    par_file_path = tmp_path / "Par_file"
    par_file_path.write_text("NPROC = 4\n")
    par_file = provenance.parse_par_file(par_file_path)

    with pytest.raises(ValueError, match="output_sampling_kind"):
        provenance.assemble_provenance(
            output_sampling_kind="bogus",
            par_file=par_file,
            mesh_input_hash="sha256:aaaa",
            mesh_input_manifest="text",
            build_identity=_minimal_build_identity(),
            runtime_manifest=_minimal_runtime_manifest(tmp_path),
            pml_mask_hash="sha256:cccc",
        )
