"""Provenance collection for the SPECFEM3D Cartesian GLL certified export.

Implements the reusable pieces of gragra's solver-import contract (the
common and SPECFEM3D-specific required provenance items, plus the
``element_id`` construction and ``t0`` acquisition contracts) that this
module owns: ``Par_file`` parsing, file-manifest/hash construction (including
``mesh_input_hash``), solver build identity (``solver_build_identity``),
runtime executable/library capture (``runtime_binary_manifest``), the ``t0``/
``f_resolved_estimate`` extraction contracts, the ``pml_mask_hash`` canonical
encoding, and :func:`assemble_provenance`, which folds all of the above plus
caller-supplied scalars into the single flat provenance dict that
``writer.py`` (Step 6) will later write to ``/metadata``.

Scope discipline (do not re-derive elsewhere -- see the module docstrings of
``schema.py``/``specfem_bin.py`` for the same discipline applied to their
slices):

- This module never re-defines the required-key tables, the canonical-JSON
  encoding, or the ``"algorithm:hexdigest"`` hash-string format -- all of
  those are imported from :mod:`gragra.adapters.specfem.schema` (the single
  source of truth for schema (a)).
- This module does **not** perform the provenance required-key audit
  (existence/type/value-range checks against ``schema.required_keys()`` /
  ``schema.conditionally_required_keys()``). That is ``writer.py``'s
  (Step 6, which raises ``ValueError`` at write time) and Tier A4's
  (``tests/_tier_a_checks.py::check_a4_masks_scalars_provenance``, an
  independent read-back audit) responsibility. Duplicating the check here
  would let the two drift apart -- see :func:`assemble_provenance`'s
  docstring for the explicit statement of this boundary.
- Caller-supplied scalar values whose *type/on-disk-encoding* is defined by
  ``specfem_bin.py``'s ``RecordLayout`` (``endianness``,
  ``record_marker_bytes``) or its integer-kind auto-detection
  (``integer_kind``) are threaded through :func:`assemble_provenance` as
  plain ``**scalars`` -- this module does not import ``specfem_bin`` (no
  binary-format knowledge is needed to *collect* provenance), it only
  documents the expected key names so a caller who *does* have a
  ``RecordLayout`` in hand knows what to name the corresponding provenance
  key.

This module is **numpy-only + standard library** (``hashlib``, ``re``,
``subprocess``, ``pathlib``). It must never import ``h5py`` -- that lives in
``writer.py``, behind the lazy-import convention documented in
``gragra/adapters/hdf5_field.py``. Unlike ``specfem_bin.py``, this module
also never needs to import an optional dependency at all, since provenance
collection touches only text files, small binaries (for hashing), and
``subprocess``/git-plumbing files.

Runner injection (testability): every function that would otherwise shell
out (``git status``, ``ldd``) accepts an injectable :data:`CommandRunner`
callback (``run_git=``/``run_ldd=``) instead of calling ``subprocess`` marks
directly. Production callers may omit it (falling back to
:func:`_default_run_command`, a thin ``subprocess.run`` wrapper); tests
always inject a fixed-output stub, so that the test suite never actually
spawns a subprocess.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

from gragra.adapters.specfem import schema

# ===========================================================================
# 0. Shared hashing helpers (format via schema.format_hash -- not redefined)
# ===========================================================================


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """``"sha256:<hex>"``-formatted hash of ``data`` (uses
    ``schema.format_hash`` -- the prefix convention is not redefined here).
    """
    return schema.format_hash(_sha256_hex(data))


def sha256_file(path) -> str:
    """``"sha256:<hex>"``-formatted hash of a file's contents.

    Symlinks are followed transparently by ``Path.read_bytes()`` (it opens
    the resolved target), which satisfies the contract's "resolve a symlink
    to its target before hashing" requirement for every hashing entry point
    in this module without any special-casing.
    """
    return sha256_bytes(Path(path).read_bytes())


# ===========================================================================
# 1. Par_file parsing
#    ref: solver_import_contract.md, required-provenance section
#    (par_file_hash/par_file_text).
# ===========================================================================

_TRUE_LITERALS: Final[frozenset[str]] = frozenset({".true.", ".t.", "true"})
_FALSE_LITERALS: Final[frozenset[str]] = frozenset({".false.", ".f.", "false"})
_INT_LITERAL_RE: Final[re.Pattern[str]] = re.compile(r"[+-]?\d+")


def parse_par_file(path) -> dict[str, object]:
    """Parse a SPECFEM3D ``DATA/Par_file``.

    Format: one ``KEY = value`` assignment per logical line. Both full-line
    comments (a line whose first non-blank character is ``#``) and trailing
    inline comments (e.g. ``LOCAL_PATH = OUTPUT_FILES/DATABASES_MPI   #
    ...``, seen in the pilot Par_file) are stripped before checking for
    blank lines. Values are kept as **text** -- use
    :func:`par_file_value_as_bool` / :func:`par_file_value_as_number` to
    convert entries known to be Fortran logicals/numbers; this function does
    not guess a value's intended type.

    Returns
    -------
    dict with keys:
        ``"values"`` : dict[str, str]
            ``KEY -> raw string value`` (whitespace-trimmed, comment-
            stripped).
        ``"text"`` : str
            The full, unmodified file text -- this is the ``par_file_hash``
            hash input and is what gets stored verbatim as the
            ``par_file_text`` provenance dataset (a plain-text dataset, not
            JSON-wrapped -- schema.py §7).
        ``"par_file_hash"`` : str
            ``sha256_bytes(text.encode("utf-8"))``.

    Raises
    ------
    ValueError
        If a non-blank, non-comment-only line does not contain ``"="``.
    """
    text = Path(path).read_text()
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue
        if "=" not in content:
            raise ValueError(f"{path}: expected 'KEY = value', got line: {raw_line!r}")
        key, _, value = content.partition("=")
        values[key.strip()] = value.strip()
    return {
        "values": values,
        "text": text,
        "par_file_hash": sha256_bytes(text.encode("utf-8")),
    }


def par_file_value_as_bool(value: str) -> bool:
    """Convert a Fortran logical literal (``.true.``/``.false.``, tolerating
    the ``.t.``/``.f.`` and bare ``true``/``false`` spellings, case-
    insensitive) to ``bool``. Raises ``ValueError`` for anything else."""
    normalized = value.strip().lower()
    if normalized in _TRUE_LITERALS:
        return True
    if normalized in _FALSE_LITERALS:
        return False
    raise ValueError(f"not a Fortran logical literal: {value!r}")


def par_file_value_as_number(value: str) -> int | float:
    """Convert a Par_file numeric literal to ``int`` (bare optionally-signed
    digits) or ``float`` (everything else -- including Fortran's ``d``/``D``
    double-precision exponent marker, e.g. ``"1.0d-3"``, normalized to ``e``
    before parsing). Raises ``ValueError`` for non-numeric input."""
    stripped = value.strip()
    if _INT_LITERAL_RE.fullmatch(stripped):
        return int(stripped)
    normalized = stripped.replace("D", "E").replace("d", "e")
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"not a numeric Par_file value: {value!r}") from exc


# ===========================================================================
# 2. File manifest / mesh_input_hash
#    ref: solver_import_contract.md, general hash-notation rules (the
#    multi-file "path + sha256, sorted, hashed, manifest also stored" rule).
# ===========================================================================

DEFAULT_PAR_FILE_RELPATH: Final[str] = "DATA/Par_file"


def build_file_manifest(root, files: Sequence[str]) -> tuple[str, str]:
    """Build a canonical multi-file hash manifest, per the contract's
    general "*_hash of a file set" rule.

    Each ``files`` entry is a path **relative to** ``root``. The manifest is
    built by: (1) sorting the relative paths by the byte value of their
    UTF-8 encoding, (2) hashing each file's contents (symlinks resolved --
    see :func:`sha256_file`), (3) writing one ``"sha256:<hex>  <relpath>\\n"``
    line per file, in that sorted order, joined with ``"\\n"`` (LF).

    Returns
    -------
    (manifest_text, manifest_hash) : tuple[str, str]
        ``manifest_text`` must itself be preserved by the caller (as the
        ``*_manifest`` provenance dataset) -- the hash alone cannot recover
        which file set it was computed over. ``manifest_hash`` is
        ``sha256_bytes(manifest_text.encode("utf-8"))``.

    Raises
    ------
    ValueError
        If ``files`` is empty, contains a duplicate relative path (after
        POSIX-normalizing each entry), or names a path that is not a
        regular file under ``root``.
    """
    root = Path(root)
    rel_paths = [Path(f).as_posix() for f in files]
    if not rel_paths:
        raise ValueError("build_file_manifest: files must be non-empty")
    duplicates = sorted({p for p in rel_paths if rel_paths.count(p) > 1})
    if duplicates:
        raise ValueError(
            f"build_file_manifest: duplicate relative path(s): {duplicates}"
        )

    lines: list[str] = []
    for rel in sorted(rel_paths, key=lambda p: p.encode("utf-8")):
        file_path = root / rel
        if not file_path.is_file():
            raise ValueError(f"build_file_manifest: {file_path} is not a file")
        lines.append(f"sha256:{_sha256_hex(file_path.read_bytes())}  {rel}")
    manifest_text = "\n".join(lines) + "\n"
    return manifest_text, sha256_bytes(manifest_text.encode("utf-8"))


def compute_mesh_input_hash(
    root,
    mesh_files: Sequence[str],
    *,
    par_file_relpath: str = DEFAULT_PAR_FILE_RELPATH,
) -> tuple[str, str]:
    """Compute the external-mesh-route ``mesh_input_hash`` (and its backing
    manifest text): the file set is ``DATA/Par_file`` plus the external mesh
    input files supplied by the caller (``mesh_file``/``nodes_coords_file``/
    ``materials_file``/etc. under e.g. ``MESH-default/`` -- the exact file
    list is a decompose_mesh-experiment-specific detail the caller must
    supply). Thin wrapper over :func:`build_file_manifest` with
    ``par_file_relpath`` prepended.
    """
    if not mesh_files:
        raise ValueError("compute_mesh_input_hash: mesh_files must be non-empty")
    return build_file_manifest(root, [par_file_relpath, *mesh_files])


# ===========================================================================
# 3. solver_build_identity
#    ref: solver_import_contract.md, common required items (source-build
#    identity + dirty-tree source_tree_hash canonicalization/scope limit).
# ===========================================================================

#: (argv, cwd) -> captured stdout text. Injected by callers/tests instead of
#: calling ``subprocess`` directly, so tests never actually invoke `git`/
#: `ldd` (S1 plan Step 5 test list).
CommandRunner = Callable[[Sequence[str], Path], str]


def _default_run_command(argv: Sequence[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            list(argv), cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"command {list(argv)!r} (cwd={cwd}) failed: {exc}") from exc
    return completed.stdout


_COMMIT_SHA_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-fA-F]{40}")


def _resolve_git_dir(solver_root: Path) -> Path:
    """Resolve ``solver_root``'s ``.git`` metadata directory, handling both
    the ordinary case (``.git`` is a directory) and the worktree/submodule
    case (``.git`` is a file containing ``"gitdir: <path>"``)."""
    dotgit = solver_root / ".git"
    if dotgit.is_dir():
        return dotgit
    if dotgit.is_file():
        content = dotgit.read_text().strip()
        m = re.fullmatch(r"gitdir:\s*(\S+)", content)
        if not m:
            raise ValueError(f"{dotgit}: unrecognized .git file content: {content!r}")
        gitdir_path = Path(m.group(1))
        if not gitdir_path.is_absolute():
            gitdir_path = (solver_root / gitdir_path).resolve()
        return gitdir_path
    raise ValueError(f"{solver_root}: no .git directory or file found")


def _read_git_head_commit(git_dir: Path) -> str:
    """Resolve the exact commit SHA HEAD points to by reading ``.git/HEAD``
    (and, if it is a symbolic ref, the referenced loose-ref file or
    ``packed-refs``) directly -- deliberately *not* by shelling out to
    ``git rev-parse HEAD``, to avoid depending on an external ``git``
    invocation for a value this simple to read from disk.
    """
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        raise ValueError(f"{head_path}: not found (not a git worktree?)")
    head_text = head_path.read_text().strip()

    if _COMMIT_SHA_RE.fullmatch(head_text):
        return head_text.lower()

    m = re.fullmatch(r"ref:\s*(\S+)", head_text)
    if not m:
        raise ValueError(f"{head_path}: unrecognized HEAD content: {head_text!r}")
    ref = m.group(1)

    ref_path = git_dir / ref
    if ref_path.is_file():
        sha = ref_path.read_text().strip()
        if not _COMMIT_SHA_RE.fullmatch(sha):
            raise ValueError(f"{ref_path}: not a 40-hex commit sha: {sha!r}")
        return sha.lower()

    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "^")):
                continue
            parts = stripped.split()
            if (
                len(parts) == 2
                and parts[1] == ref
                and _COMMIT_SHA_RE.fullmatch(parts[0])
            ):
                return parts[0].lower()

    raise ValueError(f"{git_dir}: could not resolve ref {ref!r} to a commit sha")


#: Build-input paths included in ``source_tree_hash`` when the source tree
#: is dirty -- the SPECFEM3D-specific scope limitation excluding run-output
#: directories (``OUTPUT_FILES/``, ``DATABASES_MPI/``, ``bin/``, ``obj/``),
#: ref: solver_import_contract.md, source-build identity section.
_BUILD_INPUT_DIRS: Final[tuple[str, ...]] = ("src", "setup")
_BUILD_INPUT_FILE_GLOBS: Final[tuple[str, ...]] = (
    "configure*",
    "Makefile*",
    "flags.guess",
)
_BUILD_CONFIG_FILES: Final[tuple[str, ...]] = ("config.log", "config.status")


def _iter_build_input_relpaths(solver_root: Path) -> list[str]:
    relpaths: list[str] = []
    for dirname in _BUILD_INPUT_DIRS:
        base = solver_root / dirname
        if not base.is_dir():
            continue
        relpaths.extend(
            p.relative_to(solver_root).as_posix()
            for p in base.rglob("*")
            if p.is_file()
        )
    for pattern in _BUILD_INPUT_FILE_GLOBS:
        relpaths.extend(
            p.relative_to(solver_root).as_posix()
            for p in solver_root.glob(pattern)
            if p.is_file()
        )
    return relpaths


def _collect_build_configuration(solver_root: Path) -> dict[str, object]:
    return {
        name: sha256_file(solver_root / name)
        if (solver_root / name).is_file()
        else None
        for name in _BUILD_CONFIG_FILES
    }


def collect_build_identity(
    solver_root, *, run_git: CommandRunner | None = None
) -> dict[str, object]:
    """Collect the ``build_kind="source"`` solver build identity for a
    SPECFEM3D source worktree (ref: solver_import_contract.md, common
    required items, source-build branch).

    Only the ``source`` build form is implemented -- the certified
    profile builds SPECFEM3D from a source worktree, not a packaged binary
    (``build_kind == "package"`` is out of this function's scope; a caller
    targeting that form must construct its own dict per the contract's
    packaged-binary field list).

    Parameters
    ----------
    solver_root
        Path to the SPECFEM3D source worktree root (containing ``.git``).
    run_git
        Injected ``(argv, cwd) -> stdout`` callable used for the
        ``git status --porcelain`` clean/dirty check. Production callers may
        omit it (falls back to :func:`_default_run_command`); tests must
        always inject a fixed-output stub (no real ``git`` invocation).

    Returns
    -------
    dict with keys ``build_kind``, ``specfem_git_commit``,
    ``source_tree_clean``, ``build_configuration``, and -- only when the
    tree is dirty -- ``source_tree_hash`` (a content hash of the build-input
    files under ``src/``/``setup/``/``configure*``/``Makefile*``/
    ``flags.guess``, deliberately excluding run-output directories so that a
    solver run's side effects do not change this value -- see
    :data:`_BUILD_INPUT_DIRS`/:data:`_BUILD_INPUT_FILE_GLOBS`).

    Raises
    ------
    ValueError
        If ``.git`` cannot be resolved/read, or if the tree is reported
        dirty but no build-input files exist to hash (a dirty tree with
        nothing to hash indicates the dirty state is entirely in
        run-output directories -- see the contract's rationale for
        excluding those from this hash).
    """
    solver_root = Path(solver_root)
    git_dir = _resolve_git_dir(solver_root)
    commit = _read_git_head_commit(git_dir)

    runner = run_git if run_git is not None else _default_run_command
    status_output = runner(["git", "status", "--porcelain"], solver_root)
    clean = status_output.strip() == ""

    result: dict[str, object] = {
        "build_kind": "source",
        "specfem_git_commit": commit,
        "source_tree_clean": clean,
        "build_configuration": _collect_build_configuration(solver_root),
    }

    if not clean:
        relpaths = _iter_build_input_relpaths(solver_root)
        if not relpaths:
            raise ValueError(
                f"{solver_root}: source tree reported dirty but no build-input "
                "files (src/, setup/, configure*, Makefile*, flags.guess) were "
                "found to hash -- the dirty state must be confined to "
                "run-output directories, which this hash intentionally excludes"
            )
        _manifest_text, manifest_hash = build_file_manifest(solver_root, relpaths)
        result["source_tree_hash"] = manifest_hash

    return result


# ===========================================================================
# 4. runtime_binary_manifest
#    ref: solver_import_contract.md, common required items (executables[]/
#    linked_libraries/resolution_method + script-exporter identity).
# ===========================================================================

RUNTIME_MANIFEST_RESOLUTION_METHOD: Final[str] = "ldd"

_LDD_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<name>\S+)(?:\s+=>\s+(?P<target>not found|\S+))?(?:\s+\(0x[0-9a-fA-F]+\))?$"
)


def _parse_ldd_output(output: str) -> list[dict[str, object]]:
    libs: list[dict[str, object]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _LDD_LINE_RE.fullmatch(stripped)
        if not m:
            continue
        name = m.group("name")
        target = m.group("target")
        if target is None:
            # No "=>" clause: either the vdso (no backing file) or the
            # dynamic loader itself, named by its own absolute path.
            resolved = name if name.startswith("/") else None
        elif target == "not found":
            resolved = None
        else:
            resolved = target

        entry: dict[str, object] = {"name": name, "path": resolved, "sha256": None}
        if resolved is not None and Path(resolved).is_file():
            entry["sha256"] = sha256_file(resolved)
        libs.append(entry)
    return libs


def capture_executable(
    path, role: str, *, run_ldd: CommandRunner | None = None
) -> dict[str, object]:
    """Capture one ``executables[]`` entry: resolved path, content hash, and
    linked libraries, for a single executable.

    **Usage note (do not weaken)**: this function must be called
    **immediately before** the executable actually runs (the ``capture-run``
    CLI wrapper, Step 6, is the intended caller) -- capturing "the binary
    currently on PATH" after the fact (e.g. via a later ``which``) does not
    guarantee it is the binary that produced the output being described,
    per solver_import_contract.md's discussion of why
    ``runtime_binary_manifest`` exists at all (a stale binary, a half-failed
    rebuild, or a different-prefix binary resolved earlier on PATH can all
    make "the source that was built" and "the binary that ran" diverge).

    Parameters
    ----------
    path
        Path to the executable to capture.
    role
        One of ``"mesher"``/``"database"``/``"solver"``/``"exporter"`` (the
        contract's role vocabulary; not validated here -- multiple entries
        may share a role, and role-completeness ("solver"/"exporter" must
        both be present) is a reader-side, not a capture-side, concern).
    run_ldd
        Injected ``(argv, cwd) -> stdout`` callable for ``ldd``. Production
        callers may omit it; tests must always inject a fixed-output stub.

    Returns
    -------
    dict with keys ``role``, ``resolved_path``, ``sha256``, ``version``
    (always ``None`` here, plus ``version_unavailable_reason`` -- SPECFEM3D
    executables expose no reliable ``--version`` flag; the solver version is
    instead recorded via the ``solver_version`` provenance key, populated
    from a source the caller already has), and ``linked_libraries`` (a list
    of ``{"name", "path", "sha256"}`` dicts parsed from ``ldd`` output).
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: executable not found")
    resolved_path = str(path.resolve())

    runner = run_ldd if run_ldd is not None else _default_run_command
    ldd_output = runner(["ldd", resolved_path], path.parent)

    return {
        "role": role,
        "resolved_path": resolved_path,
        "sha256": sha256_file(path),
        "version": None,
        "version_unavailable_reason": (
            "SPECFEM3D executables expose no reliable --version flag; see "
            "the solver_version provenance key instead."
        ),
        "linked_libraries": _parse_ldd_output(ldd_output),
    }


def collect_exporter_identity(
    *,
    exporter_code_ref: str,
    gragra_version: str | None = None,
    python_version: str | None = None,
    numpy_version: str | None = None,
) -> dict[str, object]:
    """Build the ``role="exporter"`` ``executables[]`` entry for a
    script-based exporter (this exporter, run as a Python module rather than
    a compiled binary), per solver_import_contract.md L958-962: exporter
    identity for a script substitutes ``exporter_code_ref`` (the exporter
    code's commit SHA or content hash) and interpreter/package versions for
    ``resolved_path``/``sha256``.

    Parameters
    ----------
    exporter_code_ref
        The exporter code's identity (a commit SHA or content hash). This
        function does not compute it -- "the exporter code" is the whole
        gragra checkout that produced this HDF5, not a single file this
        module could canonically hash on its own; the caller (Step 6)
        supplies it (e.g. by resolving gragra's own repo HEAD, or an
        installed-package content hash for a non-editable install).
    gragra_version, python_version, numpy_version
        Overridable for testing; default to ``importlib.metadata.version
        ("gragra")`` (``"unknown"`` if not installed as a package),
        ``sys.version``'s leading token, and ``numpy.__version__``
        respectively.
    """
    import sys
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    if gragra_version is None:
        try:
            gragra_version = _pkg_version("gragra")
        except PackageNotFoundError:
            gragra_version = "unknown"
    if python_version is None:
        python_version = sys.version.split()[0]
    if numpy_version is None:
        numpy_version = np.__version__

    return {
        "role": "exporter",
        "resolved_path": None,
        "sha256": None,
        "version": None,
        "exporter_code_ref": exporter_code_ref,
        "gragra_version": gragra_version,
        "python_version": python_version,
        "numpy_version": numpy_version,
    }


def build_runtime_manifest(
    executables: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Wrap a list of :func:`capture_executable`/:func:`collect_exporter_identity`
    entries into the ``runtime_binary_manifest`` structure
    (``{"executables": [...], "resolution_method": "ldd"}``).

    See :func:`capture_executable`'s docstring for the "capture immediately
    before running" usage discipline this manifest depends on -- this
    function only assembles entries that were (by convention, not by any
    check this function performs) captured at the right time.
    """
    if not executables:
        raise ValueError("build_runtime_manifest: executables must be non-empty")
    return {
        "executables": [dict(e) for e in executables],
        "resolution_method": RUNTIME_MANIFEST_RESOLUTION_METHOD,
    }


def runtime_manifest_json(manifest: Mapping[str, object]) -> str:
    """Canonical-JSON-encode a :func:`build_runtime_manifest` result via
    ``schema.canonical_json`` -- a thin wrapper so callers needing the
    ``executables`` provenance key's on-disk text do not have to import
    ``schema.canonical_json`` directly."""
    return schema.canonical_json(dict(manifest))


# ===========================================================================
# 5. t0 / f_resolved_estimate
#    ref: specfem_import.md, element_id-construction / t0-acquisition
#    contract (t0 source regex contract, t0_crosscheck semantics, and how
#    f_resolved_estimate is obtained).
# ===========================================================================

T0_SOURCE_OUTPUT_SOLVER_TXT_START_TIME: Final[str] = "output_solver_txt_start_time"
T0_CROSSCHECK_SEMD_COLUMN1: Final[str] = "semd_column1"
DEFAULT_T0_CROSSCHECK_ATOL: Final[float] = 2e-5

#: ref: specfem_import.md L336-339 -- anchored so that "set new simulation
#: start time: ... s" (USER_T0 > 0) and similar strings, which end in a bare
#: "s" rather than "seconds", never match.
_START_TIME_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*start time:\s*(\S+)\s+seconds\s*$", re.MULTILINE
)
_MIN_PERIOD_RESOLVED_RE: Final[re.Pattern[str]] = re.compile(
    r"Minimum period resolved\s*=\s*(\S+)"
)


def parse_t0(output_solver_txt_path) -> tuple[float, str]:
    """Extract the source time shift ``t0`` from
    ``OUTPUT_FILES/output_solver.txt``'s ``"start time: ... seconds"`` line.

    The logged value is **-t0** (``prepare_timerun.F90`` writes
    ``sngl(-t0)``) -- this function negates it before returning. Requires
    **exactly one** regex match; 0 or >=2 matches raise ``ValueError`` (an
    ambiguous or missing t0 must not be silently guessed at).

    Returns
    -------
    (t0, t0_source) : tuple[float, str]
        ``t0_source`` is always
        :data:`T0_SOURCE_OUTPUT_SOLVER_TXT_START_TIME`.
    """
    text = Path(output_solver_txt_path).read_text()
    matches = _START_TIME_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"{output_solver_txt_path}: expected exactly 1 'start time: ... "
            f"seconds' match, found {len(matches)}"
        )
    t0 = -float(matches[0])
    return t0, T0_SOURCE_OUTPUT_SOLVER_TXT_START_TIME


def crosscheck_t0_semd(
    semd_path, dt: float, t0: float, *, atol: float = DEFAULT_T0_CROSSCHECK_ATOL
) -> float:
    """Cross-check ``t0`` against a ``*.semd`` seismogram file's first
    (time) column: row ``n`` (0-based) is expected to read
    ``n * dt - t0`` (``dt`` is the Par_file ``DT`` value -- **never** derive
    ``dt`` from the semd column differences themselves, per
    specfem_import.md's explicit warning that Par_file-vs-semd-derived DT
    differ by a jitter of order 1e-5 s).

    Bit-exact agreement is **not** required or expected: semd files are
    float32-precision, list-directed-ASCII-formatted output, so a few times
    ``1e-6`` s of rounding is normal (the pilot measured a 6.0e-6 s maximum
    deviation against ``atol=2e-5``, the default here).

    Returns
    -------
    float
        The observed maximum absolute deviation (seconds) -- always
        returned (not just on failure), so callers can record it as
        supplementary provenance (e.g. via
        ``assemble_provenance(t0_crosscheck_max_deviation=...)``).

    Raises
    ------
    ValueError
        If ``dt <= 0``, the file has no data rows, or the maximum deviation
        exceeds ``atol``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    times: list[float] = []
    for line in Path(semd_path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        times.append(float(stripped.split()[0]))
    if not times:
        raise ValueError(f"{semd_path}: no data rows found")

    times_arr = np.asarray(times, dtype=np.float64)
    expected = np.arange(len(times_arr), dtype=np.float64) * dt - t0
    max_deviation = float(np.abs(times_arr - expected).max())
    if max_deviation > atol:
        raise ValueError(
            f"{semd_path}: t0 crosscheck max deviation {max_deviation:.6e} s "
            f"exceeds atol {atol:.6e} s (dt={dt}, t0={t0})"
        )
    return max_deviation


def parse_f_resolved_estimate(output_generate_databases_txt_path) -> float:
    """Extract ``f_resolved_estimate`` (Hz) as the reciprocal of
    ``OUTPUT_FILES/output_generate_databases.txt``'s ``"Minimum period
    resolved"`` value (seconds). Requires **exactly one** match; 0 or >=2
    matches raise ``ValueError``. This is a non-sharp mesh-resolution
    estimate, not a hard cutoff -- see the first-stage cadence discussion
    in solver_import_contract.md/specfem_import.md for how a reader is
    allowed to use (or not use) this value."""
    text = Path(output_generate_databases_txt_path).read_text()
    matches = _MIN_PERIOD_RESOLVED_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"{output_generate_databases_txt_path}: expected exactly 1 "
            f"'Minimum period resolved' match, found {len(matches)}"
        )
    period_s = float(matches[0])
    if period_s <= 0:
        raise ValueError(
            f"{output_generate_databases_txt_path}: non-positive minimum "
            f"period resolved {period_s}"
        )
    return 1.0 / period_s


# ===========================================================================
# 6. pml_mask_hash
#    ref: solver_import_contract.md, new-dataset rationale /
#    SPECFEM3D certified-specific items (pml_mask_hash canonical encoding).
# ===========================================================================


def pml_mask_hash(is_pml_sorted_by_element_id) -> str:
    """Canonical hash of an ``element_id``-ascending ``is_pml`` bool column.

    Canonical byte representation (fixed by this function, not merely by
    convention): the bool array cast to ``np.uint8`` (0/1) and
    ``.tobytes()``'d, then sha256-hashed and ``sha256:``-prefixed
    (``schema.format_hash``). The caller is responsible for having already
    sorted the input by ``element_id`` ascending -- this function does not
    re-sort (it has no ``element_id`` array to sort by).

    Raises
    ------
    ValueError
        If the input is not 1-D, or contains values that are not cleanly
        representable as 0/1 (a silent miscast into the hash would be worse
        than raising).
    """
    arr = np.asarray(is_pml_sorted_by_element_id)
    if arr.ndim != 1:
        raise ValueError(
            f"is_pml_sorted_by_element_id must be 1-D, got shape {arr.shape}"
        )
    if arr.dtype != np.bool_:
        as_bool = arr.astype(np.bool_)
        if not np.array_equal(as_bool.astype(arr.dtype), arr):
            raise ValueError(
                "is_pml_sorted_by_element_id contains values not cleanly "
                "representable as bool (expected only 0/1 or True/False)"
            )
        arr = as_bool
    return sha256_bytes(arr.astype(np.uint8).tobytes())


# ===========================================================================
# 7. assemble_provenance
# ===========================================================================


def assemble_provenance(
    *,
    output_sampling_kind: str,
    par_file: Mapping[str, object],
    mesh_input_hash: str,
    mesh_input_manifest: str,
    build_identity: Mapping[str, object],
    runtime_manifest: Mapping[str, object],
    pml_mask_hash: str,
    pml_exclusion_applied: bool = True,
    t0: float | None = None,
    t0_source: str | None = None,
    t0_crosscheck: str | None = None,
    t0_crosscheck_max_deviation: float | None = None,
    time_step_origin: str | None = None,
    native_step_origin: int | None = None,
    cadence: Mapping[str, object] | None = None,
    cadence_provenance: str = "placeholder",
    upstream_constants: Mapping[str, object] | None = None,
    coordinate_convention: str = "ENU",
    schema_version: int = schema.SCHEMA_VERSION,
    mesh_type: str = schema.MESH_TYPE,
    layout: str = schema.LAYOUT_SINGLE_FILE,
    **scalars: object,
) -> dict[str, object]:
    """Assemble every Step 5 provenance component into a single flat dict.

    This function **does not validate required-ness, type, or value range**
    of any key (neither its own explicit parameters nor ``**scalars``) --
    that is entirely the responsibility of ``writer.py`` (Step 6, which
    raises ``ValueError`` from ``schema.required_keys()`` /
    ``schema.conditionally_required_keys()`` at write time) and Tier A4's
    independent read-back audit
    (``tests/_tier_a_checks.py::check_a4_masks_scalars_provenance``).
    Re-deriving that check here would create a second, driftable copy of
    the required-key logic that ``schema.py`` already owns as the single
    source of truth -- this function's only job is to *collect and merge*,
    not to gate.

    Structured components (each produced by another function in this
    module):

    - ``par_file``: the dict :func:`parse_par_file` returns (only its
      ``"par_file_hash"``/``"text"`` entries are pulled into the result, as
      the ``par_file_hash``/``par_file_text`` provenance keys).
    - ``build_identity``: :func:`collect_build_identity`'s result, merged in
      as-is (``build_kind``, ``specfem_git_commit``, ``source_tree_clean``,
      ``build_configuration``, and -- if dirty -- ``source_tree_hash``).
    - ``runtime_manifest``: :func:`build_runtime_manifest`'s result; its
      ``"executables"``/``"resolution_method"`` entries become the
      like-named top-level provenance keys.
    - ``cadence`` (optional): a caller-built dict of the cadence-group keys
      (``output_interval_steps``, ``f_max_analysis``, ``f_content_max``,
      ...; see ``schema.CADENCE_PROVENANCE_KEYS``) for a ``time_sampled``
      smoke output -- merged in verbatim when given. Omit entirely for a
      ``static`` (mesh-only) output: these keys have no applicable target
      at all there.
    - ``upstream_constants`` (optional): the actually-measured upstream
      constant values (e.g. from the ``GRAGRA_SPECFEM_SOURCE``-gated
      cross-check in ``test_specfem_schema.py``) recorded for traceability
      under the ``upstream_constants`` key -- supplementary, not a
      ``schema.py``-defined required key.

    Everything else the caller supplies via ``**scalars`` and is merged in
    verbatim under its given key name. Expected names (non-exhaustive; see
    ``schema.ALL_PROVENANCE_KEYS`` for the authoritative required-key list)
    include: ``solver_kind``, ``solver_version``, ``nproc``,
    ``float_precision``, ``endianness``, ``record_marker_bytes``,
    ``integer_kind`` (the latter three come from a
    ``specfem_bin.RecordLayout`` plus its integer-kind auto-detection, which
    this module does not import), ``density_source``,
    ``suppress_utm_projection``, ``field_units``, ``dt_s``, ``nstep``,
    ``element_id_method``, ``element_id_min``, ``element_id_max``,
    ``n_elements_total``, ``f_resolved_estimate``,
    ``total_quadrature_volume_before_exclusion``,
    ``total_quadrature_volume_after_exclusion``,
    ``total_mass_before_exclusion``, ``total_mass_after_exclusion``.

    Raises
    ------
    ValueError
        If ``output_sampling_kind`` is not one of
        ``schema.VALID_OUTPUT_SAMPLING_KINDS`` (a basic sanity check on a
        value this function itself branches on -- not a required-key
        audit).
    """
    if output_sampling_kind not in schema.VALID_OUTPUT_SAMPLING_KINDS:
        raise ValueError(
            "output_sampling_kind must be one of "
            f"{schema.VALID_OUTPUT_SAMPLING_KINDS}, got {output_sampling_kind!r}"
        )

    result: dict[str, object] = {
        "schema_version": schema_version,
        "mesh_type": mesh_type,
        "layout": layout,
        "output_sampling_kind": output_sampling_kind,
        "coordinate_convention": coordinate_convention,
        "mesh_input_hash": mesh_input_hash,
        "mesh_input_manifest": mesh_input_manifest,
        "par_file_hash": par_file["par_file_hash"],
        "par_file_text": par_file["text"],
        "pml_mask_hash": pml_mask_hash,
        "pml_exclusion_applied": pml_exclusion_applied,
        "cadence_provenance": cadence_provenance,
    }

    result.update(build_identity)
    result["executables"] = runtime_manifest["executables"]
    result["resolution_method"] = runtime_manifest["resolution_method"]

    optional_scalars = {
        "t0": t0,
        "t0_source": t0_source,
        "t0_crosscheck": t0_crosscheck,
        "t0_crosscheck_max_deviation": t0_crosscheck_max_deviation,
        "time_step_origin": time_step_origin,
        "native_step_origin": native_step_origin,
    }
    result.update({k: v for k, v in optional_scalars.items() if v is not None})

    if cadence is not None:
        result.update(cadence)
    if upstream_constants is not None:
        result["upstream_constants"] = dict(upstream_constants)

    result.update(scalars)
    return result
