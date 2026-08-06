"""Fortran unformatted sequential-record binary readers for SPECFEM3D Cartesian.

Implements the "static data" reading half of the certified exporter path
for the decompose (``xdecompose_mesh``) output tree: ``proc*_ibool.bin`` /
``proc*_{x,y,z}.bin`` / ``proc*_rho.bin`` / ``proc*_displ_{X,Y,Z}_it*.bin``
(per-rank movie volume snapshots, primary -- real pilot-verified 2026-08-03,
S1 Step 8) / ``displ_{X,Y,Z}_it*.bin`` (whole-mesh movie volume snapshots,
fallback -- audit-recorded but not reproduced by the S1 pilot build) /
``proc*_Database`` / ``part_array.vtk``.

Provenance: this module is an **independent implementation** written from
the documented Fortran unformatted sequential-record layout and from
byte-level inspection of real solver output. No SPECFEM3D source code is
copied, translated, or otherwise derived here; the upstream file and routine
names that appear below are cited only to identify *which* on-disk artifact
is being read.

This module is **numpy-only**. It must never import ``h5py`` -- that lives
in ``writer.py``, behind the lazy-import convention documented in
``gragra/adapters/hdf5_field.py``.

Fortran unformatted sequential record layout
---------------------------------------------

Every record on disk is ``[marker][payload][marker]``: a leading integer
byte-count, the payload bytes, and a trailing copy of the same count. A
**negative** leading marker is not a valid record start under this
convention -- see the ``ValueError`` raised by :func:`detect_record_layout`,
which explicitly flags this as a possible **subrecord split** (some Fortran
runtimes split a single WRITE whose payload exceeds ~2 GiB into several
physical sub-records; this module does not attempt to reassemble those).

Auto-detection ladder (per the S1 plan Step 4): endianness, record-marker
width (4 or 8 bytes), and default ``INTEGER`` kind are none of them assumed
in advance. :func:`detect_record_layout` tries the four marker-dtype
candidates (``<i4``, ``>i4``, ``<i8``, ``>i8``) against the **whole file**:
for each candidate, it walks the file as a sequence of
``[marker][payload][marker]`` records (head marker == tail marker, at every
step) and accepts the candidate only if doing so consumes the file
**exactly to EOF** with no leftover or overrun bytes -- an exact file-size
match, not merely a plausible first record. This is a strictly stronger
check than validating only the first record, and it works uniformly for
both single-record files (``ibool.bin``, coordinate files) and
multi-record files (``proc*_Database``). The detected
(endianness, record_marker_bytes) is returned in a :class:`RecordLayout` so
callers can carry it into ``provenance.py``.

The default ``INTEGER`` kind used *inside* a payload (e.g. each ``ibool``
entry, or each field of a Database element/material record) is a separate
property from the record-marker width, and is detected independently by
each reader from payload-length divisibility (see :func:`read_ibool` and
:func:`_detect_database_integer_kind` for the exact rule): 4-byte integers
are tried first (the common case), falling back to 8-byte only if 4-byte
does not divide evenly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

# ===========================================================================
# 1. Generic Fortran unformatted sequential record reading
# ===========================================================================

#: (endianness, marker_bytes) candidates tried, in order, by
#: ``detect_record_layout``. Order does not affect correctness (only one
#: candidate should ever validate for a well-formed file) but little-endian
#: 4-byte is tried first since it is by far the most common gfortran/ifort
#: default.
_MARKER_CANDIDATES: Final[tuple[tuple[str, int], ...]] = (
    ("<", 4),
    (">", 4),
    ("<", 8),
    (">", 8),
)


@dataclass(frozen=True)
class RecordLayout:
    """Auto-detected Fortran unformatted sequential record layout for one
    file.

    Attributes
    ----------
    endianness : str
        ``"<"`` (little) or ``">"`` (big) -- a numpy dtype byte-order
        character.
    record_marker_bytes : int
        4 or 8 -- width in bytes of the leading/trailing record-length
        marker.
    """

    endianness: str
    record_marker_bytes: int

    @property
    def marker_dtype(self) -> np.dtype:
        return np.dtype(f"{self.endianness}i{self.record_marker_bytes}")


def _split_records(data: bytes, endian: str, marker_bytes: int) -> list[bytes] | None:
    """Try to parse ``data`` as a sequence of Fortran unformatted records
    under the given (endian, marker_bytes) candidate. Returns the list of
    payload byte-strings if the *entire* buffer is consumed exactly by a
    sequence of self-consistent ``[marker][payload][marker]`` records
    (head marker == tail marker at every step), else ``None``.
    """
    marker_dtype = np.dtype(f"{endian}i{marker_bytes}")
    size = len(data)
    offset = 0
    payloads: list[bytes] = []
    while offset < size:
        if offset + marker_bytes > size:
            return None
        head = int(np.frombuffer(data, dtype=marker_dtype, count=1, offset=offset)[0])
        if head < 0:
            return None
        payload_start = offset + marker_bytes
        payload_end = payload_start + head
        tail_end = payload_end + marker_bytes
        if tail_end > size:
            return None
        tail = int(
            np.frombuffer(data, dtype=marker_dtype, count=1, offset=payload_end)[0]
        )
        if tail != head:
            return None
        payloads.append(data[payload_start:payload_end])
        offset = tail_end
    return payloads


def detect_record_layout(
    data: bytes, *, source: str = "<bytes>"
) -> tuple[RecordLayout, list[bytes]]:
    """Auto-detect (endianness, record_marker_bytes) for ``data`` and return
    the fully-parsed list of record payloads.

    See the module docstring for the detection ladder. ``source`` is used
    only to make the raised ``ValueError`` identify the offending file.
    """
    negative_marker_seen = False
    for endian, marker_bytes in _MARKER_CANDIDATES:
        if len(data) >= marker_bytes:
            head0 = int(
                np.frombuffer(data, dtype=f"{endian}i{marker_bytes}", count=1)[0]
            )
            if head0 < 0:
                negative_marker_seen = True
        payloads = _split_records(data, endian, marker_bytes)
        if payloads is not None:
            return RecordLayout(endian, marker_bytes), payloads

    message = (
        f"{source}: could not auto-detect Fortran unformatted record layout "
        "-- tried little/big-endian 4-byte and 8-byte record markers, and "
        "none produced a sequence of [marker][payload][marker] records that "
        "consumes the file exactly to EOF."
    )
    if negative_marker_seen:
        message += (
            " A negative leading marker was observed for at least one "
            "candidate -- this may indicate a subrecord split (a single "
            "Fortran WRITE whose payload exceeds ~2 GiB, which some "
            "compilers/runtimes split into multiple physical sub-records; "
            "this module does not reassemble those)."
        )
    raise ValueError(message)


def read_records(path) -> tuple[RecordLayout, list[bytes]]:
    """Read an entire Fortran unformatted sequential file, auto-detecting
    its record layout, and return ``(layout, [payload0, payload1, ...])``.
    """
    data = Path(path).read_bytes()
    return detect_record_layout(data, source=str(path))


def _require_single_record(path, payloads: list[bytes]) -> bytes:
    if len(payloads) != 1:
        raise ValueError(
            f"{path}: expected exactly one Fortran record, found {len(payloads)}"
        )
    return payloads[0]


def _detect_float_size(path, payload: bytes, expected_count: int) -> int:
    """Determine whether ``payload`` holds ``expected_count`` float32 or
    float64 values by exact length match. Raises ``ValueError`` if neither
    matches.
    """
    if expected_count <= 0:
        raise ValueError(f"expected_count must be positive, got {expected_count}")
    if len(payload) == expected_count * 4:
        return 4
    if len(payload) == expected_count * 8:
        return 8
    raise ValueError(
        f"{path}: payload length {len(payload)} does not match "
        f"expected_count={expected_count} for float32 ({expected_count * 4} "
        f"bytes) or float64 ({expected_count * 8} bytes)"
    )


# ===========================================================================
# 2. proc*_ibool.bin / proc*_{x,y,z}.bin / proc*_rho.bin
# ===========================================================================


def read_ibool(path, ngll: int) -> tuple[np.ndarray, RecordLayout, int]:
    """Read a ``proc*_ibool.bin`` file.

    SPECFEM3D declares this array as ``ibool(NGLLX,NGLLY,NGLLZ,NSPEC)``,
    Fortran (column-major) order, one-based. Because Fortran column-major
    order varies the *first* declared axis (NGLLX) fastest, then NGLLY,
    then NGLLZ, then NSPEC slowest, the flat on-disk layout is *already*
    exactly ``schema.py``'s ``(k, j, i)`` = ``(z, y, x)`` C-order flatten
    convention within each element's contiguous ``ngll``-length block,
    followed by the next element's block -- so a plain (C-order) reshape to
    ``(nspec, ngll)`` recovers ``ibool[element, flat_gll_index]`` directly,
    with no ``order="F"`` transpose needed.

    ``ngll`` (= NGLLX*NGLLY*NGLLZ) is an external input (from
    ``ExportConfig``/``constants.h``, not stored in this file). ``NSPEC`` is
    *derived*: ``NSPEC = payload_len / (integer_kind * ngll)`` -- the
    integer kind (4 or 8 bytes) is auto-detected by trying 4 bytes first
    (falling back to 8 only if 4 does not divide the payload evenly).

    Returns
    -------
    ibool : np.ndarray, shape (nspec, ngll), int64, zero-based
        Validated to be a surjection onto ``0 .. nglob-1`` (equivalently:
        the one-based values on disk are validated to be a surjection onto
        ``1 .. nglob``, per "nglob = max(ibool)" and "ibool is a surjection
        onto 1..nglob").
    layout : RecordLayout
        The auto-detected endianness/marker width for this file.
    integer_kind : int
        4 or 8 -- the auto-detected width, in bytes, of each ibool entry.
    """
    if ngll <= 0:
        raise ValueError(f"ngll must be positive, got {ngll}")
    layout, payloads = read_records(path)
    payload = _require_single_record(path, payloads)

    integer_kind: int | None = None
    for candidate in (4, 8):
        denom = candidate * ngll
        if len(payload) % denom == 0 and len(payload) // denom > 0:
            integer_kind = candidate
            break
    if integer_kind is None:
        raise ValueError(
            f"{path}: payload length {len(payload)} is not evenly divisible "
            f"by ngll={ngll} for either 4-byte or 8-byte integers"
        )
    nspec = len(payload) // (integer_kind * ngll)

    dtype = np.dtype(f"{layout.endianness}i{integer_kind}")
    flat = np.frombuffer(payload, dtype=dtype)
    ibool_one_based = flat.reshape((nspec, ngll)).astype(np.int64)

    min_val = int(ibool_one_based.min())
    if min_val < 1:
        raise ValueError(
            f"{path}: ibool contains non-positive (one-based) value {min_val}"
        )
    nglob = int(ibool_one_based.max())
    unique_count = len(np.unique(ibool_one_based))
    if unique_count != nglob:
        # Pigeonhole: unique_count distinct integers all lying in [1, nglob]
        # equal nglob in count iff they are exactly {1, ..., nglob} with no
        # gaps -- so this single comparison is both the "out of range" and
        # the "not a full surjection" check.
        raise ValueError(
            f"{path}: ibool is not a surjection onto 1..{nglob} (found "
            f"{unique_count} unique values, expected {nglob})"
        )

    ibool_zero_based = ibool_one_based - 1
    return ibool_zero_based, layout, integer_kind


def read_coords(path, nglob: int) -> tuple[np.ndarray, RecordLayout]:
    """Read a ``proc*_{x,y,z}.bin`` file: a single Fortran record holding
    ``nglob`` scalars -- **one per global DOF (GLL point), not one per mesh
    element**. float32 vs float64 is determined by exact payload-length
    match against ``nglob``.

    Returns
    -------
    values : np.ndarray, shape (nglob,), float64
        Upcast from the on-disk precision (float32 upcast to float64 is
        exact -- no precision is invented).
    layout : RecordLayout
    """
    if nglob <= 0:
        raise ValueError(f"nglob must be positive, got {nglob}")
    layout, payloads = read_records(path)
    payload = _require_single_record(path, payloads)
    float_size = _detect_float_size(path, payload, nglob)
    dtype = np.dtype(f"{layout.endianness}f{float_size}")
    values = np.frombuffer(payload, dtype=dtype).astype(np.float64)
    return values, layout


def read_rho(path, nspec: int, ngll: int) -> tuple[np.ndarray, RecordLayout]:
    """Read a ``proc*_rho.bin`` file: a single Fortran record holding an
    element-local ``(ngll, nspec)`` density array (Fortran column-major,
    i.e. C-order ``(nspec, ngll)`` after a plain reshape -- same layout
    argument as :func:`read_ibool`).

    IMPORTANT -- ``rho.bin`` is **not** the raw background density in
    general: SPECFEM3D's ``save_arrays_solver`` mesher output writes
    ``rho_vp**2 / (4/3*mu + kappa)`` (``vp`` here is the **P-wave velocity**
    material parameter, not a particle velocity), a *derived* quantity that
    differs from the true ``rho0`` by a small systematic relative offset
    (~9e-8 in the pilot measurement -- ``docs/validation/specfem_s1_tier_a.md``
    §5 A3 / ``docs/design/specfem_upstream_gravity_audit.md:301``). This
    near-equivalence holds only under the certified profile's isotropic
    elastic, self-consistent-material-parameters assumption -- it must not
    be assumed for a non-certified material configuration. Which quantity a
    given file actually holds is recorded by the exporter under the
    ``density_source`` provenance key -- this reader has no way to know or guess
    which one it is; callers must track ``density_source`` separately from
    this function's return value, and must not treat "the file is named
    rho.bin" as equivalent to "this is usable as the Newtonian-gravity
    background density rho0" without checking that key.

    Raises ``ValueError`` for non-finite or non-positive density values.
    """
    if nspec <= 0 or ngll <= 0:
        raise ValueError(f"nspec/ngll must be positive, got nspec={nspec}, ngll={ngll}")
    layout, payloads = read_records(path)
    payload = _require_single_record(path, payloads)
    expected_count = nspec * ngll
    float_size = _detect_float_size(path, payload, expected_count)
    dtype = np.dtype(f"{layout.endianness}f{float_size}")
    flat = np.frombuffer(payload, dtype=dtype).astype(np.float64)
    rho = flat.reshape((nspec, ngll))
    if not np.isfinite(rho).all():
        raise ValueError(f"{path}: rho contains non-finite values")
    if (rho <= 0.0).any():
        raise ValueError(f"{path}: rho must be strictly positive (rho<=0 found)")
    return rho, layout


def gather_element_coordinates(ibool: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Combine a zero-based ``ibool`` (shape ``(nspec, ngll)``) with a
    per-DOF coordinate table (shape ``(nglob, 3)`` -- typically built by
    stacking three :func:`read_coords` results) into per-element GLL point
    coordinates, shape ``(nspec, ngll, 3)``.

    Raises ``ValueError`` if any ``ibool`` entry indexes outside ``xyz``'s
    range -- the defensive guard against an ``ibool`` file that references
    more DOFs than the coordinate file actually has (a cross-file
    consistency check that :func:`read_ibool` alone cannot perform, since
    it does not see the coordinate file).
    """
    if ibool.min() < 0:
        raise ValueError("ibool (zero-based) must not contain negative values")
    if ibool.max() >= len(xyz):
        raise ValueError(
            f"ibool references DOF index {int(ibool.max())} but the "
            f"coordinate array only has {len(xyz)} entries (nglob mismatch)"
        )
    return xyz[ibool]


# ===========================================================================
# 3. displ_{X,Y,Z}_it*.bin (volume displacement snapshots)
#
# Two real on-disk layouts exist for the movie volume displacement snapshot
# files (2026-08-03, S1 Step 8 real-pilot findings):
#
# - **per-rank (primary)**: ``proc{rank:06d}_displ_{X,Y,Z}_it{it:06d}.bin``
#   under LOCAL_PATH, one file set per MPI rank. Each file is a single
#   Fortran record, ``(NGLLX,NGLLY,NGLLZ,nspec_rank)`` column-major --
#   exactly the same on-disk shape convention as ``proc*_rho.bin`` (see
#   :func:`read_rho`). Byte-size verified against the real pilot: rank 0
#   with ``nspec_rank=5184`` gives ``4 + 4*125*5184 + 4 = 2_592_008`` bytes,
#   an exact match. The rank-local element order is the *same* order as
#   this rank's ``proc*_Database`` element block (:class:`DatabaseContents`
#   .elements) -- i.e. the same order :func:`recover_element_ids` recovers
#   for this rank, so re-combining into the canonical element_id-ascending
#   mesh array uses the identical index permutation the mesh read already
#   computed. This is the layout the S1 pilot's actual SPECFEM3D CPU build
#   writes; treat it as the default expectation for new solver builds.
# - **whole-mesh (fallback)**: ``displ_{X,Y,Z}_it{it:06d}.bin``, a single
#   file set covering all ``NSPEC`` elements at once (movie-volume binary
#   format, real-run byte-size verified: ``NSPEC=12800`` gives
#   ``4 + 4*125*12800 + 4 = 6_400_008`` bytes). This is a **different**
#   real SPECFEM3D run/build than the S1 pilot -- it was not reproduced by
#   the pilot's own build, but is real and documented, so this reader keeps
#   it as a fallback for solver configurations that write this way instead.
#
# :func:`detect_displacement_layout` distinguishes the two purely from
# filename presence (no file content is read for the decision).
# ===========================================================================

_DISPL_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^displ_([XYZ])_it(\d{6})\.bin$"
)

_RANK_DISPL_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^proc(\d{6})_displ_([XYZ])_it(\d{6})\.bin$"
)

DisplacementLayout = Literal["per_rank", "whole_mesh"]


def detect_displacement_layout(dirpath) -> DisplacementLayout:
    """Auto-detect which of the two real movie-volume displacement snapshot
    file layouts is present under ``dirpath`` (see the section-3 module
    comment above for the two layouts' full description).

    Detection is by filename presence only: if any ``proc*_displ_*_it*.bin``
    file exists, the per-rank layout is used (this takes priority even if
    whole-mesh files happen to coexist in the same directory); otherwise, if
    any ``displ_*_it*.bin`` file exists, the whole-mesh layout is used.
    Raises ``ValueError`` if neither pattern matches any file.
    """
    dirpath = Path(dirpath)
    if any(dirpath.glob("proc*_displ_*_it*.bin")):
        return "per_rank"
    if any(dirpath.glob("displ_*_it*.bin")):
        return "whole_mesh"
    raise ValueError(
        f"{dirpath}: no displacement snapshot files found -- expected "
        "either 'proc{rank:06d}_displ_{X,Y,Z}_it{it:06d}.bin' (per-rank, "
        "primary) or 'displ_{X,Y,Z}_it{it:06d}.bin' (whole-mesh, fallback)"
    )


def _read_snapshot_records(
    by_it: dict[int, dict[str, Path]],
    nspec: int,
    ngll: int,
    *,
    missing_component_context: str = "",
) -> tuple[np.ndarray, list[int], RecordLayout]:
    """Shared record-reading core for both :func:`read_displacement_snapshots`
    (whole-mesh) and :func:`read_rank_displacement_snapshots` (per-rank):
    given an already-grouped ``{it: {component: path}}`` mapping, reads every
    component file and bundles each timestep's 3 components into one array.

    ``missing_component_context`` is prepended verbatim to the "missing
    component(s)" error message (e.g. ``f"rank {rank}, "``) so a per-rank
    caller's error identifies which rank failed; the whole-mesh caller
    passes the default empty string, keeping its error message unchanged
    from before this helper was factored out.
    """
    expected_count = nspec * ngll
    layout_ref: RecordLayout | None = None
    its_sorted = sorted(by_it)
    snapshots: list[np.ndarray] = []
    for it in its_sorted:
        comps = by_it[it]
        missing = sorted({"X", "Y", "Z"} - set(comps))
        if missing:
            raise ValueError(
                f"{missing_component_context}it={it:06d}: missing "
                f"displacement component(s) {missing} (or an inconsistent "
                "'it' set across X/Y/Z components)"
            )
        axis_arrays = []
        for comp in ("X", "Y", "Z"):
            comp_path = comps[comp]
            layout, payloads = read_records(comp_path)
            payload = _require_single_record(comp_path, payloads)
            float_size = _detect_float_size(comp_path, payload, expected_count)
            dtype = np.dtype(f"{layout.endianness}f{float_size}")
            arr = (
                np.frombuffer(payload, dtype=dtype)
                .astype(np.float64)
                .reshape((nspec, ngll))
            )
            axis_arrays.append(arr)
            if layout_ref is None:
                layout_ref = layout
        snapshots.append(np.stack(axis_arrays, axis=-1))

    displacement = np.stack(snapshots, axis=0)
    assert layout_ref is not None  # by_it non-empty => at least one iteration
    return displacement, its_sorted, layout_ref


def read_displacement_snapshots(
    dirpath, nspec: int, ngll: int, glob_pattern: str = "displ_*_it*.bin"
) -> tuple[np.ndarray, list[int], RecordLayout]:
    """Read a set of ``displ_{X,Y,Z}_it######.bin`` **whole-mesh** volume
    movie snapshot files (six-digit ``it``, per the audit-confirmed naming)
    and bundle each timestep's 3 components into one array.

    This is the **fallback** layout -- see the section-3 module comment
    above. Callers that do not already know which layout is on disk should
    go through :func:`detect_displacement_layout` first.

    Component-letter -> axis mapping is a direct X/Y/Z write (this is only
    valid under ``SUPPRESS_UTM_PROJECTION=.true.`` -- with UTM projection
    active the components are E/N/Z in a projected frame instead; enforcing
    that precondition is a certified-profile gate concern (Step 6), not
    this reader's).

    Raises ``ValueError`` if any ``it`` value is missing one or more of the
    three components (this also transitively catches an inconsistent ``it``
    set across components: an ``it`` present for only some components shows
    up here as "missing component(s)" for that ``it``).

    Returns
    -------
    displacement : np.ndarray, shape (ntime, nspec, ngll, 3), float64
    its : list[int]
        The (ascending, deduplicated) six-digit ``it`` values found,
        parallel to ``displacement``'s leading axis.
    layout : RecordLayout
        Layout detected from the first component file read (all files are
        expected to share one file layout; this is not independently
        re-verified per file beyond each file's own record parse).
    """
    dirpath = Path(dirpath)
    by_it: dict[int, dict[str, Path]] = {}
    for p in sorted(dirpath.glob(glob_pattern)):
        m = _DISPL_FILENAME_RE.match(p.name)
        if m is None:
            continue
        comp, it_str = m.group(1), m.group(2)
        by_it.setdefault(int(it_str), {})[comp] = p

    if not by_it:
        raise ValueError(
            f"{dirpath}: no displ_{{X,Y,Z}}_it######.bin files matched "
            f"pattern {glob_pattern!r}"
        )

    return _read_snapshot_records(by_it, nspec, ngll)


def read_rank_displacement_snapshots(
    dirpath,
    rank: int,
    nspec_rank: int,
    ngll: int,
    glob_pattern: str | None = None,
) -> tuple[np.ndarray, list[int], RecordLayout]:
    """Read one rank's ``proc{rank:06d}_displ_{X,Y,Z}_it######.bin``
    **per-rank** volume movie snapshot files -- the **primary** layout, real
    pilot-verified 2026-08-03 (see the section-3 module comment above).

    ``nspec_rank`` is this rank's local element count (the same value
    :func:`recover_element_ids` returns the length of for this rank, and
    the same value ``proc*_rho.bin``/``proc*_ibool.bin`` use as their
    element axis) -- **not** the global ``n_elements_total``. The returned
    array's element axis is in this rank's *local* (Database) element
    order; callers reassembling the full mesh must scatter it via this
    rank's ``element_id`` array (:func:`recover_element_ids`), exactly as
    they already do for ``corners8``/``coordinates``/``density`` in the
    mesh read.

    Raises ``ValueError`` for a negative ``rank``, for no matching files
    (rank file(s) entirely missing), and -- via :func:`_read_snapshot_records`
    -- for a missing component / inconsistent ``it`` set within this rank.

    Returns
    -------
    displacement : np.ndarray, shape (ntime, nspec_rank, ngll, 3), float64
    its : list[int]
        The (ascending, deduplicated) six-digit ``it`` values found for
        this rank, parallel to ``displacement``'s leading axis.
    layout : RecordLayout
    """
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    dirpath = Path(dirpath)
    pattern = (
        glob_pattern if glob_pattern is not None else f"proc{rank:06d}_displ_*_it*.bin"
    )
    by_it: dict[int, dict[str, Path]] = {}
    for p in sorted(dirpath.glob(pattern)):
        m = _RANK_DISPL_FILENAME_RE.match(p.name)
        if m is None:
            continue
        m_rank, comp, it_str = m.group(1), m.group(2), m.group(3)
        if int(m_rank) != rank:
            continue
        by_it.setdefault(int(it_str), {})[comp] = p

    if not by_it:
        raise ValueError(
            f"{dirpath}: no proc{rank:06d}_displ_{{X,Y,Z}}_it######.bin "
            f"files matched pattern {pattern!r}"
        )

    return _read_snapshot_records(
        by_it, nspec_rank, ngll, missing_component_context=f"rank {rank}, "
    )


# ===========================================================================
# 4. proc*_Database parsing (decompose_mesh output)
# ===========================================================================


@dataclass(frozen=True)
class DatabaseMaterial:
    """One defined-material record: 17 float64 values, no ``material_id``
    field -- the k-th record (0-based here) corresponds to
    ``material_id = k + 1`` by the pure ordering convention.
    """

    values: np.ndarray  # shape (17,), float64

    @property
    def domain_id(self) -> float:
        """0-based index 6 of the 17-field record (Fortran
        ``mat_prop(7,·)`` -- the 1-based 7th field). Verified against the
        real ``proc000000_Database`` on 2026-08-03: fields are
        ``(rho, vp, vs, Q_kappa, Q_mu, anisotropy_flag, domain_id, ...)``
        and the pilot's elastic material stores ``2.0`` here. Earlier
        1-based/0-based ambiguity ("index 7") caused a real read bug that
        only surfaced on real pilot data -- do not change without
        re-probing an actual Database file.
        """
        return float(self.values[6])


@dataclass(frozen=True)
class DatabaseElement:
    """One element record: ``iloc`` (this rank's local element index, as
    stored -- redundant with record position but kept for round-trip
    fidelity), ``mat1``/``mat2`` (material record indices, one-based;
    ``mat2`` is 0 on the decompose path -- never assume it equals 1), and
    the 8 local (one-based, this rank's local node numbering)
    ``usual_hex_nodes``-ordered corner node ids.
    """

    iloc: int
    mat1: int
    mat2: int
    node_loc_ids: tuple[int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class DatabaseContents:
    """Parsed contents of one ``proc*_Database`` file (the subset Step 4
    needs: node block, material block, element block). Everything after
    the element block (boundary headers/records, CPML count, MPI
    interfaces) is consumed generically by :func:`read_records` (which
    already validates the *entire* file's record structure against one
    consistent layout, all the way to an exact EOF) but not semantically
    interpreted. The supported scope is deliberately "parse
    nnodes/nodes/materials/nspec/elements, then skip the remainder and
    confirm EOF consistency".
    """

    nnodes_loc: int
    node_iloc: np.ndarray  # shape (nnodes_loc,), int64, one-based as stored
    node_xyz: np.ndarray  # shape (nnodes_loc, 3), float64
    materials: tuple[DatabaseMaterial, ...]
    nspec_local: int
    elements: tuple[DatabaseElement, ...]
    layout: RecordLayout
    integer_kind: int


def _detect_database_integer_kind(path, first_payload: bytes) -> int:
    """The first Database record (``nnodes_loc``, a lone scalar) is used to
    detect the default INTEGER kind: its length must be exactly 4 or 8
    bytes."""
    if len(first_payload) == 4:
        return 4
    if len(first_payload) == 8:
        return 8
    raise ValueError(
        f"{path}: nnodes_loc record has length {len(first_payload)}, "
        "expected 4 or 8 bytes (a single INTEGER(4) or INTEGER(8) value)"
    )


def parse_database(path) -> DatabaseContents:
    """Parse a ``proc*_Database`` file's node/material/element blocks.

    Record structure consumed (byte-verified against real solver output):
    ``nnodes_loc`` -> node records x nnodes_loc (each: ``iloc`` [int] +
    ``x,y,z`` [float64]) -> ``ndef, nundef`` -> defined-material records x
    ``ndef`` (17 float64 each) -> (``nundef`` undefined/tomography material
    records, skipped generically -- their layout is solver-version-specific
    and irrelevant to the certified MVP profile) -> ``nspec_local`` ->
    element records x ``nspec_local`` (``iloc, mat1, mat2`` + 8 local node
    ids) -> everything else (boundary/CPML/interfaces), read but not
    interpreted.
    """
    layout, payloads = read_records(path)
    idx = 0

    def next_payload() -> bytes:
        nonlocal idx
        if idx >= len(payloads):
            raise ValueError(f"{path}: unexpected EOF while parsing Database")
        p = payloads[idx]
        idx += 1
        return p

    nnodes_payload = next_payload()
    integer_kind = _detect_database_integer_kind(path, nnodes_payload)
    int_dtype = np.dtype(f"{layout.endianness}i{integer_kind}")
    float_dtype = np.dtype(f"{layout.endianness}f8")

    nnodes_loc = int(np.frombuffer(nnodes_payload, dtype=int_dtype)[0])
    if nnodes_loc <= 0:
        raise ValueError(f"{path}: nnodes_loc must be positive, got {nnodes_loc}")

    node_iloc = np.empty(nnodes_loc, dtype=np.int64)
    node_xyz = np.empty((nnodes_loc, 3), dtype=np.float64)
    expected_node_len = integer_kind + 24
    for n in range(nnodes_loc):
        payload = next_payload()
        if len(payload) != expected_node_len:
            raise ValueError(
                f"{path}: node record {n} has length {len(payload)}, "
                f"expected {expected_node_len} (iloc[{integer_kind}B] + "
                "xyz[24B, float64x3])"
            )
        node_iloc[n] = int(np.frombuffer(payload, dtype=int_dtype, count=1)[0])
        node_xyz[n] = np.frombuffer(
            payload, dtype=float_dtype, count=3, offset=integer_kind
        )

    ndef_payload = next_payload()
    if len(ndef_payload) != 2 * integer_kind:
        raise ValueError(
            f"{path}: ndef/nundef record has unexpected length "
            f"{len(ndef_payload)}, expected {2 * integer_kind}"
        )
    ndef, nundef = (
        int(v) for v in np.frombuffer(ndef_payload, dtype=int_dtype, count=2)
    )
    if ndef < 0 or nundef < 0:
        raise ValueError(
            f"{path}: negative material counts ndef={ndef}, nundef={nundef}"
        )

    materials: list[DatabaseMaterial] = []
    for m in range(ndef):
        payload = next_payload()
        if len(payload) != 17 * 8:
            raise ValueError(
                f"{path}: defined-material record {m} has length "
                f"{len(payload)}, expected {17 * 8} (17 float64 values)"
            )
        values = np.frombuffer(payload, dtype=float_dtype, count=17).copy()
        materials.append(DatabaseMaterial(values=values))
    for _ in range(nundef):
        next_payload()  # undefined/tomography materials -- skipped generically

    nspec_payload = next_payload()
    if len(nspec_payload) != integer_kind:
        raise ValueError(
            f"{path}: nspec_local record has unexpected length "
            f"{len(nspec_payload)}, expected {integer_kind}"
        )
    nspec_local = int(np.frombuffer(nspec_payload, dtype=int_dtype)[0])
    if nspec_local <= 0:
        raise ValueError(f"{path}: nspec_local must be positive, got {nspec_local}")

    elements: list[DatabaseElement] = []
    expected_elem_len = 11 * integer_kind
    for e in range(nspec_local):
        payload = next_payload()
        if len(payload) != expected_elem_len:
            raise ValueError(
                f"{path}: element record {e} has length {len(payload)}, "
                f"expected {expected_elem_len} (11 integers: iloc, mat1, "
                "mat2, 8 node-loc-ids)"
            )
        values = np.frombuffer(payload, dtype=int_dtype, count=11)
        iloc = int(values[0])
        mat1 = int(values[1])
        mat2 = int(values[2])
        node_loc_ids = tuple(int(v) for v in values[3:11])
        if mat1 < 1 or mat1 > ndef:
            raise ValueError(
                f"{path}: element record {e} has mat1={mat1} out of range "
                f"[1, {ndef}] -- material record count/mat1 field mismatch"
            )
        elements.append(
            DatabaseElement(iloc=iloc, mat1=mat1, mat2=mat2, node_loc_ids=node_loc_ids)
        )

    # Everything past this point (boundary headers/records, nspec_cpml,
    # interfaces) is intentionally not semantically parsed (Step 4 scope).
    # `read_records()` already consumed and validated the ENTIRE file's
    # record structure up front (every remaining [marker][payload][marker]
    # against this same detected layout, exactly to EOF) -- so reaching
    # this point with a fully-populated `payloads` list already constitutes
    # the "read past + EOF alignment" check; there is nothing left to do.
    return DatabaseContents(
        nnodes_loc=nnodes_loc,
        node_iloc=node_iloc,
        node_xyz=node_xyz,
        materials=tuple(materials),
        nspec_local=nspec_local,
        elements=tuple(elements),
        layout=layout,
        integer_kind=integer_kind,
    )


# ===========================================================================
# 5. part_array.vtk parsing and element_id recovery
# ===========================================================================


@dataclass(frozen=True)
class PartArray:
    """Parsed ``part_array.vtk`` (ASCII legacy VTK, decompose_mesh output).

    Attributes
    ----------
    points : np.ndarray, shape (n_points, 3), float64
        Global (zero-based-indexed) mesh node coordinates.
    cells : np.ndarray, shape (nspec, 8), int64
        VTK_HEXAHEDRON connectivity: zero-based global node ids, one row
        per global mesh element, in the file's cell order.
    elem_flag : np.ndarray, shape (nspec,), int64
        ``CELL_DATA`` scalar: the MPI rank owning each global element
        (parallel to ``cells``' row order).
    """

    points: np.ndarray
    cells: np.ndarray
    elem_flag: np.ndarray


_VTK_POINTS_RE: Final[re.Pattern[str]] = re.compile(r"^POINTS\s+(\d+)\s+(\S+)\s*$")
_VTK_CELLS_RE: Final[re.Pattern[str]] = re.compile(r"^CELLS\s+(\d+)\s+(\d+)\s*$")
_VTK_CELL_DATA_RE: Final[re.Pattern[str]] = re.compile(r"^CELL_DATA\s+(\d+)\s*$")


def read_part_array_vtk(path) -> PartArray:
    """Parse an ASCII legacy VTK ``part_array.vtk`` file: ``POINTS``
    (global mesh node coordinates), ``CELLS`` (``nspec`` rows of 9 values:
    leading ``8`` + 8 zero-based global node ids, VTK_HEXAHEDRON order),
    and ``CELL_DATA`` scalar ``elem_flag`` (the rank owning each global
    element).
    """
    lines = Path(path).read_text().splitlines()
    n_lines = len(lines)

    points: np.ndarray | None = None
    cells: np.ndarray | None = None
    elem_flag: np.ndarray | None = None

    i = 0
    while i < n_lines:
        line = lines[i].strip()

        m_points = _VTK_POINTS_RE.match(line)
        if m_points:
            n_points = int(m_points.group(1))
            values: list[float] = []
            i += 1
            while len(values) < n_points * 3:
                if i >= n_lines:
                    raise ValueError(
                        f"{path}: POINTS block truncated -- declared {n_points} "
                        f"points but ran out of file data (index {i} >= "
                        f"{n_lines} lines; collected {len(values)}/{n_points * 3} "
                        "coordinate values)"
                    )
                values.extend(float(v) for v in lines[i].split())
                i += 1
            points = np.array(values, dtype=np.float64).reshape((n_points, 3))
            continue

        m_cells = _VTK_CELLS_RE.match(line)
        if m_cells:
            n_cells = int(m_cells.group(1))
            rows: list[list[int]] = []
            i += 1
            for row_idx in range(n_cells):
                if i >= n_lines:
                    raise ValueError(
                        f"{path}: CELLS block truncated -- declared {n_cells} "
                        f"rows but ran out of file data after {row_idx}/{n_cells} "
                        f"rows read (index {i} >= {n_lines} lines)"
                    )
                parts = [int(v) for v in lines[i].split()]
                if not parts or parts[0] != 8:
                    raise ValueError(
                        f"{path}: CELLS row {row_idx} does not start with "
                        f"8 (VTK_HEXAHEDRON), got {parts[:1]!r}"
                    )
                rows.append(parts[1:9])
                i += 1
            cells = np.array(rows, dtype=np.int64)
            continue

        m_cell_data = _VTK_CELL_DATA_RE.match(line)
        if m_cell_data:
            n_cell_data = int(m_cell_data.group(1))
            i += 1
            while i < n_lines and not lines[i].strip().upper().startswith("SCALARS"):
                i += 1
            if i >= n_lines:
                raise ValueError(f"{path}: CELL_DATA block missing SCALARS header")
            i += 1  # consume SCALARS line
            if i < n_lines and lines[i].strip().upper().startswith("LOOKUP_TABLE"):
                i += 1  # consume LOOKUP_TABLE line
            values_int: list[int] = []
            while len(values_int) < n_cell_data:
                if i >= n_lines:
                    raise ValueError(
                        f"{path}: CELL_DATA block truncated -- declared "
                        f"{n_cell_data} scalar values but ran out of file data "
                        f"(index {i} >= {n_lines} lines; collected "
                        f"{len(values_int)}/{n_cell_data} values)"
                    )
                values_int.extend(int(float(v)) for v in lines[i].split())
                i += 1
            elem_flag = np.array(values_int, dtype=np.int64)
            continue

        i += 1

    if points is None or cells is None or elem_flag is None:
        missing = [
            name
            for name, val in (
                ("POINTS", points),
                ("CELLS", cells),
                ("CELL_DATA/elem_flag", elem_flag),
            )
            if val is None
        ]
        raise ValueError(f"{path}: missing VTK section(s): {missing}")
    if len(elem_flag) != len(cells):
        raise ValueError(
            f"{path}: elem_flag length {len(elem_flag)} != CELLS row count {len(cells)}"
        )
    return PartArray(points=points, cells=cells, elem_flag=elem_flag)


def recover_element_ids(part: PartArray, rank: int) -> np.ndarray:
    """Recover the zero-based, partition-independent ``element_id`` array
    for the elements owned by ``rank``
    (``element_id_method = "pre_partition_global_index"``).

    Algorithm (integer arithmetic only -- no coordinate matching, hashing,
    or quantization): ``G_r = sorted({i : part.elem_flag[i] == rank})``,
    where ``i`` is the zero-based row index into ``part.cells`` /
    ``part.elem_flag``. Because upstream writes both ``part_array.vtk`` and
    each rank's Database element block by walking the *global* element
    index in ascending order (``part_decompose_mesh.F90:310-401,972-991``),
    the k-th (zero-based) local element record in ``proc{rank}_Database``
    corresponds to ``element_id = G_r[k]`` -- this zero-based array
    position *is* the "global index - 1" the design doc's algorithm
    description defines ``element_id`` as (the one-based "global index"
    equals this position + 1).
    """
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    matches = np.nonzero(part.elem_flag == rank)[0]
    if matches.size == 0:
        raise ValueError(f"No elements found for rank {rank} in elem_flag")
    return np.sort(matches).astype(np.int64)
