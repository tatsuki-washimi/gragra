"""Orchestration entry point for the SPECFEM3D Cartesian GLL certified/smoke
export (Step 6): ``ExportConfig`` + the certified-profile gate + the
multi-rank read/reconstruct/self-check pipeline + provenance assembly +
delegation to ``writer.py``.

Pipeline:

1. Parse ``DATA/Par_file`` and run the **profile gate** first (before any
   binary is touched) -- ``NGNOD != 8``, ``ATTENUATION``,
   ``PML_INSTEAD_OF_FREE_SURFACE``, ``SUPPRESS_UTM_PROJECTION``,
   ``SAVE_MESH_FILES``, ``SIMULATION_TYPE``, and (time_sampled only)
   ``SAVE_DISPLACEMENT`` all raise ``ValueError`` on violation.
2. Read ``part_array.vtk`` + every rank's ``proc*_Database`` + ``proc*_{ibool,
   x,y,z,rho}.bin`` (``specfem_bin``), recover ``element_id`` per rank, and
   combine all ranks into one element_id-ascending canonical table.
3. Compute ``/mesh/quadrature_volume`` from the Database anchor (control)
   nodes via ``gll.elements_quadrature_volume``.
4. Run the **anchor<->GLL bridging self-check** (the only real-data check
   that the Database's element_id ordering and the ibool-derived GLL field
   array's ordering agree).
5. Assemble provenance (``provenance.assemble_provenance``).
6. Delegate to ``writer.write_gll_hdf5``.

Layout note (docstrings.md index): as in ``writer.py``, the on-disk
``/field/displacement`` leading axis is **time** -- an on-disk file-layout
convention distinct from gragra's in-memory API shape convention (component
axis trailing). This module never returns an in-memory field array, so the
two conventions do not collide, but callers building a reader on this schema
must convert explicitly rather than assume they are the same axis
convention.

This module (like ``writer.py``) must not import ``h5py`` at module scope --
``writer.py``'s lazy-import convention is the only place h5py is touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np

from gragra.adapters.specfem import gll, provenance, schema, specfem_bin
from gragra.adapters.specfem import writer as _writer

#: Explanatory tail appended to every certified-profile gate ``ValueError``.
#: Deliberately prose rather than a document path: the gate's meaning is that
#: the run is outside the *named solver profile* gragra self-certifies against
#: its own written contract, and that statement must stay readable to someone
#: who has only the installed package.
_PROFILE_DOC_REF: Final[str] = (
    "'certified' here means self-certified against gragra's own written"
    " contract for this named SPECFEM3D Cartesian profile -- not a"
    " third-party audit"
)

#: Relative to ExportConfig.specfem_dir -- auto-parse fallbacks for t0 /
#: f_resolved_estimate when the config does not supply them explicitly.
_OUTPUT_SOLVER_TXT_RELPATH: Final[str] = "OUTPUT_FILES/output_solver.txt"

_DATABASE_FILENAME_RE: Final[re.Pattern[str]] = re.compile(r"^proc(\d{6})_Database$")

OutputSamplingKind = Literal["static", "time_sampled"]


# ===========================================================================
# ExportConfig
# ===========================================================================


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for one SPECFEM3D Cartesian GLL certified/smoke export.

    Only ``specfem_dir``/``output_path``/``output_sampling_kind`` have no
    generic default -- everything else is either mechanically derivable
    (``nproc`` is auto-detected from ``proc*_Database`` files;
    ``LOCAL_PATH`` is auto-resolved from ``DATA/Par_file`` unless
    ``local_path`` overrides it) or a scientific/build-identity input this
    module deliberately refuses to fabricate (``cadence``, ``t0``,
    ``f_resolved_estimate`` when no matching solver log file exists --
    raising ``ValueError`` instead of guessing rather than fabricating a
    plausible-looking default).

    Attributes
    ----------
    specfem_dir : Path
        Root of the SPECFEM3D run tree (contains ``DATA/Par_file``, and --
        via ``LOCAL_PATH`` -- the decompose_mesh output tree).
    output_path : Path
        Destination canonical HDF5 path.
    output_sampling_kind : "static" | "time_sampled"
        ``"static"`` writes a mesh-only artifact (no ``/field/displacement``
        dataset); ``"time_sampled"`` additionally writes movie-volume
        displacement snapshots under the configured sampling cadence.
    ngll_shape : tuple[int, int, int]
        ``(NGLLX, NGLLY, NGLLZ)``, default ``schema.EXPECTED_NGLL``.
    local_path : Path | None
        Override for the decompose_mesh output directory (``proc*_Database``
        etc.); when ``None``, resolved from ``Par_file``'s ``LOCAL_PATH``
        (which ``xdecompose_mesh`` treats as the write target, not a CLI
        argument).
    par_file_relpath : str
        Path to ``Par_file`` relative to ``specfem_dir``.
    density_source, solver_kind, solver_version, float_precision, field_units
        Recorded verbatim into provenance (``density_source`` documents
        which quantity ``rho.bin`` actually holds -- see
        ``specfem_bin.read_rho``'s docstring; ``float_precision`` is a
        build-configuration fact this module has no independent way to
        verify from the binaries alone). The default,
        ``"proc_rho_bin_derived"``, is only a valid stand-in for the
        Newtonian-gravity background density rho0 under the certified
        profile's isotropic-elastic, self-consistent-material-parameters
        assumption -- do not reuse this default for a non-certified
        material configuration without re-deriving that the equivalence
        still holds.
    f_resolved_estimate : float | None
        If ``None``, parsed from
        ``specfem_dir/output_generate_databases_relpath`` when that file
        exists; otherwise the export raises ``ValueError`` (this key is
        unconditionally required by the schema).
    build_identity : Mapping | None
        If ``None``, auto-collected via ``provenance.collect_build_identity``
        (requires a real ``.git`` under ``specfem_dir``). Callers without a
        git-tracked solver checkout (e.g. synthetic/test fixtures) must
        supply this explicitly.
    exporter_code_ref : str
        Passed to ``provenance.collect_exporter_identity``.
    runtime_manifest_entries : tuple[Mapping, ...]
        Pre-captured ``executables[]`` entries (e.g. loaded from
        ``capture-run``-produced manifest JSON files for the mesher/
        decompose/solver executables). The exporter's own identity entry is
        always appended automatically.
    mesh_input_files : tuple[str, ...]
        External mesh input files (relative to ``specfem_dir``, e.g. under
        ``MESH-default/``) folded into ``mesh_input_hash`` alongside
        ``Par_file``. Empty means "hash ``Par_file`` alone" -- a degraded,
        non-certified-profile fallback documented here rather than silently
        assumed equivalent to the real external-mesh-route hash.
    t0, t0_source, t0_crosscheck, t0_crosscheck_max_deviation, cadence,
    cadence_provenance
        ``time_sampled``-only. ``t0``/``t0_source`` auto-parsed from
        ``OUTPUT_FILES/output_solver.txt`` when not given. ``cadence`` has
        no auto-fallback (scientific input -- see class docstring); only
        ``output_interval_steps`` (mechanically read from
        ``NTSTEP_BETWEEN_FRAMES``) is filled in automatically.
    material_id_source, domain_id_source, is_pml_source
        Fixed documentation tags recorded into provenance (extra, non-
        schema-required keys) naming where each mask/label actually came
        from. This exporter version always reads ``material_id``/
        ``domain_id`` from the per-rank Database (generalizes the MVP
        single-material profile without a separate code path) and always
        writes ``is_pml`` as all-False (no CPML support in this exporter
        version -- the Database's boundary/CPML tail records are
        deliberately not parsed, per ``specfem_bin.parse_database``'s
        docstring).
    chunk_element_size, gzip_level
        Passed through to ``writer.write_gll_hdf5``.
    """

    specfem_dir: Path
    output_path: Path
    output_sampling_kind: OutputSamplingKind

    ngll_shape: tuple[int, int, int] = schema.EXPECTED_NGLL
    local_path: Path | None = None
    par_file_relpath: str = provenance.DEFAULT_PAR_FILE_RELPATH

    density_source: str = "proc_rho_bin_derived"
    solver_kind: str = "specfem3d_cartesian"
    solver_version: str = "unknown"
    float_precision: str = "single"
    field_units: dict[str, str] = field(default_factory=lambda: {"displacement": "m"})

    f_resolved_estimate: float | None = None
    output_generate_databases_relpath: str = (
        "OUTPUT_FILES/output_generate_databases.txt"
    )

    build_identity: dict[str, object] | None = None
    exporter_code_ref: str = "unknown"
    runtime_manifest_entries: tuple[dict[str, object], ...] = ()

    mesh_input_files: tuple[str, ...] = ()

    t0: float | None = None
    t0_source: str | None = None
    t0_crosscheck: str | None = None
    t0_crosscheck_max_deviation: float | None = None
    cadence: dict[str, object] | None = None
    cadence_provenance: str = "placeholder"

    material_id_source: Literal["database"] = "database"
    domain_id_source: Literal["database"] = "database"
    is_pml_source: Literal["profile_constant_all_false"] = "profile_constant_all_false"

    chunk_element_size: int | None = None
    gzip_level: int = 4

    def __post_init__(self) -> None:
        if self.output_sampling_kind not in ("static", "time_sampled"):
            raise ValueError(
                "output_sampling_kind must be 'static' or 'time_sampled', "
                f"got {self.output_sampling_kind!r}"
            )
        object.__setattr__(self, "specfem_dir", Path(self.specfem_dir))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if self.local_path is not None:
            object.__setattr__(self, "local_path", Path(self.local_path))


# ===========================================================================
# Profile gate
# ===========================================================================


def _require_par_value(values: dict[str, str], key: str) -> str:
    if key not in values:
        raise ValueError(
            f"Par_file is missing required key {key!r} for the certified "
            f"profile gate ({_PROFILE_DOC_REF})"
        )
    return values[key]


def _check_profile_gate(values: dict[str, str], output_sampling_kind: str) -> None:
    """Reject a SPECFEM3D configuration outside the certified profile.

    This is the exporter's only defence against several structural blind
    spots the rest of the pipeline cannot detect on its own -- most notably
    ``NGNOD != 8``: the A2 volume oracle and ``gll.py``'s Jacobian both
    assume an 8-node trilinear hexahedron and would silently mis-integrate a
    27-node element without ever raising on their own.
    """
    ngnod = int(
        provenance.par_file_value_as_number(_require_par_value(values, "NGNOD"))
    )
    if ngnod != 8:
        raise ValueError(
            "certified profile requires NGNOD=8 (8-node trilinear hexahedra "
            "-- the A2 volume oracle's structural blind spot for other node "
            f"counts), got NGNOD={ngnod} ({_PROFILE_DOC_REF})"
        )
    if provenance.par_file_value_as_bool(_require_par_value(values, "ATTENUATION")):
        raise ValueError(
            "certified profile requires ATTENUATION=.false. (attenuation "
            "introduces a systematic difference from the elastic analytic "
            f"solutions Tier B validates against), got .true. ({_PROFILE_DOC_REF})"
        )
    if provenance.par_file_value_as_bool(
        _require_par_value(values, "PML_INSTEAD_OF_FREE_SURFACE")
    ):
        raise ValueError(
            "certified profile requires PML_INSTEAD_OF_FREE_SURFACE=.false. "
            "(a free surface must exist for the surface mass-sheet term the "
            f"profile validates), got .true. ({_PROFILE_DOC_REF})"
        )
    if not provenance.par_file_value_as_bool(
        _require_par_value(values, "SUPPRESS_UTM_PROJECTION")
    ):
        raise ValueError(
            "certified profile requires SUPPRESS_UTM_PROJECTION=.true. (the "
            "displ_{X,Y,Z} component-letter -> axis mapping is only a "
            f"direct X/Y/Z write under this setting), got .false. ({_PROFILE_DOC_REF})"
        )
    if not provenance.par_file_value_as_bool(
        _require_par_value(values, "SAVE_MESH_FILES")
    ):
        raise ValueError(
            "certified profile requires SAVE_MESH_FILES=.true. (the static "
            "mesh output the mass-weight reconstruction reads from), got "
            f".false. ({_PROFILE_DOC_REF})"
        )
    sim_type = int(
        provenance.par_file_value_as_number(
            _require_par_value(values, "SIMULATION_TYPE")
        )
    )
    if sim_type != 1:
        raise ValueError(
            "certified profile requires SIMULATION_TYPE=1 (an adjoint run, "
            "SIMULATION_TYPE=3, relabels time as (NSTEP-it)*DT-t0, breaking "
            f"the t0_source/t0_crosscheck contract), got {sim_type} "
            f"({_PROFILE_DOC_REF})"
        )
    if output_sampling_kind == "time_sampled" and not provenance.par_file_value_as_bool(
        _require_par_value(values, "SAVE_DISPLACEMENT")
    ):
        raise ValueError(
            "output_sampling_kind='time_sampled' requires "
            "SAVE_DISPLACEMENT=.true. (.false. writes a velocity field into "
            f"the same-named movie files instead), got .false. ({_PROFILE_DOC_REF})"
        )


# ===========================================================================
# LOCAL_PATH / nproc resolution
# ===========================================================================


def _resolve_local_path(
    specfem_dir: Path, values: dict[str, str], override: Path | None
) -> Path:
    if override is not None:
        return override
    raw = _require_par_value(values, "LOCAL_PATH")
    raw_path = Path(raw)
    return raw_path if raw_path.is_absolute() else (specfem_dir / raw_path).resolve()


def _detect_nproc(local_path: Path) -> int:
    ranks = sorted(
        int(m.group(1))
        for p in local_path.glob("proc*_Database")
        if (m := _DATABASE_FILENAME_RE.match(p.name)) is not None
    )
    if not ranks:
        raise ValueError(f"{local_path}: no proc??????_Database files found")
    if ranks != list(range(len(ranks))):
        raise ValueError(
            f"{local_path}: proc*_Database ranks are not a contiguous "
            f"0..N-1 range: {ranks}"
        )
    return len(ranks)


# ===========================================================================
# Multi-rank read + element_id-ascending canonical reassembly
# ===========================================================================


@dataclass(frozen=True)
class _AllRanksData:
    element_id: np.ndarray
    corners8: np.ndarray
    coordinates: np.ndarray
    material_id: np.ndarray
    domain_id: np.ndarray
    partition_id: np.ndarray
    density: np.ndarray
    endianness: str
    record_marker_bytes: int
    integer_kind: int
    #: Per-rank (rank-order, NOT element_id-ascending) recovered element_id
    #: arrays -- i.e. recover_element_ids(part, rank) for rank in
    #: range(nproc), kept around so the time_sampled per-rank displacement
    #: read (_read_displacement_time_sampled) can scatter each rank's
    #: locally-ordered snapshot data into the canonical element_id-ascending
    #: array using the exact same index permutation this mesh read already
    #: computed, without re-parsing part_array.vtk.
    element_id_by_rank: tuple[np.ndarray, ...]


def _to_domain_id(value: float) -> int:
    as_int = int(round(value))
    if abs(value - as_int) > 1e-9 or as_int not in schema.VALID_DOMAIN_IDS:
        raise ValueError(
            f"Database material domain_id value {value} does not match any "
            f"of {sorted(schema.VALID_DOMAIN_IDS)}"
        )
    return as_int


def _read_all_ranks(local_path: Path, nproc: int, ngll: int) -> _AllRanksData:
    part = specfem_bin.read_part_array_vtk(local_path / "part_array.vtk")

    element_id_chunks: list[np.ndarray] = []
    corners8_chunks: list[np.ndarray] = []
    coords_chunks: list[np.ndarray] = []
    material_id_chunks: list[np.ndarray] = []
    domain_id_chunks: list[np.ndarray] = []
    partition_id_chunks: list[np.ndarray] = []
    density_chunks: list[np.ndarray] = []
    endianness: str | None = None
    record_marker_bytes: int | None = None
    integer_kind: int | None = None

    for rank in range(nproc):
        element_ids_rank = specfem_bin.recover_element_ids(part, rank)
        n_local = len(element_ids_rank)

        db = specfem_bin.parse_database(local_path / f"proc{rank:06d}_Database")
        if db.nspec_local != n_local:
            raise ValueError(
                f"rank {rank}: Database nspec_local={db.nspec_local} != "
                f"part_array.vtk element count {n_local}"
            )

        node_xyz_by_loc = np.empty((db.nnodes_loc, 3), dtype=np.float64)
        for n in range(db.nnodes_loc):
            node_xyz_by_loc[int(db.node_iloc[n]) - 1] = db.node_xyz[n]

        corners8_rank = np.empty((n_local, 8, 3), dtype=np.float64)
        material_id_rank = np.empty(n_local, dtype=np.int32)
        domain_id_rank = np.empty(n_local, dtype=np.int32)
        for k, elem in enumerate(db.elements):
            corners8_rank[k] = node_xyz_by_loc[[nid - 1 for nid in elem.node_loc_ids]]
            material_id_rank[k] = elem.mat1
            domain_id_rank[k] = _to_domain_id(db.materials[elem.mat1 - 1].domain_id)

        ibool, layout_ibool, int_kind_ibool = specfem_bin.read_ibool(
            local_path / f"proc{rank:06d}_ibool.bin", ngll
        )
        if ibool.shape[0] != n_local:
            raise ValueError(
                f"rank {rank}: ibool element count {ibool.shape[0]} != "
                f"part_array.vtk element count {n_local}"
            )
        nglob = int(ibool.max()) + 1
        x_vals, _ = specfem_bin.read_coords(local_path / f"proc{rank:06d}_x.bin", nglob)
        y_vals, _ = specfem_bin.read_coords(local_path / f"proc{rank:06d}_y.bin", nglob)
        z_vals, _ = specfem_bin.read_coords(local_path / f"proc{rank:06d}_z.bin", nglob)
        xyz = np.stack([x_vals, y_vals, z_vals], axis=-1)
        coords_rank = specfem_bin.gather_element_coordinates(ibool, xyz)

        rho_rank, _ = specfem_bin.read_rho(
            local_path / f"proc{rank:06d}_rho.bin", n_local, ngll
        )

        if endianness is None:
            endianness = layout_ibool.endianness
            record_marker_bytes = layout_ibool.record_marker_bytes
            integer_kind = int_kind_ibool
        elif (endianness, record_marker_bytes) != (
            layout_ibool.endianness,
            layout_ibool.record_marker_bytes,
        ):
            raise ValueError(
                f"rank {rank}: record layout ({layout_ibool.endianness}, "
                f"{layout_ibool.record_marker_bytes}) disagrees with rank "
                f"0's ({endianness}, {record_marker_bytes}) -- inconsistent "
                "file set"
            )

        element_id_chunks.append(element_ids_rank)
        corners8_chunks.append(corners8_rank)
        coords_chunks.append(coords_rank)
        material_id_chunks.append(material_id_rank)
        domain_id_chunks.append(domain_id_rank)
        partition_id_chunks.append(np.full(n_local, rank, dtype=np.int32))
        density_chunks.append(rho_rank)

    element_id_all = np.concatenate(element_id_chunks)
    n_elements_total = len(element_id_all)
    if set(element_id_all.tolist()) != set(range(n_elements_total)):
        raise ValueError(
            "recovered element_id values are not exactly "
            f"{{0, ..., {n_elements_total - 1}}} -- partition/element "
            "recovery is incomplete or overlapping across ranks"
        )
    order = np.argsort(element_id_all)

    def _ordered(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks, axis=0)[order]

    assert endianness is not None and record_marker_bytes is not None
    assert integer_kind is not None
    return _AllRanksData(
        element_id=element_id_all[order],
        corners8=_ordered(corners8_chunks),
        coordinates=_ordered(coords_chunks),
        material_id=_ordered(material_id_chunks),
        domain_id=_ordered(domain_id_chunks),
        partition_id=_ordered(partition_id_chunks),
        density=_ordered(density_chunks),
        endianness=endianness,
        record_marker_bytes=record_marker_bytes,
        integer_kind=integer_kind,
        element_id_by_rank=tuple(element_id_chunks),
    )


# ===========================================================================
# Anchor <-> GLL bridging self-check
# ===========================================================================

#: float32 relative resolution budget for the anchor(float64, Database)
#: vs. corner-GLL-point(float32-storage-roundtrip) comparison. A factor of 4
#: over the bare float32 ULP (2**-23) leaves headroom for the trilinear
#: evaluation's rounding and the 3-axis combination of independent
#: per-component roundoffs (pilot measurement: 3.5e-3 m observed vs. a
#: 7.8e-3 m budget at the pilot's coordinate scale).
_ANCHOR_GLL_ATOL_REL: Final[float] = 4.0 * 2.0**-23


def _usual_hex_nodes_flat_indices(ngll_shape: tuple[int, int, int]) -> list[int]:
    """Flat GLL indices of the 8 corner nodes, in SPECFEM's usual_hex_nodes
    order (the same order Database element records list their 8 corner node
    ids in -- see ``specfem_bin.DatabaseElement``'s docstring)."""
    nx, ny, nz = ngll_shape
    combos = (
        (0, 0, 0),
        (nx - 1, 0, 0),
        (nx - 1, ny - 1, 0),
        (0, ny - 1, 0),
        (0, 0, nz - 1),
        (nx - 1, 0, nz - 1),
        (nx - 1, ny - 1, nz - 1),
        (0, ny - 1, nz - 1),
    )
    return [schema.flatten_index(i, j, k, ngll_shape) for (i, j, k) in combos]


def _check_anchor_gll_bridge(
    corners8: np.ndarray, coordinates: np.ndarray, ngll_shape: tuple[int, int, int]
) -> None:
    """Verify that the Database anchor nodes and the ibool-derived corner GLL
    points describe the same physical corners, in the same order, for every
    element -- the one real-data check that the element_id (Database-
    derived) and the GLL field array (ibool-derived) orderings agree."""
    corner_idx = _usual_hex_nodes_flat_indices(ngll_shape)
    corner_gll = coordinates[:, corner_idx, :]
    max_abs = float(np.max(np.abs(coordinates))) if coordinates.size else 0.0
    atol = _ANCHOR_GLL_ATOL_REL * max(max_abs, 1.0)
    dist = np.linalg.norm(corner_gll - corners8, axis=-1)
    bad = np.argwhere(dist > atol)
    if bad.size:
        e, c = (int(v) for v in bad[0])
        raise ValueError(
            f"element {e}: Database anchor node vs. ibool-derived corner "
            f"GLL point {c} distance {dist[e, c]:.6e} m exceeds the float32 "
            f"storage-roundtrip budget atol={atol:.6e} m -- possible "
            "element_id/GLL-array ordering mismatch between the Database "
            "and ibool node numbering systems"
        )


# ===========================================================================
# time_sampled displacement snapshot reading (per-rank primary / whole-mesh
# fallback -- see specfem_bin.py's section-3 module comment for the two
# real on-disk layouts this auto-detects between)
# ===========================================================================


def _read_displacement_time_sampled(
    local_path: Path,
    nproc: int,
    ngll: int,
    n_elements_total: int,
    element_id_by_rank: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, list[int], str]:
    """Read the time_sampled movie volume displacement snapshots, auto-
    detecting which of the two real on-disk layouts
    (``specfem_bin.detect_displacement_layout``) is present under
    ``local_path``, and return ``(displacement, its, displacement_layout)``
    with ``displacement`` shape ``(ntime, n_elements_total, ngll, 3)`` in
    element_id-ascending canonical order (matching ``/mesh/*``'s order).

    Real pilot data (2026-08-03, S1 Step 8) confirmed the **per-rank**
    layout (``proc{rank:06d}_displ_{X,Y,Z}_it######.bin``) is what the
    pilot's SPECFEM3D CPU build actually writes; the **whole-mesh** layout
    (``displ_{X,Y,Z}_it######.bin``) remains a real, audit-recorded
    fallback for solver builds/configurations that write that way
    instead. ``displacement_layout`` (the auto-detection result) is
    recorded into provenance as a supplementary key (not schema-required)
    so a reader can tell which real-world layout an export was produced
    from.

    For the per-rank layout, each rank's local snapshot array is scattered
    into the canonical array via that rank's ``element_id`` array (the same
    one ``_read_all_ranks`` already recovered via
    ``specfem_bin.recover_element_ids`` for the mesh read -- reused here,
    not recomputed) -- this is valid because the rank-local element order
    in the movie volume binary is the same order as this rank's
    ``proc*_Database`` element block (``specfem_bin.py``'s section-3 module
    comment).

    Raises ``ValueError`` (via ``specfem_bin.read_rank_displacement_
    snapshots``) if any rank's displacement files are entirely missing, and
    directly here if the ranks' recovered ``it`` sets disagree with each
    other (an inconsistency across ranks that a single rank's own read
    cannot detect on its own).
    """
    layout_kind = specfem_bin.detect_displacement_layout(local_path)

    if layout_kind == "whole_mesh":
        displacement, its, _layout = specfem_bin.read_displacement_snapshots(
            local_path, nspec=n_elements_total, ngll=ngll
        )
        return displacement, its, layout_kind

    # per_rank: read each rank's local snapshots and scatter them into the
    # element_id-ascending canonical array using the SAME per-rank
    # element_id array the mesh read already recovered.
    its_ref: list[int] | None = None
    displacement: np.ndarray | None = None
    for rank in range(nproc):
        element_ids_rank = element_id_by_rank[rank]
        n_local = len(element_ids_rank)
        rank_displacement, its_rank, _layout = (
            specfem_bin.read_rank_displacement_snapshots(
                local_path, rank, nspec_rank=n_local, ngll=ngll
            )
        )
        if its_ref is None:
            its_ref = its_rank
            displacement = np.empty(
                (len(its_ref), n_elements_total, ngll, 3), dtype=np.float64
            )
        elif its_rank != its_ref:
            raise ValueError(
                f"rank {rank}: displacement snapshot 'it' set {its_rank} "
                f"disagrees with rank 0's {its_ref} -- inconsistent movie "
                "volume output across ranks"
            )
        assert displacement is not None
        displacement[:, element_ids_rank, :, :] = rank_displacement

    assert displacement is not None and its_ref is not None
    return displacement, its_ref, layout_kind


# ===========================================================================
# t0 / f_resolved_estimate / cadence resolution
# ===========================================================================


def _resolve_f_resolved_estimate(config: ExportConfig) -> float:
    if config.f_resolved_estimate is not None:
        return float(config.f_resolved_estimate)
    candidate = config.specfem_dir / config.output_generate_databases_relpath
    if candidate.is_file():
        return provenance.parse_f_resolved_estimate(candidate)
    raise ValueError(
        "f_resolved_estimate could not be determined: ExportConfig."
        f"f_resolved_estimate is None and {candidate} does not exist -- "
        "supply it explicitly or ensure "
        "OUTPUT_FILES/output_generate_databases.txt is present"
    )


def _resolve_t0(config: ExportConfig) -> tuple[float, str]:
    if config.t0 is not None:
        if config.t0_source is None:
            raise ValueError("ExportConfig.t0 was given without t0_source")
        return float(config.t0), config.t0_source
    candidate = config.specfem_dir / _OUTPUT_SOLVER_TXT_RELPATH
    if candidate.is_file():
        return provenance.parse_t0(candidate)
    raise ValueError(
        "t0 could not be determined for output_sampling_kind="
        "'time_sampled': ExportConfig.t0/t0_source were not given and "
        f"{candidate} does not exist"
    )


def _resolve_cadence(config: ExportConfig, values: dict[str, str]) -> dict[str, object]:
    if config.cadence is None:
        raise ValueError(
            "ExportConfig.cadence must be supplied for output_sampling_kind="
            "'time_sampled' -- f_max_analysis/f_min_analysis/minimum_cycles/"
            "sampling_oversampling_factor/antialias_filter_applied are "
            "scientific inputs this exporter cannot fabricate and refuses "
            "to guess"
        )
    m = int(
        provenance.par_file_value_as_number(
            _require_par_value(values, "NTSTEP_BETWEEN_FRAMES")
        )
    )
    # output_interval_steps is mechanically known from Par_file -- the
    # config-supplied value (if any) is overridden, not merely defaulted.
    return {**config.cadence, "output_interval_steps": m}


# ===========================================================================
# Public entry point
# ===========================================================================


def export_specfem_gll_hdf5(config: ExportConfig) -> Path:
    """Export one canonical ``mesh_type="gll"`` HDF5 file from a SPECFEM3D
    Cartesian decompose_mesh output tree, per ``config``.

    See the module docstring for the 6-step pipeline. Raises ``ValueError``
    for any certified-profile violation, any structural inconsistency
    between the mesh/binary files (mismatched element counts, disagreeing
    record layouts across ranks, non-contiguous recovered ``element_id``
    values, ...), or a missing required provenance input the config could
    not auto-resolve.
    """
    specfem_dir = config.specfem_dir
    par_file_path = specfem_dir / config.par_file_relpath
    if not par_file_path.is_file():
        raise ValueError(f"Par_file not found: {par_file_path}")
    par_file = provenance.parse_par_file(par_file_path)
    values = par_file["values"]

    _check_profile_gate(values, config.output_sampling_kind)

    local_path = _resolve_local_path(specfem_dir, values, config.local_path)
    nproc = _detect_nproc(local_path)
    ngll = schema.ngll_from_shape(config.ngll_shape)

    all_ranks = _read_all_ranks(local_path, nproc, ngll)
    n_elements_total = len(all_ranks.element_id)

    _check_anchor_gll_bridge(
        all_ranks.corners8, all_ranks.coordinates, config.ngll_shape
    )

    # Snap the 8 corner GLL points of /mesh/coordinates to the Database
    # anchor nodes' exact float64 values, now that the bridging check above
    # has confirmed they describe the same physical corner within the
    # float32 storage-roundtrip budget. Without this, the corner entries
    # read from ibool/x/y/z.bin carry only float32 precision while
    # quadrature_volume (below) is computed from the float64 anchors --  a
    # genuine solver-side precision asymmetry (Database node coordinates are
    # double precision; the GLL DOF arrays are float32) that would otherwise
    # make Tier A2's rtol<1e-12 self-consistency gate structurally
    # unsatisfiable (the residual would be a float32 storage artifact, not
    # a computation error). Only the 8 corners are replaced; the interior
    # GLL points have no higher-precision source and keep their
    # ibool/x/y/z.bin-derived values.
    corner_flat_idx = _usual_hex_nodes_flat_indices(config.ngll_shape)
    coordinates = all_ranks.coordinates.copy()
    coordinates[:, corner_flat_idx, :] = all_ranks.corners8

    quadrature_volume = gll.elements_quadrature_volume(
        all_ranks.corners8, config.ngll_shape
    )

    # MVP profile: no CPML support in this exporter version (the Database's
    # boundary/CPML tail records are deliberately not parsed -- see
    # specfem_bin.parse_database's docstring), so is_pml is always all-False.
    is_pml = np.zeros(n_elements_total, dtype=bool)
    pml_hash = provenance.pml_mask_hash(is_pml)

    if config.mesh_input_files:
        mesh_input_manifest, mesh_input_hash = provenance.compute_mesh_input_hash(
            specfem_dir,
            config.mesh_input_files,
            par_file_relpath=config.par_file_relpath,
        )
    else:
        mesh_input_manifest, mesh_input_hash = provenance.build_file_manifest(
            specfem_dir, [config.par_file_relpath]
        )

    build_identity = (
        dict(config.build_identity)
        if config.build_identity is not None
        else provenance.collect_build_identity(specfem_dir)
    )
    executables = [dict(e) for e in config.runtime_manifest_entries]
    executables.append(
        provenance.collect_exporter_identity(exporter_code_ref=config.exporter_code_ref)
    )
    runtime_manifest = provenance.build_runtime_manifest(executables)

    suppress_utm = provenance.par_file_value_as_bool(
        _require_par_value(values, "SUPPRESS_UTM_PROJECTION")
    )
    f_resolved_estimate = _resolve_f_resolved_estimate(config)

    scalars: dict[str, object] = {
        "solver_kind": config.solver_kind,
        "solver_version": config.solver_version,
        "nproc": nproc,
        "float_precision": config.float_precision,
        "density_source": config.density_source,
        "element_id_method": schema.ADOPTED_ELEMENT_ID_METHOD,
        "f_resolved_estimate": f_resolved_estimate,
        "field_units": dict(config.field_units),
        "suppress_utm_projection": suppress_utm,
        "endianness": all_ranks.endianness,
        "record_marker_bytes": all_ranks.record_marker_bytes,
        "integer_kind": all_ranks.integer_kind,
        "material_id_source": config.material_id_source,
        "domain_id_source": config.domain_id_source,
        "is_pml_source": config.is_pml_source,
    }

    displacement: np.ndarray | None = None
    time_step: np.ndarray | None = None
    t0: float | None = None
    t0_source: str | None = None
    cadence: dict[str, object] | None = None

    if config.output_sampling_kind == "time_sampled":
        dt_s = float(
            provenance.par_file_value_as_number(_require_par_value(values, "DT"))
        )
        nstep_par = int(
            provenance.par_file_value_as_number(_require_par_value(values, "NSTEP"))
        )
        scalars["dt_s"] = dt_s
        scalars["nstep"] = nstep_par
        t0, t0_source = _resolve_t0(config)
        cadence = _resolve_cadence(config, values)

        # Real-pilot-verified layout handling (2026-08-03, S1 Step 8): the
        # movie volume snapshot file set is per-rank by default (real pilot
        # data), with the whole-mesh layout audit-recorded as a fallback --
        # see _read_displacement_time_sampled's docstring and
        # specfem_bin.py's section-3 module comment for the two layouts.
        displacement, its, displacement_layout = _read_displacement_time_sampled(
            local_path, nproc, ngll, n_elements_total, all_ranks.element_id_by_rank
        )
        time_step = np.asarray(its, dtype=np.int64) - 1  # n = it - 1
        scalars["displacement_layout"] = displacement_layout

    prov = provenance.assemble_provenance(
        output_sampling_kind=config.output_sampling_kind,
        par_file=par_file,
        mesh_input_hash=mesh_input_hash,
        mesh_input_manifest=mesh_input_manifest,
        build_identity=build_identity,
        runtime_manifest=runtime_manifest,
        pml_mask_hash=pml_hash,
        t0=t0,
        t0_source=t0_source,
        t0_crosscheck=config.t0_crosscheck,
        t0_crosscheck_max_deviation=config.t0_crosscheck_max_deviation,
        time_step_origin=(
            "zero_based_normalized"
            if config.output_sampling_kind == "time_sampled"
            else None
        ),
        native_step_origin=(
            1 if config.output_sampling_kind == "time_sampled" else None
        ),
        cadence=cadence,
        cadence_provenance=config.cadence_provenance,
        **scalars,
    )

    return _writer.write_gll_hdf5(
        config.output_path,
        output_sampling_kind=config.output_sampling_kind,
        ngll_shape=config.ngll_shape,
        coordinates=coordinates,
        quadrature_volume=quadrature_volume,
        element_id=all_ranks.element_id,
        partition_id=all_ranks.partition_id,
        domain_id=all_ranks.domain_id,
        material_id=all_ranks.material_id,
        is_pml=is_pml,
        density=all_ranks.density,
        provenance=prov,
        displacement=displacement,
        time_step=time_step,
        chunk_element_size=config.chunk_element_size,
        gzip_level=config.gzip_level,
    )
