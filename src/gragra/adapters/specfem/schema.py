"""Canonical schema definition for the SPECFEM3D Cartesian GLL certified export.

This module is the **single source of truth** for the ``mesh_type="gll"``
canonical HDF5 schema ("schema (a)") defined in gragra's solver-import
contract, and for the SPECFEM3D-specific provenance/value conventions that
accompany it. It intentionally duplicates *nothing* that those documents
define -- every constant/table below carries a ``contract_ref`` (or an
inline comment) pointing at the section read while writing this module. If
the contract is revised, this module's refs may drift; treat the *documents*
as authoritative and this module's refs as a pointer, not a copy.

Design goal: both ``writer.py`` (which raises ``ValueError`` on missing
provenance) and the independent Tier A4 provenance audit in the test suite
must consume the *same* required-key table -- so a key added/removed here
automatically changes both the writer's validation and the audit, instead of
the two drifting apart.

This module is **numpy-only**. It must never import ``h5py`` (or any other
optional dependency) -- see ``gragra/adapters/hdf5_field.py`` for the
lazy-import convention used by modules that actually touch HDF5 files.
"""

import json
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

# ===========================================================================
# 1. Schema identity (/metadata attributes)
#    ref: solver_import_contract.md L281-299 (attribute table + version<->
#    commit correspondence table).
# ===========================================================================

#: Canonical schema version for schema (a) (GLL). Monotonically increasing;
#: bumped only on a breaking change to schema (a) (new dataset, type change,
#: semantic change). The version<->contract-commit table lives in
#: solver_import_contract.md (this module does not duplicate it).
SCHEMA_VERSION: Final[int] = 1

#: `/metadata` attribute identifying this schema variant (dispatch key for
#: readers). ref: solver_import_contract.md L288.
MESH_TYPE: Final[str] = "gll"

#: `/metadata` attribute; v1 always uses a single HDF5 file (sharding is a
#: reserved future value). ref: solver_import_contract.md L290.
LAYOUT_SINGLE_FILE: Final[str] = "single_file"

#: The three ``output_sampling_kind`` values the common contract defines.
#: ``"frequency_domain"`` has no defined semantics yet (contract explicitly
#: declines to define it) -- ``required_keys()`` below only ever adds the
#: unconditional ("always") keys for it.
#: ref: solver_import_contract.md L1021-1023, L590-603.
OutputSamplingKind = Literal["time_sampled", "static", "frequency_domain"]
VALID_OUTPUT_SAMPLING_KINDS: Final[tuple[str, ...]] = (
    "time_sampled",
    "static",
    "frequency_domain",
)

# ===========================================================================
# 2. domain_id / material_id value conventions
#    ref: solver_import_contract.md L364-372.
# ===========================================================================

#: SPECFEM3D `IDOMAIN_ACOUSTIC`/`IDOMAIN_ELASTIC`/`IDOMAIN_POROELASTIC` --
#: also serves double duty as the "expected value" for the upstream
#: equivalence test (test_specfem_schema.py), since the schema's domain_id
#: enum *is* upstream's IDOMAIN_* by definition (not merely a coincidence to
#: cross-check).
IDOMAIN_ACOUSTIC: Final[int] = 1
IDOMAIN_ELASTIC: Final[int] = 2
IDOMAIN_POROELASTIC: Final[int] = 3
VALID_DOMAIN_IDS: Final[frozenset[int]] = frozenset(
    {IDOMAIN_ACOUSTIC, IDOMAIN_ELASTIC, IDOMAIN_POROELASTIC}
)

# ===========================================================================
# 3. element_id construction method
#    ref: solver_import_contract.md L341-345 (three allowed methods);
#    specfem_import.md, element_id-construction contract (adopted method).
#
#    Method-scoping (solver_import_contract.md L355-363): element_id
#    numbering is NOT invariant across ELEMENT_ID_METHODS -- for the same
#    mesh, "pre_partition_global_index" and "corner_node_tuple" agree on
#    only 4 of 20,736 elements (measured). Golden invariants that depend on
#    element_id ordering/values (element_id_min/_max, pml_mask_hash, the
#    ascending-order total_* scalars, ...) are therefore meaningful only
#    *within* a single element_id_method; they cannot be compared across a
#    method change. See solver_import_contract.md L355-363 for the full
#    rationale.
# ===========================================================================

ELEMENT_ID_METHOD_PRE_PARTITION_GLOBAL_INDEX: Final[str] = "pre_partition_global_index"
ELEMENT_ID_METHOD_EXPLICIT_GLOB2LOC_MAP: Final[str] = "explicit_glob2loc_map"
ELEMENT_ID_METHOD_CORNER_NODE_TUPLE: Final[str] = "corner_node_tuple"

ELEMENT_ID_METHODS: Final[tuple[str, ...]] = (
    ELEMENT_ID_METHOD_PRE_PARTITION_GLOBAL_INDEX,
    ELEMENT_ID_METHOD_EXPLICIT_GLOB2LOC_MAP,
    ELEMENT_ID_METHOD_CORNER_NODE_TUPLE,
)

#: The method the S1 certified profile (external mesh + xdecompose_mesh) has
#: frozen on, per specfem_import.md's 2026-08-03 real-data investigation
#: (Step 0.5): partition round-trip verified for NPROC=3/4/8 on the pilot
#: mesh. ``"corner_node_tuple"`` remains a defined (but unused) fallback.
ADOPTED_ELEMENT_ID_METHOD: Final[str] = ELEMENT_ID_METHOD_PRE_PARTITION_GLOBAL_INDEX

# ===========================================================================
# 4. Upstream constant expectations (cross-checked against the real
#    SPECFEM3D source tree in test_specfem_schema.py when
#    GRAGRA_SPECFEM_SOURCE is set).
#    ref: specfem_import.md, implementation-facts section, and the pilot
#    source tree's setup/constants.h (NGLLX=5 at L62-64, GAUSSALPHA/GAUSSBETA=0 at
#    L736, USE_MESH_COLORING_GPU=.false. at L385, IDOMAIN_* at L863-865, in
#    the pilot commit cc2e9ffa layout).
# ===========================================================================

#: (NGLLX, NGLLY, NGLLZ) expected for the certified profile.
EXPECTED_NGLL: Final[tuple[int, int, int]] = (5, 5, 5)
EXPECTED_GAUSSALPHA: Final[float] = 0.0
EXPECTED_GAUSSBETA: Final[float] = 0.0
EXPECTED_USE_MESH_COLORING_GPU: Final[bool] = False
# EXPECTED_IDOMAIN_* intentionally omitted: IDOMAIN_ACOUSTIC/ELASTIC/
# POROELASTIC above already *are* the expected values (the schema's domain_id
# enum is defined to equal upstream's IDOMAIN_* constants) -- see §2.

# ===========================================================================
# 5. Dataset path constants + dtype table (schema (a))
#    ref: solver_import_contract.md L263-279 (dataset block).
# ===========================================================================

PATH_MESH_COORDINATES: Final[str] = "/mesh/coordinates"
PATH_MESH_QUADRATURE_VOLUME: Final[str] = "/mesh/quadrature_volume"
PATH_MESH_ELEMENT_ID: Final[str] = "/mesh/element_id"
PATH_MESH_PARTITION_ID: Final[str] = "/mesh/partition_id"
PATH_MESH_DOMAIN_ID: Final[str] = "/mesh/domain_id"
PATH_MESH_MATERIAL_ID: Final[str] = "/mesh/material_id"
PATH_MESH_IS_PML: Final[str] = "/mesh/is_pml"
PATH_MESH_ELEMENT_CORNER_NODES: Final[str] = "/mesh/element_corner_nodes"
PATH_MATERIAL_DENSITY: Final[str] = "/material/density"
PATH_FIELD_DISPLACEMENT: Final[str] = "/field/displacement"
PATH_TIME: Final[str] = "/time"
PATH_TIME_STEP: Final[str] = "/time_step"
PATH_DIAGNOSTICS_DIVERGENCE: Final[str] = "/diagnostics/divergence"
PATH_DIAGNOSTICS_CURL: Final[str] = "/diagnostics/curl"
METADATA_GROUP: Final[str] = "/metadata"


@dataclass(frozen=True)
class DatasetSpec:
    """One row of the schema (a) dataset dtype table.

    Attributes
    ----------
    path : str
        Absolute HDF5 path (one of the ``PATH_*`` constants above).
    dtype : np.dtype
        On-disk dtype (in the array-conventions.md sense -- see the module
        docstring of ``array_types.py``: coordinates/physical quantities are
        float64, integer labels are int64/int32, masks are bool).
    shape : tuple[str, ...]
        Symbolic dimension names, outermost first (e.g. ``("element",
        "gll", "3")``). ``"3"`` denotes the fixed component axis.
    unit : str | None
        Physical unit, or ``None`` for dimensionless/label datasets.
    output_modes : frozenset[str]
        Which ``output_sampling_kind`` value(s) this dataset is present
        under. Static (mesh-only) output omits ``/field/displacement``,
        ``/time``, ``/time_step`` and the ``/diagnostics/*`` datasets.
    required : bool
        Whether the dataset is mandatory *given* that its output mode
        applies (a dataset with ``output_modes == {"time_sampled"}`` is
        simply absent -- not "required but missing" -- when the output is
        static). ``/mesh/element_corner_nodes`` is the one exception: it is
        conditionally required only when
        ``element_id_method == "corner_node_tuple"`` (see
        ``element_corner_nodes_required()`` below); it is marked
        ``required=False`` here and the condition is documented in notes.
    notes : str
        Free-text notes, including the contract line reference.
    """

    path: str
    dtype: np.dtype
    shape: tuple[str, ...]
    unit: str | None
    output_modes: frozenset[str]
    required: bool
    notes: str


_STATIC_AND_TIME_SAMPLED: Final[frozenset[str]] = frozenset({"static", "time_sampled"})
_TIME_SAMPLED_ONLY: Final[frozenset[str]] = frozenset({"time_sampled"})

DATASET_SPECS: Final[dict[str, DatasetSpec]] = {
    PATH_MESH_COORDINATES: DatasetSpec(
        path=PATH_MESH_COORDINATES,
        dtype=np.dtype(np.float64),
        shape=("element", "gll", "3"),
        unit="m",
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=True,
        notes=(
            "Absolute coordinates (not recentred -- recentring is a "
            "verification-time-only step, see hex_volume oracle and A1/A2). "
            "ref: solver_import_contract.md L264."
        ),
    ),
    PATH_MESH_QUADRATURE_VOLUME: DatasetSpec(
        path=PATH_MESH_QUADRATURE_VOLUME,
        dtype=np.dtype(np.float64),
        shape=("element", "gll"),
        unit="m3",
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=True,
        notes="J*w_x*w_y*w_z per GLL point. ref: solver_import_contract.md L265.",
    ),
    PATH_MESH_ELEMENT_ID: DatasetSpec(
        path=PATH_MESH_ELEMENT_ID,
        dtype=np.dtype(np.int64),
        shape=("element",),
        unit=None,
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=True,
        notes=(
            "Zero-based, partition-independent global element id. Must be "
            "unique across all elements. ref: solver_import_contract.md "
            "L266, L320-345."
        ),
    ),
    PATH_MESH_PARTITION_ID: DatasetSpec(
        path=PATH_MESH_PARTITION_ID,
        dtype=np.dtype(np.int32),
        shape=("element",),
        unit=None,
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=True,
        notes="MPI rank; all-0 if NPROC=1. ref: solver_import_contract.md L267.",
    ),
    PATH_MESH_DOMAIN_ID: DatasetSpec(
        path=PATH_MESH_DOMAIN_ID,
        dtype=np.dtype(np.int32),
        shape=("element",),
        unit=None,
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=True,
        notes=(
            "Values restricted to VALID_DOMAIN_IDS (IDOMAIN_ACOUSTIC/"
            "ELASTIC/POROELASTIC). ref: solver_import_contract.md L268, "
            "L364-368."
        ),
    ),
    PATH_MESH_MATERIAL_ID: DatasetSpec(
        path=PATH_MESH_MATERIAL_ID,
        dtype=np.dtype(np.int32),
        shape=("element",),
        unit=None,
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=True,
        notes=(
            "One-based material label (SPECFEM3D convention; values < 1 "
            "are rejected for the certified profile). ref: "
            "solver_import_contract.md L269, L368-372."
        ),
    ),
    PATH_MESH_IS_PML: DatasetSpec(
        path=PATH_MESH_IS_PML,
        dtype=np.dtype(np.bool_),
        shape=("element",),
        unit=None,
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=True,
        notes=(
            "True = absorbing-layer element (excluded from mass source). "
            "Written as all-False (not omitted) when the profile has no "
            "PML. ref: solver_import_contract.md L270, L373-375."
        ),
    ),
    PATH_MESH_ELEMENT_CORNER_NODES: DatasetSpec(
        path=PATH_MESH_ELEMENT_CORNER_NODES,
        dtype=np.dtype(np.int64),
        shape=("element", "n_corner"),
        unit=None,
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=False,
        notes=(
            "Required only when element_id_method == "
            "'corner_node_tuple' (see element_corner_nodes_required()); "
            "optional for all other methods. The S1 certified profile uses "
            "ADOPTED_ELEMENT_ID_METHOD (pre_partition_global_index), so "
            "this dataset is not written by S1. ref: "
            "solver_import_contract.md L271, L346-356."
        ),
    ),
    PATH_MATERIAL_DENSITY: DatasetSpec(
        path=PATH_MATERIAL_DENSITY,
        dtype=np.dtype(np.float64),
        shape=("element", "gll"),
        unit="kg/m3",
        output_modes=_STATIC_AND_TIME_SAMPLED,
        required=True,
        notes="Background density rho0 per GLL point. ref: L272.",
    ),
    PATH_FIELD_DISPLACEMENT: DatasetSpec(
        path=PATH_FIELD_DISPLACEMENT,
        dtype=np.dtype(np.float64),
        shape=("time", "element", "gll", "3"),
        unit="m",
        output_modes=_TIME_SAMPLED_ONLY,
        required=True,
        notes=(
            "Only present for time_sampled output (mesh-only static "
            "output has no /field group). ref: L273; specfem_import.md, "
            "mesh-only static output section."
        ),
    ),
    PATH_TIME: DatasetSpec(
        path=PATH_TIME,
        dtype=np.dtype(np.float64),
        shape=("time",),
        unit="s",
        output_modes=_TIME_SAMPLED_ONLY,
        required=True,
        notes="Coordinate metadata; core does not read it. ref: L274.",
    ),
    PATH_TIME_STEP: DatasetSpec(
        path=PATH_TIME_STEP,
        dtype=np.dtype(np.int64),
        shape=("time",),
        unit=None,
        output_modes=_TIME_SAMPLED_ONLY,
        required=True,
        notes=(
            "Zero-based normalized step index n = it - 1 (SPECFEM native "
            "step is 1-based). ref: L275; time-axis dataset consistency "
            "rules, L765-777."
        ),
    ),
    PATH_DIAGNOSTICS_DIVERGENCE: DatasetSpec(
        path=PATH_DIAGNOSTICS_DIVERGENCE,
        dtype=np.dtype(np.float64),
        shape=("time", "element", "gll"),
        unit=None,
        output_modes=_TIME_SAMPLED_ONLY,
        required=False,
        notes="Diagnostic-only; never used for certified integration. ref: L276.",
    ),
    PATH_DIAGNOSTICS_CURL: DatasetSpec(
        path=PATH_DIAGNOSTICS_CURL,
        dtype=np.dtype(np.float64),
        shape=("time", "element", "gll", "3"),
        unit=None,
        output_modes=_TIME_SAMPLED_ONLY,
        required=False,
        notes="Diagnostic-only; never used for certified integration. ref: L277.",
    ),
}


def element_corner_nodes_required(element_id_method: str) -> bool:
    """Whether ``/mesh/element_corner_nodes`` must be present for a given
    ``element_id_method`` (ref: solver_import_contract.md L271, L346-356)."""
    return element_id_method == ELEMENT_ID_METHOD_CORNER_NODE_TUPLE


def required_dataset_paths(output_sampling_kind: str) -> frozenset[str]:
    """Return the dataset paths that must exist for a given
    ``output_sampling_kind`` (excluding the conditional
    ``/mesh/element_corner_nodes`` -- see ``element_corner_nodes_required``).
    """
    _validate_output_sampling_kind(output_sampling_kind)
    return frozenset(
        spec.path
        for spec in DATASET_SPECS.values()
        if spec.required
        and output_sampling_kind in spec.output_modes
        and spec.path != PATH_MESH_ELEMENT_CORNER_NODES
    )


# ===========================================================================
# 6. GLL axis flatten convention
#    ref: solver_import_contract.md L387-395 ("(k, j, i) = (z, y, x)"
#    C-order, x innermost).
# ===========================================================================


def _validate_ngll_shape(ngll_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(ngll_shape) != 3:
        raise ValueError(f"ngll_shape must have length 3, got {ngll_shape!r}")
    nx, ny, nz = (int(v) for v in ngll_shape)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"ngll_shape entries must be positive, got {ngll_shape!r}")
    return nx, ny, nz


def ngll_from_shape(ngll_shape: tuple[int, int, int]) -> int:
    """``ngll = NGLLX * NGLLY * NGLLZ`` (contract: total must match this
    product, not merely be assumed factorizable as ``n**3``)."""
    nx, ny, nz = _validate_ngll_shape(ngll_shape)
    return nx * ny * nz


def reshape_shape_zyx(ngll_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """``(NGLLZ, NGLLY, NGLLX)`` -- the C-order reshape target for a flat
    ``(ngll,)`` (or ``(ngll, 3)``) array, per the flatten convention (x
    innermost). Reshaping ``/mesh/coordinates[e]`` (shape ``(ngll, 3)``) with
    ``reshape((*reshape_shape_zyx(ngll_shape), 3))`` recovers the per-axis
    grid used by A1.
    """
    nx, ny, nz = _validate_ngll_shape(ngll_shape)
    return (nz, ny, nx)


def flatten_index(i: int, j: int, k: int, ngll_shape: tuple[int, int, int]) -> int:
    """Map a per-axis GLL index ``(i, j, k)`` (x, y, z; each 0-based) within
    one element to the flat GLL index used by ``/mesh/coordinates`` etc.

    Convention (contract L387-395): flatten order is ``(k, j, i)`` =
    ``(z, y, x)`` in C-order, i.e. x is the innermost (fastest-varying) axis:
    ``flat = k * (ny * nx) + j * nx + i``.
    """
    nx, ny, nz = _validate_ngll_shape(ngll_shape)
    if not (0 <= i < nx):
        raise ValueError(f"i={i} out of range [0, {nx})")
    if not (0 <= j < ny):
        raise ValueError(f"j={j} out of range [0, {ny})")
    if not (0 <= k < nz):
        raise ValueError(f"k={k} out of range [0, {nz})")
    return k * (ny * nx) + j * nx + i


def unflatten_index(
    flat: int, ngll_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Inverse of ``flatten_index``: returns ``(i, j, k)`` for a flat index."""
    nx, ny, nz = _validate_ngll_shape(ngll_shape)
    total = nx * ny * nz
    if not (0 <= flat < total):
        raise ValueError(f"flat index {flat} out of range [0, {total})")
    k, rem = divmod(flat, ny * nx)
    j, i = divmod(rem, nx)
    return i, j, k


# ===========================================================================
# 7. Provenance key tables (3 groups, per the S1 plan Step 1)
# ===========================================================================

#: Free-text tags describing when a key becomes required *purely as a
#: function of output_sampling_kind* (consumed by ``required_keys()``).
#: ``"never_base"`` means the key's necessity depends on some *other*
#: already-collected provenance value (build_kind, antialias_filter_applied,
#: output_interval_steps, f_content_max_method, ...) rather than on
#: output_sampling_kind alone -- those cases are resolved by
#: ``conditionally_required_keys()`` instead.
BaseRequiredWhen = Literal["always", "time_sampled", "never_base"]

#: How a key's value is encoded on disk (§provenance encoding, S1 plan Step
#: 1 item 5): true scalars (and the fixed-shape ngll_shape int64[3]) are
#: `/metadata` *attributes*; structured objects are canonical-JSON *vlen str
#: datasets* under `/metadata/<key>`; full-text content (Par_file text, the
#: mesh/manifest file-listing text that mesh_input_hash/par_file_hash were
#: computed over) is stored as plain-text vlen str *datasets* rather than
#: JSON-wrapped, so that re-hashing the stored text reproduces the recorded
#: hash without a JSON round-trip in between.
ProvenanceEncoding = Literal["scalar_attr", "json_dataset", "text_dataset"]

#: Coarse value-type tag used by ``check_value_type()`` for the A4
#: existence+type audit. ``"json"`` only checks presence (deep structural
#: validation of nested keys is out of Step 1/2 scope -- see module notes).
ValueTypeName = Literal["int", "float", "str", "bool", "int_array3", "json"]


@dataclass(frozen=True)
class ProvenanceKeySpec:
    """One row of the provenance requirement table.

    Attributes
    ----------
    name : str
        Provenance key name (as written under `/metadata`).
    value_type : ValueTypeName
        Coarse type tag, checked by ``check_value_type()``.
    encoding : ProvenanceEncoding
        On-disk encoding (attribute vs. JSON dataset vs. text dataset).
    base_required_when : BaseRequiredWhen
        Whether ``required_keys(output_sampling_kind)`` includes this key.
    contract_ref : str
        Pointer into solver_import_contract.md / specfem_import.md.
    notes : str
        Free-text notes (allowed values, units, etc.).
    recommended : bool
        If True, the key is a *recommended* (not machine-enforced) key --
        excluded from ``required_keys()``/``conditionally_required_keys()``
        but still tracked here for documentation and optional auditing.
    """

    name: str
    value_type: ValueTypeName
    encoding: ProvenanceEncoding
    base_required_when: BaseRequiredWhen
    contract_ref: str
    notes: str = ""
    recommended: bool = False


# --- 7a. Common required items (all solvers) ------------------------------
# ref: solver_import_contract.md L914-1042 (required provenance, common items).
COMMON_PROVENANCE_KEYS: Final[dict[str, ProvenanceKeySpec]] = {
    "schema_version": ProvenanceKeySpec(
        "schema_version",
        "int",
        "scalar_attr",
        "always",
        "solver_import_contract.md L254,289",
        "Must equal SCHEMA_VERSION for schema (a).",
    ),
    "mesh_type": ProvenanceKeySpec(
        "mesh_type",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L288",
        "Reader dispatch key; must equal MESH_TYPE ('gll').",
    ),
    "layout": ProvenanceKeySpec(
        "layout",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L290",
        "Must equal LAYOUT_SINGLE_FILE ('single_file') for v1.",
    ),
    "solver_kind": ProvenanceKeySpec(
        "solver_kind",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L923",
        "Fixed 'specfem3d_cartesian' for this exporter.",
    ),
    "solver_version": ProvenanceKeySpec(
        "solver_version",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L923",
    ),
    "build_kind": ProvenanceKeySpec(
        "build_kind",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L929-946",
        "One of 'source' | 'package'.",
    ),
    "source_tree_clean": ProvenanceKeySpec(
        "source_tree_clean",
        "bool",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L935",
        "Required iff build_kind == 'source'.",
    ),
    "source_tree_hash": ProvenanceKeySpec(
        "source_tree_hash",
        "str",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L937-940,1005",
        (
            "One of {patch_hash, source_tree_hash} required iff build_kind "
            "== 'source' and source_tree_clean is False -- see "
            "source_tree_dirty_hash_satisfied()."
        ),
    ),
    "patch_hash": ProvenanceKeySpec(
        "patch_hash",
        "str",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L937-940,1005",
        "Alternative to source_tree_hash -- see source_tree_hash notes.",
    ),
    "build_configuration": ProvenanceKeySpec(
        "build_configuration",
        "json",
        "json_dataset",
        "always",
        "solver_import_contract.md L941,945",
        "configure options / compiler+flags / float precision, or 'unknown'.",
    ),
    "package_version": ProvenanceKeySpec(
        "package_version",
        "str",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L943",
        "Required iff build_kind == 'package'.",
    ),
    "package_digest": ProvenanceKeySpec(
        "package_digest",
        "str",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L944",
        "Required iff build_kind == 'package'.",
    ),
    "executables": ProvenanceKeySpec(
        "executables",
        "json",
        "json_dataset",
        "always",
        "solver_import_contract.md L948-956",
        (
            "runtime_binary_manifest executables[] list (role/"
            "resolved_path/sha256/version/linked_libraries)."
        ),
    ),
    "resolution_method": ProvenanceKeySpec(
        "resolution_method",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L956",
        "e.g. 'ldd'.",
    ),
    "par_file_hash": ProvenanceKeySpec(
        "par_file_hash",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1016",
    ),
    "par_file_text": ProvenanceKeySpec(
        "par_file_text",
        "str",
        "text_dataset",
        "always",
        "solver_import_contract.md L1016-1017",
        "Full text of DATA/Par_file.",
    ),
    "mesh_input_hash": ProvenanceKeySpec(
        "mesh_input_hash",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1018,L1011-1015",
    ),
    "mesh_input_manifest": ProvenanceKeySpec(
        "mesh_input_manifest",
        "str",
        "text_dataset",
        "always",
        "solver_import_contract.md L1011-1015",
        (
            "Canonical (path, sha256) file manifest text mesh_input_hash "
            "was computed over."
        ),
    ),
    "nproc": ProvenanceKeySpec(
        "nproc",
        "int",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1020",
    ),
    "output_sampling_kind": ProvenanceKeySpec(
        "output_sampling_kind",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1021-1023",
        "One of 'time_sampled' | 'static' | 'frequency_domain'.",
    ),
    "dt_s": ProvenanceKeySpec(
        "dt_s",
        "float",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L1024",
    ),
    "nstep": ProvenanceKeySpec(
        "nstep",
        "int",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L1024",
    ),
    "float_precision": ProvenanceKeySpec(
        "float_precision",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1038",
        "One of 'single' | 'double'.",
    ),
    "coordinate_convention": ProvenanceKeySpec(
        "coordinate_convention",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1040",
        "Fixed 'ENU' for SPECFEM3D Cartesian.",
    ),
    "field_units": ProvenanceKeySpec(
        "field_units",
        "json",
        "json_dataset",
        "always",
        "solver_import_contract.md L1039",
        "dict mapping field name -> unit, e.g. {'displacement': 'm'}.",
    ),
    "time_step_origin": ProvenanceKeySpec(
        "time_step_origin",
        "str",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L773",
        "Fixed 'zero_based_normalized'.",
    ),
    "native_step_origin": ProvenanceKeySpec(
        "native_step_origin",
        "int",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L773",
        "0 or 1 (SPECFEM3D uses 1).",
    ),
}

# --- 7b. SPECFEM3D-specific required items ---------------------------------
# ref: solver_import_contract.md L1044-1125 (SPECFEM3D certified-specific
# items); specfem_import.md, element_id-construction / t0-acquisition
# contract (t0/t0_source/t0_crosscheck).
SPECFEM_PROVENANCE_KEYS: Final[dict[str, ProvenanceKeySpec]] = {
    "specfem_git_commit": ProvenanceKeySpec(
        "specfem_git_commit",
        "str",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L932-934",
        "Required iff build_kind == 'source'.",
    ),
    "ngll": ProvenanceKeySpec(
        "ngll",
        "int",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1061-1062",
        "Must equal ngll_from_shape(ngll_shape).",
    ),
    "ngll_shape": ProvenanceKeySpec(
        "ngll_shape",
        "int_array3",
        "scalar_attr",
        "always",
        "solver_import_contract.md L389-395",
        "[NGLLX, NGLLY, NGLLZ] -- note this is an array-valued attribute, "
        "not a JSON dataset (S1 plan item 5).",
    ),
    "element_id_method": ProvenanceKeySpec(
        "element_id_method",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L341-345",
        "One of ELEMENT_ID_METHODS.",
    ),
    "element_id_min": ProvenanceKeySpec(
        "element_id_min",
        "int",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1076-1078",
    ),
    "element_id_max": ProvenanceKeySpec(
        "element_id_max",
        "int",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1076-1078",
    ),
    "density_source": ProvenanceKeySpec(
        "density_source",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1064-1069",
    ),
    "f_resolved_estimate": ProvenanceKeySpec(
        "f_resolved_estimate",
        "float",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1079-1081",
        "Hz; a non-sharp mesh-resolution estimate, not a hard cutoff.",
    ),
    "total_quadrature_volume_before_exclusion": ProvenanceKeySpec(
        "total_quadrature_volume_before_exclusion",
        "float",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1082-1089",
    ),
    "total_quadrature_volume_after_exclusion": ProvenanceKeySpec(
        "total_quadrature_volume_after_exclusion",
        "float",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1082-1089",
    ),
    "total_mass_before_exclusion": ProvenanceKeySpec(
        "total_mass_before_exclusion",
        "float",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1082-1089",
    ),
    "total_mass_after_exclusion": ProvenanceKeySpec(
        "total_mass_after_exclusion",
        "float",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1082-1089",
    ),
    "n_elements_total": ProvenanceKeySpec(
        "n_elements_total",
        "int",
        "scalar_attr",
        "always",
        "solver_import_contract.md L1070-1078",
    ),
    "pml_exclusion_applied": ProvenanceKeySpec(
        "pml_exclusion_applied",
        "bool",
        "scalar_attr",
        "always",
        "solver_import_contract.md L458",
        "Always True (even with an all-False is_pml mask).",
    ),
    "pml_mask_hash": ProvenanceKeySpec(
        "pml_mask_hash",
        "str",
        "scalar_attr",
        "always",
        "solver_import_contract.md L459",
        "sha256 of the element_id-ascending is_pml bool column.",
    ),
    "t0": ProvenanceKeySpec(
        "t0",
        "float",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L1049-1053",
    ),
    "t0_source": ProvenanceKeySpec(
        "t0_source",
        "str",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L1054-1060",
        "e.g. 'output_solver_txt_start_time'.",
    ),
    "t0_crosscheck": ProvenanceKeySpec(
        "t0_crosscheck",
        "str",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L1054-1060; specfem_import.md L340-346",
        "Recommended (not machine-enforced), e.g. 'semd_column1'.",
        recommended=True,
    ),
    "t0_crosscheck_max_deviation": ProvenanceKeySpec(
        "t0_crosscheck_max_deviation",
        "float",
        "scalar_attr",
        "never_base",
        "specfem_import.md L340-346",
        (
            "Recommended (not machine-enforced): observed max |semd_time - "
            "(n*dt - t0)| in seconds, from provenance.crosscheck_t0_semd. "
            "Only meaningful alongside t0_crosscheck='semd_column1'."
        ),
        recommended=True,
    ),
    # --- Step 6 addition (2026-08-03): these four keys were collected by
    # Step 5's specfem_bin.RecordLayout / integer-kind auto-detection but not
    # yet registered in this required-key table (Step 5 commit note). They
    # are SPECFEM-specific structural facts about the on-disk binary layout
    # that hold regardless of output_sampling_kind, so they are
    # unconditionally required ("always"), not "recommended".
    "suppress_utm_projection": ProvenanceKeySpec(
        "suppress_utm_projection",
        "bool",
        "scalar_attr",
        "always",
        "specfem_import.md, certified profile / implementation facts",
        (
            "Must be True for the certified profile -- the displ_{X,Y,Z} "
            "component-letter -> axis mapping is only a direct X/Y/Z write "
            "under SUPPRESS_UTM_PROJECTION=.true.."
        ),
    ),
    "endianness": ProvenanceKeySpec(
        "endianness",
        "str",
        "scalar_attr",
        "always",
        "specfem_bin.RecordLayout",
        "One of '<' (little) | '>' (big) -- auto-detected record layout.",
    ),
    "record_marker_bytes": ProvenanceKeySpec(
        "record_marker_bytes",
        "int",
        "scalar_attr",
        "always",
        "specfem_bin.RecordLayout",
        "4 or 8 -- auto-detected Fortran record marker width.",
    ),
    "integer_kind": ProvenanceKeySpec(
        "integer_kind",
        "int",
        "scalar_attr",
        "always",
        "specfem_bin.read_ibool / specfem_bin.parse_database",
        "4 or 8 -- auto-detected default INTEGER width used in payloads.",
    ),
}

# --- 7c. Cadence group (time_sampled only) ---------------------------------
# ref: solver_import_contract.md L605-620 (base table), L642-716 (value
# ranges / f_content_max_method variants).
CADENCE_PROVENANCE_KEYS: Final[dict[str, ProvenanceKeySpec]] = {
    "output_interval_steps": ProvenanceKeySpec(
        "output_interval_steps",
        "int",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L610",
        "M >= 1.",
    ),
    "f_max_analysis": ProvenanceKeySpec(
        "f_max_analysis",
        "float",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L611",
    ),
    "f_min_analysis": ProvenanceKeySpec(
        "f_min_analysis",
        "float",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L618",
    ),
    "minimum_cycles": ProvenanceKeySpec(
        "minimum_cycles",
        "float",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L619",
    ),
    "sampling_oversampling_factor": ProvenanceKeySpec(
        "sampling_oversampling_factor",
        "float",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L617",
        "q >= 2.",
    ),
    "antialias_filter_applied": ProvenanceKeySpec(
        "antialias_filter_applied",
        "bool",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L612",
    ),
    "antialias_filter_spec": ProvenanceKeySpec(
        "antialias_filter_spec",
        "json",
        "json_dataset",
        "never_base",
        "solver_import_contract.md L613",
        "Required iff antialias_filter_applied is True.",
    ),
    "f_content_max": ProvenanceKeySpec(
        "f_content_max",
        "float",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L614",
        "Required iff output_interval_steps > 1 (regardless of filter use).",
    ),
    "f_content_max_method": ProvenanceKeySpec(
        "f_content_max_method",
        "str",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L615,662-693",
        (
            "Required under the same condition as f_content_max. One of "
            "'solver_step_nyquist' | 'measured_spectrum' | 'filter_cutoff' | "
            "'analytic_manufactured' | 'source_spectrum_bound'."
        ),
    ),
    "f_content_max_evidence": ProvenanceKeySpec(
        "f_content_max_evidence",
        "json",
        "json_dataset",
        "never_base",
        "solver_import_contract.md L669-675",
        (
            "Required iff f_content_max_method == 'measured_spectrum' "
            "(sub-key existence only is checked at this scope -- deep "
            "sub-key validation is deferred, see module notes)."
        ),
    ),
    "alias_error_budget": ProvenanceKeySpec(
        "alias_error_budget",
        "float",
        "scalar_attr",
        "never_base",
        "solver_import_contract.md L616",
        "Required iff f_content_max_method != 'solver_step_nyquist'.",
    ),
    "cadence_provenance": ProvenanceKeySpec(
        "cadence_provenance",
        "str",
        "scalar_attr",
        "time_sampled",
        "solver_import_contract.md L620,627-640",
        "One of 'placeholder' | 'confirmed'.",
    ),
}

ALL_PROVENANCE_KEYS: Final[dict[str, ProvenanceKeySpec]] = {
    **COMMON_PROVENANCE_KEYS,
    **SPECFEM_PROVENANCE_KEYS,
    **CADENCE_PROVENANCE_KEYS,
}


def _validate_output_sampling_kind(output_sampling_kind: str) -> None:
    if output_sampling_kind not in VALID_OUTPUT_SAMPLING_KINDS:
        raise ValueError(
            f"output_sampling_kind must be one of {VALID_OUTPUT_SAMPLING_KINDS}, "
            f"got {output_sampling_kind!r}"
        )


def required_keys(output_sampling_kind: str) -> frozenset[str]:
    """Return the provenance keys that are unconditionally required given
    only ``output_sampling_kind`` (the "base" required set).

    This does **not** include keys whose necessity depends on some *other*
    provenance value already collected (build_kind, antialias_filter_applied,
    output_interval_steps, f_content_max_method) -- see
    ``conditionally_required_keys()`` for those.

    Both ``writer.py`` (Step 6) and Tier A4 (``check_a4_masks_scalars_
    provenance``) must call this function rather than re-deriving the set,
    so that a key added/removed here changes both consumers at once.
    """
    _validate_output_sampling_kind(output_sampling_kind)
    result: set[str] = set()
    for spec in ALL_PROVENANCE_KEYS.values():
        is_always = spec.base_required_when == "always"
        is_active_time_sampled = (
            spec.base_required_when == "time_sampled"
            and output_sampling_kind == "time_sampled"
        )
        if is_always or is_active_time_sampled:
            result.add(spec.name)
    return frozenset(result)


def recommended_keys(output_sampling_kind: str) -> frozenset[str]:
    """Return keys that are *recommended* (not machine-enforced) for the
    given ``output_sampling_kind`` (currently only ``t0_crosscheck``, which
    only applies when ``output_sampling_kind == "time_sampled"``)."""
    _validate_output_sampling_kind(output_sampling_kind)
    if output_sampling_kind != "time_sampled":
        return frozenset()
    return frozenset(
        spec.name for spec in ALL_PROVENANCE_KEYS.values() if spec.recommended
    )


def conditionally_required_keys(provenance) -> frozenset[str]:
    """Return additional provenance keys required given the *values* already
    present in ``provenance`` (a ``Mapping[str, object]``) -- the
    value-dependent complement to ``required_keys()``.

    Handles: build_kind branching (source/package), the antialias filter
    spec, the ``output_interval_steps > 1`` -> f_content_max(_method)
    requirement, the f_content_max_method -> alias_error_budget /
    f_content_max_evidence requirements. The build_kind == 'source' AND dirty
    tree "one of {patch_hash, source_tree_hash}" requirement is *not*
    representable as a single required-key set (it is an "at least one of"
    condition) -- use ``source_tree_dirty_hash_satisfied()`` for that check.
    """
    extra: set[str] = set()

    build_kind = provenance.get("build_kind")
    if build_kind == "source":
        extra.add("specfem_git_commit")
        extra.add("source_tree_clean")
    elif build_kind == "package":
        extra.add("package_version")
        extra.add("package_digest")

    if provenance.get("antialias_filter_applied") is True:
        extra.add("antialias_filter_spec")

    m = provenance.get("output_interval_steps")
    if isinstance(m, (int, np.integer)) and not isinstance(m, bool) and m > 1:
        extra.add("f_content_max")
        extra.add("f_content_max_method")

    method = provenance.get("f_content_max_method")
    if method is not None and method != "solver_step_nyquist":
        extra.add("alias_error_budget")
    if method == "measured_spectrum":
        extra.add("f_content_max_evidence")

    return frozenset(extra)


def source_tree_dirty_hash_satisfied(provenance) -> bool:
    """Check the "one of {patch_hash, source_tree_hash}" requirement for a
    dirty source-build tree (ref: solver_import_contract.md L937-940).

    Returns True when the requirement does not apply (non-source build, or
    a clean source tree) or when at least one of the two keys is present.
    """
    if provenance.get("build_kind") != "source":
        return True
    if provenance.get("source_tree_clean") is not False:
        return True
    return ("patch_hash" in provenance) or ("source_tree_hash" in provenance)


def provenance_encoding(key: str) -> str:
    """Look up the on-disk encoding ("scalar_attr" | "json_dataset" |
    "text_dataset") for a provenance key. Raises ``KeyError`` for unknown
    keys."""
    spec = ALL_PROVENANCE_KEYS.get(key)
    if spec is None:
        raise KeyError(f"Unknown provenance key: {key!r}")
    return spec.encoding


def check_value_type(name: str, value: object) -> bool:
    """Coarse runtime type check for a provenance value against its
    ``ProvenanceKeySpec.value_type``. Used by Tier A4's required-provenance-
    key type audit. ``"json"``-typed values are only checked for presence
    (deep structural validation of e.g. ``executables[]`` entries is out of
    scope here).
    """
    spec = ALL_PROVENANCE_KEYS.get(name)
    if spec is None:
        raise KeyError(f"Unknown provenance key: {name!r}")
    vt = spec.value_type
    if vt == "int":
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
    if vt == "float":
        is_numeric = isinstance(value, (int, float, np.floating, np.integer))
        return is_numeric and not isinstance(value, bool)
    if vt == "str":
        return isinstance(value, str)
    if vt == "bool":
        return isinstance(value, (bool, np.bool_))
    if vt == "int_array3":
        arr = np.asarray(value)
        return arr.shape == (3,) and np.issubdtype(arr.dtype, np.integer)
    if vt == "json":
        return True
    # pragma: no cover -- unreachable unless a new ValueTypeName is added
    # without updating this dispatcher.
    raise AssertionError(f"unhandled value_type {vt!r} for key {name!r}")


# ===========================================================================
# 8. Canonical JSON encoding (for `encoding == "json_dataset"` keys)
#    ref: S1 plan Step 1 item 6.
# ===========================================================================


def canonical_json(obj: object) -> str:
    """Serialize ``obj`` to the canonical JSON text used for
    ``encoding == "json_dataset"`` provenance values: deterministic key
    order, no incidental whitespace, non-ASCII preserved, NaN/Infinity
    rejected (a structure containing a float NaN/Inf raises ``ValueError``,
    since provenance must not silently encode non-finite values).
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


# ===========================================================================
# 9. Hash formatting convention ("sha256:..." prefix)
#    ref: solver_import_contract.md L1007-1015 (general hash-notation rules).
# ===========================================================================

DEFAULT_HASH_ALGORITHM: Final[str] = "sha256"


def format_hash(hexdigest: str, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """Format a hex digest as ``"<algorithm>:<hexdigest>"`` (the contract's
    fixed hash-string convention for every ``*_hash`` key)."""
    if not hexdigest:
        raise ValueError("hexdigest must not be empty")
    if ":" in algorithm:
        raise ValueError(f"algorithm must not contain ':': {algorithm!r}")
    return f"{algorithm}:{hexdigest}"


def parse_hash(value: str) -> tuple[str, str]:
    """Parse an ``"<algorithm>:<hexdigest>"`` string. Raises ``ValueError``
    for malformed input (missing separator, empty algorithm/hexdigest)."""
    if not isinstance(value, str) or ":" not in value:
        raise ValueError(
            f"Malformed hash string (expected 'algorithm:hexdigest'): {value!r}"
        )
    algorithm, _, hexdigest = value.partition(":")
    if not algorithm or not hexdigest:
        raise ValueError(
            f"Malformed hash string (expected 'algorithm:hexdigest'): {value!r}"
        )
    return algorithm, hexdigest


def is_hash_string(value: object) -> bool:
    """Return whether ``value`` is a syntactically well-formed
    ``"algorithm:hexdigest"`` hash string."""
    if not isinstance(value, str):
        return False
    try:
        parse_hash(value)
    except ValueError:
        return False
    return True
