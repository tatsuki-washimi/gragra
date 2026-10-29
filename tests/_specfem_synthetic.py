"""Synthetic SPECFEM3D Cartesian decompose_mesh output tree generator.

Builds, under a caller-supplied ``tmp_path``, a small but structurally
faithful synthetic replica of an ``xdecompose_mesh`` output tree (NPROC=2):
``part_array.vtk``, ``proc*_Database``, ``proc*_{x,y,z,ibool,rho}.bin``,
``DATA/Par_file``, and a dummy ``bin/xspecfem3D`` executable -- everything
``gragra.adapters.specfem.specfem_bin`` needs to be exercised
end-to-end without a real SPECFEM3D run.

Design notes (see ``docs/design/specfem_import.md`` and
``docs/validation/specfem_pilot_checks.md``):

- Global mesh: 6 hexahedral elements (NGLL=5), covering affine/stretched/
  rotated/warped shapes, split into 2 materials (rho=2000, rho=3000).
  Element 0 ("E0") is an axis-aligned box with 3 *distinct* edge lengths
  (100 x 200 x 400 m) -- the A1 direction-fixed anchor element. Element 1
  ("E1") shares its x=100 face with E0 (same y/z extents), so that when
  both are assigned to the same MPI rank, the rank's GLL DOF numbering
  reduces (``nglob < ngll * NSPEC``) via the shared-face indirection.
- Two independent node-numbering systems are modelled, matching real
  SPECFEM3D: (a) coarse *anchor* mesh nodes (8 per hex, shared only at
  element *corners* -- these are what ``proc*_Database`` node records
  hold), and (b) fine GLL DOF points (125 per hex, shared across *entire*
  faces -- these are what ``ibool.bin``/``x.bin``/``y.bin``/``z.bin`` hold).
  These are deliberately kept separate; they are not the same numbering.
- Two named partition variants ("a", "b") of the *same* global mesh are
  provided (different element->rank assignments) for partition-independence
  testing (``element_id`` recovery must agree between them).
- rho=3000 (material 2) is written to ``rho.bin`` as a float32 value
  perturbed by 1 ulp from the exact profile density (``np.nextafter``) --
  the Database *material record* itself keeps the exact value. This
  exercises A3's "systematic bias" tolerance regime deliberately.

Independence / core-purity note: this module is test-only (not under
``src/``) and may freely import ``numpy``, ``hashlib``, and
``_hex_volume_oracle`` (used only for a self-check that no synthetic
element is degenerate/inverted -- not as a dependency of the readers under
test). It must NOT import ``gragra.adapters.specfem.specfem_bin`` (the
module under test) or ``gragra.adapters.specfem.export`` at module level --
the exporter is imported only when the static fixture is built; see
``build_static_fixture`` for the function-level import.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from _hex_volume_oracle import hex_volume_gauss2

# ===========================================================================
# Fixed mesh constants
# ===========================================================================

_NGLL_SHAPE: Final[tuple[int, int, int]] = (5, 5, 5)
_NGLL: Final[int] = 125  # 5*5*5

#: n=5 Gauss-Lobatto-Legendre abscissae on [-1, 1] (analytic, well-known
#: values -- also used as the Step 3 analytic test values). Hardcoded here
#: (rather than computed via a general Newton solve, or imported from a
#: verification helper) since NGLL=5 is fixed for this fixture and the
#: values are exactly known in closed form.
_GLL_ABSCISSAE: Final[np.ndarray] = np.array(
    [-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0], dtype=np.float64
)

#: Reference (xi, eta, zeta) corner coordinates in usual_hex_nodes /
#: VTK_HEXAHEDRON order (matches ``tests/_hex_volume_oracle.py``'s
#: ``_REF_CORNERS`` -- duplicated here as a small local constant rather than
#: imported, to keep this generator module self-contained).
_REF_CORNERS: Final[np.ndarray] = np.array(
    [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ],
    dtype=np.float64,
)

_RHO_BY_MATERIAL: Final[dict[int, float]] = {1: 2000.0, 2: 3000.0}
_IDOMAIN_ELASTIC: Final[float] = 2.0

#: element_id (0-based, canonical global order) -> MPI rank, for each named
#: partition variant. Both variants keep the same 6-element global mesh;
#: only the rank assignment differs (element_id recovery must agree).
_PARTITION_VARIANTS: Final[dict[str, dict[int, int]]] = {
    "a": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
    "b": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0},
}

#: Fixed dummy "executable" byte content for bin/xspecfem3D, and its
#: hardcoded expected sha256 (self-verified at generation time so an
#: accidental future edit to the byte content is caught immediately rather
#: than silently producing a stale hash).
_DUMMY_EXECUTABLE_BYTES: Final[bytes] = (
    b"gragra-specfem-synthetic-dummy-executable-fixture-v1\n" * 8
)
_DUMMY_EXECUTABLE_SHA256: Final[str] = (
    "98dc1b2d6546d9af87c25570fa14858746aaec98edbd183a26d9a929298e33a1"
)


# ===========================================================================
# Geometry helpers
# ===========================================================================


def _box_corners(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> np.ndarray:
    """Axis-aligned box corners in usual_hex_nodes order."""
    return np.array(
        [
            (xmin, ymin, zmin),
            (xmax, ymin, zmin),
            (xmax, ymax, zmin),
            (xmin, ymax, zmin),
            (xmin, ymin, zmax),
            (xmax, ymin, zmax),
            (xmax, ymax, zmax),
            (xmin, ymax, zmax),
        ],
        dtype=np.float64,
    )


def _rotation_matrix(theta_z: float, theta_x: float) -> np.ndarray:
    """A composed (about-z then about-x) rotation matrix -- just needs to
    be a nontrivial 3D rotation for the "rotated element" fixture case."""
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rx @ rz


def _trilinear_position(
    corners8: np.ndarray, xi: float, eta: float, zeta: float
) -> np.ndarray:
    """Trilinear-mapped physical position at parametric (xi, eta, zeta),
    from the 8 corner nodes (usual_hex_nodes order).

    Implemented as an explicit sequential loop over the 8 reference
    corners (rather than a vectorized ``np.dot``/``@``) so the floating-
    point summation order is fully deterministic and independent of numpy's
    BLAS backend -- this determinism is what makes the E0/E1 shared-face
    GLL points come out bit-identical between the two elements (see the
    module docstring): for a corner whose shape-function weight is exactly
    zero, adding its zero contribution never perturbs the running sum, so
    the two elements' surviving nonzero terms are added in the same
    relative order.
    """
    total = np.zeros(3, dtype=np.float64)
    for idx in range(8):
        xi_i, eta_i, zeta_i = _REF_CORNERS[idx]
        weight = 0.125 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i)
        total = total + weight * corners8[idx]
    return total


def _element_gll_grid(corners8: np.ndarray) -> np.ndarray:
    """Full (125, 3) physical GLL point grid for one element, in
    ``schema.py`` flatten order (``flat = k*25 + j*5 + i``, x innermost)."""
    grid = np.empty((_NGLL, 3), dtype=np.float64)
    for k in range(5):
        zeta = _GLL_ABSCISSAE[k]
        for j in range(5):
            eta = _GLL_ABSCISSAE[j]
            for i in range(5):
                xi = _GLL_ABSCISSAE[i]
                grid[k * 25 + j * 5 + i] = _trilinear_position(corners8, xi, eta, zeta)
    return grid


# ===========================================================================
# Global mesh definition
# ===========================================================================


def _build_global_mesh() -> list[dict]:
    """The 6-element global mesh (canonical order == element_id). Returns a
    list of dicts with keys ``name``, ``corners8``, ``material_id``.
    """
    elements: list[dict] = []

    # E0: axis-aligned box, 3 distinct edge lengths (100 x 200 x 400 m) --
    # the A1 direction-fixed anchor element.
    elements.append(
        {
            "name": "E0_axis_box",
            "corners8": _box_corners(0.0, 100.0, 0.0, 200.0, 0.0, 400.0),
            "material_id": 1,
        }
    )

    # E1: shares the x=100 face with E0 (same y/z extents) -- the required
    # face-sharing pair.
    elements.append(
        {
            "name": "E1_shared_face",
            "corners8": _box_corners(100.0, 200.0, 0.0, 200.0, 0.0, 400.0),
            "material_id": 1,
        }
    )

    # E2: stretched box, isolated, material 2.
    elements.append(
        {
            "name": "E2_stretched",
            "corners8": _box_corners(2000.0, 2050.0, 2000.0, 2180.0, 2000.0, 2090.0),
            "material_id": 2,
        }
    )

    # E3: rotated affine parallelepiped, isolated, material 2.
    local_e3 = _box_corners(-30.0, 30.0, -40.0, 40.0, -20.0, 20.0)
    rot = _rotation_matrix(np.deg2rad(25.0), np.deg2rad(15.0))
    e3 = local_e3 @ rot.T + np.array([3000.0, 3000.0, 3000.0])
    elements.append({"name": "E3_rotated", "corners8": e3, "material_id": 2})

    # E4: warped (non-planar faces) box, isolated, material 1.
    e4 = _box_corners(-20.0, 20.0, -25.0, 25.0, -17.5, 17.5) + np.array(
        [4000.0, 4000.0, 4000.0]
    )
    e4 = e4.copy()
    e4[6] += np.array([6.0, -4.0, 3.0])
    e4[0] += np.array([-2.0, 1.0, -4.0])
    elements.append({"name": "E4_warped", "corners8": e4, "material_id": 1})

    # E5: warped + stretched box, isolated, material 2.
    e5 = _box_corners(-22.5, 22.5, -27.5, 27.5, -32.5, 32.5) + np.array(
        [5000.0, 5000.0, 5000.0]
    )
    e5 = e5.copy()
    e5[6] += np.array([7.0, -5.0, 4.0])
    e5[0] += np.array([-3.0, 2.0, -5.0])
    elements.append({"name": "E5_warped_stretched", "corners8": e5, "material_id": 2})

    return elements


def _assign_global_anchor_ids(
    elements: list[dict],
) -> tuple[list[list[int]], np.ndarray]:
    """Assign global (0-based) anchor-node ids to every element's 8
    corners, sharing E1's 4 face corners with E0's matching corners (see
    the module docstring's face-sharing design). Returns
    ``(corner_global_ids, global_anchor_coords)`` where
    ``corner_global_ids[e][local]`` is the global anchor node id for
    element ``e``'s local corner ``local`` (usual_hex_nodes order), and
    ``global_anchor_coords`` is the deduplicated ``(n_anchor_nodes, 3)``
    coordinate table indexed by global anchor node id.
    """
    n_elements = len(elements)
    corner_global_ids: list[list[int]] = [[-1] * 8 for _ in range(n_elements)]
    coords_by_id: list[np.ndarray] = []
    next_id = 0

    # E0 (index 0): all 8 corners fresh.
    for local in range(8):
        corner_global_ids[0][local] = next_id
        coords_by_id.append(elements[0]["corners8"][local])
        next_id += 1

    # E1 (index 1): local corners {0, 3, 4, 7} reuse E0's {1, 2, 5, 6}
    # (the shared x=100 face); local corners {1, 2, 5, 6} are fresh.
    share_map = {0: 1, 3: 2, 4: 5, 7: 6}  # E1 local -> E0 local
    for local in range(8):
        if local in share_map:
            e0_local = share_map[local]
            if not np.array_equal(
                elements[1]["corners8"][local], elements[0]["corners8"][e0_local]
            ):
                raise AssertionError(
                    f"E1 local corner {local} does not physically coincide "
                    f"with E0 local corner {e0_local} -- fixture geometry bug"
                )
            corner_global_ids[1][local] = corner_global_ids[0][e0_local]
        else:
            corner_global_ids[1][local] = next_id
            coords_by_id.append(elements[1]["corners8"][local])
            next_id += 1

    # E2..E5: all corners fresh (isolated elements).
    for e_idx in range(2, n_elements):
        for local in range(8):
            corner_global_ids[e_idx][local] = next_id
            coords_by_id.append(elements[e_idx]["corners8"][local])
            next_id += 1

    global_anchor_coords = np.array(coords_by_id, dtype=np.float64)
    return corner_global_ids, global_anchor_coords


# ===========================================================================
# Fortran unformatted record writing (test-only; the production reader
# side lives in gragra.adapters.specfem.specfem_bin)
# ===========================================================================


def _fortran_record_bytes(
    payload: bytes, *, endian: str = "<", marker_bytes: int = 4
) -> bytes:
    marker = np.array([len(payload)], dtype=f"{endian}i{marker_bytes}").tobytes()
    return marker + payload + marker


def _write_records_file(
    path: Path, payloads: list[bytes], *, endian: str = "<", marker_bytes: int = 4
) -> None:
    chunks = [
        _fortran_record_bytes(p, endian=endian, marker_bytes=marker_bytes)
        for p in payloads
    ]
    path.write_bytes(b"".join(chunks))


def _write_single_record(
    path: Path, payload: bytes, *, endian: str = "<", marker_bytes: int = 4
) -> None:
    path.write_bytes(
        _fortran_record_bytes(payload, endian=endian, marker_bytes=marker_bytes)
    )


def _int_payload(value: int, *, dtype: str = "<i4") -> bytes:
    return np.array([value], dtype=dtype).tobytes()


# ===========================================================================
# part_array.vtk writing
# ===========================================================================


def _write_part_array_vtk(
    path: Path, points: np.ndarray, cells: np.ndarray, elem_flag: np.ndarray
) -> None:
    n_points = len(points)
    n_cells = len(cells)
    lines = [
        "# vtk DataFile Version 3.0",
        "gragra synthetic SPECFEM3D decompose_mesh part_array fixture",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_points} double",
    ]
    for p in points:
        lines.append(" ".join(f"{float(c):.17g}" for c in p))
    lines.append(f"CELLS {n_cells} {n_cells * 9}")
    for row in cells:
        lines.append("8 " + " ".join(str(int(v)) for v in row))
    lines.append(f"CELL_TYPES {n_cells}")
    lines.extend(["12"] * n_cells)  # VTK_HEXAHEDRON
    lines.append(f"CELL_DATA {n_cells}")
    lines.append("SCALARS elem_flag int 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(str(int(v)) for v in elem_flag)
    path.write_text("\n".join(lines) + "\n")


# ===========================================================================
# proc*_Database writing
# ===========================================================================


def _write_rank_database(
    path: Path,
    *,
    global_ids_this_rank: list[int],
    corner_global_ids: list[list[int]],
    global_anchor_coords: np.ndarray,
    raw_elements: list[dict],
) -> None:
    used_global_node_ids = sorted(
        {
            gid
            for elem_gid in global_ids_this_rank
            for gid in corner_global_ids[elem_gid]
        }
    )
    local_id_of = {gid: i + 1 for i, gid in enumerate(used_global_node_ids)}  # 1-based

    payloads: list[bytes] = []

    # nnodes_loc
    payloads.append(_int_payload(len(used_global_node_ids)))

    # node records: iloc (int32) + x,y,z (float64) each.
    for local_idx, gid in enumerate(used_global_node_ids, start=1):
        coord = global_anchor_coords[gid]
        payloads.append(
            np.array([local_idx], dtype="<i4").tobytes() + coord.astype("<f8").tobytes()
        )

    # ndef, nundef (2 defined materials, 0 undefined/tomography).
    payloads.append(np.array([2, 0], dtype="<i4").tobytes())

    # 2 defined-material records (17 float64 each). No material_id field --
    # the k-th record (0-based here) <-> material_id = k+1. domain_id sits
    # at 0-based index 6 (Fortran mat_prop(7,.), the 1-based 7th field --
    # verified against a real proc*_Database on 2026-08-03; IDOMAIN_ELASTIC
    # = 2.0). Index 0 holds rho for documentation purposes only (the
    # SPECFEM exporter reads density from rho.bin, not from this record).
    for material_id in (1, 2):
        values = np.zeros(17, dtype="<f8")
        values[0] = _RHO_BY_MATERIAL[material_id]
        values[6] = _IDOMAIN_ELASTIC
        payloads.append(values.tobytes())

    # nspec_local
    payloads.append(_int_payload(len(global_ids_this_rank)))

    # element records: iloc, mat1, mat2(=0 on the decompose path), 8 local
    # node ids (usual_hex_nodes order), local element index assigned by
    # walking global_ids_this_rank in ascending order (already sorted by
    # the caller) -- matches upstream's global-ascending local numbering.
    for local_elem_idx, gid in enumerate(global_ids_this_rank, start=1):
        mat1 = raw_elements[gid]["material_id"]
        node_loc_ids = [
            local_id_of[corner_gid] for corner_gid in corner_global_ids[gid]
        ]
        values = [local_elem_idx, mat1, 0, *node_loc_ids]
        payloads.append(np.array(values, dtype="<i4").tobytes())

    # Generic tail (not semantically parsed by specfem_bin.parse_database):
    # 6 boundary-type element-count headers (all 0 -- no absorbing
    # boundary in this MVP fixture) + nspec_cpml=0. This exercises the
    # "read past + EOF alignment" path with real intervening records rather
    # than an immediate EOF right after the element block.
    for _ in range(6):
        payloads.append(_int_payload(0))
    payloads.append(_int_payload(0))  # nspec_cpml

    _write_records_file(path, payloads)


# ===========================================================================
# proc*_{x,y,z,ibool,rho}.bin writing
# ===========================================================================


def _rho_on_disk_value(material_id: int) -> np.float32:
    """The float32 value written to rho.bin for a given material: material
    1 (rho=2000) is stored exactly; material 2 (rho=3000) is perturbed by
    1 ulp (``np.nextafter``) to exercise A3's systematic-bias tolerance
    regime deliberately (the Database material record keeps the exact
    3000.0 -- only this on-disk rho.bin value is shifted).
    """
    rho = np.float32(_RHO_BY_MATERIAL[material_id])
    if material_id == 2:
        rho = np.nextafter(rho, np.float32(np.inf))
    return rho


def _write_rank_bin_files(
    local_path: Path,
    rank: int,
    global_ids_this_rank: list[int],
    element_grids: dict[int, np.ndarray],
    raw_elements: list[dict],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Write ``proc{rank}_{ibool,x,y,z,rho}.bin`` for one rank's element
    set, building the rank-LOCAL GLL DOF numbering (distinct from the
    anchor-node numbering used by the Database) by walking elements in
    ascending global element_id order and merging any GLL point whose
    physical coordinate has already been seen within this rank.

    Returns ``(ibool_zero_based, xyz, nglob)`` for the caller's own
    bookkeeping/self-checks.
    """
    coord_to_iglob: dict[bytes, int] = {}
    ibool_rows: list[np.ndarray] = []
    next_iglob = 0
    for gid in global_ids_this_rank:
        grid = element_grids[gid]
        row = np.empty(_NGLL, dtype=np.int64)
        for flat in range(_NGLL):
            key = grid[flat].tobytes()
            existing = coord_to_iglob.get(key)
            if existing is None:
                coord_to_iglob[key] = next_iglob
                row[flat] = next_iglob
                next_iglob += 1
            else:
                row[flat] = existing
        ibool_rows.append(row)
    ibool_zero_based = np.stack(ibool_rows, axis=0)
    nglob = next_iglob

    xyz = np.empty((nglob, 3), dtype=np.float64)
    rho_local = np.empty((len(global_ids_this_rank), _NGLL), dtype=np.float32)
    for row_idx, gid in enumerate(global_ids_this_rank):
        grid = element_grids[gid]
        for flat in range(_NGLL):
            xyz[ibool_zero_based[row_idx, flat]] = grid[flat]
        rho_local[row_idx, :] = _rho_on_disk_value(raw_elements[gid]["material_id"])

    ibool_one_based = (ibool_zero_based + 1).astype("<i4")
    _write_single_record(
        local_path / f"proc{rank:06d}_ibool.bin", ibool_one_based.tobytes()
    )
    for axis_idx, name in enumerate("xyz"):
        values = xyz[:, axis_idx].astype("<f4")
        _write_single_record(
            local_path / f"proc{rank:06d}_{name}.bin", values.tobytes()
        )
    _write_single_record(
        local_path / f"proc{rank:06d}_rho.bin", rho_local.astype("<f4").tobytes()
    )

    return ibool_zero_based, xyz, nglob


# ===========================================================================
# Par_file writing
# ===========================================================================


def _par_file_text(nproc: int) -> str:
    """A minimal (not fully realistic) ``DATA/Par_file`` populated only
    with the documented input-profile keys this fixture / exporter path
    needs (``docs/design/specfem_import.md`` §input profile)."""
    lines = [
        "# Synthetic Par_file for gragra SPECFEM3D pilot-check fixtures.",
        "# Not a full/realistic Par_file -- only documented input-profile keys.",
        f"NPROC                           = {nproc}",
        "NGNOD                           = 8",
        "ATTENUATION                     = .false.",
        "PML_INSTEAD_OF_FREE_SURFACE     = .false.",
        "SUPPRESS_UTM_PROJECTION         = .true.",
        "SIMULATION_TYPE                 = 1",
        "SAVE_MESH_FILES                 = .true.",
        "SAVE_DISPLACEMENT               = .false.",
        "MOVIE_VOLUME                    = .false.",
        "DT                              = 0.05",
        "NSTEP                           = 10",
        "NTSTEP_BETWEEN_FRAMES           = 1",
        "LOCAL_PATH                      = ./OUTPUT_FILES/DATABASES_MPI",
    ]
    return "\n".join(lines) + "\n"


# ===========================================================================
# Public dataclasses + entry point
# ===========================================================================


@dataclass(frozen=True)
class SyntheticElementInfo:
    """Ground-truth record for one of the 6 global mesh elements."""

    element_id: int
    name: str
    corners8: np.ndarray
    material_id: int
    rho_profile: float
    rank: int


@dataclass(frozen=True)
class SyntheticMeshInfo:
    """Everything a test needs to read back and cross-check the synthetic
    tree written by :func:`write_synthetic_specfem_tree`."""

    root: Path
    local_path: Path
    nproc: int
    ngll_shape: tuple[int, int, int]
    ngll: int
    n_elements_total: int
    elements: tuple[SyntheticElementInfo, ...]  # canonical order == element_id
    rho_by_material: dict[int, float]
    domain_id_by_material: dict[int, float]
    partition_variant: str
    shared_face_element_ids: tuple[int, int]
    dummy_executable_path: Path
    dummy_executable_sha256: str
    par_file_path: Path
    expected_axis_vectors: dict[int, dict[str, tuple[float, float, float]]]

    def rank_of(self, element_id: int) -> int:
        return self.elements[element_id].rank

    def elements_for_rank(self, rank: int) -> list[int]:
        """Zero-based global element ids assigned to ``rank``, ascending
        (the ground-truth ``G_r`` that ``recover_element_ids`` should
        reproduce)."""
        return sorted(e.element_id for e in self.elements if e.rank == rank)

    def database_path(self, rank: int) -> Path:
        return self.local_path / f"proc{rank:06d}_Database"

    def part_array_path(self) -> Path:
        return self.local_path / "part_array.vtk"

    def bin_path(self, rank: int, kind: str) -> Path:
        if kind not in ("x", "y", "z", "ibool", "rho"):
            raise ValueError(f"Unknown bin file kind {kind!r}")
        return self.local_path / f"proc{rank:06d}_{kind}.bin"


def write_synthetic_specfem_tree(
    root, *, partition_variant: str = "a"
) -> SyntheticMeshInfo:
    """Write a full synthetic ``xdecompose_mesh`` output tree under
    ``root`` and return a :class:`SyntheticMeshInfo` describing it.

    ``partition_variant`` selects one of two different rank assignments
    (``"a"`` or ``"b"``) of the *same* 6-element global mesh, for
    partition-independence (``element_id`` recovery) testing.
    """
    if partition_variant not in _PARTITION_VARIANTS:
        raise ValueError(
            f"Unknown partition_variant {partition_variant!r}, expected one "
            f"of {sorted(_PARTITION_VARIANTS)}"
        )
    root = Path(root)
    local_path = root / "OUTPUT_FILES" / "DATABASES_MPI"
    local_path.mkdir(parents=True, exist_ok=True)
    (root / "DATA").mkdir(parents=True, exist_ok=True)
    (root / "bin").mkdir(parents=True, exist_ok=True)

    raw_elements = _build_global_mesh()
    corner_global_ids, global_anchor_coords = _assign_global_anchor_ids(raw_elements)
    element_grids = {
        i: _element_gll_grid(raw_elements[i]["corners8"])
        for i in range(len(raw_elements))
    }

    # Self-check: no synthetic element may be degenerate/inverted -- a bug
    # here would silently invalidate every downstream Tier A comparison
    # built on this fixture. Uses the already-validated A2 oracle
    # (tests/_hex_volume_oracle.py) purely as a sanity check, not as a
    # dependency of the readers under test.
    for i, elem in enumerate(raw_elements):
        vol = hex_volume_gauss2(elem["corners8"])
        if vol <= 0.0:
            raise AssertionError(
                f"synthetic element {i} ({elem['name']!r}) has non-positive "
                f"oracle volume {vol} -- fixture geometry bug"
            )

    rank_by_element = _PARTITION_VARIANTS[partition_variant]
    nproc = len(set(rank_by_element.values()))

    cells = np.array(corner_global_ids, dtype=np.int64)
    elem_flag = np.array(
        [rank_by_element[i] for i in range(len(raw_elements))], dtype=np.int64
    )
    _write_part_array_vtk(
        local_path / "part_array.vtk", global_anchor_coords, cells, elem_flag
    )

    for rank in range(nproc):
        global_ids_this_rank = sorted(
            i for i in range(len(raw_elements)) if rank_by_element[i] == rank
        )
        _write_rank_database(
            local_path / f"proc{rank:06d}_Database",
            global_ids_this_rank=global_ids_this_rank,
            corner_global_ids=corner_global_ids,
            global_anchor_coords=global_anchor_coords,
            raw_elements=raw_elements,
        )
        _write_rank_bin_files(
            local_path, rank, global_ids_this_rank, element_grids, raw_elements
        )

    dummy_path = root / "bin" / "xspecfem3D"
    dummy_path.write_bytes(_DUMMY_EXECUTABLE_BYTES)
    actual_sha256 = hashlib.sha256(_DUMMY_EXECUTABLE_BYTES).hexdigest()
    if actual_sha256 != _DUMMY_EXECUTABLE_SHA256:
        raise AssertionError(
            "dummy executable byte content changed but the hardcoded "
            f"expected sha256 was not updated (got {actual_sha256}, "
            f"expected {_DUMMY_EXECUTABLE_SHA256})"
        )

    par_file_path = root / "DATA" / "Par_file"
    par_file_path.write_text(_par_file_text(nproc))

    elements_info = tuple(
        SyntheticElementInfo(
            element_id=i,
            name=raw_elements[i]["name"],
            corners8=raw_elements[i]["corners8"],
            material_id=raw_elements[i]["material_id"],
            rho_profile=_RHO_BY_MATERIAL[raw_elements[i]["material_id"]],
            rank=rank_by_element[i],
        )
        for i in range(len(raw_elements))
    )

    return SyntheticMeshInfo(
        root=root,
        local_path=local_path,
        nproc=nproc,
        ngll_shape=_NGLL_SHAPE,
        ngll=_NGLL,
        n_elements_total=len(raw_elements),
        elements=elements_info,
        rho_by_material=dict(_RHO_BY_MATERIAL),
        domain_id_by_material={1: _IDOMAIN_ELASTIC, 2: _IDOMAIN_ELASTIC},
        partition_variant=partition_variant,
        shared_face_element_ids=(0, 1),
        dummy_executable_path=dummy_path,
        dummy_executable_sha256=_DUMMY_EXECUTABLE_SHA256,
        par_file_path=par_file_path,
        expected_axis_vectors={
            0: {
                "i": (100.0, 0.0, 0.0),
                "j": (0.0, 200.0, 0.0),
                "k": (0.0, 0.0, 400.0),
            }
        },
    )


# ===========================================================================
# Standalone displacement-snapshot fixture (for read_displacement_snapshots)
# ===========================================================================


def write_displacement_snapshot_files(
    dirpath,
    *,
    nspec: int,
    ngll: int,
    its: list[int],
    omit_component_for_last_it: str | None = None,
) -> np.ndarray:
    """Write a small standalone set of ``displ_{X,Y,Z}_it######.bin`` files
    (independent of the main mesh fixture -- arbitrary ``nspec``/``ngll``)
    for exercising :func:`gragra.adapters.specfem.specfem_bin.
    read_displacement_snapshots` directly.

    If ``omit_component_for_last_it`` is one of ``"X"``/``"Y"``/``"Z"``,
    that component's file is *not* written for the last ``it`` in ``its``
    (to test the missing-component / inconsistent-it-set error path); the
    returned expected array is still the full (all-components) content for
    documentation purposes -- callers testing the omission path should not
    compare against it for that ``it``.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    rng_like = np.arange(nspec * ngll, dtype=np.float64).reshape((nspec, ngll))

    expected = np.empty((len(its), nspec, ngll, 3), dtype=np.float64)
    for t, it in enumerate(its):
        is_last = t == len(its) - 1
        for comp_idx, comp in enumerate(("X", "Y", "Z")):
            values = (rng_like + it * 1000.0 + comp_idx * 10.0).astype(np.float32)
            expected[t, :, :, comp_idx] = values.astype(np.float64)
            if is_last and comp == omit_component_for_last_it:
                continue
            path = dirpath / f"displ_{comp}_it{it:06d}.bin"
            _write_single_record(path, values.astype("<f4").tobytes())
    return expected


# ===========================================================================
# Per-rank displacement-snapshot fixture (movie volume, PRIMARY real-pilot
# layout -- proc{rank:06d}_displ_{X,Y,Z}_it######.bin, 2026-08-03 findings)
# ===========================================================================


def write_rank_displacement_snapshot_files(
    info: SyntheticMeshInfo, *, its: list[int]
) -> np.ndarray:
    """Write ``proc{rank:06d}_displ_{X,Y,Z}_it######.bin`` movie volume
    snapshot files for every rank of ``info`` -- the **primary**, real
    pilot-verified (2026-08-03) per-rank layout (see
    ``gragra.adapters.specfem.specfem_bin``'s section-3 module comment for
    the two on-disk layouts and their provenance).

    Written into ``info.local_path`` (the same ``LOCAL_PATH`` directory that
    already holds ``proc*_{ibool,x,y,z,rho}.bin`` and ``proc*_Database``),
    using each rank's local element order -- ``info.elements_for_rank(rank)``
    (ascending) -- the exact same order ``_write_rank_database``/
    ``_write_rank_bin_files`` already use for this rank's Database/ibool/rho
    files. This mirrors the real invariant this fixture exists to exercise:
    the movie volume binary's rank-local element order equals this rank's
    ``proc*_Database`` element block order (so ``recover_element_ids``'
    per-rank ``element_id`` array is the correct permutation to scatter it
    into the canonical mesh order with).

    Values are a deterministic, non-trivial function of
    (rank, local element index, GLL flat index, component, ``it``) -- not a
    uniform or all-zero fixture -- so a test comparing the exporter's
    element_id-ascending re-combination against the returned ground-truth
    array actually exercises the element_id permutation: a bug that left a
    rank's data in rank-local order, or swapped two ranks' data, would not
    be caught by a shape/dtype-only check but would fail a value comparison
    against this fixture's ``expected`` array.

    Returns
    -------
    expected : np.ndarray, shape (len(its), n_elements_total, ngll, 3), float64
        Ground truth indexed by GLOBAL (canonical, element_id-ascending)
        element index -- what ``/field/displacement`` should equal after
        the exporter's per-rank read + element_id-based scatter.
    """
    ngll = info.ngll
    n_elements_total = info.n_elements_total
    expected = np.empty((len(its), n_elements_total, ngll, 3), dtype=np.float64)

    for rank in range(info.nproc):
        global_ids_this_rank = info.elements_for_rank(rank)
        n_local = len(global_ids_this_rank)
        # (n_local, 1) / (1, ngll) broadcast grids. Kept as small integer
        # multiples of rank/local-index/gll-index/it (well within float32's
        # 24-bit mantissa exact-integer range for this fixture's small
        # sizes) so the on-disk float32 round-trip is bit-exact and the test
        # can assert equality rather than approximate closeness.
        local_elem_idx = np.arange(n_local, dtype=np.float64).reshape(-1, 1)
        gll_idx = np.arange(ngll, dtype=np.float64).reshape(1, -1)
        for t, it in enumerate(its):
            for comp_idx, comp in enumerate(("X", "Y", "Z")):
                values = (
                    rank * 100_000.0
                    + local_elem_idx * 1_000.0
                    + gll_idx * 10.0
                    + float(comp_idx)
                    + it * 1_000_000.0
                ).astype(np.float32)
                path = info.local_path / f"proc{rank:06d}_displ_{comp}_it{it:06d}.bin"
                _write_single_record(path, values.astype("<f4").tobytes())
                expected[t, global_ids_this_rank, :, comp_idx] = values.astype(
                    np.float64
                )

    return expected


def write_rank_displacement_snapshot_files_cg_continuous(
    info: SyntheticMeshInfo, *, its: list[int]
) -> np.ndarray:
    """Like :func:`write_rank_displacement_snapshot_files`, but the written
    values are a deterministic function of each GLL point's PHYSICAL
    POSITION (never of rank or local-element index) -- so any two elements
    sharing a face necessarily produce bit-identical displacement at their
    shared GLL points.

    This distinction matters for Tier A4's CG-continuity check
    (``tests/_tier_a_checks.py:_check_cg_continuity``): the plain
    ``write_rank_displacement_snapshot_files`` formula is keyed on
    ``(rank, local_elem_idx, gll_idx)``, so E0/E1's shared-face points (which
    belong to different local element rows) get *different* values by
    construction -- confirmed empirically (2026-08-04 investigation) to make
    ``check_a4_masks_scalars_provenance`` raise ``ValueError: CG continuity
    violated`` on that fixture, rather than exercise the intended "should
    PASS for genuinely continuous data" path. This function exists so
    ``tests/test_specfem_tier_a.py``'s
    ``test_tier_a4_cg_continuity_displacement_smoke`` check can exercise the
    displacement branch of ``_check_cg_continuity`` on data that is actually
    CG-continuous, as required by the scientific continuity check.

    Uses ``_element_gll_grid(info.elements[gid].corners8)`` -- the same
    pure, deterministic function ``write_synthetic_specfem_tree`` used to
    build the mesh -- to recompute each element's physical GLL grid. Per
    that function's docstring, two elements sharing a face compute
    bit-identical grid values at the shared points; a value formula that
    only ever reads ``grid`` (never rank/local index) therefore inherits
    that bit-identical property elementwise (no summation-order dependence,
    since the formula is applied independently per grid point).

    Returns
    -------
    expected : np.ndarray, shape (len(its), n_elements_total, ngll, 3), float64
        Ground truth indexed by GLOBAL (canonical, element_id-ascending)
        element index, matching ``write_rank_displacement_snapshot_files``'s
        return convention.
    """
    ngll = info.ngll
    n_elements_total = info.n_elements_total
    expected = np.empty((len(its), n_elements_total, ngll, 3), dtype=np.float64)

    element_grids = {
        gid: _element_gll_grid(info.elements[gid].corners8)
        for gid in range(n_elements_total)
    }

    for rank in range(info.nproc):
        global_ids_this_rank = info.elements_for_rank(rank)
        # (n_local, ngll, 3) -- stacking preserves each element's own grid
        # values untouched (no cross-element arithmetic), so bit-identical
        # shared-point coordinates carry through to this array unchanged.
        grids_stacked = np.stack(
            [element_grids[gid] for gid in global_ids_this_rank], axis=0
        )
        position_term = (
            grids_stacked[:, :, 0]
            + 2.0 * grids_stacked[:, :, 1]
            + 3.0 * grids_stacked[:, :, 2]
        )
        for t, it in enumerate(its):
            for comp_idx, comp in enumerate(("X", "Y", "Z")):
                values = (position_term + float(comp_idx) + it * 1_000_000.0).astype(
                    np.float32
                )
                path = info.local_path / f"proc{rank:06d}_displ_{comp}_it{it:06d}.bin"
                _write_single_record(path, values.astype("<f4").tobytes())
                expected[t, global_ids_this_rank, :, comp_idx] = values.astype(
                    np.float64
                )

    return expected


# ===========================================================================
# Static fixture construction: build_static_fixture(tmp_path)
# ===========================================================================


@dataclass(frozen=True)
class _StaticFixture:
    """Return type matching the interface ``tests/test_specfem_tier_a.py``
    (the test helper assumes for
    ``build_static_fixture`` -- see that file's ``_synthetic_static_fixture``
    docstring for the exact attribute contract this must satisfy.
    """

    h5_path: Path
    rho_by_material: dict[int, float]
    rtol_by_material: dict[int, float]
    profile_density_range: tuple[float, float]
    expected_axis_vectors: dict[int, dict[str, tuple[float, float, float]]]


def build_static_fixture(tmp_path) -> _StaticFixture:
    """Build the synthetic *static-mode* canonical HDF5 fixture
    that ``tests/test_specfem_tier_a.py``'s numerical checks consume.

    Uses a function-level import of ``gragra.adapters.specfem.export`` so
    importing *this module* never depends on that subpackage's public API
    being available at collection time. ``ExportConfig`` receives the
    synthetic solver tree and auto-detects the local database layout.
    (``ExportConfig`` takes ``specfem_dir`` -- not a pre-resolved
    ``local_path`` -- and auto-detects ``nproc`` from the ``proc*_Database``
    files under the resolved ``LOCAL_PATH``). The synthetic tree has no real
    ``.git`` checkout or ``output_generate_databases.txt``, so
    ``build_identity``/``f_resolved_estimate`` (which a real SPECFEM3D
    checkout/run would otherwise supply) are given explicitly here instead
    of being auto-collected.
    """
    from gragra.adapters.specfem import export

    tree_root = Path(tmp_path) / "specfem_static_fixture_tree"
    info = write_synthetic_specfem_tree(tree_root, partition_variant="a")
    h5_path = Path(tmp_path) / "specfem_static.h5"

    config = export.ExportConfig(
        specfem_dir=info.root,
        output_path=h5_path,
        output_sampling_kind="static",
        ngll_shape=info.ngll_shape,
        build_identity={
            "build_kind": "package",
            "package_version": "gragra-synthetic-fixture",
            "package_digest": "sha256:" + hashlib.sha256(b"synthetic").hexdigest(),
            "build_configuration": {},
        },
        exporter_code_ref="gragra-synthetic-fixture",
        f_resolved_estimate=1.0,
    )
    export.export_specfem_gll_hdf5(config)

    return _StaticFixture(
        h5_path=h5_path,
        rho_by_material=dict(info.rho_by_material),
        # Two tolerance regimes for the A3 numerical check: exact
        # profile density (material 1, stored bit-for-bit) ~1e-12; the
        # deliberately 1-ulp-perturbed derived-density stand-in (material 2)
        # ~1e-6 (systematic bias, not averaged away -- corrected 2026-08-05
        # from a prior 2e-7 declaration that sat below the measured
        # worst-case derived-density bias of 3.0e-7, see
        # docs/validation/tier_error_budget.md §2 M7).
        rtol_by_material={1: 1e-12, 2: 1e-6},
        profile_density_range=(1900.0, 3100.0),
        # Element row 0 is expected to be E0 (assuming the exporter writes
        # /mesh/* rows in element_id-ascending order, consistent with the
        # element_id-ascending canonical summation convention used
        # throughout Tier A -- schema.py / _tier_a_checks.py).
        expected_axis_vectors=dict(info.expected_axis_vectors),
    )
