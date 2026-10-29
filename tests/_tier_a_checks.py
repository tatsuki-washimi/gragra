"""Tier A (static) verification library for the SPECFEM3D GLL canonical HDF5.

Implements A1-A4 as described in ``docs/design/specfem_import.md`` §MVP完了
条件 and refined in the scientific tests.

Each ``check_a*`` function accepts either a filesystem path (``str`` /
``os.PathLike``) or an already-open ``h5py.File``/``h5py.Group`` and raises
``ValueError`` (with a diagnostic message identifying the offending element/
key) on the first failure it finds; a check that returns normally means it
passed.

IMPORTANT -- independence discipline (do not weaken this):

- This module **must never import** ``gragra.adapters.specfem.gll`` (the
  production GLL/Jacobian implementation, Step 3). A1's auxiliary GLL
  interior-position check and A2's oracle both need GLL-related reference
  values, and this module computes them itself (see ``_gll_nodes_fractional``
  and ``tests/_hex_volume_oracle.py``) rather than importing the code under
  test.
- ``gragra.adapters.specfem.schema`` **may** be imported (it is the
  numpy-only canonical schema definition, not the exporter/reader
  implementation).
- ``h5py`` is imported lazily, inside functions, per the
  ``adapters/hdf5_field.py`` convention (this module is a test helper, not
  part of the numpy-only core, but keeping the import lazy means importing
  this module does not require h5py to be installed).
"""

import json
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np
from _hex_volume_oracle import hex_volume_24tetra, hex_volume_gauss2

from gragra.adapters.specfem import schema

# ===========================================================================
# Shared helpers
# ===========================================================================


@contextmanager
def _open_h5(h5):
    """Yield an open h5py.File/Group for ``h5`` (a path or an already-open
    file/group). Closes the file itself only if this function opened it.
    """
    import h5py

    if isinstance(h5, (h5py.File, h5py.Group)):
        yield h5
        return
    f = h5py.File(h5, "r")
    try:
        yield f
    finally:
        f.close()


def _decode_scalar(value):
    """Decode an h5py attribute/dataset scalar into a plain Python value
    (bytes -> str, numpy scalar -> Python scalar, numpy array -> list)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_provenance(meta_group) -> dict[str, object]:
    """Flatten `/metadata` attributes + JSON-dataset/text-dataset children
    into a single ``{key: value}`` provenance mapping, per schema.py's
    scalar-attr / json_dataset / text_dataset encoding table.
    """
    import h5py

    provenance: dict[str, object] = {}
    for key, value in meta_group.attrs.items():
        provenance[key] = _decode_scalar(value)

    for name in meta_group:
        item = meta_group[name]
        if not isinstance(item, h5py.Dataset):
            continue
        raw = item[()]
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        try:
            encoding = schema.provenance_encoding(name)
        except KeyError:
            encoding = None
        if encoding == "json_dataset":
            provenance[name] = json.loads(text)
        else:
            # text_dataset (or an unrecognized dataset -- kept as raw text
            # rather than silently dropped, so A4's full-count check can
            # still see it if it happens to collide with a known key name).
            provenance[name] = text
    return provenance


# ===========================================================================
# A1: GLL ordering
# ===========================================================================


def _lgl_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Legendre-Gauss-Lobatto nodes/weights on [-1, 1] for n >= 3 points.

    Independent implementation (fixed-point iteration on the Legendre-
    Vandermonde matrix, the well-known "lglnodes" construction): the n-2
    interior nodes are the roots of P'_{n-1}, found via Newton-style
    iteration seeded from the Chebyshev-Gauss-Lobatto points; exact to
    polynomial degree 2n-3 (n=5 -> degree 7, matching gll.py's own Step 3
    analytic test expectations -- computed independently here, not
    cross-imported).
    """
    big_n = n - 1
    x = np.cos(np.pi * np.arange(n) / big_n)
    p = np.zeros((n, n), dtype=np.float64)
    x_old = np.full(n, 2.0)
    max_iter = 200
    for _ in range(max_iter):
        if np.max(np.abs(x - x_old)) <= np.finfo(np.float64).eps * 10:
            break
        x_old = x.copy()
        p[:, 0] = 1.0
        p[:, 1] = x
        for k in range(2, big_n + 1):
            p[:, k] = ((2 * k - 1) * x * p[:, k - 1] - (k - 1) * p[:, k - 2]) / k
        x = x_old - (x * p[:, big_n] - p[:, big_n - 1]) / (n * p[:, big_n])
    else:
        raise RuntimeError(f"LGL node iteration did not converge for n={n}")
    weights = 2.0 / (big_n * n * p[:, big_n] ** 2)
    order = np.argsort(x)
    return x[order], weights[order]


def _gll_nodes_fractional(n: int) -> np.ndarray:
    """The n Gauss-Lobatto-Legendre abscissae, mapped from [-1, 1] to the
    fractional edge position [0, 1] (ascending). Used only by A1's auxiliary
    edge-interior-position check.
    """
    if n < 2:
        raise ValueError(f"GLL requires at least 2 points (endpoints), got n={n}")
    if n == 2:
        return np.array([0.0, 1.0], dtype=np.float64)
    x, _w = _lgl_nodes_weights(n)
    return (x + 1.0) / 2.0


def _check_element_edges(elem_grid, nodes_x, nodes_y, nodes_z, atol, elem_index):
    """Auxiliary check for one element: every GLL point along each of the
    12 index-grid edges must sit at the GLL-fractional interpolated
    position between that edge's two corner points. This is an *exact*
    (not merely warp-tolerant) identity: a trilinear hexahedron's edges are
    exactly straight lines (fixing two of xi/eta/zeta at +-1 collapses the
    shape functions to a purely linear function of the third coordinate),
    so ``atol`` here should only need to absorb storage precision, not
    warp-induced curvature.

    Note (per the S1 plan's A1 redesign): this check alone does **not**
    disambiguate the 48 index-cube symmetries for an isotropic element --
    see ``_check_axis_fixed_directions`` for the primary, disambiguating
    check.
    """
    nz, ny, nx, _ = elem_grid.shape

    for j in (0, ny - 1):
        for k in (0, nz - 1):
            p0 = elem_grid[k, j, 0]
            p1 = elem_grid[k, j, nx - 1]
            for i, t in enumerate(nodes_x):
                expected = p0 + t * (p1 - p0)
                actual = elem_grid[k, j, i]
                if not np.allclose(actual, expected, atol=atol, rtol=0.0):
                    raise ValueError(
                        f"element {elem_index}: x-edge (j={j},k={k}) GLL "
                        f"point i={i} at {actual} deviates from expected "
                        f"interpolated position {expected} by more than "
                        f"atol={atol}"
                    )

    for i in (0, nx - 1):
        for k in (0, nz - 1):
            p0 = elem_grid[k, 0, i]
            p1 = elem_grid[k, ny - 1, i]
            for j, t in enumerate(nodes_y):
                expected = p0 + t * (p1 - p0)
                actual = elem_grid[k, j, i]
                if not np.allclose(actual, expected, atol=atol, rtol=0.0):
                    raise ValueError(
                        f"element {elem_index}: y-edge (i={i},k={k}) GLL "
                        f"point j={j} at {actual} deviates from expected "
                        f"interpolated position {expected} by more than "
                        f"atol={atol}"
                    )

    for i in (0, nx - 1):
        for j in (0, ny - 1):
            p0 = elem_grid[0, j, i]
            p1 = elem_grid[nz - 1, j, i]
            for k, t in enumerate(nodes_z):
                expected = p0 + t * (p1 - p0)
                actual = elem_grid[k, j, i]
                if not np.allclose(actual, expected, atol=atol, rtol=0.0):
                    raise ValueError(
                        f"element {elem_index}: z-edge (i={i},j={j}) GLL "
                        f"point k={k} at {actual} deviates from expected "
                        f"interpolated position {expected} by more than "
                        f"atol={atol}"
                    )


_AXIS_NAMES = ("i", "j", "k")


def _axis_span_endpoints(axis_name, elem, nx, ny, nz):
    """Return (p0, p1), the two GLL points spanning ``elem`` end-to-end
    along the given index axis ("i"=x, "j"=y, "k"=z)."""
    origin = elem[0, 0, 0]
    if axis_name == "i":
        return origin, elem[0, 0, nx - 1]
    if axis_name == "j":
        return origin, elem[0, ny - 1, 0]
    if axis_name == "k":
        return origin, elem[nz - 1, 0, 0]
    raise ValueError(f"Unknown axis name {axis_name!r}; expected one of {_AXIS_NAMES}")


def _check_axis_fixed_directions(grid, expected_axis_vectors, atol):
    """Primary, direction-fixed A1 check: for specific designated elements
    (keyed by their row position in ``/mesh/coordinates``, *not* necessarily
    their ``/mesh/element_id`` value), assert that the end-to-end
    displacement vector along a given index axis equals an expected vector.

    This is the check that fully disambiguates GLL ordering: for an element
    with three *distinct* edge lengths, only the correct (index axis) <->
    (physical direction) assignment (out of the 48 index-cube symmetries)
    produces the expected displacement vector -- an accidental
    transposition or axis flip produces a visibly different vector.

    Parameters
    ----------
    expected_axis_vectors : Mapping[int, Mapping[str, Sequence[float]]]
        ``{element_row: {"i"|"j"|"k": (dx, dy, dz), ...}, ...}``. Only
        elements present as keys are checked this way (other elements are
        only covered by the auxiliary edge-linearity check).
    """
    n_element, nz, ny, nx, _ = grid.shape
    for elem_row, axis_vectors in expected_axis_vectors.items():
        if not (0 <= elem_row < n_element):
            raise ValueError(
                f"expected_axis_vectors element row {elem_row} out of range "
                f"[0, {n_element})"
            )
        elem = grid[elem_row]
        for axis_name, expected_vec in axis_vectors.items():
            p0, p1 = _axis_span_endpoints(axis_name, elem, nx, ny, nz)
            actual_vec = p1 - p0
            expected_arr = np.asarray(expected_vec, dtype=np.float64)
            if not np.allclose(actual_vec, expected_arr, atol=atol, rtol=0.0):
                raise ValueError(
                    f"element row {elem_row}: axis {axis_name!r} "
                    f"end-to-end displacement {actual_vec.tolist()} does "
                    f"not match expected {expected_arr.tolist()} within "
                    f"atol={atol} (GLL ordering direction-fixed check "
                    "failed -- possible transposed/rotated axis assignment)"
                )


def check_a1_gll_ordering(
    h5,
    *,
    expected_axis_vectors: Mapping[int, Mapping[str, object]] | None = None,
    atol_rel_budget: float = 3 * 2**-23,
) -> None:
    """A1: verify the GLL axis ordering of ``/mesh/coordinates``.

    1. Reshapes each element's flat ``(gll, 3)`` coordinates using
       ``/metadata/ngll_shape`` (never assuming the total factors as
       ``n**3``, per ``schema.reshape_shape_zyx``).
    2. Runs the auxiliary edge-linearity + GLL-interior-position check on
       every element (exact identity; tolerance only absorbs storage
       precision -- see ``_check_element_edges``). NOTE: for a fully
       isotropic element this check alone cannot reject all 48 index-cube
       symmetries (see the module-level and function-level notes) -- it is
       an auxiliary check, not the primary one.
    3. If ``expected_axis_vectors`` is given, runs the primary,
       disambiguating direction-fixed check (see
       ``_check_axis_fixed_directions``) against the designated element(s).

    ``atol_rel_budget`` defaults to a float32 storage-precision budget
    (``3 * 2**-23``); the effective ``atol`` is
    ``atol_rel_budget * max(|coordinates|, 1.0)``.
    """
    with _open_h5(h5) as f:
        ngll_shape = tuple(int(v) for v in f[schema.METADATA_GROUP].attrs["ngll_shape"])
        coords = f[schema.PATH_MESH_COORDINATES][()]

    n_element, ngll, three = coords.shape
    if three != 3:
        raise ValueError(f"/mesh/coordinates last axis must be 3, got {three}")
    expected_ngll = schema.ngll_from_shape(ngll_shape)
    if ngll != expected_ngll:
        raise ValueError(
            f"/mesh/coordinates gll axis {ngll} != ngll_shape product "
            f"{expected_ngll} (ngll_shape={ngll_shape})"
        )

    nx, ny, nz = ngll_shape
    reshape_shape = schema.reshape_shape_zyx(ngll_shape)
    grid = coords.reshape((n_element, *reshape_shape, 3))

    max_abs_coord = float(np.max(np.abs(coords))) if coords.size else 0.0
    atol = atol_rel_budget * max(max_abs_coord, 1.0)

    nodes_x = _gll_nodes_fractional(nx)
    nodes_y = _gll_nodes_fractional(ny)
    nodes_z = _gll_nodes_fractional(nz)
    for e in range(n_element):
        _check_element_edges(grid[e], nodes_x, nodes_y, nodes_z, atol, e)

    if expected_axis_vectors is not None:
        _check_axis_fixed_directions(grid, expected_axis_vectors, atol)


# ===========================================================================
# A2: element quadrature volume
# ===========================================================================


def _corner_flat_indices(ngll_shape: tuple[int, int, int]) -> list[int]:
    """Flat GLL indices of the 8 corner nodes, in usual_hex_nodes /
    VTK_HEXAHEDRON order (matching ``tests/_hex_volume_oracle.py``'s
    expected ``corners8`` order)."""
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


def check_a2_quadrature_volume(h5, *, rtol: float = 1e-12) -> None:
    """A2: for every element, ``sum(/mesh/quadrature_volume[e])`` must match
    the element's analytic 8-corner-hexahedron volume within ``rtol``, and
    the two independent oracles (``hex_volume_24tetra``, ``hex_volume_
    gauss2``) must agree with each other within the same ``rtol`` (oracle
    self-consistency, checked per element rather than once globally, so a
    single degenerate element cannot hide behind an otherwise-passing
    fixture).
    """
    with _open_h5(h5) as f:
        ngll_shape = tuple(int(v) for v in f[schema.METADATA_GROUP].attrs["ngll_shape"])
        coords = f[schema.PATH_MESH_COORDINATES][()]
        qvol = f[schema.PATH_MESH_QUADRATURE_VOLUME][()]

    n_element = coords.shape[0]
    corner_idx = _corner_flat_indices(ngll_shape)

    for e in range(n_element):
        corners8 = coords[e, corner_idx, :]
        vol_24tetra = hex_volume_24tetra(corners8)
        vol_gauss2 = hex_volume_gauss2(corners8)
        if vol_24tetra <= 0.0 or vol_gauss2 <= 0.0:
            raise ValueError(
                f"element {e}: non-positive oracle volume "
                f"(24tetra={vol_24tetra}, gauss2={vol_gauss2})"
            )
        oracle_scale = max(abs(vol_24tetra), abs(vol_gauss2))
        oracle_rel = abs(vol_24tetra - vol_gauss2) / oracle_scale
        if oracle_rel >= rtol:
            raise ValueError(
                f"element {e}: independent oracle mismatch "
                f"(24tetra={vol_24tetra}, gauss2={vol_gauss2}, "
                f"rel={oracle_rel}) exceeds rtol={rtol}"
            )

        summed = float(np.sum(qvol[e]))
        rel_main = abs(summed - vol_24tetra) / abs(vol_24tetra)
        if rel_main >= rtol:
            raise ValueError(
                f"element {e}: sum(quadrature_volume)={summed} does not "
                f"match oracle volume={vol_24tetra} within rtol={rtol} "
                f"(rel={rel_main})"
            )


# ===========================================================================
# A3: total mass (per-material partial-sum cross-check)
# ===========================================================================


def check_a3_material_mass(
    h5,
    *,
    rho_by_material: Mapping[int, float],
    rtol_by_material: Mapping[int, float],
) -> None:
    """A3: for every material present, ``material_id``-partitioned
    ``sum(density * quadrature_volume)`` must match
    ``rho_by_material[material_id] * sum(quadrature_volume)`` (for that
    material's elements) within ``rtol_by_material[material_id]``.

    Callers must declare a per-material tolerance (this function does not
    hardcode one): analytically, the residual is
    ``rho_stored / rho_profile - 1`` volume-weighted-averaged over the
    material's elements. Two regimes are expected in practice:
    exact-representation density (e.g. a synthetic fixture that stores the
    profile density bit-for-bit) gives a residual near machine epsilon
    (~1e-12); a *derived* density quantity (e.g. SPECFEM3D's
    ``rho_vp^2/(4/3 mu + kappa)``, or a fixture that deliberately perturbs
    one material's stored density by 1 ulp) gives a small but non-epsilon
    systematic bias that does *not* average away with more elements (it is
    a systematic offset in the derived quantity itself, not random
    per-element noise). The measured worst case for this derived-density
    regime is ``3.0e-7`` (5 float32 ops at half-ulp each, see
    ``docs/validation/tier_error_budget.md`` §2 M7); callers should declare
    ``rtol >= 1e-6`` (~3x safety margin) for materials in this regime -- a
    prior ``2e-7`` declaration sat *below* the measured worst case and
    risked false FAILs on correctly-implemented exporters.

    This function trusts ``/mesh/quadrature_volume`` as already verified by
    ``check_a2_quadrature_volume`` -- it does not re-derive volumes from the
    corner-node oracle.
    """
    with _open_h5(h5) as f:
        material_id = f[schema.PATH_MESH_MATERIAL_ID][()]
        density = f[schema.PATH_MATERIAL_DENSITY][()]
        qvol = f[schema.PATH_MESH_QUADRATURE_VOLUME][()]

    element_total_volume = np.sum(qvol, axis=1)
    element_total_mass = np.sum(density * qvol, axis=1)

    seen_materials = {int(v) for v in np.unique(material_id).tolist()}
    expected_materials = set(rho_by_material.keys())
    if seen_materials != expected_materials:
        raise ValueError(
            f"material_id values in file {sorted(seen_materials)} do not "
            f"match rho_by_material keys {sorted(expected_materials)}"
        )

    for mat_id, rho0 in rho_by_material.items():
        mask = material_id == mat_id
        if not np.any(mask):
            raise ValueError(f"material_id {mat_id} declared but has no elements")
        mat_volume = float(np.sum(element_total_volume[mask]))
        mat_mass_actual = float(np.sum(element_total_mass[mask]))
        mat_mass_expected = rho0 * mat_volume
        rtol = rtol_by_material[mat_id]
        rel = abs(mat_mass_actual - mat_mass_expected) / abs(mat_mass_expected)
        if rel >= rtol:
            raise ValueError(
                f"material_id {mat_id}: mass {mat_mass_actual} vs expected "
                f"{mat_mass_expected} (rho0={rho0}, volume={mat_volume}) "
                f"exceeds rtol={rtol} (rel={rel})"
            )


# ===========================================================================
# A4: masks / scalars / provenance
# ===========================================================================


def _check_cg_continuity(coords_flat, displacement_flat=None) -> None:
    """Group points by bit-exact coordinate value (a stand-in for a shared
    solver-side iglob, which is not itself stored in the HDF5 -- see the
    contract's note on this) and, if a displacement snapshot is supplied,
    assert that grouped points also carry bit-identical displacement.
    """
    groups: dict[bytes, list[int]] = {}
    for idx in range(coords_flat.shape[0]):
        key = coords_flat[idx].tobytes()
        groups.setdefault(key, []).append(idx)

    if displacement_flat is None:
        return

    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        base = displacement_flat[idxs[0]]
        for other in idxs[1:]:
            if not np.array_equal(displacement_flat[other], base):
                raise ValueError(
                    "CG continuity violated: points sharing coordinates "
                    f"(flat indices {idxs}) have different displacement "
                    f"values (index {idxs[0]} vs {other})"
                )


def check_a4_masks_scalars_provenance(
    h5, *, profile_density_range: tuple[float, float] | None
) -> None:
    """A4: mask/scalar/provenance integrity.

    - shape/value-domain checks for ``domain_id``/``material_id``/
      ``partition_id``/``is_pml``.
    - ``element_id`` integrity: no duplicates,
      ``n_elements_total == len(element_id)``,
      ``element_id_min``/``element_id_max`` match the dataset.
    - the 4 total-volume/mass scalars, recomputed in **element_id-ascending
      canonical summation order** and cross-checked against the recorded
      provenance values; verification rule 2 (before==after iff no
      exclusion) and rule 3 (average density within the caller-supplied
      profile range, if given).

    ``profile_density_range`` is a **required keyword** (no default) --
    this is a deliberate fail-closed contract, not an oversight. The
    exported canonical HDF5 does not itself carry this value (it is not
    derivable from the file alone; see
    ``docs/design/specfem_import.md``'s "profile_density_range の
    provenance キー化" gap entry, resolved 2026-08-05 in favor of keeping
    it a reader/caller-side external input rather than an exporter-written
    key). Passing ``None`` explicitly is how a caller declares "rule 3
    does not apply to this fixture" -- an *omitted* argument (which would
    previously default to ``None`` and silently skip rule 3) is now a
    ``TypeError`` instead, so a caller can no longer skip rule 3 by
    forgetting to pass this argument.
    - the full required-provenance-key audit, delegating entirely to
      ``schema.required_keys`` / ``schema.conditionally_required_keys`` /
      ``schema.source_tree_dirty_hash_satisfied`` / ``schema.
      check_value_type`` (never re-deriving the required-key set locally --
      this is the point of the single shared schema.py table).
    - ``element_id_method == "corner_node_tuple"``-specific checks (tuple
      uniqueness + lexicographic rank agreement), when applicable.
    - CG continuity (see ``_check_cg_continuity``).
    """
    with _open_h5(h5) as f:
        element_id = f[schema.PATH_MESH_ELEMENT_ID][()]
        partition_id = f[schema.PATH_MESH_PARTITION_ID][()]
        domain_id = f[schema.PATH_MESH_DOMAIN_ID][()]
        material_id = f[schema.PATH_MESH_MATERIAL_ID][()]
        is_pml = f[schema.PATH_MESH_IS_PML][()]
        qvol = f[schema.PATH_MESH_QUADRATURE_VOLUME][()]
        density = f[schema.PATH_MATERIAL_DENSITY][()]
        coords = f[schema.PATH_MESH_COORDINATES][()]
        corner_nodes = (
            f[schema.PATH_MESH_ELEMENT_CORNER_NODES][()]
            if schema.PATH_MESH_ELEMENT_CORNER_NODES in f
            else None
        )
        has_displacement = schema.PATH_FIELD_DISPLACEMENT in f
        displacement_snapshots: list[np.ndarray] = []
        if has_displacement:
            displacement_ds = f[schema.PATH_FIELD_DISPLACEMENT]
            displacement_snapshots = [
                displacement_ds[t][()] for t in range(displacement_ds.shape[0])
            ]
        meta = f[schema.METADATA_GROUP]
        nproc = int(meta.attrs["nproc"])
        provenance = _read_provenance(meta)

    n_element = len(element_id)
    output_sampling_kind = provenance.get("output_sampling_kind")

    # --- provenance required-key full audit (schema.py is the sole source
    # of truth; this function never re-derives the required-key set). This
    # runs FIRST, before any of the direct provenance[key] lookups below,
    # so a missing key always surfaces as a clean ValueError here rather
    # than a raw KeyError from deeper in this function. ---
    if output_sampling_kind is None:
        raise ValueError("Missing required provenance key: 'output_sampling_kind'")
    required = schema.required_keys(output_sampling_kind)
    missing = required - set(provenance.keys())
    if missing:
        raise ValueError(f"Missing required provenance keys: {sorted(missing)}")

    extra_required = schema.conditionally_required_keys(provenance)
    missing_conditional = extra_required - set(provenance.keys())
    if missing_conditional:
        raise ValueError(
            f"Missing conditionally-required provenance keys: "
            f"{sorted(missing_conditional)}"
        )

    if not schema.source_tree_dirty_hash_satisfied(provenance):
        raise ValueError(
            "build_kind == 'source' with a dirty tree, but neither "
            "'patch_hash' nor 'source_tree_hash' is present"
        )

    for key in required | extra_required:
        if key in provenance and not schema.check_value_type(key, provenance[key]):
            raise ValueError(
                f"provenance key {key!r} has unexpected type for value "
                f"{provenance[key]!r}"
            )

    # --- shape / value-domain checks ---
    for name, arr in (
        ("partition_id", partition_id),
        ("domain_id", domain_id),
        ("material_id", material_id),
        ("is_pml", is_pml),
    ):
        if arr.shape != (n_element,):
            raise ValueError(f"{name} shape {arr.shape} != ({n_element},)")

    if not np.isin(domain_id, list(schema.VALID_DOMAIN_IDS)).all():
        raise ValueError(
            f"domain_id contains values outside {sorted(schema.VALID_DOMAIN_IDS)}"
        )
    if (material_id < 1).any():
        raise ValueError("material_id must be one-based (>= 1)")
    if not np.isin(partition_id, np.arange(nproc)).all():
        raise ValueError(f"partition_id values must be in [0, {nproc})")
    if is_pml.dtype != np.bool_:
        raise ValueError(f"is_pml dtype must be bool, got {is_pml.dtype}")

    # --- element_id integrity ---
    if len(set(element_id.tolist())) != n_element:
        raise ValueError("element_id contains duplicates")
    n_elements_total = int(provenance["n_elements_total"])
    if n_elements_total != n_element:
        raise ValueError(
            f"n_elements_total provenance ({n_elements_total}) != "
            f"len(element_id) ({n_element})"
        )
    element_id_min = int(provenance["element_id_min"])
    element_id_max = int(provenance["element_id_max"])
    if element_id_min != int(element_id.min()) or element_id_max != int(
        element_id.max()
    ):
        raise ValueError("element_id_min/max provenance does not match dataset")

    # --- 4-scalar recomputation (canonical element_id-ascending order) ---
    order = np.argsort(element_id)
    qvol_ordered = qvol[order]
    density_ordered = density[order]
    is_pml_ordered = is_pml[order]

    total_qvol_before = float(np.sum(qvol_ordered))
    total_mass_before = float(np.sum(density_ordered * qvol_ordered))
    physical_mask = ~is_pml_ordered
    total_qvol_after = float(np.sum(qvol_ordered[physical_mask]))
    total_mass_after = float(np.sum((density_ordered * qvol_ordered)[physical_mask]))

    computed = {
        "total_quadrature_volume_before_exclusion": total_qvol_before,
        "total_quadrature_volume_after_exclusion": total_qvol_after,
        "total_mass_before_exclusion": total_mass_before,
        "total_mass_after_exclusion": total_mass_after,
    }
    for key, computed_val in computed.items():
        recorded_val = float(provenance[key])
        if not np.isfinite(recorded_val) or recorded_val <= 0.0:
            raise ValueError(f"{key} must be finite and positive, got {recorded_val}")
        denom = abs(computed_val) if computed_val != 0.0 else 1.0
        rel = abs(computed_val - recorded_val) / denom
        if rel >= 1e-9:
            raise ValueError(
                f"{key}: recorded {recorded_val} vs recomputed {computed_val} "
                f"mismatch (rel={rel})"
            )

    any_pml = bool(is_pml.any())
    vol_equal = total_qvol_before == total_qvol_after
    mass_equal = total_mass_before == total_mass_after
    if any_pml and (vol_equal or mass_equal):
        raise ValueError(
            "is_pml has True entries but before==after totals "
            "(exclusion rule 2 not satisfied)"
        )
    if (not any_pml) and not (vol_equal and mass_equal):
        raise ValueError(
            "is_pml is all False but before != after totals "
            "(spurious exclusion, rule 2 not satisfied)"
        )

    if profile_density_range is not None:
        lo, hi = profile_density_range
        avg_before = total_mass_before / total_qvol_before
        avg_after = total_mass_after / total_qvol_after
        if not (lo <= avg_before <= hi):
            raise ValueError(
                f"average density before-exclusion {avg_before} outside "
                f"profile range [{lo}, {hi}]"
            )
        if not (lo <= avg_after <= hi):
            raise ValueError(
                f"average density after-exclusion {avg_after} outside "
                f"profile range [{lo}, {hi}]"
            )

    # --- corner_node_tuple specific checks ---
    element_id_method = provenance.get("element_id_method")
    if element_id_method == schema.ELEMENT_ID_METHOD_CORNER_NODE_TUPLE:
        if corner_nodes is None:
            raise ValueError(
                "element_id_method is 'corner_node_tuple' but "
                "/mesh/element_corner_nodes is missing"
            )
        tuples = [tuple(sorted(row.tolist())) for row in corner_nodes]
        if len(set(tuples)) != len(tuples):
            raise ValueError("element_corner_nodes tuples contain duplicates")
        rank_of = {t: r for r, t in enumerate(sorted(tuples))}
        for e, t in enumerate(tuples):
            if rank_of[t] != int(element_id[e]):
                raise ValueError(
                    f"element {e}: lexicographic rank {rank_of[t]} != "
                    f"element_id {element_id[e]}"
                )

    # --- CG continuity ---
    coords_flat = coords.reshape(-1, 3)
    _check_cg_continuity(coords_flat)
    for disp_snapshot in displacement_snapshots:
        _check_cg_continuity(coords_flat, disp_snapshot.reshape(-1, 3))
