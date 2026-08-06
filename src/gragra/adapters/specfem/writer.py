"""HDF5 writer for the SPECFEM3D Cartesian GLL certified export.

Implements the ``mesh_type="gll"`` canonical HDF5 write path: dataset
creation per ``schema.DATASET_SPECS``, the ``/metadata`` provenance encoding
dispatch (scalar attribute / JSON dataset / plain-text dataset -- schema.py
section 7), the required-provenance-key gate (delegating to ``schema.required_keys``
/ ``schema.conditionally_required_keys`` rather than a hand-written list),
and the 4 total-volume/mass scalar invariants (computed here, in
element_id-ascending canonical order, from the arrays the caller already
assembled).

Layout note (an importer/exporter must distinguish *file layout* from *API
shape*): ``/field/displacement``'s leading axis is **time**, i.e.
``(time, element, gll, 3)`` -- this is an on-disk *file layout* convention
("schema (a)"), not an instance of gragra's in-memory API shape convention
(component axis trailing, parametric axes immediately before it). Nothing
in this module returns an in-memory field array; it only writes on-disk
datasets, so the two conventions never collide here, but a reader built on
top of this schema must not assume the on-disk leading time axis is also
the in-memory convention's leading batch axis without an explicit
conversion step (external_field_schema.md Batch Axis §1's "storage layout
vs in-memory API shape" distinction applies to this schema too).

``h5py`` is imported only lazily, inside :func:`write_gll_hdf5`, following
the ``adapters/hdf5_field.py`` convention -- importing this module (or
``gragra``/``gragra.adapters.specfem``) must never load h5py as a side
effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np

from gragra.adapters.specfem import schema

#: pip install hint shared with adapters/hdf5_field.py's lazy-import
#: convention (kept identical so the two modules give one consistent error
#: message across the codebase).
_H5PY_INSTALL_HINT: Final[str] = (
    "h5py is required for HDF5 field features. "
    'Please install it using: pip install "gragra[field-io]"'
)


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(_H5PY_INSTALL_HINT) from exc
    return h5py


def _write_provenance(meta_group, values: Mapping[str, object], h5py_mod) -> None:
    """Write a flat provenance dict to ``/metadata``, dispatching each key's
    on-disk encoding via ``schema.provenance_encoding`` (the single source of
    truth -- this function never hardcodes which keys are attributes vs.
    datasets).

    Keys absent from ``schema.ALL_PROVENANCE_KEYS`` (caller-supplied,
    documentation-only extras such as the ``*_source`` provenance tags this
    exporter records -- e.g. ``material_id_source``) are not silently
    dropped: they default to a scalar attribute for HDF5/JSON-primitive
    values, or a canonical-JSON dataset otherwise.
    """
    str_dtype = h5py_mod.string_dtype(encoding="utf-8")
    for key, value in values.items():
        try:
            encoding = schema.provenance_encoding(key)
        except KeyError:
            is_primitive = isinstance(value, (str, int, float, bool, np.generic))
            encoding = "scalar_attr" if is_primitive else "json_dataset"

        if encoding == "scalar_attr":
            if key == "ngll_shape":
                meta_group.attrs[key] = np.asarray(value, dtype=np.int64)
            else:
                meta_group.attrs[key] = value
        elif encoding == "json_dataset":
            meta_group.create_dataset(
                key, data=schema.canonical_json(value), dtype=str_dtype
            )
        elif encoding == "text_dataset":
            meta_group.create_dataset(key, data=str(value), dtype=str_dtype)
        else:  # pragma: no cover -- unreachable unless schema.py adds a new
            # ProvenanceEncoding value without updating this dispatcher.
            raise AssertionError(f"unhandled provenance encoding {encoding!r}")


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values -- refusing to write")


def _resolve_chunk_element_count(n_element: int, chunk_element_size: int | None) -> int:
    """Chunk size along the element axis: the caller's request, clamped into
    ``[1, n_element]`` (an oversized request degrades to "one chunk"; a
    caller-omitted request defaults to "one chunk" too -- both are
    conservative, valid h5py chunk shapes)."""
    if n_element <= 0:
        return 1
    requested = chunk_element_size if chunk_element_size else n_element
    return max(1, min(int(requested), n_element))


def write_gll_hdf5(
    output_path,
    *,
    output_sampling_kind: str,
    ngll_shape: tuple[int, int, int],
    coordinates: np.ndarray,
    quadrature_volume: np.ndarray,
    element_id: np.ndarray,
    partition_id: np.ndarray,
    domain_id: np.ndarray,
    material_id: np.ndarray,
    is_pml: np.ndarray,
    density: np.ndarray,
    provenance: Mapping[str, object],
    displacement: np.ndarray | None = None,
    time_step: np.ndarray | None = None,
    element_corner_nodes: np.ndarray | None = None,
    chunk_element_size: int | None = None,
    gzip_level: int = 4,
) -> Path:
    """Write one canonical ``mesh_type="gll"`` HDF5 file.

    All ``(element, ...)``-shaped arrays must already be in **element_id
    ascending canonical order** (``element_id == arange(n_element)``); this
    is a precondition this function asserts rather than silently re-sorting,
    so a caller-side ordering bug surfaces immediately instead of being
    papered over here (``export.py`` owns the multi-rank read + sort that
    establishes this order).

    This function derives and injects into a *copy* of ``provenance``
    (immutability -- the caller's mapping is never mutated) the facts that
    are mechanically computable from the arrays it already has: ``ngll``,
    ``ngll_shape``, ``n_elements_total``, ``element_id_min``/``_max``, and
    the four ``total_{quadrature_volume,mass}_{before,after}_exclusion``
    scalars (computed here in element_id-ascending order, per
    ``docs/design/specfem_import.md`` §Tier A A3/A4). The caller
    (``export.py``) must supply everything else (build identity, hashes,
    solver facts, ...) via ``provenance`` already.

    Method-scoped golden values (``docs/design/solver_import_contract.md``
    L355-363): ``element_id_min``/``_max`` and ``pml_mask_hash`` (and the
    element_id-ascending total_* scalars above) are ordering-dependent on
    ``provenance["element_id_method"]`` -- they are golden invariants only
    *within* a fixed ``element_id_method`` and are not comparable across a
    method change.

    ``output_sampling_kind == "time_sampled"`` requires ``displacement`` and
    ``time_step``; this function constructs ``/time`` itself as
    ``time_step * dt_s - t0`` (both read from the (already-validated)
    ``provenance`` dict) -- this construction makes the "/time is internally
    self-consistent with time_step/dt_s/t0" check tautological by
    definition; the independent verification (semd crosscheck) is Tier A/B's
    job, not this writer's.

    Missing required provenance keys raise ``ValueError`` generated from
    ``schema.required_keys(output_sampling_kind)`` /
    ``schema.conditionally_required_keys`` -- never a hand-written list, so
    a schema.py key addition/removal changes this check automatically.
    """
    h5py = _import_h5py()
    output_path = Path(output_path)

    element_id = np.asarray(element_id)
    n_element = element_id.shape[0]

    shape_checks = (
        ("coordinates", coordinates, n_element),
        ("quadrature_volume", quadrature_volume, n_element),
        ("partition_id", partition_id, n_element),
        ("domain_id", domain_id, n_element),
        ("material_id", material_id, n_element),
        ("is_pml", is_pml, n_element),
        ("density", density, n_element),
    )
    for name, arr, expected in shape_checks:
        if arr.shape[0] != expected:
            raise ValueError(
                f"{name} has leading dimension {arr.shape[0]}, expected "
                f"{expected} (== len(element_id))"
            )
    if is_pml.dtype != np.bool_:
        raise ValueError(f"is_pml dtype must be bool, got {is_pml.dtype}")
    if n_element and not np.array_equal(element_id, np.arange(n_element)):
        raise ValueError(
            "element_id must already be ascending 0..n_element-1 when passed "
            "to the writer (export.py's multi-rank read establishes this "
            "order -- see this function's docstring)"
        )

    for name, arr in (
        ("coordinates", coordinates),
        ("quadrature_volume", quadrature_volume),
        ("density", density),
    ):
        _check_finite(name, arr)

    ngll = schema.ngll_from_shape(ngll_shape)

    total_qvol_before = float(np.sum(quadrature_volume))
    total_mass_before = float(np.sum(density * quadrature_volume))
    physical_mask = ~is_pml
    total_qvol_after = float(np.sum(quadrature_volume[physical_mask]))
    total_mass_after = float(np.sum((density * quadrature_volume)[physical_mask]))

    full_provenance: dict[str, object] = dict(provenance)
    full_provenance.update(
        {
            "ngll": ngll,
            "ngll_shape": np.asarray(ngll_shape, dtype=np.int64),
            "n_elements_total": n_element,
            "element_id_min": int(element_id.min()) if n_element else 0,
            "element_id_max": int(element_id.max()) if n_element else 0,
            "total_quadrature_volume_before_exclusion": total_qvol_before,
            "total_quadrature_volume_after_exclusion": total_qvol_after,
            "total_mass_before_exclusion": total_mass_before,
            "total_mass_after_exclusion": total_mass_after,
        }
    )

    required = schema.required_keys(output_sampling_kind)
    missing = required - set(full_provenance)
    if missing:
        raise ValueError(f"Missing required provenance keys: {sorted(missing)}")
    extra_required = schema.conditionally_required_keys(full_provenance)
    missing_conditional = extra_required - set(full_provenance)
    if missing_conditional:
        raise ValueError(
            f"Missing conditionally-required provenance keys: "
            f"{sorted(missing_conditional)}"
        )
    if not schema.source_tree_dirty_hash_satisfied(full_provenance):
        raise ValueError(
            "build_kind == 'source' with a dirty source_tree requires "
            "'patch_hash' or 'source_tree_hash' in provenance"
        )

    element_id_method = full_provenance.get("element_id_method") or ""
    corner_nodes_required = schema.element_corner_nodes_required(element_id_method)
    if corner_nodes_required and element_corner_nodes is None:
        raise ValueError(
            "element_id_method == 'corner_node_tuple' requires "
            "element_corner_nodes to be supplied"
        )

    dataset_paths = schema.required_dataset_paths(output_sampling_kind)
    ce = _resolve_chunk_element_count(n_element, chunk_element_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:

        def _write(
            path: str, data: np.ndarray, extra_chunk_dims: tuple[int, ...]
        ) -> None:
            if path not in dataset_paths:
                return
            if n_element == 0:
                f.create_dataset(path, data=data)
                return
            f.create_dataset(
                path,
                data=data,
                chunks=(ce, *extra_chunk_dims),
                compression="gzip",
                compression_opts=gzip_level,
            )

        _write(schema.PATH_MESH_COORDINATES, coordinates, (ngll, 3))
        _write(schema.PATH_MESH_QUADRATURE_VOLUME, quadrature_volume, (ngll,))
        _write(schema.PATH_MESH_ELEMENT_ID, element_id, ())
        _write(schema.PATH_MESH_PARTITION_ID, partition_id, ())
        _write(schema.PATH_MESH_DOMAIN_ID, domain_id, ())
        _write(schema.PATH_MESH_MATERIAL_ID, material_id, ())
        _write(schema.PATH_MESH_IS_PML, is_pml, ())
        _write(schema.PATH_MATERIAL_DENSITY, density, (ngll,))
        if element_corner_nodes is not None:
            f.create_dataset(
                schema.PATH_MESH_ELEMENT_CORNER_NODES, data=element_corner_nodes
            )

        if output_sampling_kind == "time_sampled":
            if displacement is None or time_step is None:
                raise ValueError(
                    "output_sampling_kind='time_sampled' requires both "
                    "displacement and time_step"
                )
            n_time = displacement.shape[0]
            if time_step.shape != (n_time,):
                raise ValueError(
                    f"time_step shape {time_step.shape} must match "
                    f"displacement's leading time axis ({n_time},)"
                )
            _check_finite("displacement", displacement)
            if n_time >= 2:
                diffs = np.diff(time_step)
                if not np.all(diffs > 0):
                    raise ValueError("/time_step must be strictly increasing")
                m = full_provenance.get("output_interval_steps")
                if m is not None and not np.all(diffs == int(m)):
                    raise ValueError(
                        f"/time_step spacing must equal "
                        f"output_interval_steps={m}, got diffs={diffs.tolist()}"
                    )

            dt_s = float(full_provenance["dt_s"])
            t0 = float(full_provenance.get("t0", 0.0))
            # /time is *constructed* from time_step/dt_s/t0 (not read from
            # anywhere else), so this identity always holds by definition --
            # independent verification (semd crosscheck) is Tier A/B's job.
            time = time_step.astype(np.float64) * dt_s - t0

            f.create_dataset(
                schema.PATH_FIELD_DISPLACEMENT,
                data=displacement,
                chunks=(1, ce, ngll, 3),
                compression="gzip",
                compression_opts=gzip_level,
            )
            f.create_dataset(schema.PATH_TIME, data=time)
            f.create_dataset(schema.PATH_TIME_STEP, data=time_step)

        meta = f.create_group(schema.METADATA_GROUP)
        _write_provenance(meta, full_provenance, h5py)

    return output_path
