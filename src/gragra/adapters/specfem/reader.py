"""Reader for the canonical SPECFEM3D Cartesian GLL HDF5 schema.

The reader loads a closed, immutable in-memory representation of schema v1.
Rows remain ordered by ``element_id`` and GLL index; shared coordinates are
deliberately not merged because each element's quadrature contribution is a
separate source mass.
"""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gragra.adapters.specfem import schema


@dataclass(frozen=True)
class SpecfemGLLFileIdentity:
    """Stable identity facts captured when a GLL HDF5 file is read."""

    path: Path
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class SpecfemGLLBlock:
    """One contiguous element-id block from :class:`SpecfemGLLDataset`."""

    element_id: np.ndarray
    coordinates_m: np.ndarray
    mass_kg: np.ndarray
    displacement_m: np.ndarray | None


def _readonly(array: np.ndarray) -> np.ndarray:
    copied = np.asarray(array).copy()
    copied.setflags(write=False)
    return copied


def _import_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "h5py is required for SPECFEM GLL HDF5 input. "
            'Install it with: pip install "gragra[field-io]"'
        ) from error
    return h5py


def _decode_utf8_scalar(dataset, name: str, h5py_mod) -> str:
    """Read one scalar UTF-8 metadata dataset without coercion."""
    if dataset.shape != ():
        raise ValueError(f"/metadata/{name} must be a scalar dataset")
    string_info = h5py_mod.check_string_dtype(dataset.dtype)
    if string_info is None or string_info.encoding != "utf-8":
        raise ValueError(f"/metadata/{name} must be a UTF-8 string dataset")
    value = dataset[()]
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"/metadata/{name} must be valid UTF-8") from error
    raise ValueError(f"/metadata/{name} must be a UTF-8 string dataset")


def _read_provenance_value(metadata, name: str, encoding: str, h5py_mod):
    """Read a schema-defined provenance value from its sole valid location."""
    has_attr = name in metadata.attrs
    has_member = name in metadata
    if encoding == "scalar_attr":
        if has_member:
            raise ValueError(
                f"/metadata/{name} must be a scalar attribute, not a dataset"
            )
        if not has_attr:
            raise ValueError(f"Missing required /metadata attribute {name!r}")
        value = metadata.attrs[name]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"/metadata attribute {name!r} must be UTF-8"
                ) from error
        return value

    if has_attr:
        raise ValueError(f"/metadata/{name} must be a dataset, not an attribute")
    if not has_member:
        raise ValueError(f"Missing required /metadata dataset {name!r}")
    member = metadata[name]
    if not hasattr(member, "dtype"):
        raise ValueError(f"/metadata/{name} must be a dataset")
    text = _decode_utf8_scalar(member, name, h5py_mod)
    if encoding == "text_dataset":
        return text
    if encoding == "json_dataset":
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"/metadata/{name} must contain valid JSON") from error
    raise AssertionError(f"unknown provenance encoding {encoding!r}")


def _read_required_provenance(metadata, h5py_mod) -> dict[str, object]:
    """Decode and type-check the complete rc1 time-sampled provenance set."""
    output_sampling_kind = _read_provenance_value(
        metadata,
        "output_sampling_kind",
        schema.provenance_encoding("output_sampling_kind"),
        h5py_mod,
    )
    if output_sampling_kind != "time_sampled":
        raise ValueError("rc1 reader requires output_sampling_kind='time_sampled'")

    required = set(schema.required_keys(output_sampling_kind))
    provenance: dict[str, object] = {}
    while True:
        for key in required - set(provenance):
            provenance[key] = _read_provenance_value(
                metadata, key, schema.provenance_encoding(key), h5py_mod
            )
        expanded = required | set(schema.conditionally_required_keys(provenance))
        if expanded == required:
            break
        required = expanded

    if provenance.get("f_content_max_method") in {
        "filter_cutoff",
        "source_spectrum_bound",
    } and (
        "f_content_max_evidence" in metadata.attrs
        or "f_content_max_evidence" in metadata
    ):
        provenance["f_content_max_evidence"] = _read_provenance_value(
            metadata,
            "f_content_max_evidence",
            schema.provenance_encoding("f_content_max_evidence"),
            h5py_mod,
        )

    if (
        provenance.get("build_kind") == "source"
        and provenance.get("source_tree_clean") is False
    ):
        for key in ("patch_hash", "source_tree_hash"):
            if key in metadata.attrs or key in metadata:
                provenance[key] = _read_provenance_value(
                    metadata, key, schema.provenance_encoding(key), h5py_mod
                )

    type_checked_keys = set(required) | {
        key for key in ("patch_hash", "source_tree_hash") if key in provenance
    }
    for key in type_checked_keys:
        if not schema.check_value_type(key, provenance[key]):
            raise ValueError(
                f"provenance key {key!r} has unexpected type for value "
                f"{provenance[key]!r}"
            )
    if not schema.source_tree_dirty_hash_satisfied(provenance):
        raise ValueError(
            "build_kind == 'source' with a dirty source_tree requires "
            "'patch_hash' or 'source_tree_hash'"
        )
    return provenance


def _require_dataset(h5, path: str) -> np.ndarray:
    if path not in h5:
        raise ValueError(f"Missing required dataset {path!r}")
    dataset = h5[path]
    if not hasattr(dataset, "dtype"):
        raise ValueError(f"Required path {path!r} must be a dataset")
    spec = schema.DATASET_SPECS[path]
    if np.dtype(dataset.dtype) != spec.dtype:
        raise ValueError(
            f"{path} dtype must be {spec.dtype}, got {np.dtype(dataset.dtype)}"
        )
    return np.asarray(dataset[()])


def _require_shape(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")


def _validate_density_range(
    profile_density_range: tuple[float, float] | None, density: np.ndarray
) -> None:
    if profile_density_range is None:
        return
    if len(profile_density_range) != 2:
        raise ValueError("profile_density_range must contain exactly (low, high)")
    low, high = (float(value) for value in profile_density_range)
    if not np.isfinite((low, high)).all() or low > high:
        raise ValueError("profile_density_range must be finite with low <= high")
    if np.any(density < low) or np.any(density > high):
        raise ValueError(
            f"density is outside the supplied profile_density_range ({low}, {high})"
        )


def _validate_provenance_against_mesh(
    provenance: dict[str, object],
    *,
    element_id: np.ndarray,
    partition_id: np.ndarray,
    quadrature_volume: np.ndarray,
    density: np.ndarray,
    is_pml: np.ndarray,
) -> None:
    """Check the required mesh-derived provenance scalars against the file."""
    n_elements = element_id.size
    if int(provenance["n_elements_total"]) != n_elements:
        raise ValueError("metadata n_elements_total does not match /mesh/element_id")
    if int(provenance["element_id_min"]) != int(element_id.min()):
        raise ValueError("metadata element_id_min does not match /mesh/element_id")
    if int(provenance["element_id_max"]) != int(element_id.max()):
        raise ValueError("metadata element_id_max does not match /mesh/element_id")
    nproc = int(provenance["nproc"])
    if not np.isin(partition_id, np.arange(nproc, dtype=np.int32)).all():
        raise ValueError("/mesh/partition_id must be in [0, metadata nproc)")
    if provenance["pml_exclusion_applied"] is not True:
        raise ValueError("rc1 reader requires pml_exclusion_applied=True")
    expected_pml_hash = (
        "sha256:" + hashlib.sha256(is_pml.astype(np.uint8).tobytes()).hexdigest()
    )
    if provenance["pml_mask_hash"] != expected_pml_hash:
        raise ValueError("metadata pml_mask_hash does not match /mesh/is_pml")

    computed = {
        "total_quadrature_volume_before_exclusion": float(np.sum(quadrature_volume)),
        "total_quadrature_volume_after_exclusion": float(
            np.sum(quadrature_volume[~is_pml])
        ),
        "total_mass_before_exclusion": float(np.sum(density * quadrature_volume)),
        "total_mass_after_exclusion": float(
            np.sum((density * quadrature_volume)[~is_pml])
        ),
    }
    for name, expected in computed.items():
        actual = float(provenance[name])
        matches = np.isclose(actual, expected, rtol=1e-9, atol=0.0)
        if not np.isfinite(actual) or not matches:
            raise ValueError(f"metadata {name} does not match mesh data")


def _validate_cadence(provenance: dict[str, object], time_step: np.ndarray) -> None:
    """Validate deterministic cadence bounds for a time-sampled rc1 file."""
    dt_s = float(provenance["dt_s"])
    interval = int(provenance["output_interval_steps"])
    f_max = float(provenance["f_max_analysis"])
    f_min = float(provenance["f_min_analysis"])
    cycles = float(provenance["minimum_cycles"])
    oversampling = float(provenance["sampling_oversampling_factor"])
    values = (dt_s, f_max, f_min, cycles, oversampling)
    if not np.isfinite(values).all() or min(values) <= 0.0:
        raise ValueError("cadence values must be finite and positive")
    if f_min >= f_max:
        raise ValueError("f_min_analysis must be less than f_max_analysis")
    if oversampling < 2.0:
        raise ValueError("sampling_oversampling_factor must be at least 2")

    sampling_frequency = 1.0 / (interval * dt_s)
    if sampling_frequency < oversampling * f_max:
        raise ValueError("sampling frequency is below q * f_max_analysis")
    record_duration = float(time_step[-1] - time_step[0]) * dt_s
    if record_duration < cycles / f_min:
        raise ValueError("record duration is below minimum_cycles / f_min_analysis")

    if interval == 1:
        return
    f_content = float(provenance["f_content_max"])
    if not np.isfinite(f_content) or not 0.0 < f_content <= 1.0 / (2.0 * dt_s):
        raise ValueError("f_content_max must be within the solver Nyquist band")
    if sampling_frequency < 2.0 * f_content:
        raise ValueError("sampling frequency is below 2 * f_content_max")
    method = provenance["f_content_max_method"]
    allowed_methods = {
        "solver_step_nyquist",
        "measured_spectrum",
        "filter_cutoff",
        "analytic_manufactured",
        "source_spectrum_bound",
    }
    if method not in allowed_methods:
        raise ValueError("metadata f_content_max_method is invalid")
    solver_nyquist = 1.0 / (2.0 * dt_s)
    if method == "solver_step_nyquist" and not np.isclose(
        f_content, solver_nyquist, rtol=1e-12, atol=0.0
    ):
        raise ValueError(
            "solver_step_nyquist requires f_content_max equal to the solver Nyquist"
        )
    if method == "filter_cutoff" and provenance["antialias_filter_applied"] is not True:
        raise ValueError("filter_cutoff requires antialias_filter_applied=True")
    if method != "solver_step_nyquist":
        budget = float(provenance["alias_error_budget"])
        if not np.isfinite(budget) or not 0.0 < budget <= 1.0:
            raise ValueError("alias_error_budget must be in (0, 1]")
    if method in {"measured_spectrum", "filter_cutoff", "source_spectrum_bound"}:
        evidence = provenance.get("f_content_max_evidence")
        required_evidence = {
            "channel_scope",
            "measurement",
            "criterion",
            "artifact_hash",
            "full_rate_run",
        }
        if not isinstance(evidence, dict) or not required_evidence <= set(evidence):
            raise ValueError("f_content_max_evidence is incomplete")
        if not schema.is_hash_string(evidence["artifact_hash"]):
            raise ValueError("f_content_max_evidence artifact_hash is invalid")


@dataclass(frozen=True)
class SpecfemGLLDataset:
    """Closed immutable canonical GLL data, ordered by element then GLL.

    ``mass_kg`` is not stored independently.  It is always reconstructed as
    ``density_kg_m3 * quadrature_volume_m3`` so the unit conversion is
    visible at the reader boundary.  rc1 loads all time snapshots before
    closing the HDF5 file; it is not a low-memory streaming reader.
    """

    file_identity: SpecfemGLLFileIdentity
    time_s: np.ndarray
    time_step: np.ndarray
    element_id: np.ndarray
    coordinates_m: np.ndarray
    quadrature_volume_m3: np.ndarray
    density_kg_m3: np.ndarray
    displacement_m: np.ndarray | None

    def iter_blocks(self, *, element_block_size: int | None = None):
        """Yield contiguous element-id blocks without merging shared GLL points."""
        n_elements = self.element_id.size
        if element_block_size is None:
            block_size = max(1, n_elements)
        else:
            if isinstance(element_block_size, (bool, np.bool_)):
                raise ValueError("element_block_size must be a positive integer")
            try:
                block_size = operator.index(element_block_size)
            except TypeError as error:
                raise ValueError(
                    "element_block_size must be a positive integer"
                ) from error
            if block_size <= 0:
                raise ValueError("element_block_size must be a positive integer")

        for start in range(0, n_elements, block_size):
            stop = min(start + block_size, n_elements)
            mass = _readonly(
                self.density_kg_m3[start:stop] * self.quadrature_volume_m3[start:stop]
            )
            displacement = self.displacement_m
            if displacement is not None:
                displacement = displacement[:, start:stop]
            yield SpecfemGLLBlock(
                element_id=self.element_id[start:stop],
                coordinates_m=self.coordinates_m[start:stop],
                mass_kg=mass,
                displacement_m=displacement,
            )


def read_specfem_gll_hdf5(
    path: str | Path,
    *,
    profile_density_range: tuple[float, float] | None,
) -> SpecfemGLLDataset:
    """Read a canonical schema-v1 SPECFEM GLL file into closed immutable arrays.

    The certified rc1 profile accepts only time-sampled, elastic, single-material,
    non-PML data with ``NGNOD=8`` exporter output.  ``profile_density_range`` is
    deliberately required: passing ``None`` explicitly documents that an external
    density-profile range is unavailable for this input.
    """
    h5py = _import_h5py()
    resolved_path = Path(path).expanduser().resolve()

    with h5py.File(resolved_path, "r") as h5:
        if schema.METADATA_GROUP not in h5:
            raise ValueError("Missing required /metadata group")
        metadata = h5[schema.METADATA_GROUP]
        provenance = _read_required_provenance(metadata, h5py)
        if provenance["schema_version"] != schema.SCHEMA_VERSION:
            raise ValueError("Unsupported SPECFEM GLL schema_version")
        if provenance["mesh_type"] != schema.MESH_TYPE:
            raise ValueError("Expected metadata mesh_type='gll'")
        if provenance["layout"] != schema.LAYOUT_SINGLE_FILE:
            raise ValueError("Expected metadata layout='single_file'")
        if provenance["solver_kind"] != "specfem3d_cartesian":
            raise ValueError("rc1 reader requires solver_kind='specfem3d_cartesian'")
        if provenance["coordinate_convention"] != "ENU":
            raise ValueError("metadata coordinate_convention must be 'ENU'")
        if provenance["element_id_method"] != schema.ADOPTED_ELEMENT_ID_METHOD:
            raise ValueError("rc1 reader requires the adopted element_id_method")

        ngll_shape_array = np.asarray(provenance["ngll_shape"])
        ngll_shape = tuple(int(value) for value in ngll_shape_array)
        ngll = schema.ngll_from_shape(ngll_shape)
        if int(provenance["ngll"]) != ngll:
            raise ValueError("metadata ngll must match metadata ngll_shape")
        dt_s = float(provenance["dt_s"])
        t0 = float(provenance["t0"])
        output_interval_steps = int(provenance["output_interval_steps"])
        if not np.isfinite((dt_s, t0)).all() or dt_s <= 0.0:
            raise ValueError("metadata dt_s must be finite and positive")
        if output_interval_steps <= 0:
            raise ValueError("metadata output_interval_steps must be positive")
        if int(provenance["nproc"]) <= 0:
            raise ValueError("metadata nproc must be positive")
        if int(provenance["nstep"]) < 0:
            raise ValueError("metadata nstep must be non-negative")
        if provenance["time_step_origin"] != "zero_based_normalized":
            raise ValueError("rc1 reader requires zero-based normalized time steps")
        if int(provenance["native_step_origin"]) not in (0, 1):
            raise ValueError("metadata native_step_origin must be 0 or 1")
        if float(provenance["sampling_oversampling_factor"]) < 2.0:
            raise ValueError("sampling_oversampling_factor must be at least 2")
        if provenance["cadence_provenance"] not in {"placeholder", "confirmed"}:
            raise ValueError("metadata cadence_provenance is invalid")

        dataset_paths = schema.required_dataset_paths("time_sampled")
        arrays = {path: _require_dataset(h5, path) for path in dataset_paths}

        coordinates = arrays[schema.PATH_MESH_COORDINATES]
        quadrature_volume = arrays[schema.PATH_MESH_QUADRATURE_VOLUME]
        element_id = arrays[schema.PATH_MESH_ELEMENT_ID]
        partition_id = arrays[schema.PATH_MESH_PARTITION_ID]
        domain_id = arrays[schema.PATH_MESH_DOMAIN_ID]
        material_id = arrays[schema.PATH_MESH_MATERIAL_ID]
        is_pml = arrays[schema.PATH_MESH_IS_PML]
        density = arrays[schema.PATH_MATERIAL_DENSITY]
        displacement = arrays[schema.PATH_FIELD_DISPLACEMENT]
        time_s = arrays[schema.PATH_TIME]
        time_step = arrays[schema.PATH_TIME_STEP]

    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("/mesh/coordinates must have shape (element, gll, 3)")
    n_elements, file_ngll, _ = coordinates.shape
    if n_elements == 0:
        raise ValueError("SPECFEM GLL input must contain at least one element")
    if file_ngll != ngll:
        raise ValueError(
            f"GLL axis has length {file_ngll}, but ngll_shape implies {ngll}"
        )
    _require_shape("/mesh/quadrature_volume", quadrature_volume, (n_elements, ngll))
    _require_shape("/material/density", density, (n_elements, ngll))
    _require_shape("/mesh/element_id", element_id, (n_elements,))
    _require_shape("/mesh/partition_id", partition_id, (n_elements,))
    _require_shape("/mesh/domain_id", domain_id, (n_elements,))
    _require_shape("/mesh/material_id", material_id, (n_elements,))
    _require_shape("/mesh/is_pml", is_pml, (n_elements,))
    if is_pml.dtype != np.bool_:
        raise ValueError(f"/mesh/is_pml dtype must be bool, got {is_pml.dtype}")
    if not np.array_equal(element_id, np.arange(n_elements, dtype=element_id.dtype)):
        raise ValueError("/mesh/element_id must be ascending 0..N-1")
    if np.any(is_pml):
        raise ValueError("PML elements are not supported by the rc1 reader")
    if not np.all(domain_id == schema.IDOMAIN_ELASTIC):
        raise ValueError("rc1 reader requires a single elastic domain")
    if np.unique(material_id).size != 1:
        raise ValueError("rc1 reader requires a single material")
    if np.any(material_id < 1):
        raise ValueError("/mesh/material_id must be one-based and positive")
    if not np.isfinite(coordinates).all() or not np.isfinite(quadrature_volume).all():
        raise ValueError("mesh coordinates and quadrature volume must be finite")
    if np.any(quadrature_volume <= 0.0):
        raise ValueError("quadrature volume must be positive")
    if not np.isfinite(density).all() or np.any(density <= 0.0):
        raise ValueError("density must be finite and positive")
    _validate_density_range(profile_density_range, density)
    _validate_provenance_against_mesh(
        provenance,
        element_id=element_id,
        partition_id=partition_id,
        quadrature_volume=quadrature_volume,
        density=density,
        is_pml=is_pml,
    )

    if displacement.ndim != 4 or displacement.shape[1:] != (n_elements, ngll, 3):
        raise ValueError("/field/displacement must have shape (time, element, gll, 3)")
    n_time = displacement.shape[0]
    _require_shape("/time", time_s, (n_time,))
    _require_shape("/time_step", time_step, (n_time,))
    if (
        n_time == 0
        or not np.isfinite(displacement).all()
        or not np.isfinite(time_s).all()
    ):
        raise ValueError("displacement and time must be finite and non-empty")
    if n_time > 1 and (
        np.any(np.diff(time_s) <= 0.0) or np.any(np.diff(time_step) <= 0)
    ):
        raise ValueError("/time and /time_step must be strictly increasing")
    if not np.allclose(time_s, time_step.astype(np.float64) * dt_s - t0):
        raise ValueError("/time must equal time_step * dt_s - t0")
    if n_time > 1 and not np.all(np.diff(time_step) == output_interval_steps):
        raise ValueError("/time_step spacing must equal output_interval_steps")
    _validate_cadence(provenance, time_step)

    stat = resolved_path.stat()
    return SpecfemGLLDataset(
        file_identity=SpecfemGLLFileIdentity(
            path=resolved_path,
            device=stat.st_dev,
            inode=stat.st_ino,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        ),
        time_s=_readonly(time_s.astype(np.float64, copy=False)),
        time_step=_readonly(time_step.astype(np.int64, copy=False)),
        element_id=_readonly(element_id.astype(np.int64, copy=False)),
        coordinates_m=_readonly(coordinates.astype(np.float64, copy=False)),
        quadrature_volume_m3=_readonly(
            quadrature_volume.astype(np.float64, copy=False)
        ),
        density_kg_m3=_readonly(density.astype(np.float64, copy=False)),
        displacement_m=_readonly(displacement.astype(np.float64, copy=False)),
    )
