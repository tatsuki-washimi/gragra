"""Shared bounded validation for eager and streaming SPECFEM readers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from gragra.adapters.specfem import schema
from gragra.integrators.streaming import _bounded_slice, _positive_count


@dataclass(frozen=True)
class ValidatedSpecfemLayout:
    provenance: dict[str, object]
    n_elements: int
    ngll: int
    n_time: int
    sample_slice: slice
    time_s: np.ndarray
    time_step: np.ndarray
    element_id: np.ndarray


def _readonly(value: Any, dtype: Any = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _decode_utf8_scalar(dataset: Any, name: str, h5py_mod: Any) -> str:
    if dataset.shape != ():
        raise ValueError(f"/metadata/{name} must be a scalar dataset")
    info = h5py_mod.check_string_dtype(dataset.dtype)
    if info is None or info.encoding != "utf-8":
        raise ValueError(f"/metadata/{name} must be a UTF-8 string dataset")
    holder = np.empty((), dtype=dataset.dtype)
    dataset.read_direct(holder)
    value = holder[()]
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"/metadata/{name} must be valid UTF-8") from error
    raise ValueError(f"/metadata/{name} must be a UTF-8 string dataset")


def _read_provenance_value(metadata: Any, name: str, encoding: str, h5py_mod: Any):
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


def _read_required_provenance(metadata: Any, h5py_mod: Any) -> dict[str, object]:
    sampling = _read_provenance_value(
        metadata,
        "output_sampling_kind",
        schema.provenance_encoding("output_sampling_kind"),
        h5py_mod,
    )
    if sampling != "time_sampled":
        raise ValueError("SPECFEM reader requires output_sampling_kind='time_sampled'")
    required = set(schema.required_keys(sampling))
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
    checked = set(required) | {
        key for key in ("patch_hash", "source_tree_hash") if key in provenance
    }
    for key in checked:
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


def _require_dataset(h5: Any, path: str):
    if path not in h5:
        raise ValueError(f"Missing required dataset {path!r}")
    dataset = h5[path]
    if not hasattr(dataset, "dtype"):
        raise ValueError(f"Required path {path!r} must be a dataset")
    expected = schema.DATASET_SPECS[path].dtype
    if np.dtype(dataset.dtype) != expected:
        raise ValueError(
            f"{path} dtype must be {expected}, got {np.dtype(dataset.dtype)}"
        )
    return dataset


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


def _validate_cadence(provenance: dict[str, object], time_step: np.ndarray) -> None:
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
    if float(time_step[-1] - time_step[0]) * dt_s < cycles / f_min:
        raise ValueError("record duration is below minimum_cycles / f_min_analysis")
    if interval == 1:
        return
    f_content = float(provenance["f_content_max"])
    if not np.isfinite(f_content) or not 0.0 < f_content <= 1.0 / (2.0 * dt_s):
        raise ValueError("f_content_max must be within the solver Nyquist band")
    if sampling_frequency < 2.0 * f_content:
        raise ValueError("sampling frequency is below 2 * f_content_max")
    method = provenance["f_content_max_method"]
    allowed = {
        "solver_step_nyquist",
        "measured_spectrum",
        "filter_cutoff",
        "analytic_manufactured",
        "source_spectrum_bound",
    }
    if method not in allowed:
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
        fields = {
            "channel_scope",
            "measurement",
            "criterion",
            "artifact_hash",
            "full_rate_run",
        }
        if not isinstance(evidence, dict) or not fields <= set(evidence):
            raise ValueError("f_content_max_evidence is incomplete")
        if not schema.is_hash_string(evidence["artifact_hash"]):
            raise ValueError("f_content_max_evidence artifact_hash is invalid")


def validate_specfem_h5(
    h5: Any,
    h5py_mod: Any,
    *,
    profile_density_range: tuple[float, float] | None,
    slab_elements: int = 512,
    sample_slice: slice | None = None,
) -> ValidatedSpecfemLayout:
    """Validate the eager reader's full profile using bounded numeric slabs."""
    slab_elements = _positive_count(slab_elements, "slab_elements")
    if schema.METADATA_GROUP not in h5:
        raise ValueError("Missing required /metadata group")
    provenance = _read_required_provenance(h5[schema.METADATA_GROUP], h5py_mod)
    field_units = provenance.get("field_units")
    if isinstance(field_units, dict) and field_units.get("displacement") != "m":
        raise ValueError(
            "metadata field_units.displacement must be 'm' for the SPECFEM reader"
        )
    checks = (
        (
            "schema_version",
            schema.SCHEMA_VERSION,
            "Unsupported SPECFEM GLL schema_version",
        ),
        ("mesh_type", schema.MESH_TYPE, "Expected metadata mesh_type='gll'"),
        ("layout", schema.LAYOUT_SINGLE_FILE, "Expected metadata layout='single_file'"),
        (
            "solver_kind",
            "specfem3d_cartesian",
            "SPECFEM reader requires solver_kind='specfem3d_cartesian'",
        ),
        (
            "coordinate_convention",
            "ENU",
            "metadata coordinate_convention must be 'ENU'",
        ),
        (
            "element_id_method",
            schema.ADOPTED_ELEMENT_ID_METHOD,
            "SPECFEM reader requires the documented element_id_method",
        ),
    )
    for key, expected, message in checks:
        if provenance[key] != expected:
            raise ValueError(message)
    ngll_shape = tuple(int(value) for value in np.asarray(provenance["ngll_shape"]))
    ngll = schema.ngll_from_shape(ngll_shape)
    if int(provenance["ngll"]) != ngll:
        raise ValueError("metadata ngll must match metadata ngll_shape")
    dt_s = float(provenance["dt_s"])
    t0 = float(provenance["t0"])
    interval = int(provenance["output_interval_steps"])
    if not np.isfinite((dt_s, t0)).all() or dt_s <= 0.0:
        raise ValueError("metadata dt_s must be finite and positive")
    if interval <= 0:
        raise ValueError("metadata output_interval_steps must be positive")
    if int(provenance["nproc"]) <= 0:
        raise ValueError("metadata nproc must be positive")
    if int(provenance["nstep"]) < 0:
        raise ValueError("metadata nstep must be non-negative")
    if provenance["time_step_origin"] != "zero_based_normalized":
        raise ValueError("SPECFEM reader requires zero-based normalized time steps")
    if int(provenance["native_step_origin"]) not in (0, 1):
        raise ValueError("metadata native_step_origin must be 0 or 1")
    if float(provenance["sampling_oversampling_factor"]) < 2.0:
        raise ValueError("sampling_oversampling_factor must be at least 2")
    if provenance["cadence_provenance"] not in {"placeholder", "confirmed"}:
        raise ValueError("metadata cadence_provenance is invalid")

    datasets = {
        path: _require_dataset(h5, path)
        for path in schema.required_dataset_paths("time_sampled")
    }
    coordinates = datasets[schema.PATH_MESH_COORDINATES]
    volume = datasets[schema.PATH_MESH_QUADRATURE_VOLUME]
    element_ids = datasets[schema.PATH_MESH_ELEMENT_ID]
    partition = datasets[schema.PATH_MESH_PARTITION_ID]
    domain = datasets[schema.PATH_MESH_DOMAIN_ID]
    material = datasets[schema.PATH_MESH_MATERIAL_ID]
    is_pml = datasets[schema.PATH_MESH_IS_PML]
    density = datasets[schema.PATH_MATERIAL_DENSITY]
    displacement = datasets[schema.PATH_FIELD_DISPLACEMENT]
    time_dataset = datasets[schema.PATH_TIME]
    step_dataset = datasets[schema.PATH_TIME_STEP]
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("/mesh/coordinates must have shape (element, gll, 3)")
    n_elements, file_ngll, _ = coordinates.shape
    if n_elements == 0:
        raise ValueError("SPECFEM GLL input must contain at least one element")
    if file_ngll != ngll:
        raise ValueError(
            f"GLL axis has length {file_ngll}, but ngll_shape implies {ngll}"
        )
    expected_shapes = {
        schema.PATH_MESH_QUADRATURE_VOLUME: (n_elements, ngll),
        schema.PATH_MATERIAL_DENSITY: (n_elements, ngll),
        schema.PATH_MESH_ELEMENT_ID: (n_elements,),
        schema.PATH_MESH_PARTITION_ID: (n_elements,),
        schema.PATH_MESH_DOMAIN_ID: (n_elements,),
        schema.PATH_MESH_MATERIAL_ID: (n_elements,),
        schema.PATH_MESH_IS_PML: (n_elements,),
    }
    for path, shape in expected_shapes.items():
        if datasets[path].shape != shape:
            raise ValueError(
                f"{path} has shape {datasets[path].shape}, expected {shape}"
            )
    if is_pml.dtype != np.bool_:
        raise ValueError(f"/mesh/is_pml dtype must be bool, got {is_pml.dtype}")
    if displacement.ndim != 4 or displacement.shape[1:] != (n_elements, ngll, 3):
        raise ValueError("/field/displacement must have shape (time, element, gll, 3)")
    n_time = displacement.shape[0]
    if time_dataset.shape != (n_time,) or step_dataset.shape != (n_time,):
        raise ValueError("/time and /time_step must match displacement time axis")
    if n_time == 0:
        raise ValueError("displacement and time must be finite and non-empty")
    requested = (
        slice(0, n_time)
        if sample_slice is None
        else _bounded_slice(sample_slice, n_time, "sample_slice")
    )
    time_s = np.asarray(time_dataset[slice(0, n_time)])
    time_step = np.asarray(step_dataset[slice(0, n_time)])
    if not np.isfinite(time_s).all():
        raise ValueError("displacement and time must be finite and non-empty")
    if n_time > 1 and (
        np.any(np.diff(time_s) <= 0.0) or np.any(np.diff(time_step) <= 0)
    ):
        raise ValueError("/time and /time_step must be strictly increasing")
    if not np.allclose(time_s, time_step.astype(np.float64) * dt_s - t0):
        raise ValueError("/time must equal time_step * dt_s - t0")
    if n_time > 1 and not np.all(np.diff(time_step) == interval):
        raise ValueError("/time_step spacing must equal output_interval_steps")
    _validate_cadence(provenance, time_step)

    ids = np.empty(n_elements, dtype=np.int64)
    pml_hash = hashlib.sha256()
    material_values: set[int] = set()
    total_volume = np.float64(0.0)
    total_mass = np.float64(0.0)
    for start in range(0, n_elements, slab_elements):
        stop = min(start + slab_elements, n_elements)
        key = slice(start, stop)
        xyz = np.asarray(coordinates[key])
        vol = np.asarray(volume[key])
        rho = np.asarray(density[key])
        elem = np.asarray(element_ids[key])
        part = np.asarray(partition[key])
        dom = np.asarray(domain[key])
        mat = np.asarray(material[key])
        pml = np.asarray(is_pml[key])
        ids[key] = elem
        if not np.array_equal(elem, np.arange(start, stop, dtype=elem.dtype)):
            raise ValueError("/mesh/element_id must be ascending 0..N-1")
        if np.any(pml):
            raise ValueError("PML elements are not supported by this SPECFEM reader")
        if not np.all(dom == schema.IDOMAIN_ELASTIC):
            raise ValueError("SPECFEM reader requires a single elastic domain")
        material_values.update(int(value) for value in np.unique(mat))
        if np.any(mat < 1):
            raise ValueError("/mesh/material_id must be one-based and positive")
        if not np.isin(part, np.arange(int(provenance["nproc"]), dtype=np.int32)).all():
            raise ValueError("/mesh/partition_id must be in [0, metadata nproc)")
        if not np.isfinite(xyz).all() or not np.isfinite(vol).all():
            raise ValueError("mesh coordinates and quadrature volume must be finite")
        if np.any(vol <= 0.0):
            raise ValueError("quadrature volume must be positive")
        if not np.isfinite(rho).all() or np.any(rho <= 0.0):
            raise ValueError("density must be finite and positive")
        _validate_density_range(profile_density_range, rho)
        pml_hash.update(pml.astype(np.uint8, copy=False).tobytes())
        total_volume += np.sum(vol, dtype=np.float64)
        total_mass += np.sum(rho * vol, dtype=np.float64)
    # Reuse one explicit output buffer and align it to the stored chunks.  A
    # time-major HDF selection across many full-width chunks can otherwise
    # retain every decoded chunk until the selection completes.
    if displacement.chunks is None:
        time_slab = 1
        field_element_slab = slab_elements
    else:
        time_slab = displacement.chunks[0]
        field_element_slab = max(slab_elements, displacement.chunks[1])
    time_slab = min(time_slab, requested.stop - requested.start)
    field_element_slab = min(field_element_slab, n_elements)
    field_buffer = np.empty((time_slab, field_element_slab, ngll, 3), dtype=np.float64)
    for sample_start in range(requested.start, requested.stop, time_slab):
        sample_stop = min(sample_start + time_slab, requested.stop)
        sample_width = sample_stop - sample_start
        for element_start in range(0, n_elements, field_element_slab):
            element_stop = min(element_start + field_element_slab, n_elements)
            element_width = element_stop - element_start
            displacement.read_direct(
                field_buffer,
                source_sel=(
                    slice(sample_start, sample_stop),
                    slice(element_start, element_stop),
                    slice(0, ngll),
                    slice(0, 3),
                ),
                dest_sel=(
                    slice(0, sample_width),
                    slice(0, element_width),
                    slice(0, ngll),
                    slice(0, 3),
                ),
            )
            if not np.isfinite(
                field_buffer[
                    slice(0, sample_width),
                    slice(0, element_width),
                    slice(0, ngll),
                    slice(0, 3),
                ]
            ).all():
                raise ValueError("displacement and time must be finite and non-empty")
    if len(material_values) != 1:
        raise ValueError("SPECFEM reader requires a single material")
    if provenance["pml_exclusion_applied"] is not True:
        raise ValueError("SPECFEM reader requires pml_exclusion_applied=True")
    expected_hash = "sha256:" + pml_hash.hexdigest()
    if provenance["pml_mask_hash"] != expected_hash:
        raise ValueError("metadata pml_mask_hash does not match /mesh/is_pml")
    scalar_checks = {
        "n_elements_total": n_elements,
        "element_id_min": int(ids.min()),
        "element_id_max": int(ids.max()),
    }
    for key, expected in scalar_checks.items():
        if int(provenance[key]) != expected:
            raise ValueError(f"metadata {key} does not match /mesh/element_id")
    totals = {
        "total_quadrature_volume_before_exclusion": float(total_volume),
        "total_quadrature_volume_after_exclusion": float(total_volume),
        "total_mass_before_exclusion": float(total_mass),
        "total_mass_after_exclusion": float(total_mass),
    }
    for key, expected in totals.items():
        actual = float(provenance[key])
        if not np.isfinite(actual) or not np.isclose(
            actual, expected, rtol=1e-9, atol=0.0
        ):
            raise ValueError(f"metadata {key} does not match mesh data")
    return ValidatedSpecfemLayout(
        provenance=provenance,
        n_elements=n_elements,
        ngll=ngll,
        n_time=n_time,
        sample_slice=requested,
        time_s=_readonly(time_s[requested], np.float64),
        time_step=_readonly(time_step[requested], np.int64),
        element_id=_readonly(ids, np.int64),
    )


def validate_numeric_slab(*arrays: np.ndarray) -> None:
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("SPECFEM numeric slab contains non-finite values")
