"""Public reader and acceleration path for canonical SPECFEM GLL HDF5."""

from __future__ import annotations

import os

import numpy as np
import pytest

from gragra.adapters.specfem import provenance as specfem_provenance
from gragra.adapters.specfem import schema, writer
from gragra.adapters.specfem.reader import read_specfem_gll_hdf5
from gragra.constants import G_SI
from gragra.observables import displacement_field_acceleration


def _explicit_linearized_acceleration(
    coordinates: np.ndarray,
    mass_kg: np.ndarray,
    displacement_m: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Independent first-order point-mass acceleration formula for tests."""
    result = np.zeros((targets.shape[0], 3), dtype=np.float64)
    for source, mass, displacement in zip(
        coordinates.reshape(-1, 3),
        mass_kg.reshape(-1),
        displacement_m.reshape(-1, 3),
        strict=True,
    ):
        r = source - targets
        r2 = np.einsum("ij,ij->i", r, r)
        r3 = r2 * np.sqrt(r2)
        r5 = r3 * r2
        result += (
            G_SI
            * mass
            * (
                displacement / r3[:, None]
                - 3.0 * r * np.einsum("ij,j->i", r, displacement)[:, None] / r5[:, None]
            )
        )
    return result


def _explicit_point_acceleration(
    coordinates: np.ndarray, mass_kg: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Direct point-mass field used only by the finite-difference oracle."""
    result = np.zeros((targets.shape[0], 3), dtype=np.float64)
    for source, mass in zip(
        coordinates.reshape(-1, 3), mass_kg.reshape(-1), strict=True
    ):
        r = source - targets
        r2 = np.einsum("ij,ij->i", r, r)
        result += G_SI * mass * r / (r2 * np.sqrt(r2))[:, None]
    return result


def _time_sampled_provenance(is_pml: np.ndarray) -> dict[str, object]:
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "mesh_type": schema.MESH_TYPE,
        "layout": schema.LAYOUT_SINGLE_FILE,
        "solver_kind": "specfem3d_cartesian",
        "solver_version": "fixture",
        "build_kind": "package",
        "package_version": "fixture",
        "package_digest": "sha256:" + "0" * 64,
        "build_configuration": {},
        "executables": [{"role": "fixture"}],
        "resolution_method": "fixture",
        "par_file_hash": "sha256:" + "0" * 64,
        "par_file_text": "synthetic fixture",
        "mesh_input_hash": "sha256:" + "0" * 64,
        "mesh_input_manifest": "synthetic fixture",
        "nproc": 1,
        "output_sampling_kind": "time_sampled",
        "dt_s": 0.25,
        "nstep": 1,
        "float_precision": "double",
        "coordinate_convention": "ENU",
        "field_units": {"displacement": "m"},
        "time_step_origin": "zero_based_normalized",
        "native_step_origin": 1,
        "element_id_method": schema.ADOPTED_ELEMENT_ID_METHOD,
        "density_source": "synthetic",
        "f_resolved_estimate": 1.0,
        "pml_exclusion_applied": True,
        "pml_mask_hash": specfem_provenance.pml_mask_hash(is_pml),
        "t0": 0.0,
        "t0_source": "synthetic",
        "suppress_utm_projection": True,
        "endianness": "<",
        "record_marker_bytes": 4,
        "integer_kind": 4,
        "output_interval_steps": 1,
        "f_max_analysis": 1.5,
        "f_min_analysis": 1.0,
        "minimum_cycles": 0.25,
        "sampling_oversampling_factor": 2.0,
        "antialias_filter_applied": False,
        "cadence_provenance": "confirmed",
    }


def _write_custom_time_sampled_gll(
    path,
    *,
    ngll_shape: tuple[int, int, int],
    coordinates: np.ndarray,
    density: np.ndarray,
    quadrature_volume: np.ndarray,
    displacement: np.ndarray,
    is_pml: np.ndarray | None = None,
) -> None:
    n_elements = coordinates.shape[0]
    if is_pml is None:
        is_pml = np.zeros(n_elements, dtype=np.bool_)
    writer.write_gll_hdf5(
        path,
        output_sampling_kind="time_sampled",
        ngll_shape=ngll_shape,
        coordinates=coordinates,
        quadrature_volume=quadrature_volume,
        element_id=np.arange(n_elements, dtype=np.int64),
        partition_id=np.zeros(n_elements, dtype=np.int32),
        domain_id=np.full(n_elements, schema.IDOMAIN_ELASTIC, dtype=np.int32),
        material_id=np.ones(n_elements, dtype=np.int32),
        is_pml=is_pml,
        density=density,
        provenance=_time_sampled_provenance(is_pml),
        displacement=displacement,
        time_step=np.arange(displacement.shape[0], dtype=np.int64),
    )


def _write_time_sampled_gll(path, *, is_pml=(False, False)):
    coordinates = np.array([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=np.float64)
    density = np.array([[2000.0], [3000.0]], dtype=np.float64)
    quadrature_volume = np.array([[4.0], [5.0]], dtype=np.float64)
    displacement = np.array(
        [
            [[[0.01, 0.0, 0.0]], [[0.0, 0.02, 0.0]]],
            [[[0.02, 0.0, 0.0]], [[0.0, 0.04, 0.0]]],
        ],
        dtype=np.float64,
    )
    _write_custom_time_sampled_gll(
        path,
        ngll_shape=(1, 1, 1),
        coordinates=coordinates,
        density=density,
        quadrature_volume=quadrature_volume,
        displacement=displacement,
        is_pml=np.asarray(is_pml, dtype=np.bool_),
    )
    return coordinates, density, quadrature_volume, displacement


def test_reader_returns_closed_immutable_dataset_and_element_order(tmp_path):
    path = tmp_path / "gll.h5"
    coordinates, density, volume, displacement = _write_time_sampled_gll(path)

    dataset = read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))

    assert dataset.time_s.flags.writeable is False
    assert dataset.file_identity.path == path.resolve()
    blocks = list(dataset.iter_blocks(element_block_size=1))
    assert [block.element_id.tolist() for block in blocks] == [[0], [1]]
    np.testing.assert_array_equal(blocks[0].coordinates_m, coordinates[:1])
    np.testing.assert_array_equal(blocks[1].mass_kg, density[1:] * volume[1:])
    np.testing.assert_array_equal(blocks[0].displacement_m, displacement[:, :1])
    assert blocks[0].coordinates_m.flags.writeable is False


def test_reader_rejects_pml(tmp_path):
    path = tmp_path / "pml.h5"
    _write_time_sampled_gll(path, is_pml=(False, True))

    with pytest.raises(ValueError, match="PML"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_non_boolean_pml_mask(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "pml_dtype.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        del h5["/mesh/is_pml"]
        h5.create_dataset("/mesh/is_pml", data=np.array([0, 0], dtype=np.int8))

    with pytest.raises(ValueError, match="bool"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_non_enu_coordinate_convention(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong_coordinates.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        h5["/metadata"].attrs["coordinate_convention"] = "NED"

    with pytest.raises(ValueError, match="coordinate_convention"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_missing_required_provenance_key(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "missing_provenance.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        del h5["/metadata"].attrs["solver_kind"]

    with pytest.raises(ValueError, match="solver_kind"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_required_dataset_with_wrong_disk_dtype(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong_dtype.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        values = h5[schema.PATH_MESH_ELEMENT_ID][()]
        del h5[schema.PATH_MESH_ELEMENT_ID]
        h5.create_dataset(schema.PATH_MESH_ELEMENT_ID, data=values.astype(np.float64))

    with pytest.raises(ValueError, match="dtype"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_required_json_key_in_attribute_location(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong_metadata_encoding.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        metadata = h5["/metadata"]
        del metadata["field_units"]
        metadata.attrs["field_units"] = "{}"

    with pytest.raises(ValueError, match="field_units.*dataset"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_required_json_key_with_non_utf8_encoding(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong_json_encoding.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        metadata = h5["/metadata"]
        del metadata["field_units"]
        metadata.create_dataset(
            "field_units", data="{}", dtype=h5py.string_dtype(encoding="ascii")
        )

    with pytest.raises(ValueError, match="UTF-8"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_dirty_source_provenance_without_hash(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "dirty_source_without_hash.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        metadata = h5["/metadata"]
        metadata.attrs["build_kind"] = "source"
        metadata.attrs["source_tree_clean"] = False
        metadata.attrs["specfem_git_commit"] = "0" * 40

    with pytest.raises(ValueError, match="patch_hash.*source_tree_hash"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_accepts_dirty_source_provenance_with_patch_hash(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "dirty_source_with_hash.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        metadata = h5["/metadata"]
        metadata.attrs["build_kind"] = "source"
        metadata.attrs["source_tree_clean"] = False
        metadata.attrs["specfem_git_commit"] = "0" * 40
        metadata.attrs["patch_hash"] = "sha256:" + "0" * 64

    dataset = read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))
    assert dataset.time_step.tolist() == [0, 1]


def test_reader_rejects_multistage_conditional_cadence_key_gap(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "missing_measured_evidence.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        metadata = h5["/metadata"]
        metadata.attrs["output_interval_steps"] = 2
        metadata.attrs["f_content_max"] = 1.0
        metadata.attrs["f_content_max_method"] = "measured_spectrum"
        h5[schema.PATH_TIME_STEP][...] = [0, 2]
        h5[schema.PATH_TIME][...] = [0.0, 0.5]

    with pytest.raises(ValueError, match="alias_error_budget|f_content_max_evidence"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_cadence_above_sampling_bound(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "invalid_cadence.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        h5["/metadata"].attrs["f_max_analysis"] = 3.0

    with pytest.raises(ValueError, match="sampling frequency"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def _configure_valid_interval_two_cadence(h5) -> None:
    metadata = h5["/metadata"]
    metadata.attrs["output_interval_steps"] = 2
    metadata.attrs["f_max_analysis"] = 0.75
    metadata.attrs["f_min_analysis"] = 0.5
    metadata.attrs["minimum_cycles"] = 0.25
    metadata.attrs["f_content_max"] = 0.5
    h5[schema.PATH_TIME_STEP][...] = [0, 2]
    h5[schema.PATH_TIME][...] = [0.0, 0.5]


def test_reader_rejects_solver_nyquist_method_with_non_nyquist_content(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong_solver_nyquist_method.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        _configure_valid_interval_two_cadence(h5)
        h5["/metadata"].attrs["f_content_max_method"] = "solver_step_nyquist"

    with pytest.raises(ValueError, match="solver_step_nyquist"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_filter_cutoff_method_without_filter(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "filter_without_filter.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        _configure_valid_interval_two_cadence(h5)
        metadata = h5["/metadata"]
        metadata.attrs["f_content_max_method"] = "filter_cutoff"
        metadata.attrs["alias_error_budget"] = 0.1

    with pytest.raises(ValueError, match="filter_cutoff"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_dirty_source_hash_with_wrong_type(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "dirty_source_bad_hash_type.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        metadata = h5["/metadata"]
        metadata.attrs["build_kind"] = "source"
        metadata.attrs["source_tree_clean"] = False
        metadata.attrs["specfem_git_commit"] = "0" * 40
        metadata.attrs["patch_hash"] = 42

    with pytest.raises(ValueError, match="patch_hash.*unexpected type"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_measured_spectrum_evidence_without_required_fields(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "empty_measured_evidence.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        _configure_valid_interval_two_cadence(h5)
        metadata = h5["/metadata"]
        metadata.attrs["f_content_max_method"] = "measured_spectrum"
        metadata.attrs["alias_error_budget"] = 0.1
        metadata.create_dataset(
            "f_content_max_evidence", data="{}", dtype=h5py.string_dtype("utf-8")
        )

    with pytest.raises(ValueError, match="f_content_max_evidence"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_total_mass_provenance_mismatch(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong_total_mass.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        h5["/metadata"].attrs["total_mass_after_exclusion"] = 1.0

    with pytest.raises(ValueError, match="total_mass_after_exclusion"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_pml_mask_hash_mismatch(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong_pml_hash.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        h5["/metadata"].attrs["pml_mask_hash"] = "sha256:" + "f" * 64

    with pytest.raises(ValueError, match="pml_mask_hash"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_displacement_shape_and_time_relation_mutations(tmp_path):
    h5py = pytest.importorskip("h5py")
    shape_path = tmp_path / "wrong_displacement_shape.h5"
    _write_time_sampled_gll(shape_path)
    with h5py.File(shape_path, "a") as h5:
        values = h5[schema.PATH_FIELD_DISPLACEMENT][()]
        del h5[schema.PATH_FIELD_DISPLACEMENT]
        h5.create_dataset(schema.PATH_FIELD_DISPLACEMENT, data=values[..., :2])

    with pytest.raises(ValueError, match="displacement.*shape"):
        read_specfem_gll_hdf5(shape_path, profile_density_range=(1900.0, 3100.0))

    time_path = tmp_path / "wrong_time_relation.h5"
    _write_time_sampled_gll(time_path)
    with h5py.File(time_path, "a") as h5:
        h5[schema.PATH_TIME][...] = [0.0, 0.3]

    with pytest.raises(ValueError, match="/time must equal"):
        read_specfem_gll_hdf5(time_path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_density_outside_inclusive_profile_range(tmp_path):
    path = tmp_path / "density_range.h5"
    _write_time_sampled_gll(path)

    with pytest.raises(ValueError, match="profile_density_range"):
        read_specfem_gll_hdf5(path, profile_density_range=(2000.1, 3000.0))


def test_reader_accepts_density_at_inclusive_profile_range_bounds(tmp_path):
    path = tmp_path / "density_range_bounds.h5"
    _write_time_sampled_gll(path)

    dataset = read_specfem_gll_hdf5(path, profile_density_range=(2000.0, 3000.0))
    np.testing.assert_array_equal(dataset.density_kg_m3, [[2000.0], [3000.0]])


def test_reader_rejects_invalid_json_and_missing_required_dataset(tmp_path):
    h5py = pytest.importorskip("h5py")
    json_path = tmp_path / "invalid_json.h5"
    _write_time_sampled_gll(json_path)
    with h5py.File(json_path, "a") as h5:
        metadata = h5["/metadata"]
        del metadata["field_units"]
        metadata.create_dataset(
            "field_units", data="{", dtype=h5py.string_dtype("utf-8")
        )

    with pytest.raises(ValueError, match="valid JSON"):
        read_specfem_gll_hdf5(json_path, profile_density_range=(1900.0, 3100.0))

    missing_path = tmp_path / "missing_dataset.h5"
    _write_time_sampled_gll(missing_path)
    with h5py.File(missing_path, "a") as h5:
        del h5[schema.PATH_TIME]

    with pytest.raises(ValueError, match="Missing required dataset"):
        read_specfem_gll_hdf5(missing_path, profile_density_range=(1900.0, 3100.0))


def test_reader_rejects_time_step_interval_mismatch(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "wrong_time_step_interval.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        h5[schema.PATH_TIME_STEP][...] = [0, 3]
        h5[schema.PATH_TIME][...] = [0.0, 0.75]

    with pytest.raises(ValueError, match="spacing"):
        read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))


def test_displacement_acceleration_matches_independent_linearized_formula(tmp_path):
    path = tmp_path / "gll.h5"
    coordinates, density, volume, displacement = _write_time_sampled_gll(path)
    dataset = read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))
    targets = np.array([[10.0, 1.0, -2.0]], dtype=np.float64)

    actual = displacement_field_acceleration(dataset, targets, chunk_size=1)
    expected = np.stack(
        [
            _explicit_linearized_acceleration(
                coordinates,
                density * volume,
                snapshot,
                targets,
            )
            for snapshot in displacement
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)
    assert actual.shape == (2, 1, 3)
    assert actual.flags.writeable is False


def test_displacement_acceleration_matches_independent_central_difference(tmp_path):
    path = tmp_path / "gll.h5"
    coordinates, density, volume, displacement = _write_time_sampled_gll(path)
    dataset = read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))
    targets = np.array([[10.0, 1.0, -2.0]], dtype=np.float64)
    scale = 1.0e-4

    actual = displacement_field_acceleration(dataset, targets)[0]
    expected = (
        _explicit_point_acceleration(
            coordinates + scale * displacement[0], density * volume, targets
        )
        - _explicit_point_acceleration(
            coordinates - scale * displacement[0], density * volume, targets
        )
    ) / (2.0 * scale)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=0.0)


def test_displacement_acceleration_matches_rigid_translation_oracle(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "rigid_translation.h5"
    coordinates, density, volume, _ = _write_time_sampled_gll(path)
    translation = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    with h5py.File(path, "a") as h5:
        h5[schema.PATH_FIELD_DISPLACEMENT][...] = translation

    dataset = read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))
    targets = np.array([[10.0, 1.0, -2.0]], dtype=np.float64)
    displacement = np.broadcast_to(translation, coordinates.shape)
    actual = displacement_field_acceleration(dataset, targets)[0]
    expected = _explicit_linearized_acceleration(
        coordinates, density * volume, displacement, targets
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)


def test_displacement_acceleration_is_exact_zero_for_zero_displacement(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "zero_displacement.h5"
    _write_time_sampled_gll(path)
    with h5py.File(path, "a") as h5:
        h5[schema.PATH_FIELD_DISPLACEMENT][...] = 0.0

    dataset = read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))
    actual = displacement_field_acceleration(
        dataset, np.array([[10.0, 1.0, -2.0]], dtype=np.float64)
    )

    np.testing.assert_array_equal(actual, np.zeros_like(actual))


def test_ngll_order_duplicate_coordinates_and_chunking_match_independent_sum(tmp_path):
    path = tmp_path / "ngll_order.h5"
    ngll_shape = (2, 2, 1)
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    density = np.arange(2, 10, dtype=np.float64).reshape(2, 4)
    volume = np.arange(1, 9, dtype=np.float64).reshape(2, 4)
    displacement = np.full((2, 2, 4, 3), [0.001, -0.002, 0.003], dtype=np.float64)
    _write_custom_time_sampled_gll(
        path,
        ngll_shape=ngll_shape,
        coordinates=coordinates,
        density=density,
        quadrature_volume=volume,
        displacement=displacement,
    )
    dataset = read_specfem_gll_hdf5(path, profile_density_range=(2.0, 9.0))
    targets = np.array([[10.0, -1.0, 3.0]], dtype=np.float64)

    blocks = list(dataset.iter_blocks(element_block_size=1))
    assert [block.element_id.tolist() for block in blocks] == [[0], [1]]
    for i in range(2):
        for j in range(2):
            flat = schema.flatten_index(i, j, 0, ngll_shape)
            np.testing.assert_array_equal(blocks[0].coordinates_m[0, flat], [i, j, 0])
    repeated_origin = np.all(
        dataset.coordinates_m.reshape(-1, 3) == [0.0, 0.0, 0.0], axis=1
    )
    assert np.count_nonzero(repeated_origin) == 2

    expected = _explicit_linearized_acceleration(
        coordinates, density * volume, displacement[0], targets
    )
    actual_small_chunk = displacement_field_acceleration(dataset, targets, chunk_size=1)
    actual_large_chunk = displacement_field_acceleration(dataset, targets, chunk_size=7)
    np.testing.assert_allclose(actual_small_chunk[0], expected, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(
        actual_small_chunk, actual_large_chunk, rtol=1e-13, atol=0.0
    )


def test_displacement_acceleration_backend_matches_numpy_when_required(tmp_path):
    """Dedicated CI runs this once per compiled optional backend.

    The ordinary suite deliberately leaves ``GRAGRA_SPECFEM_BACKEND`` unset;
    the two strict CI jobs set it and therefore cannot satisfy this gate by
    skipping an unavailable backend.
    """
    backend = os.environ.get("GRAGRA_SPECFEM_BACKEND")
    if backend is None:
        return
    if backend == "numba":
        __import__("numba")
    elif backend == "cpp":
        __import__("gragra._cpp_backend")
    else:
        raise AssertionError(f"unsupported required backend {backend!r}")

    path = tmp_path / f"{backend}.h5"
    _write_time_sampled_gll(path)
    dataset = read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))
    targets = np.array([[10.0, 1.0, -2.0]], dtype=np.float64)

    expected = displacement_field_acceleration(dataset, targets, backend="numpy")
    actual = displacement_field_acceleration(dataset, targets, backend=backend)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)


def test_displacement_acceleration_rejects_non_integral_chunk_size(tmp_path):
    path = tmp_path / "gll.h5"
    _write_time_sampled_gll(path)
    dataset = read_specfem_gll_hdf5(path, profile_density_range=(1900.0, 3100.0))

    with pytest.raises(ValueError, match="positive integer"):
        displacement_field_acceleration(dataset, [[10.0, 1.0, -2.0]], chunk_size=1.5)
