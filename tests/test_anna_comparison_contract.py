"""Contract tests for the neutral ANNA comparison preparation.

These tests exercise geometry and physical invariants independently of the
manifest-writing implementation.  They do not execute ANNA or infer a native
ANNA file format.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gragra.fields import UnstructuredDisplacementField, piecewise_tetra_linear_sources
from gragra.observables import displacement_transfer_acceleration
from scripts import prepare_anna_comparison as prep

EXPECTED_TETRA = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=np.int64,
)
EXPECTED_BOUNDARY_FACES = np.array(
    [
        [0, 3, 2],
        [0, 2, 1],
        [4, 5, 6],
        [4, 6, 7],
        [0, 7, 3],
        [0, 4, 7],
        [1, 2, 6],
        [1, 6, 5],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
    ],
    dtype=np.int64,
)


def _make_sources(displacement: np.ndarray):
    field = UnstructuredDisplacementField(prep.NODES_M, EXPECTED_TETRA, displacement)
    return piecewise_tetra_linear_sources(field, prep.RHO_KG_M3)


def _independent_p1_divergence(displacement: np.ndarray) -> np.ndarray:
    """Compute P1 gradients with an independent row-wise linear solve."""

    values = []
    for cell in EXPECTED_TETRA:
        coordinates = prep.NODES_M[cell]
        nodal_displacement = displacement[cell]
        edge_rows = coordinates[1:] - coordinates[0]
        displacement_rows = nodal_displacement[1:] - nodal_displacement[0]
        gradient = np.linalg.solve(edge_rows, displacement_rows).T
        values.append(np.trace(gradient))
    return np.asarray(values)


def test_positive_jacobian_cube_and_outward_boundary_contract():
    np.testing.assert_array_equal(prep.TETRA, EXPECTED_TETRA)
    points = prep.NODES_M[EXPECTED_TETRA]
    jacobian = np.stack(
        [
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ],
        axis=-1,
    )
    np.testing.assert_allclose(np.linalg.det(jacobian), np.ones(6))
    np.testing.assert_allclose(np.abs(np.linalg.det(jacobian)) / 6.0, 1.0 / 6.0)
    assert np.sum(np.abs(np.linalg.det(jacobian)) / 6.0) == pytest.approx(1.0)

    normals = prep.boundary_normals()
    np.testing.assert_array_equal(prep.BOUNDARY_FACES, EXPECTED_BOUNDARY_FACES)
    np.testing.assert_array_equal(
        normals,
        np.array(
            [
                [0, 0, -1],
                [0, 0, -1],
                [0, 0, 1],
                [0, 0, 1],
                [-1, 0, 0],
                [-1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=np.float64,
        ),
    )
    face_centres = np.mean(prep.NODES_M[EXPECTED_BOUNDARY_FACES], axis=1)
    outward = face_centres - np.array([0.5, 0.5, 0.5])
    assert np.all(np.einsum("ij,ij->i", normals, outward) > 0.0)

    sources = _make_sources(prep.displacement_cases()["constant"])
    orientation = sources.interface_field.check_orientation()
    assert orientation.is_closed
    assert orientation.is_consistent
    assert orientation.boundary_edge_count == 0
    assert orientation.nonmanifold_edge_count == 0
    actual_faces = sources.interface_field.triangles
    assert {tuple(face) for face in np.sort(actual_faces, axis=1)} == {
        tuple(face) for face in np.sort(EXPECTED_BOUNDARY_FACES, axis=1)
    }
    actual_points = prep.NODES_M[actual_faces]
    actual_normals = np.cross(
        actual_points[:, 1] - actual_points[:, 0],
        actual_points[:, 2] - actual_points[:, 0],
    )
    actual_normals /= np.linalg.norm(actual_normals, axis=1)[:, None]
    for face, normal in zip(actual_faces, actual_normals, strict=True):
        matching = np.where(
            np.all(np.sort(EXPECTED_BOUNDARY_FACES, axis=1) == np.sort(face), axis=1)
        )[0]
        np.testing.assert_allclose(normal, normals[matching[0]])


def test_constant_and_affine_fields_have_expected_discrete_bulk_terms():
    cases = prep.displacement_cases()
    constant = _make_sources(cases["constant"])
    np.testing.assert_allclose(
        constant.mass_positions_m,
        np.mean(prep.NODES_M[EXPECTED_TETRA], axis=1),
    )
    np.testing.assert_allclose(
        constant.mass_displacement_m,
        np.mean(cases["constant"][EXPECTED_TETRA], axis=1),
    )
    np.testing.assert_allclose(constant.bulk_source.density_kg_m3, 0.0, atol=0.0)
    np.testing.assert_allclose(
        constant.bulk_source.as_weighted_source().weights_kg, 0.0
    )
    np.testing.assert_allclose(
        np.sum(constant.interface_source.weights_kg), 0.0, atol=1e-18
    )

    affine = _make_sources(cases["affine"])
    expected_divergence = np.full(6, 0.6 * prep.U_REF_M)
    np.testing.assert_allclose(
        -np.asarray(affine.bulk_source.density_kg_m3) / prep.RHO_KG_M3,
        expected_divergence,
        rtol=0.0,
        atol=1e-20,
    )
    np.testing.assert_allclose(
        affine.bulk_source.as_weighted_source().weights_kg,
        -prep.RHO_KG_M3 * expected_divergence / 6.0,
        rtol=0.0,
        atol=1e-18,
    )


def test_complex_p_and_s_are_p1_cases_with_recorded_discrete_divergences():
    cases = prep.displacement_cases()
    assert set(cases) == {"constant", "affine", "complex_p", "complex_s"}
    assert np.iscomplexobj(cases["complex_p"])
    assert np.iscomplexobj(cases["complex_s"])
    assert np.vdot(prep.K_HAT, prep.S_POLARIZATION) == pytest.approx(0.0)

    p_sources = _make_sources(cases["complex_p"])
    s_sources = _make_sources(cases["complex_s"])
    p_div = -np.asarray(p_sources.bulk_source.density_kg_m3) / prep.RHO_KG_M3
    s_div = -np.asarray(s_sources.bulk_source.density_kg_m3) / prep.RHO_KG_M3
    assert p_div.shape == (6,)
    assert s_div.shape == (6,)
    assert np.isfinite(p_div).all()
    assert np.isfinite(s_div).all()
    assert np.any(np.abs(s_div) > 1e-12)
    assert not np.allclose(s_div, 0.0, atol=1e-12)


def test_wave_nodal_phasors_match_independent_literal_formula():
    """Check both nodal wave cases without calling gragra's wave helper."""

    positions = prep.NODES_M - prep.COORDINATE_ORIGIN_M
    phase = prep.WAVENUMBER_RAD_M * (positions @ prep.K_HAT) + prep.WAVE_PHASE0
    phasor = np.exp(1j * phase)
    expected_p = prep.U_REF_M * phasor[:, None] * prep.K_HAT[None, :]
    expected_s = prep.U_REF_M * phasor[:, None] * prep.S_POLARIZATION[None, :]
    cases = prep.displacement_cases()
    np.testing.assert_allclose(cases["complex_p"], expected_p)
    np.testing.assert_allclose(cases["complex_s"], expected_s)


def test_wave_bulk_terms_match_independent_p1_gradients_on_all_tetrahedra():
    cases = prep.displacement_cases()
    for case_id in ("complex_p", "complex_s"):
        displacement = cases[case_id]
        sources = _make_sources(displacement)
        expected_divergence = _independent_p1_divergence(displacement)
        expected_bulk_density = -prep.RHO_KG_M3 * expected_divergence
        expected_bulk_mass = expected_bulk_density / 6.0
        np.testing.assert_allclose(
            np.asarray(sources.bulk_source.density_kg_m3),
            expected_bulk_density,
            rtol=1e-12,
            atol=1e-20,
        )
        np.testing.assert_allclose(
            sources.bulk_source.as_weighted_source().weights_kg,
            expected_bulk_mass,
            rtol=1e-12,
            atol=1e-20,
        )


def test_case_metadata_freezes_output_and_phase_contract():
    cases = prep.displacement_cases()
    for case_id in cases:
        metadata = prep.comparison_metadata(cases, case_id)
        expected_dtype = "complex128" if case_id.startswith("complex_") else "float64"
        expected_phase = prep.WAVE_PHASE0 if case_id.startswith("complex_") else None
        assert metadata["case_id"] == case_id
        assert metadata["observable"] == "acceleration"
        assert metadata["output_name"] == "acceleration"
        assert metadata["units"] == "m/s^2"
        assert metadata["component_order"] == ["x", "y", "z"]
        assert metadata["expected_dtype"] == expected_dtype
        assert metadata["target_count"] == 3
        assert metadata["coordinate_origin_m"] == [0.0, 0.0, 0.0]
        assert metadata["wave_phase0_rad"] == expected_phase
        assert metadata["case_nodal_hash"] == prep.array_hash(cases[case_id])
        if expected_phase is None:
            assert metadata["wave_type"] is None
            assert metadata["wavenumber_rad_m"] is None
            assert metadata["k_hat"] is None
            assert metadata["polarization"] is None
        else:
            assert metadata["phase_convention"] == prep.WAVE_PHASE_CONVENTION
            assert metadata["wavenumber_rad_m"] == prep.WAVENUMBER_RAD_M
            assert metadata["k_hat"] == prep.K_HAT.tolist()
            expected_polarization = (
                prep.K_HAT if case_id == "complex_p" else prep.S_POLARIZATION
            )
            assert metadata["polarization"] == expected_polarization.tolist()
            assert metadata["wave_type"] == (
                "P" if case_id == "complex_p" else "custom"
            )


def test_complex_observable_is_linear_and_raw_si_is_kept():
    displacement = prep.displacement_cases()["complex_p"]
    reference = _make_sources(displacement)
    doubled = _make_sources(2.0 * displacement)
    reference_acceleration = displacement_transfer_acceleration(
        reference.mass_positions_m,
        reference.mass_kg,
        reference.mass_displacement_m,
        prep.TARGETS_M,
        backend="numpy",
    )
    doubled_acceleration = displacement_transfer_acceleration(
        doubled.mass_positions_m,
        doubled.mass_kg,
        doubled.mass_displacement_m,
        prep.TARGETS_M,
        backend="numpy",
    )
    np.testing.assert_allclose(doubled_acceleration, 2.0 * reference_acceleration)
    np.testing.assert_allclose(
        reference_acceleration / prep.ACCELERATION_SCALE * prep.ACCELERATION_SCALE,
        reference_acceleration,
    )


def test_export_roundtrip_hashes_and_synthetic_mismatch_are_explicit(tmp_path):
    paths = prep.write_evidence(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["state"] == ["PREPARATION_COMPLETE", "ANNA_EXECUTION_NOT_RUN"]
    assert manifest["candidate_source"] == "gragra_numpy_reference"
    assert manifest["normalization_scale_m_s2"] == prep.ACCELERATION_SCALE
    assert manifest["field_contract"]["interior_representation"] == (
        "P1_nodal_interpolation"
    )
    assert not manifest["gragra_quadrature_contract"]["exact_fe_volume_integral"]
    assert manifest["quadrature_comparison_status"] == "NOT_ASSUMED_IDENTICAL"
    assert len(manifest["future_quadrature_budget"]) == 3
    assert manifest["field_contract"]["discrete_divergence_units"] == "1"

    exported = prep.read_saved_data(paths["results"], manifest)
    assert set(exported) == set(manifest["array_metadata"])
    for key, array in exported.items():
        saved_metadata = manifest["array_metadata"][key]
        assert saved_metadata["dtype"] == array.dtype.str
        assert saved_metadata["shape"] == list(array.shape)
        assert saved_metadata["sha256"] == prep.array_hash(array)
    np.testing.assert_array_equal(exported["nodes_m"], prep.NODES_M)
    np.testing.assert_array_equal(exported["tetra"], EXPECTED_TETRA)
    np.testing.assert_array_equal(exported["boundary_faces"], EXPECTED_BOUNDARY_FACES)
    np.testing.assert_array_equal(exported["targets_m"], prep.TARGETS_M)
    assert "complex_p_divergence" in exported
    assert not any("divergence_s-1" in key for key in exported)
    prep.read_and_verify_input_contract(paths["input"], manifest)
    readback = json.loads(paths["readback"].read_text())
    assert readback["all_array_hashes_match"] is True
    assert readback["input_json_hash_match"] is True
    assert readback["array_count"] == len(exported)
    assert readback["keys"] == sorted(exported)

    synthetic_manifest = json.loads(paths["synthetic_metadata"].read_text())
    synthetic_key = synthetic_manifest["npz_key"]
    synthetic = prep.read_saved_data(paths["synthetic"], synthetic_manifest)
    assert set(synthetic) == {synthetic_key}
    assert synthetic_manifest["array_metadata"][synthetic_key] == {
        "dtype": synthetic[synthetic_key].dtype.str,
        "shape": list(synthetic[synthetic_key].shape),
        "sha256": prep.array_hash(synthetic[synthetic_key]),
    }
    assert synthetic_manifest["array_hashes"] == {
        synthetic_key: prep.array_hash(synthetic[synthetic_key])
    }
    assert readback["synthetic"]["array_count"] == 1
    assert readback["synthetic"]["keys"] == [synthetic_key]
    assert readback["synthetic"]["all_array_hashes_match"] is True
    comparison = json.loads(paths["comparison"].read_text())
    assert comparison["candidate_loaded_from_saved_npz"] is True
    candidate_from_json = np.asarray(comparison["candidate_acceleration_si"])
    candidate_from_json = candidate_from_json[..., 0] + 1j * candidate_from_json[..., 1]
    np.testing.assert_allclose(candidate_from_json, synthetic[synthetic_key])

    synthetic_missing_path = tmp_path / "synthetic-missing-key.npz"
    np.savez(synthetic_missing_path)
    with pytest.raises(ValueError, match="missing NPZ keys"):
        prep.read_saved_data(synthetic_missing_path, synthetic_manifest)

    synthetic_corrupted = {key: value.copy() for key, value in synthetic.items()}
    synthetic_corrupted[synthetic_key][0, 0] += 1.0
    synthetic_corrupted_path = tmp_path / "synthetic-corrupted.npz"
    np.savez(synthetic_corrupted_path, **synthetic_corrupted)
    with pytest.raises(ValueError, match="hash mismatch"):
        prep.read_saved_data(synthetic_corrupted_path, synthetic_manifest)

    synthetic_wrong_dtype = {
        synthetic_key: synthetic[synthetic_key].astype(np.complex64)
    }
    synthetic_wrong_dtype_path = tmp_path / "synthetic-wrong-dtype.npz"
    np.savez(synthetic_wrong_dtype_path, **synthetic_wrong_dtype)
    with pytest.raises(ValueError, match="dtype mismatch"):
        prep.read_saved_data(synthetic_wrong_dtype_path, synthetic_manifest)

    corrupted_input = json.loads(paths["input"].read_text())
    corrupted_input["targets_m"][0][0] += 1.0
    corrupted_input_path = tmp_path / "corrupted-input.json"
    corrupted_input_path.write_text(json.dumps(corrupted_input))
    with pytest.raises(ValueError, match="input JSON hash mismatch"):
        prep.read_and_verify_input_contract(corrupted_input_path, manifest)

    missing = dict(exported)
    missing.pop(sorted(missing)[0])
    missing_path = tmp_path / "missing-key.npz"
    np.savez(missing_path, **missing)
    with pytest.raises(ValueError, match="missing NPZ keys"):
        prep.read_saved_data(missing_path, manifest)

    corrupted = {key: value.copy() for key, value in exported.items()}
    corrupted["nodes_m"][0, 0] += 1.0
    corrupted_path = tmp_path / "corrupted.npz"
    np.savez(corrupted_path, **corrupted)
    with pytest.raises(ValueError, match="hash mismatch"):
        prep.read_saved_data(corrupted_path, manifest)

    wrong_dtype = {key: value.copy() for key, value in exported.items()}
    wrong_dtype["targets_m"] = wrong_dtype["targets_m"].astype(np.float32)
    wrong_dtype_path = tmp_path / "wrong-dtype.npz"
    np.savez(wrong_dtype_path, **wrong_dtype)
    with pytest.raises(ValueError, match="dtype mismatch"):
        prep.read_saved_data(wrong_dtype_path, manifest)

    cases = prep.displacement_cases()
    metadata = prep.comparison_metadata(cases, "complex_p")
    reference = prep.build_results(cases)[0]["complex_p"]["total_acceleration_si"]
    candidate = 1.25 * reference
    comparison = prep.compare_acceleration_outputs(
        reference,
        candidate,
        {**metadata, "candidate_source": "synthetic_test_double"},
        metadata,
    )
    assert comparison["candidate_source"] == "synthetic_test_double"
    assert comparison["max_abs_delta_normalized"] > 0.0
    np.testing.assert_allclose(
        comparison["candidate_acceleration_normalized"],
        1.25 * comparison["reference_acceleration_normalized"],
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("case_id", "affine"),
        ("observable", "gradient"),
        ("output_name", "potential"),
        ("units", "SI"),
        ("component_order", ["z", "y", "x"]),
        ("expected_dtype", "float64"),
        ("target_count", 4),
        ("coordinate_origin_m", [1.0, 0.0, 0.0]),
        ("wave_phase0_rad", 0.5),
        ("phase_convention", "wrong"),
        ("wave_type", "S"),
        ("wavenumber_rad_m", 1.0),
        ("k_hat", [0.0, 0.0, 1.0]),
        ("polarization", [0.0, 0.0, 1.0]),
        ("case_nodal_hash", "wrong-case-hash"),
        ("input_hash", "wrong-input"),
        ("target_order_hash", "wrong-target-order"),
        ("G_SI", 2.0 * prep.G_SI),
        ("t_ref_s", 1.0),
    ],
)
def test_comparison_rejects_mismatched_metadata(key, value):
    cases = prep.displacement_cases()
    metadata = prep.comparison_metadata(cases, "complex_p")
    candidate_metadata = {
        **metadata,
        "candidate_source": "synthetic_test_double",
        key: value,
    }
    reference = np.zeros((len(prep.TARGETS_M), 3), dtype=np.float64)
    with pytest.raises(ValueError, match="metadata mismatch"):
        prep.compare_acceleration_outputs(
            reference, reference, candidate_metadata, metadata
        )


def test_comparison_rejects_target_shape_mismatch():
    cases = prep.displacement_cases()
    metadata = prep.comparison_metadata(cases, "complex_p")
    candidate_metadata = {**metadata, "candidate_source": "synthetic_test_double"}
    reference = np.zeros((3, 3), dtype=np.float64)
    candidate = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="target-order/shape mismatch"):
        prep.compare_acceleration_outputs(
            reference, candidate, candidate_metadata, metadata
        )


@pytest.mark.parametrize("case_id", ["constant", "affine", "complex_p", "complex_s"])
def test_comparison_rejects_wrong_array_dtype_and_target_count(case_id):
    cases = prep.displacement_cases()
    metadata = prep.comparison_metadata(cases, case_id)
    result = prep.build_results(cases)[0][case_id]["total_acceleration_si"]
    wrong_dtype = result.real if np.iscomplexobj(result) else result.astype(np.float32)
    with pytest.raises(ValueError, match="dtype mismatch"):
        prep.compare_acceleration_outputs(
            result,
            wrong_dtype,
            {**metadata, "candidate_source": "synthetic_test_double"},
            metadata,
        )

        wrong_count = np.concatenate([result, result[:1]], axis=0)
        with pytest.raises(ValueError, match="target_count"):
            prep.compare_acceleration_outputs(
                wrong_count,
                wrong_count,
                {**metadata, "candidate_source": "synthetic_test_double"},
                metadata,
            )
