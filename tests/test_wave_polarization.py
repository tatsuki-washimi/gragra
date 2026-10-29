"""Tests for wave polarization utilities."""

import numpy as np
import pytest

from gragra.waves.polarization import (
    displacement_direction,
    p_wave_polarization,
    s_wave_basis,
)


def test_p_wave_polarization():
    k_hat = np.array([3.0, 0.0, 4.0])
    p_pol = p_wave_polarization(k_hat)
    # P wave is parallel to k_hat
    assert p_pol.shape == (1, 3)
    np.testing.assert_allclose(p_pol, [[0.6, 0.0, 0.8]])
    assert not p_pol.flags.writeable


def test_s_wave_basis():
    k_hat = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    e1, e2 = s_wave_basis(k_hat)

    assert e1.shape == (2, 3)
    assert e2.shape == (2, 3)
    assert not e1.flags.writeable
    assert not e2.flags.writeable

    # Check orthogonality for each direction
    for i in range(2):
        ki = k_hat[i] / np.linalg.norm(k_hat[i])
        e1_i = e1[i]
        e2_i = e2[i]

        assert np.abs(np.dot(e1_i, ki)) < 1e-15
        assert np.abs(np.dot(e2_i, ki)) < 1e-15
        assert np.abs(np.dot(e1_i, e2_i)) < 1e-15
        np.testing.assert_allclose(np.cross(ki, e1_i), e2_i, atol=1e-15)


def test_displacement_direction():
    k_hat = np.array([0.0, 0.0, 1.0])

    p_dir = displacement_direction("P", k_hat)
    s1_dir = displacement_direction("S1", k_hat)
    s2_dir = displacement_direction("S2", k_hat)

    # Check P wave shape and value
    assert p_dir.shape == (1, 3)
    np.testing.assert_allclose(p_dir, [[0.0, 0.0, 1.0]])

    # Check S1, S2 shapes and orthogonality
    assert s1_dir.shape == (1, 3)
    assert s2_dir.shape == (1, 3)

    e1 = s1_dir[0]
    e2 = s2_dir[0]
    k = p_dir[0]

    assert np.abs(np.dot(e1, k)) < 1e-15
    assert np.abs(np.dot(e2, k)) < 1e-15
    assert np.abs(np.dot(e1, e2)) < 1e-15

    # Invalid wave type
    with pytest.raises(ValueError, match="Unknown wave_type"):
        displacement_direction("S3", k_hat)


def test_s_wave_basis_vertical_anchored():
    # non-degenerate case
    k_hat = np.array([[1.0, 2.0, 3.0], [0.3, -0.4, 0.8]])
    v = np.array([0.0, 0.0, 1.0])
    e_sh, e_sv = s_wave_basis(k_hat, vertical_unit=v)

    assert e_sh.shape == (2, 3)
    assert e_sv.shape == (2, 3)
    assert not e_sh.flags.writeable
    assert not e_sv.flags.writeable

    k_norm = k_hat / np.linalg.norm(k_hat, axis=1, keepdims=True)
    v_norm = v / np.linalg.norm(v)

    for i in range(2):
        ki = k_norm[i]
        sh = e_sh[i]
        sv = e_sv[i]

        assert np.abs(np.dot(sh, ki)) < 1e-12
        assert np.abs(np.dot(sv, ki)) < 1e-12
        assert np.abs(np.dot(sh, sv)) < 1e-12
        # SH is horizontal (orthogonal to vertical_unit)
        assert np.abs(np.dot(sh, v_norm)) < 1e-12
        assert np.abs(np.linalg.norm(sh) - 1.0) < 1e-12
        assert np.abs(np.linalg.norm(sv) - 1.0) < 1e-12


def test_displacement_direction_sh_sv():
    k_hat = np.array([1.0, 2.0, 3.0])
    v = np.array([0.0, 1.0, 0.0])

    sh_dir = displacement_direction("SH", k_hat, vertical_unit=v)
    sv_dir = displacement_direction("SV", k_hat, vertical_unit=v)

    e_sh, e_sv = s_wave_basis(k_hat, vertical_unit=v)

    np.testing.assert_allclose(sh_dir, e_sh, rtol=1e-15)
    np.testing.assert_allclose(sv_dir, e_sv, rtol=1e-15)


def test_s_wave_basis_vertical_anchored_guards():
    k_hat = np.array([1.0, 2.0, 3.0])
    v = np.array([0.0, 0.0, 1.0])

    # degeneracy_tol out of bounds or invalid
    for invalid_tol in [0, -1.0, 1.1]:
        with pytest.raises(ValueError, match="degeneracy_tol must be in interval"):
            s_wave_basis(k_hat, vertical_unit=v, degeneracy_tol=invalid_tol)

    with pytest.raises(ValueError, match="degeneracy_tol must be finite"):
        s_wave_basis(k_hat, vertical_unit=v, degeneracy_tol=np.nan)

    with pytest.raises(TypeError, match="degeneracy_tol must be a finite float or int"):
        s_wave_basis(k_hat, vertical_unit=v, degeneracy_tol=True)

    with pytest.raises(TypeError, match="degeneracy_tol must be a finite float or int"):
        s_wave_basis(k_hat, vertical_unit=v, degeneracy_tol="invalid")

    # vertical_unit shape mismatch
    with pytest.raises(ValueError, match="vertical_unit must have shape"):
        s_wave_basis(k_hat, vertical_unit=np.eye(3))

    with pytest.raises(ValueError, match="vertical_unit must have shape"):
        s_wave_basis(k_hat, vertical_unit=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))

    # vertical_unit zero vector (check that it raises ValueError)
    with pytest.raises(ValueError, match="directions must not contain zero vectors"):
        s_wave_basis(k_hat, vertical_unit=[0.0, 0.0, 0.0])


def test_displacement_direction_custom():
    k_hat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pol = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # 1. custom with polarization provided
    custom_dir = displacement_direction("custom", k_hat, polarization=pol)
    assert custom_dir.shape == (2, 3)
    assert not custom_dir.flags.writeable
    np.testing.assert_allclose(custom_dir, pol)

    # 2. Non-custom with polarization must raise ValueError
    with pytest.raises(ValueError, match="polarization must be None for wave_type='P'"):
        displacement_direction("P", k_hat, polarization=pol)

    # 3. Custom without polarization must raise ValueError
    with pytest.raises(ValueError, match="polarization must be provided"):
        displacement_direction("custom", k_hat)

    # 4. Non-orthogonal polarization must raise ValueError
    non_ortho_pol = np.array([[0.5, 0.866, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(
        ValueError, match="Custom polarization must be orthogonal to k_hat"
    ):
        displacement_direction("custom", k_hat, polarization=non_ortho_pol)

    # 5. orthogonality_tol checks
    # If we allow tolerance to be large enough, it should pass
    passed_dir = displacement_direction(
        "custom", k_hat, polarization=non_ortho_pol, orthogonality_tol=0.6
    )
    expected_p = non_ortho_pol / np.linalg.norm(non_ortho_pol, axis=1, keepdims=True)
    np.testing.assert_allclose(passed_dir, expected_p)

    # Invalid orthogonality_tol
    for invalid_tol in [0, -1.0, 1.1]:
        with pytest.raises(ValueError, match="orthogonality_tol must be in interval"):
            displacement_direction(
                "custom", k_hat, polarization=pol, orthogonality_tol=invalid_tol
            )
    with pytest.raises(ValueError, match="orthogonality_tol must be finite"):
        displacement_direction(
            "custom", k_hat, polarization=pol, orthogonality_tol=np.nan
        )
    with pytest.raises(
        TypeError, match="orthogonality_tol must be a finite float or int"
    ):
        displacement_direction(
            "custom", k_hat, polarization=pol, orthogonality_tol=True
        )

    # 6. Shape mismatch
    mismatch_pol = np.array([[0.0, 1.0, 0.0]])
    with pytest.raises(
        ValueError, match="polarization shape .* must match k_hat shape"
    ):
        displacement_direction("custom", k_hat, polarization=mismatch_pol)

    # 7. scalar (3,) -> (1, 3) promotion and mixed-rank check
    k_scalar = np.array([1.0, 0.0, 0.0])
    p_scalar = np.array([0.0, 1.0, 0.0])
    res_scalar = displacement_direction("custom", k_scalar, polarization=p_scalar)
    assert res_scalar.shape == (1, 3)
    np.testing.assert_allclose(res_scalar, [[0.0, 1.0, 0.0]])

    # Mixed rank: k_hat is (2,3) but polarization is (3,)
    with pytest.raises(
        ValueError, match="polarization shape .* must match k_hat shape"
    ):
        displacement_direction("custom", k_hat, polarization=p_scalar)
