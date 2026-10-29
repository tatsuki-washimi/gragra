import numpy as np
import pytest

from gragra.waves.stochastic import StochasticPSParameters, stochastic_ps_realization


def test_stochastic_ps_realization_rng_validation():
    # TDD: test that passing invalid RNG raises TypeError
    with pytest.raises(TypeError):
        stochastic_ps_realization(
            rng=42,
            n_realizations=5,
            positions_m=[[1.0, 0.0, 0.0]],
            wavenumber_p_rad_m=2.0,
            velocity_ratio_s_over_p=0.5,
        )


def test_t1_strict_equivalence():
    rng = np.random.default_rng(42)
    pos = np.array([[1.0, 2.0, 3.0]])
    kp = 2.0
    ratio = 0.5
    ks = kp / ratio
    v_unit = np.array([0.0, 0.0, 1.0])
    f_dir = np.array([1.0, 0.0, 0.0])

    u_p, u_s, drawn = stochastic_ps_realization(
        rng,
        n_realizations=1,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        s_wave_type="SV",
        amplitude_p_m=1.5,
        amplitude_s_m=0.8,
        vertical_unit=v_unit,
        fixed_direction=f_dir,
        correlated_phase=True,
    )

    # Corresponding plane_wave_displacement call
    from gragra.waves.displacement import plane_wave_displacement

    # For P:
    ref_u_p_raw = plane_wave_displacement(
        positions_m=pos,
        k_hat=f_dir[np.newaxis],
        wavenumber_rad_m=kp,
        wave_type="P",
        amplitude_m=1.5,
        phase0=drawn.phase0_p[0],
        vertical_unit=v_unit,
    )
    # For S:
    ref_u_s_raw = plane_wave_displacement(
        positions_m=pos,
        k_hat=f_dir[np.newaxis],
        wavenumber_rad_m=ks,
        wave_type="SV",
        amplitude_m=0.8,
        phase0=drawn.phase0_s[0],
        vertical_unit=v_unit,
    )

    assert np.array_equal(u_p[0, 0, :], ref_u_p_raw[0, 0, 0, :])
    assert np.array_equal(u_s[0, 0, :], ref_u_s_raw[0, 0, 0, :])


def test_t2_pillar1_golden():
    rng = np.random.default_rng(42)
    pos = np.array([[1.0, 0.0, 0.0]])
    kp = 2.0
    ratio = 0.5

    # phase0_p expected: [4.86290927, 2.75755456, 5.39472984]
    expected_phase0 = np.array([4.86290927, 2.75755456, 5.39472984])

    u_p, u_s, drawn = stochastic_ps_realization(
        rng,
        n_realizations=3,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        s_wave_type="SV",
        amplitude_p_m=1.0,
        amplitude_s_m=1.0,
        vertical_unit=(0, 0, 1),
        fixed_direction=[1, 0, 0],
        correlated_phase=True,
    )

    np.testing.assert_allclose(drawn.phase0_p, expected_phase0, rtol=1e-8)
    np.testing.assert_allclose(drawn.phase0_s, expected_phase0, rtol=1e-8)

    # Expected displacement for first realization (n=0):
    # u_p[0, 0, :] == (0.83661389+0.54779302j, 0, 0)
    # u_s[0, 0, :] == (0, 0, -0.84626101+0.53276853j)
    expected_u_p_0 = np.array([0.83661389 + 0.54779302j, 0.0, 0.0])
    expected_u_s_0 = np.array([0.0, 0.0, -0.84626101 + 0.53276853j])

    np.testing.assert_allclose(u_p[0, 0, :], expected_u_p_0, rtol=1e-8)
    np.testing.assert_allclose(u_s[0, 0, :], expected_u_s_0, rtol=1e-8)

    # Also check that u_p matches analytic formula for all n
    for n in range(3):
        expected_p = np.array([1.0, 0.0, 0.0]) * np.exp(
            1j * (2.0 * 1.0 + drawn.phase0_p[n])
        )
        expected_s = np.array([0.0, 0.0, 1.0]) * np.exp(
            1j * (4.0 * 1.0 + drawn.phase0_s[n])
        )
        np.testing.assert_allclose(u_p[0, n, :], expected_p, rtol=1e-8)
        np.testing.assert_allclose(u_s[0, n, :], expected_s, rtol=1e-8)


@pytest.mark.parametrize("n_real", [1, 3, 100])
def test_t3_pillar4_fixed_point(n_real):
    from gragra.coherent.ensemble import ensemble_outer_cross_average

    rng = np.random.default_rng(42)
    pos = np.array([[1.0, 0.0, 0.0]])
    kp = 2.0
    ratio = 0.5

    u_p, u_s, _ = stochastic_ps_realization(
        rng,
        n_realizations=n_real,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        s_wave_type="SV",
        amplitude_p_m=1.0,
        amplitude_s_m=1.0,
        vertical_unit=(0, 0, 1),
        fixed_direction=[1, 0, 0],
        correlated_phase=True,
    )

    avg = ensemble_outer_cross_average(u_p, u_s)
    expected = np.cos(2.0) - 1j * np.sin(2.0)
    np.testing.assert_allclose(avg[0, 0, 2], expected, rtol=1e-14, atol=1e-14)


def test_t4_pillar3_structural_zeros():
    from gragra.coherent.ensemble import ensemble_outer_cross_average

    rng = np.random.default_rng(42)
    pos = np.array([[1.0, 0.0, 0.0]])
    kp = 2.0
    ratio = 0.5

    u_p, u_s, _ = stochastic_ps_realization(
        rng,
        n_realizations=10,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        s_wave_type="SV",
        amplitude_p_m=1.0,
        amplitude_s_m=1.0,
        vertical_unit=(0, 0, 1),
        fixed_direction=[1, 0, 0],
        correlated_phase=True,
    )

    avg = ensemble_outer_cross_average(u_p, u_s)[0]
    for i in range(3):
        for j in range(3):
            if (i, j) != (0, 2):
                assert avg[i, j] == 0.0


@pytest.mark.parametrize("n_realizations", [2000, 20000, 200000])
def test_t5_pillar2_statistical_convergence(n_realizations):
    from gragra.coherent.ensemble import ensemble_outer_cross_average

    rng = np.random.default_rng(0)
    pos = np.array([[1.0, 0.0, 0.0]])
    kp = 2.0
    ratio = 0.5

    u_p, u_s, _ = stochastic_ps_realization(
        rng,
        n_realizations=n_realizations,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        s_wave_type="SV",
        amplitude_p_m=1.0,
        amplitude_s_m=1.0,
        vertical_unit=(0, 0, 1),
        fixed_direction=None,
        correlated_phase=False,
    )

    avg = ensemble_outer_cross_average(u_p, u_s)[0]
    atol_r = 5.0 / np.sqrt(n_realizations)
    zeros = np.zeros((3, 3), dtype=np.complex128)

    err_max = np.max(np.abs(avg - zeros))
    assert err_max < atol_r, f"Max error {err_max} >= {atol_r} for R={n_realizations}"
    assert np.abs(avg[0, 2]) < atol_r, (
        f"Component (0,2) error {np.abs(avg[0, 2])} >= {atol_r} for R={n_realizations}"
    )


def test_t6_rng_contract():
    # TypeErrors:
    with pytest.raises(TypeError):
        stochastic_ps_realization(
            rng=42,
            n_realizations=5,
            positions_m=[[1, 0, 0]],
            wavenumber_p_rad_m=2.0,
            velocity_ratio_s_over_p=0.5,
        )

    # Check RNG consumption sequence:
    pos = np.array([[1.0, 0.0, 0.0]])
    kp = 2.0
    ratio = 0.5

    # 4 cases:
    # Case 1: fixed_direction is None, correlated_phase=True
    rng1 = np.random.default_rng(123)
    _, _, drawn1 = stochastic_ps_realization(
        rng1,
        n_realizations=5,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        fixed_direction=None,
        correlated_phase=True,
    )

    rng_ref = np.random.default_rng(123)
    ref_raw = rng_ref.standard_normal((5, 3))
    ref_k = ref_raw / np.linalg.norm(ref_raw, axis=1, keepdims=True)
    ref_phase_p = rng_ref.uniform(0.0, 2.0 * np.pi, size=5)

    np.testing.assert_allclose(drawn1.k_hat, ref_k)
    np.testing.assert_allclose(drawn1.phase0_p, ref_phase_p)
    np.testing.assert_allclose(drawn1.phase0_s, ref_phase_p)
    assert rng1.bit_generator.state == rng_ref.bit_generator.state

    # Case 2: fixed_direction is None, correlated_phase=False
    rng2 = np.random.default_rng(123)
    _, _, drawn2 = stochastic_ps_realization(
        rng2,
        n_realizations=5,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        fixed_direction=None,
        correlated_phase=False,
    )

    rng_ref = np.random.default_rng(123)
    ref_raw = rng_ref.standard_normal((5, 3))
    ref_k = ref_raw / np.linalg.norm(ref_raw, axis=1, keepdims=True)
    ref_phase_p = rng_ref.uniform(0.0, 2.0 * np.pi, size=5)
    ref_phase_s = rng_ref.uniform(0.0, 2.0 * np.pi, size=5)

    np.testing.assert_allclose(drawn2.k_hat, ref_k)
    np.testing.assert_allclose(drawn2.phase0_p, ref_phase_p)
    np.testing.assert_allclose(drawn2.phase0_s, ref_phase_s)
    assert rng2.bit_generator.state == rng_ref.bit_generator.state

    # Case 3: fixed_direction is NOT None, correlated_phase=True
    rng3 = np.random.default_rng(123)
    _, _, drawn3 = stochastic_ps_realization(
        rng3,
        n_realizations=5,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        fixed_direction=[1.0, 0.0, 0.0],
        correlated_phase=True,
    )

    rng_ref = np.random.default_rng(123)
    ref_phase_p = rng_ref.uniform(0.0, 2.0 * np.pi, size=5)

    np.testing.assert_allclose(drawn3.phase0_p, ref_phase_p)
    np.testing.assert_allclose(drawn3.phase0_s, ref_phase_p)
    assert rng3.bit_generator.state == rng_ref.bit_generator.state

    # Case 4: fixed_direction is NOT None, correlated_phase=False
    rng4 = np.random.default_rng(123)
    _, _, drawn4 = stochastic_ps_realization(
        rng4,
        n_realizations=5,
        positions_m=pos,
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
        fixed_direction=[1.0, 0.0, 0.0],
        correlated_phase=False,
    )

    rng_ref = np.random.default_rng(123)
    ref_phase_p = rng_ref.uniform(0.0, 2.0 * np.pi, size=5)
    ref_phase_s = rng_ref.uniform(0.0, 2.0 * np.pi, size=5)

    np.testing.assert_allclose(drawn4.phase0_p, ref_phase_p)
    np.testing.assert_allclose(drawn4.phase0_s, ref_phase_s)
    assert rng4.bit_generator.state == rng_ref.bit_generator.state


def test_t7_argument_validation():
    rng = np.random.default_rng(42)
    pos = np.array([[1.0, 0.0, 0.0]])
    kp = 2.0
    ratio = 0.5

    def check_val_error(expected_err, **kwargs):
        nonlocal rng
        state_before = rng.bit_generator.state
        params = {
            "rng": rng,
            "n_realizations": 5,
            "positions_m": pos,
            "wavenumber_p_rad_m": kp,
            "velocity_ratio_s_over_p": ratio,
        }
        params.update(kwargs)
        with pytest.raises(expected_err):
            stochastic_ps_realization(**params)
        assert rng.bit_generator.state == state_before

    # wavenumber_p_rad_m validation
    check_val_error(TypeError, wavenumber_p_rad_m=True)
    check_val_error(TypeError, wavenumber_p_rad_m="invalid")
    check_val_error(ValueError, wavenumber_p_rad_m=np.nan)
    check_val_error(ValueError, wavenumber_p_rad_m=0.0)
    check_val_error(ValueError, wavenumber_p_rad_m=-2.0)

    # velocity_ratio_s_over_p validation
    check_val_error(TypeError, velocity_ratio_s_over_p=True)
    check_val_error(TypeError, velocity_ratio_s_over_p="invalid")
    check_val_error(ValueError, velocity_ratio_s_over_p=np.inf)
    check_val_error(ValueError, velocity_ratio_s_over_p=0.0)
    check_val_error(ValueError, velocity_ratio_s_over_p=1.0)
    check_val_error(ValueError, velocity_ratio_s_over_p=-0.5)

    # n_realizations validation
    check_val_error(TypeError, n_realizations=True)
    check_val_error(TypeError, n_realizations=5.5)
    check_val_error(ValueError, n_realizations=0)
    check_val_error(ValueError, n_realizations=-1)

    # amplitude validation
    check_val_error(TypeError, amplitude_p_m=True)
    check_val_error(TypeError, amplitude_p_m="invalid")
    check_val_error(ValueError, amplitude_p_m=np.nan)

    check_val_error(TypeError, amplitude_s_m=True)
    check_val_error(ValueError, amplitude_s_m=np.inf)

    # s_wave_type validation
    check_val_error(ValueError, s_wave_type="invalid")

    # correlated_phase validation
    check_val_error(TypeError, correlated_phase=42)

    # vertical_unit validation
    check_val_error(ValueError, vertical_unit=[0, 0, 0])  # Zero vector
    check_val_error(ValueError, vertical_unit=[1, 2])  # Shape error

    # fixed_direction validation
    check_val_error(ValueError, fixed_direction=[0, 0, 0])  # Zero vector
    check_val_error(ValueError, fixed_direction=[1, 2])  # Shape error

    # positions_m validation
    stochastic_ps_realization(
        rng,
        n_realizations=5,
        positions_m=np.zeros((0, 3)),
        wavenumber_p_rad_m=kp,
        velocity_ratio_s_over_p=ratio,
    )
    check_val_error(ValueError, positions_m=[1, 2, 3])  # 1D vector not allowed
    check_val_error(ValueError, positions_m=[[1, 2]])  # Shape (1,2)
    check_val_error(ValueError, positions_m=[[1, 2, np.nan]])  # non-finite


@pytest.mark.parametrize(
    "direction_kwargs",
    [
        {"fixed_direction": [1e308, 1e308, 0.0]},
        {
            "fixed_direction": [1.0, 0.0, 0.0],
            "vertical_unit": [1e308, 1e308, 0.0],
        },
        {"fixed_direction": [1e-300, 1e-300, 0.0]},
        {
            "fixed_direction": [1.0, 0.0, 0.0],
            "vertical_unit": [1e-300, 1e-300, 0.0],
        },
    ],
)
def test_t7_extreme_finite_direction_vectors_are_normalized_stably(
    direction_kwargs,
):
    rng = np.random.default_rng(42)

    _, _, drawn = stochastic_ps_realization(
        rng,
        n_realizations=2,
        positions_m=[[1.0, 0.0, 0.0]],
        wavenumber_p_rad_m=2.0,
        velocity_ratio_s_over_p=0.5,
        **direction_kwargs,
    )

    assert np.isfinite(drawn.k_hat).all()
    assert np.isfinite(drawn.vertical_unit).all()
    np.testing.assert_allclose(np.linalg.norm(drawn.k_hat, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(drawn.vertical_unit), 1.0)


def test_t8_shape_dtype_readonly():
    rng = np.random.default_rng(42)
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # N=2
    u_p, u_s, drawn = stochastic_ps_realization(
        rng,
        n_realizations=5,
        positions_m=pos,
        wavenumber_p_rad_m=2.0,
        velocity_ratio_s_over_p=0.5,
    )

    assert u_p.shape == (2, 5, 3)
    assert u_p.dtype == np.complex128
    assert not u_p.flags.writeable

    assert u_s.shape == (2, 5, 3)
    assert u_s.dtype == np.complex128
    assert not u_s.flags.writeable

    assert drawn.k_hat.shape == (5, 3)
    assert drawn.k_hat.dtype == np.float64
    assert not drawn.k_hat.flags.writeable

    assert drawn.phase0_p.shape == (5,)
    assert drawn.phase0_p.dtype == np.float64
    assert not drawn.phase0_p.flags.writeable

    assert drawn.phase0_s.shape == (5,)
    assert drawn.phase0_s.dtype == np.float64
    assert not drawn.phase0_s.flags.writeable

    assert drawn.vertical_unit.shape == (3,)
    assert drawn.vertical_unit.dtype == np.float64
    assert not drawn.vertical_unit.flags.writeable

    from dataclasses import FrozenInstanceError

    with pytest.raises((FrozenInstanceError, AttributeError)):
        drawn.amplitude_p_m = 2.0


def test_stochastic_parameters_defensively_copy_directly_constructed_arrays():
    k_hat = np.array([[2.0, 0.0, 0.0]])
    vertical_unit = np.array([0.0, 0.0, 2.0])
    phase0_p = np.array([0.25])
    phase0_s = np.array([0.5])

    params = StochasticPSParameters(
        k_hat=k_hat,
        vertical_unit=vertical_unit,
        wave_type_p="P",
        wave_type_s="SV",
        amplitude_p_m=1.0,
        amplitude_s_m=-2.0,
        wavenumber_p_rad_m=2.0,
        wavenumber_s_rad_m=4.0,
        phase0_p=phase0_p,
        phase0_s=phase0_s,
    )
    k_hat[0, 0] = 9.0
    vertical_unit[2] = 9.0
    phase0_p[0] = 9.0
    phase0_s[0] = 9.0

    np.testing.assert_array_equal(params.k_hat, [[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(params.vertical_unit, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(params.phase0_p, [0.25])
    np.testing.assert_array_equal(params.phase0_s, [0.5])
    assert not params.k_hat.flags.writeable
    assert not params.vertical_unit.flags.writeable
    assert not params.phase0_p.flags.writeable
    assert not params.phase0_s.flags.writeable


def test_t9_rng_reconstitution():
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    g_state = np.random.get_state()

    for fixed_dir in [None, [1.0, 0.0, 0.0]]:
        for corr in [True, False]:
            rng1 = np.random.default_rng(999)
            state_before = rng1.bit_generator.state

            u_p1, u_s1, drawn1 = stochastic_ps_realization(
                rng1,
                n_realizations=5,
                positions_m=pos,
                wavenumber_p_rad_m=2.0,
                velocity_ratio_s_over_p=0.5,
                fixed_direction=fixed_dir,
                correlated_phase=corr,
            )

            state_after = rng1.bit_generator.state
            assert state_before != state_after

            rng2 = np.random.default_rng(999)
            u_p2, u_s2, drawn2 = stochastic_ps_realization(
                rng2,
                n_realizations=5,
                positions_m=pos,
                wavenumber_p_rad_m=2.0,
                velocity_ratio_s_over_p=0.5,
                fixed_direction=fixed_dir,
                correlated_phase=corr,
            )

            assert np.array_equal(u_p1, u_p2)
            assert np.array_equal(u_s1, u_s2)
            assert np.array_equal(drawn1.k_hat, drawn2.k_hat)
            assert np.array_equal(drawn1.phase0_p, drawn2.phase0_p)
            assert np.array_equal(drawn1.phase0_s, drawn2.phase0_s)
            assert rng2.bit_generator.state == state_after

            from gragra.waves.displacement import plane_wave_displacement

            for r in range(5):
                ref_p = plane_wave_displacement(
                    positions_m=pos,
                    k_hat=drawn1.k_hat[r : r + 1],
                    wavenumber_rad_m=drawn1.wavenumber_p_rad_m,
                    wave_type="P",
                    amplitude_m=drawn1.amplitude_p_m,
                    phase0=drawn1.phase0_p[r],
                    vertical_unit=drawn1.vertical_unit,
                )
                np.testing.assert_allclose(
                    u_p1[:, r, :], ref_p[:, 0, 0, :], rtol=1e-13, atol=1e-15
                )

                ref_s = plane_wave_displacement(
                    positions_m=pos,
                    k_hat=drawn1.k_hat[r : r + 1],
                    wavenumber_rad_m=drawn1.wavenumber_s_rad_m,
                    wave_type=drawn1.wave_type_s,
                    amplitude_m=drawn1.amplitude_s_m,
                    phase0=drawn1.phase0_s[r],
                    vertical_unit=drawn1.vertical_unit,
                )
                np.testing.assert_allclose(
                    u_s1[:, r, :], ref_s[:, 0, 0, :], rtol=1e-13, atol=1e-15
                )

    g_state_after = np.random.get_state()
    assert g_state[0] == g_state_after[0]
    assert np.array_equal(g_state[1], g_state_after[1])
    assert g_state[2] == g_state_after[2]
    assert g_state[3] == g_state_after[3]
    assert g_state[4] == g_state_after[4]
