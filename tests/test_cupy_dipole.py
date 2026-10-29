"""Correctness tests for the CuPy-accelerated dipole/total contraction (Phase C-1)."""

import numpy as np
import pytest

# Two-stage module-level skip: cupy importable, then a usable device.
cupy = pytest.importorskip("cupy")


def _cuda_available() -> bool:
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


if not _cuda_available():
    pytest.skip("no usable CUDA device", allow_module_level=True)

from gragra.integrators import direct_point_dipole_sum  # noqa: E402
from gragra.observables import displacement_transfer_acceleration  # noqa: E402
from gragra.targets.points import TargetPoints  # noqa: E402

RTOL = 1e-12
ATOL = 1e-15


@pytest.fixture
def dipole_inputs():
    sources_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, -1.0, 0.5],
            [-0.5, 2.0, -1.5],
            [1.5, 0.0, -2.0],
        ],
        dtype=np.float64,
    )
    mass_kg = np.array([10.0, 20.0, 15.0, 8.0, 12.0], dtype=np.float64)

    # Complex displacement with non-zero imaginary parts
    disp_complex = np.array(
        [
            [1.0 + 0.5j, -0.2j, 0.3],
            [-0.1, 0.5 + 1.0j, -0.5j],
            [0.2 - 0.2j, 1.0, 0.5],
            [0.0, -0.5, 1.0 + 0.1j],
            [-1.0j, 0.2, 0.0],
        ],
        dtype=np.complex128,
    )

    # Real displacement
    disp_real = np.real(disp_complex).copy()

    targets_xyz = np.array(
        [[0.5, 0.5, 0.5], [3.0, 2.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float64
    )

    return sources_xyz, mass_kg, disp_complex, disp_real, targets_xyz


def test_cupy_dipole_parity_minimal():
    """Verify minimal dipole cupy execution."""
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets_xyz = [[1.0, 0.0, 0.0]]

    out = direct_point_dipole_sum(
        mass_kg,
        displacement_m,
        sources_xyz,
        targets_xyz,
        backend="cupy",
    )
    assert out.shape == (1, 3)


def test_cupy_dipole_complex_non_chunked(dipole_inputs):
    """Case 1: non-chunked execution with complex displacement_m."""
    sources, mass, disp_c, _, targets = dipole_inputs

    # numpy reference
    ref = direct_point_dipole_sum(
        mass,
        disp_c,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="numpy",
    )

    # cupy result
    got = direct_point_dipole_sum(
        mass,
        disp_c,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="cupy",
    )

    assert got.dtype == np.complex128
    assert not got.flags.writeable
    assert np.allclose(ref, got, rtol=RTOL, atol=ATOL)

    # Test observable layer too
    obs_ref = displacement_transfer_acceleration(
        sources,
        mass,
        disp_c,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="numpy",
    )
    obs_got = displacement_transfer_acceleration(
        sources,
        mass,
        disp_c,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="cupy",
    )
    assert obs_got.dtype == np.complex128
    assert not obs_got.flags.writeable
    assert np.allclose(obs_ref, obs_got, rtol=RTOL, atol=ATOL)


def test_cupy_dipole_complex_chunked(dipole_inputs):
    """Case 2: chunked execution with complex displacement_m."""
    sources, mass, disp_c, _, targets = dipole_inputs

    ref = direct_point_dipole_sum(
        mass,
        disp_c,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="numpy",
    )

    got = direct_point_dipole_sum(
        mass,
        disp_c,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=2,
        backend="cupy",
    )

    assert got.dtype == np.complex128
    assert np.allclose(ref, got, rtol=RTOL, atol=ATOL)


def test_cupy_dipole_real(dipole_inputs):
    """Case 3: real displacement_m asserts float64 output."""
    sources, mass, _, disp_r, targets = dipole_inputs

    # non-chunked
    got_none = direct_point_dipole_sum(
        mass,
        disp_r,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="cupy",
    )
    assert got_none.dtype == np.float64

    # chunked
    got_chunk = direct_point_dipole_sum(
        mass,
        disp_r,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=2,
        backend="cupy",
    )
    assert got_chunk.dtype == np.float64

    # numpy ref comparison
    ref = direct_point_dipole_sum(
        mass,
        disp_r,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="numpy",
    )
    assert np.allclose(ref, got_none, rtol=RTOL, atol=ATOL)
    assert np.allclose(ref, got_chunk, rtol=RTOL, atol=ATOL)


def test_cupy_dipole_dtype_driver(dipole_inputs):
    """Verify that mass_kg does not drive output dtype, only displacement_m."""
    sources, mass, disp_c, disp_r, targets = dipole_inputs

    # Mass is float64, complex displacement -> complex128
    res_c = direct_point_dipole_sum(mass, disp_c, sources, targets, backend="cupy")
    assert res_c.dtype == np.complex128

    # Mass is float64, real displacement -> float64
    res_r = direct_point_dipole_sum(mass, disp_r, sources, targets, backend="cupy")
    assert res_r.dtype == np.float64


def test_cupy_dipole_float32_input(dipole_inputs):
    """Verify float32/float64 input behavior."""
    sources, mass, _, disp_r, targets = dipole_inputs

    # Convert inputs to float32
    disp_r_32 = disp_r.astype(np.float32)

    got = direct_point_dipole_sum(mass, disp_r_32, sources, targets, backend="cupy")
    # The output should still be float64 due to normalization promotion
    assert got.dtype == np.float64

    ref = direct_point_dipole_sum(mass, disp_r_32, sources, targets, backend="numpy")
    assert np.allclose(ref, got, rtol=RTOL, atol=ATOL)


def test_cupy_dipole_target_points_parity(dipole_inputs):
    """Regression guard: TargetPoints targets must reach the cupy kernel.

    Before the point_direct.py fix, the raw (unnormalized) ``targets``
    argument (a ``TargetPoints`` instance here) was passed into
    ``_validate_dipole_inputs``, which only accepts (N, 3) array-like and
    would reject a ``TargetPoints`` instance outright. Verify numerical
    parity between numpy and cupy backends when targets is a TargetPoints.
    """
    sources, mass, disp_c, _, targets_xyz = dipole_inputs
    targets = TargetPoints(targets_xyz)

    ref = direct_point_dipole_sum(
        mass,
        disp_c,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="numpy",
    )
    got = direct_point_dipole_sum(
        mass,
        disp_c,
        sources,
        targets,
        softening_m=0.1,
        chunk_size=None,
        backend="cupy",
    )

    assert np.allclose(ref, got, rtol=RTOL, atol=ATOL)


def test_cupy_dipole_single_vector_target_parity():
    """Regression guard: a bare (3,) single-vector target must reach the kernel.

    Before the fix, a (3,) target was promoted to (1, 3) by
    ``_normalize_targets`` but the promoted array was discarded in favor
    of the raw (3,) ``targets`` argument passed to
    ``_validate_dipole_inputs`` (which strictly requires (N, 3)),
    misclassifying this case as a ValueError. Also confirms the output
    shape is (1, 3).
    """
    sources_xyz = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    mass_kg = [1.0, 2.0]
    displacement_m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]]
    target_vector = [2.0, 0.0, 0.0]  # single (3,) vector, not a table

    ref = direct_point_dipole_sum(
        mass_kg,
        displacement_m,
        sources_xyz,
        target_vector,
        softening_m=0.1,
        backend="numpy",
    )
    got = direct_point_dipole_sum(
        mass_kg,
        displacement_m,
        sources_xyz,
        target_vector,
        softening_m=0.1,
        backend="cupy",
    )

    assert ref.shape == (1, 3)
    assert got.shape == (1, 3)
    assert np.allclose(ref, got, rtol=RTOL, atol=ATOL)


def test_cupy_dipole_global_collision_indices():
    """Verify that global source index is recovered across chunk boundaries."""
    sources = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    mass = [1.0, 1.0, 1.0]
    disp = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    targets = [[5.0, 5.0, 5.0], [2.0, 2.0, 2.0]]  # target 1 hits source 2

    # collision index should be target 1 and source 2 (0-indexed)
    with pytest.raises(
        ValueError,
        match=r"(?i)collision.*target.*1.*source.*2.*zero softening",
    ):
        direct_point_dipole_sum(
            mass,
            disp,
            sources,
            targets,
            softening_m=0.0,
            chunk_size=1,
            backend="cupy",
        )


def _antipodal_dipole_cancellation_fixture():
    """8-point antipodal-pair fixture for the C-1 cancellation registry.

    8 sources on a circle of radius r=10m in the z=0 plane at 45-degree
    increments (4 antipodal pairs), target at the origin, softening=0.1m,
    mass=1000kg/source, displacement scaled radially outward from each
    source's own position ("breathing mode", h=0.001m). By antipodal
    symmetry (hand-verified in the PR design discussion and cross-checked
    against the numpy reference below), the x and y components of the
    G-free dipole contraction cancel to a structural zero; z is trivially
    zero since all sources/target share z=0. This is a cancellation-sensitive
    quantity per coherent_statistics_conventions.md §12: term_{d,i} =
    sum_j K_T_ij(Delta_d) * displacement_d[j] (source-indexed, pre-weight),
    w_d = mass_kg[d], gross_scale_i = sum_d |w_d| * |term_{d,i}|.
    """
    r = 10.0
    theta = np.deg2rad(np.arange(8) * 45.0)
    positions = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros(8)], axis=-1)
    target = [[0.0, 0.0, 0.0]]
    h = 0.001
    displacement = np.stack(
        [h * np.cos(theta), h * np.sin(theta), np.zeros(8)], axis=-1
    )
    mass_kg = np.full(8, 1000.0)
    softening_m = 0.1
    return positions, mass_kg, displacement, target, softening_m


@pytest.mark.parametrize("chunk_size", [2, 5])
def test_cupy_dipole_source_cancellation_registry(chunk_size):
    """C1-dipole-source-cancellation: gross-scale-relative cancellation atol.

    Cancellation registry entry (docs/design/cupy_backend.md §8.5):
    registry-id: C1-dipole-source-cancellation-chunk2
    registry-id: C1-dipole-source-cancellation-chunk5

    The numpy reference is a structural zero for x, y (antipodal
    cancellation) so a relative-error (max_rel) comparison is undefined;
    the pass criterion is gross-scale-relative instead:
    abs(cupy - numpy) <= coefficient * gross_scale, coefficient=1e-15
    (measured max coefficient on RTX A1000/CuPy 14.1.1 was ~9e-17 for this
    fixture; 1e-15 keeps a comfortable margin without being loose enough to
    hide a real regression). The z-component is checked with a plain
    absolute epsilon guard since its gross_scale is exactly 0 (all sources
    at z=0), making a gross-scale-relative bound undefined there too.
    """
    positions, mass_kg, displacement, target, softening_m = (
        _antipodal_dipole_cancellation_fixture()
    )

    acc_np = direct_point_dipole_sum(
        mass_kg,
        displacement,
        positions,
        target,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend="numpy",
    )
    acc_cp = direct_point_dipole_sum(
        mass_kg,
        displacement,
        positions,
        target,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend="cupy",
    )

    delta = np.asarray(positions) - np.asarray(target)[0]
    r_eff = np.sqrt(np.sum(delta**2, axis=-1) + softening_m**2)
    outer = delta[:, :, None] * delta[:, None, :]
    identity = np.eye(3)
    k_t = 3 * outer / r_eff[:, None, None] ** 5 - identity / r_eff[:, None, None] ** 3
    term_d = np.einsum("dij,dj->di", k_t, displacement)
    gross_scale = np.sum(np.abs(mass_kg)[:, None] * np.abs(term_d), axis=0)

    # registry-id: C1-dipole-source-cancellation-chunk2
    # registry-id: C1-dipole-source-cancellation-chunk5
    coefficient = 1e-15
    diff = np.abs(np.asarray(acc_cp)[0] - np.asarray(acc_np)[0])
    for i in (0, 1):
        assert diff[i] <= coefficient * gross_scale[i], (
            f"component {i}: |cupy-numpy|={diff[i]!r} exceeds "
            f"coefficient*gross_scale={coefficient * gross_scale[i]!r}"
        )
    assert diff[2] <= 1e-14  # z: gross_scale is structurally 0, use abs epsilon
