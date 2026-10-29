"""Numba parity tests for displacement-transfer potential and gradient."""

import numpy as np
import pytest

pytest.importorskip("numba")

import gragra.kernels._numba_inverse_square as numba_kernels
import gragra.kernels.inverse_square as numpy_kernels
from gragra.integrators.point_direct import direct_point_displacement_sum

SOURCES = np.array(
    [[0.2, -0.4, 0.7], [-1.1, 0.6, 0.3], [0.8, 1.2, -0.9]],
    dtype=np.float64,
)
MASS = np.array([1.5, 0.7, 2.1], dtype=np.float64)
TARGETS = np.array([[2.0, -0.3, 0.4], [-0.6, -1.4, 1.7]], dtype=np.float64)
D_REAL = np.array(
    [[0.3, -0.2, 0.5], [-0.1, 0.4, 0.2], [0.6, 0.1, -0.3]],
    dtype=np.float64,
)
D_COMPLEX = D_REAL + 1j * np.array(
    [[-0.2, 0.7, 0.1], [0.5, -0.3, 0.8], [0.2, -0.6, 0.4]],
    dtype=np.float64,
)


@pytest.mark.parametrize(
    ("observable", "shape"), [("potential", (2,)), ("gradient", (2, 3, 3))]
)
@pytest.mark.parametrize("displacement", [D_REAL, D_COMPLEX])
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_numba_displacement_derivatives_match_numpy(
    observable, shape, displacement, epsilon
):
    """The compiled derivatives match the G-free NumPy contracts."""
    numpy_contract = getattr(numpy_kernels, f"displacement_{observable}_contract")
    numba_contract = getattr(numba_kernels, f"displacement_{observable}_contract")

    reference = numpy_contract(
        SOURCES, MASS, displacement, TARGETS, softening_m=epsilon, chunk_size=2
    )
    got = numba_contract(
        SOURCES, MASS, displacement, TARGETS, softening_m=epsilon, chunk_size=2
    )

    assert got.shape == shape
    assert got.dtype == np.result_type(displacement, np.float64)
    assert got.flags.writeable is False
    exponent = 2 if observable == "potential" else 4
    displacement_size = np.max(np.linalg.norm(displacement, axis=1))
    for target_index, target in enumerate(TARGETS):
        radius = np.min(np.linalg.norm(SOURCES - target, axis=1))
        scale = MASS.sum() * displacement_size / radius**exponent
        error = np.linalg.norm(got[target_index] - reference[target_index])
        tolerance = 1e-12 * np.linalg.norm(reference[target_index]) + 1e-13 * scale
        assert error <= tolerance, (observable, epsilon, target_index, error, tolerance)


@pytest.mark.parametrize("observable", ["potential", "gradient"])
def test_numba_displacement_dispatch_matches_numpy(observable):
    numpy_contract = getattr(numpy_kernels, f"displacement_{observable}_contract")
    reference = numpy_contract(SOURCES, MASS, D_REAL, TARGETS, softening_m=0.15)
    got = direct_point_displacement_sum(
        MASS,
        D_REAL,
        SOURCES,
        TARGETS,
        observable=observable,
        softening_m=0.15,
        chunk_size=1,
        backend="numba",
    )
    np.testing.assert_allclose(got, reference, rtol=1e-12, atol=1e-13)
    assert got.flags.writeable is False


def test_numba_displacement_axis_goldens():
    potential = numba_kernels.displacement_potential_contract(
        [[0.0, 0.0, 0.0]], [1.0], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]
    )
    gradient = numba_kernels.displacement_gradient_contract(
        [[0.0, 0.0, 0.0]], [1.0], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(potential, [-1.0])
    np.testing.assert_allclose(gradient[0], np.diag([6.0, -3.0, -3.0]))


def test_numba_displacement_gradient_softened_trace_identity():
    eps = 0.35
    got = numba_kernels.displacement_gradient_contract(
        SOURCES, MASS, D_COMPLEX, TARGETS, softening_m=eps
    )
    diff = SOURCES[np.newaxis, :, :] - TARGETS[:, np.newaxis, :]
    q = np.sum(diff**2, axis=-1) + eps**2
    z = np.einsum("mni,ni->mn", diff, D_COMPLEX)
    expected = 15.0 * np.sum(MASS[np.newaxis, :] * eps**2 * z * q ** (-3.5), axis=1)
    np.testing.assert_allclose(
        np.trace(got, axis1=-2, axis2=-1), expected, rtol=1e-12, atol=1e-13
    )


@pytest.mark.parametrize("observable", ["potential", "gradient"])
def test_numba_displacement_collision_reports_first_source_index(observable):
    contract = getattr(numba_kernels, f"displacement_{observable}_contract")
    with pytest.raises(
        ValueError,
        match=r"(?i)collision.*target.*0.*source.*1.*zero softening",
    ):
        contract(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [1.0, 2.0, 3.0],
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            [[1.0, 1.0, 1.0]],
        )


@pytest.mark.parametrize("observable", ["potential", "gradient"])
def test_numba_displacement_empty_and_invalid_inputs(observable):
    contract = getattr(numba_kernels, f"displacement_{observable}_contract")
    empty = contract(
        np.empty((0, 3)),
        np.empty(0),
        np.empty((0, 3), dtype=np.complex128),
        TARGETS,
    )
    assert empty.shape[0] == len(TARGETS)
    assert empty.dtype == np.complex128
    assert np.count_nonzero(empty) == 0
    assert empty.flags.writeable is False
    empty_targets = contract(SOURCES, MASS, D_REAL, np.empty((0, 3)))
    assert empty_targets.shape == ((0,) if observable == "potential" else (0, 3, 3))
    assert empty_targets.dtype == np.float64
    assert empty_targets.flags.writeable is False

    with pytest.raises(ValueError, match="chunk_size"):
        contract(SOURCES, MASS, D_REAL, TARGETS, chunk_size=0)
    with pytest.raises(ValueError, match="softening_m"):
        contract(SOURCES, MASS, D_REAL, TARGETS, softening_m=-0.1)


def test_numba_displacement_chunk_size_is_validation_only():
    for observable in ("potential", "gradient"):
        contract = getattr(numba_kernels, f"displacement_{observable}_contract")
        none = contract(SOURCES, MASS, D_COMPLEX, TARGETS, chunk_size=None)
        small = contract(SOURCES, MASS, D_COMPLEX, TARGETS, chunk_size=1)
        np.testing.assert_array_equal(none, small)


@pytest.mark.parametrize(
    ("observable", "distance_power"), [("potential", 2), ("gradient", 4)]
)
def test_numba_displacement_matches_source_shift_central_difference(
    observable, distance_power
):
    source = np.array([0.3, -0.2, 0.4])
    target = np.array([[2.0, 0.5, -0.6]])
    mass = np.array([2.5])
    displacement = np.array([[0.4, -0.3, 0.2]])
    radius = np.linalg.norm(source - target[0])
    amplitude = np.linalg.norm(displacement[0])
    step = 1e-6 * radius / amplitude

    field_contract = getattr(numba_kernels, f"{observable}_contract")
    displacement_contract = getattr(
        numba_kernels, f"displacement_{observable}_contract"
    )
    plus = field_contract([source + step * displacement[0]], mass, target)[0]
    minus = field_contract([source - step * displacement[0]], mass, target)[0]
    finite_difference = (plus - minus) / (2.0 * step)
    exact = displacement_contract(source[None, :], mass, displacement, target)[0]
    scale = mass[0] * amplitude / radius**distance_power

    np.testing.assert_allclose(
        finite_difference / scale, exact / scale, rtol=1e-5, atol=1e-9
    )


def test_numba_displacement_gradient_rotates_covariantly():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_sources = SOURCES @ rotation.T
    rotated_targets = TARGETS @ rotation.T
    rotated_displacement = D_COMPLEX @ rotation.T

    gradient = numba_kernels.displacement_gradient_contract(
        SOURCES, MASS, D_COMPLEX, TARGETS
    )
    rotated = numba_kernels.displacement_gradient_contract(
        rotated_sources, MASS, rotated_displacement, rotated_targets
    )
    np.testing.assert_allclose(
        rotated,
        np.einsum("ij,mjk,lk->mil", rotation, gradient, rotation),
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    ("observable", "distance_power"), [("potential", 2), ("gradient", 4)]
)
def test_numba_displacement_forward_shift_has_second_order_remainder(
    observable, distance_power
):
    source = np.array([0.3, -0.2, 0.4])
    target = np.array([[2.0, 0.5, -0.6]])
    mass = np.array([2.5])
    direction = np.array([0.4, -0.3, 0.2])
    direction /= np.linalg.norm(direction)
    radius = np.linalg.norm(source - target[0])
    field_contract = getattr(numba_kernels, f"{observable}_contract")
    displacement_contract = getattr(
        numba_kernels, f"displacement_{observable}_contract"
    )
    base = field_contract([source], mass, target)[0]
    amplitudes = radius * np.array([1e-3, 5e-4, 2.5e-4])
    remainders = []
    for amplitude in amplitudes:
        displacement = amplitude * direction
        shifted = field_contract([source + displacement], mass, target)[0]
        first_order = displacement_contract(
            [source], mass, displacement[None, :], target
        )[0]
        remainders.append(np.linalg.norm(shifted - base - first_order))

    ratios = np.asarray(remainders[:-1]) / np.asarray(remainders[1:])
    assert np.all((ratios >= 3.0) & (ratios <= 5.0)), (
        observable,
        distance_power,
        ratios,
    )
