"""Tests for G-free displacement derivatives of point-mass observables."""

import numpy as np
import pytest

import gragra.kernels.inverse_square as kernels
from gragra.integrators.point_direct import (
    direct_point_dipole_sum,
    direct_point_displacement_sum,
)

SOURCES = np.array([[0.2, -0.4, 0.7], [-1.1, 0.6, 0.3], [0.8, 1.2, -0.9]])
MASS = np.array([1.5, 0.7, 2.1])
TARGETS = np.array([[2.0, -0.3, 0.4], [-0.6, -1.4, 1.7]])
D1 = np.array([[0.3, -0.2, 0.5], [-0.1, 0.4, 0.2], [0.6, 0.1, -0.3]])
D2 = np.array([[-0.2, 0.7, 0.1], [0.5, -0.3, 0.8], [0.2, -0.6, 0.4]])


@pytest.mark.parametrize(
    ("target_x", "expected_potential", "expected_gradient"),
    [
        (1.0, -1.0, np.diag([6.0, -3.0, -3.0])),
        (-1.0, 1.0, np.diag([-6.0, 3.0, 3.0])),
    ],
)
def test_displacement_kernel_axis_goldens(
    target_x, expected_potential, expected_gradient
):
    """The source-minus-target sign convention fixes both axial goldens."""
    assert hasattr(kernels, "displacement_potential_contract")
    assert hasattr(kernels, "displacement_gradient_contract")
    sources = [[0.0, 0.0, 0.0]]
    mass = [1.0]
    displacement = [[1.0, 0.0, 0.0]]
    targets = [[target_x, 0.0, 0.0]]

    potential = kernels.displacement_potential_contract(
        sources, mass, displacement, targets
    )
    acceleration = kernels.dipole_contract(sources, mass, displacement, targets)
    gradient = kernels.displacement_gradient_contract(
        sources, mass, displacement, targets
    )

    assert potential.shape == (1,)
    assert acceleration.shape == (1, 3)
    assert gradient.shape == (1, 3, 3)
    assert np.allclose(potential[0], expected_potential)
    assert np.allclose(acceleration[0], [-2.0, 0.0, 0.0])
    assert np.allclose(gradient[0], expected_gradient)


@pytest.mark.parametrize(
    "contract_name",
    ["displacement_potential_contract", "displacement_gradient_contract"],
)
def test_displacement_contract_complex_linearity_chunking_and_source_split(
    contract_name,
):
    contract = getattr(kernels, contract_name)
    complex_displacement = D1 + 1j * D2

    got = contract(
        SOURCES,
        MASS,
        complex_displacement,
        TARGETS,
        softening_m=0.15,
        chunk_size=2,
    )
    linear = contract(SOURCES, MASS, D1, TARGETS, softening_m=0.15) + 1j * contract(
        SOURCES, MASS, D2, TARGETS, softening_m=0.15
    )
    split = contract(
        SOURCES[:1], MASS[:1], complex_displacement[:1], TARGETS, softening_m=0.15
    ) + contract(
        SOURCES[1:], MASS[1:], complex_displacement[1:], TARGETS, softening_m=0.15
    )

    assert got.dtype == np.complex128
    assert got.flags.writeable is False
    assert np.allclose(got, linear, rtol=1e-12, atol=1e-13)
    assert np.allclose(got, split, rtol=1e-12, atol=1e-13)


def test_displacement_gradient_is_symmetric_and_has_softened_trace_identity():
    eps = 0.35
    displacement = D1 + 0.25j * D2
    got = kernels.displacement_gradient_contract(
        SOURCES, MASS, displacement, TARGETS, softening_m=eps, chunk_size=1
    )

    diff = SOURCES[np.newaxis, :, :] - TARGETS[:, np.newaxis, :]
    q = np.sum(diff**2, axis=-1) + eps**2
    z = np.einsum("mni,ni->mn", diff, displacement)
    expected_trace = 15.0 * np.sum(
        MASS[np.newaxis, :] * eps**2 * z * q ** (-3.5), axis=1
    )

    assert np.allclose(got, np.swapaxes(got, -1, -2), rtol=1e-12, atol=1e-13)
    assert np.allclose(
        np.trace(got, axis1=-2, axis2=-1),
        expected_trace,
        rtol=1e-12,
        atol=1e-13,
    )


def test_displacement_contracts_rotate_covariantly():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_sources = SOURCES @ rotation.T
    rotated_targets = TARGETS @ rotation.T
    rotated_displacement = D1 @ rotation.T

    potential = kernels.displacement_potential_contract(SOURCES, MASS, D1, TARGETS)
    rotated_potential = kernels.displacement_potential_contract(
        rotated_sources, MASS, rotated_displacement, rotated_targets
    )
    acceleration = kernels.dipole_contract(SOURCES, MASS, D1, TARGETS)
    rotated_acceleration = kernels.dipole_contract(
        rotated_sources, MASS, rotated_displacement, rotated_targets
    )
    gradient = kernels.displacement_gradient_contract(SOURCES, MASS, D1, TARGETS)
    rotated_gradient = kernels.displacement_gradient_contract(
        rotated_sources, MASS, rotated_displacement, rotated_targets
    )

    assert np.allclose(rotated_potential, potential, rtol=1e-12, atol=1e-13)
    assert np.allclose(rotated_acceleration, acceleration @ rotation.T)
    assert np.allclose(
        rotated_gradient,
        np.einsum("ij,mjk,lk->mil", rotation, gradient, rotation),
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    ("field_contract", "displacement_contract", "distance_power"),
    [
        (kernels.potential_contract, kernels.displacement_potential_contract, 2),
        (kernels.acceleration_contract, kernels.dipole_contract, 3),
        (kernels.gradient_contract, kernels.displacement_gradient_contract, 4),
    ],
)
def test_displacement_contracts_match_dimensionless_source_shift_finite_difference(
    field_contract, displacement_contract, distance_power
):
    source = np.array([0.3, -0.2, 0.4])
    target = np.array([2.0, 0.5, -0.6])
    mass = 2.5
    displacement = np.array([0.4, -0.3, 0.2])
    radius = np.linalg.norm(source - target)
    amplitude = np.linalg.norm(displacement)
    step = 1e-6 * radius / amplitude

    plus = field_contract([source + step * displacement], [mass], [target])[0]
    minus = field_contract([source - step * displacement], [mass], [target])[0]
    finite_difference = (plus - minus) / (2.0 * step)
    exact = displacement_contract([source], [mass], [displacement], [target])[0]
    scale = mass * amplitude / radius**distance_power

    assert np.allclose(
        finite_difference / scale,
        exact / scale,
        rtol=1e-5,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("field_contract", "displacement_contract"),
    [
        (kernels.potential_contract, kernels.displacement_potential_contract),
        (kernels.acceleration_contract, kernels.dipole_contract),
        (kernels.gradient_contract, kernels.displacement_gradient_contract),
    ],
)
def test_forward_source_shift_has_second_order_remainder(
    field_contract, displacement_contract
):
    source = np.array([0.3, -0.2, 0.4])
    target = np.array([2.0, 0.5, -0.6])
    mass = 2.5
    direction = np.array([0.4, -0.3, 0.2])
    direction /= np.linalg.norm(direction)
    radius = np.linalg.norm(source - target)
    base = field_contract([source], [mass], [target])[0]
    amplitudes = radius * np.array([1e-3, 5e-4, 2.5e-4])
    remainders = []

    for amplitude in amplitudes:
        displacement = amplitude * direction
        shifted = field_contract([source + displacement], [mass], [target])[0]
        first_order = displacement_contract(
            [source], [mass], [displacement], [target]
        )[0]
        remainders.append(np.linalg.norm(shifted - base - first_order))

    ratios = np.asarray(remainders[:-1]) / np.asarray(remainders[1:])
    assert np.all((ratios >= 3.0) & (ratios <= 5.0)), ratios


@pytest.mark.parametrize(
    "contract_name",
    ["displacement_potential_contract", "displacement_gradient_contract"],
)
def test_displacement_contract_empty_input_and_validation(contract_name):
    contract = getattr(kernels, contract_name)
    empty_sources = np.empty((0, 3))
    empty_mass = np.empty(0)
    empty_displacement = np.empty((0, 3), dtype=np.complex128)

    empty = contract(empty_sources, empty_mass, empty_displacement, TARGETS)
    assert empty.shape[0] == len(TARGETS)
    assert empty.dtype == np.complex128
    assert np.count_nonzero(empty) == 0

    with pytest.raises(TypeError, match="real-valued"):
        contract(SOURCES, MASS.astype(complex) + 1j, D1, TARGETS)
    with pytest.raises(ValueError, match="non-negative"):
        contract(SOURCES, -MASS, D1, TARGETS)
    with pytest.raises(ValueError, match="chunk_size"):
        contract(SOURCES, MASS, D1, TARGETS, chunk_size=0)
    with pytest.raises(ValueError, match="softening_m"):
        contract(SOURCES, MASS, D1, TARGETS, softening_m=-0.1)
    with pytest.raises(ValueError, match="Collision"):
        contract(SOURCES, MASS, D1, [SOURCES[1]], softening_m=0.0, chunk_size=1)


def test_displacement_dispatch_and_dipole_wrapper_compatibility():
    dispatched = direct_point_displacement_sum(MASS, D1, SOURCES, TARGETS)
    legacy = direct_point_dipole_sum(MASS, D1, SOURCES, TARGETS)
    kernel = kernels.dipole_contract(SOURCES, MASS, D1, TARGETS)

    assert np.array_equal(dispatched, legacy)
    assert np.array_equal(dispatched, kernel)


@pytest.mark.parametrize("observable", ["potential", "gradient"])
@pytest.mark.parametrize("backend", ["cpp", "cupy"])
def test_new_displacement_derivatives_reject_known_non_numpy_backends(
    observable, backend
):
    with pytest.raises(NotImplementedError, match="use backend='numpy' or 'numba'"):
        direct_point_displacement_sum(
            MASS, D1, SOURCES, TARGETS, observable=observable, backend=backend
        )


def test_displacement_dispatch_rejects_unknown_observable_and_backend():
    with pytest.raises(ValueError, match="Unsupported observable"):
        direct_point_displacement_sum(
            MASS, D1, SOURCES, TARGETS, observable="jerk"
        )
    with pytest.raises(ValueError, match="Unsupported backend"):
        direct_point_displacement_sum(
            MASS, D1, SOURCES, TARGETS, observable="potential", backend="bogus"
        )
