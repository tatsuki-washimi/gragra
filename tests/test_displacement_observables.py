"""Tests for displacement-transfer potential and gravity gradient."""

import inspect

import numpy as np
import pytest

import gragra.observables.dipole as dipole
from gragra.constants import G_SI
from gragra.kernels.inverse_square import (
    displacement_gradient_contract,
    displacement_potential_contract,
)


def test_displacement_observable_axis_goldens_are_g_scaled_and_read_only():
    assert hasattr(dipole, "displacement_transfer_potential")
    assert hasattr(dipole, "displacement_transfer_gravity_gradient")
    sources = [[0.0, 0.0, 0.0]]
    mass = [1.0]
    displacement = [[1.0, 0.0, 0.0]]
    targets = [[1.0, 0.0, 0.0]]

    potential = dipole.displacement_transfer_potential(
        sources, mass, displacement, targets
    )
    gradient = dipole.displacement_transfer_gravity_gradient(
        sources, mass, displacement, targets
    )

    assert np.allclose(potential, [-G_SI])
    assert np.allclose(gradient, [np.diag([6.0, -3.0, -3.0]) * G_SI])
    assert potential.flags.writeable is False
    assert gradient.flags.writeable is False


def test_displacement_observables_match_acceleration_signature():
    expected = inspect.signature(dipole.displacement_transfer_acceleration)
    assert inspect.signature(dipole.displacement_transfer_potential) == expected
    assert inspect.signature(dipole.displacement_transfer_gravity_gradient) == expected


@pytest.mark.parametrize(
    ("observable", "contract", "expected_shape"),
    [
        (dipole.displacement_transfer_potential, displacement_potential_contract, (2,)),
        (
            dipole.displacement_transfer_gravity_gradient,
            displacement_gradient_contract,
            (2, 3, 3),
        ),
    ],
)
def test_displacement_observables_apply_g_once_and_preserve_complex_dtype(
    observable, contract, expected_shape
):
    sources = [[0.2, -0.4, 0.7], [-1.1, 0.6, 0.3], [0.8, 1.2, -0.9]]
    mass = [1.5, 0.7, 2.1]
    displacement = [
        [0.3 + 0.2j, -0.2, 0.5j],
        [-0.1j, 0.4, 0.2],
        [0.6, 0.1j, -0.3],
    ]
    targets = [[2.0, -0.3, 0.4], [-0.6, -1.4, 1.7]]

    got = observable(
        sources,
        mass,
        displacement,
        targets,
        softening_m=0.15,
        chunk_size=2,
    )
    expected = G_SI * contract(
        sources,
        mass,
        displacement,
        targets,
        softening_m=0.15,
        chunk_size=2,
    )

    assert got.shape == expected_shape
    assert got.dtype == np.complex128
    assert got.flags.writeable is False
    assert np.array_equal(got, expected)


@pytest.mark.parametrize(
    ("observable", "contract"),
    [
        (dipole.displacement_transfer_potential, displacement_potential_contract),
        (
            dipole.displacement_transfer_gravity_gradient,
            displacement_gradient_contract,
        ),
    ],
)
def test_displacement_numba_observables_apply_g_once(observable, contract):
    pytest.importorskip("numba")
    sources = [[0.2, -0.4, 0.7], [-1.1, 0.6, 0.3], [0.8, 1.2, -0.9]]
    mass = [1.5, 0.7, 2.1]
    displacement = [
        [0.3 + 0.2j, -0.2, 0.5j],
        [-0.1j, 0.4, 0.2],
        [0.6, 0.1j, -0.3],
    ]
    targets = [[2.0, -0.3, 0.4], [-0.6, -1.4, 1.7]]

    got = observable(
        sources,
        mass,
        displacement,
        targets,
        softening_m=0.15,
        chunk_size=2,
        backend="numba",
    )
    expected = G_SI * contract(
        sources,
        mass,
        displacement,
        targets,
        softening_m=0.15,
        chunk_size=2,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)
    assert got.flags.writeable is False


@pytest.mark.parametrize(
    "observable",
    [
        dipole.displacement_transfer_potential,
        dipole.displacement_transfer_gravity_gradient,
    ],
)
@pytest.mark.parametrize("backend", ["cpp", "cupy"])
def test_new_displacement_observables_reject_unsupported_backend(observable, backend):
    with pytest.raises(NotImplementedError, match="use backend='numpy' or 'numba'"):
        observable(
            [[0.0, 0.0, 0.0]],
            [1.0],
            [[1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            backend=backend,
        )


def test_new_displacement_observables_are_not_top_level_exports():
    import gragra

    assert not hasattr(gragra, "displacement_transfer_potential")
    assert not hasattr(gragra, "displacement_transfer_gravity_gradient")
