from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gragra
from gragra.fields.thermodynamic import (
    adiabatic_pressure_density_perturbation,
    isobaric_temperature_density_perturbation,
)


def test_imports_this_worktree_source():
    expected = Path(__file__).resolve().parents[1] / "src/gragra/__init__.py"
    assert Path(gragra.__file__).resolve() == expected


def test_isobaric_temperature_relation_preserves_real_precision_and_shape():
    delta_t = np.array([[0.25, -0.5], [1.0, -2.0]], dtype=np.float32)
    result = isobaric_temperature_density_perturbation(1.2, delta_t, 300.0)

    np.testing.assert_allclose(result, -1.2 * delta_t.astype(np.float64) / 300.0)
    assert result.shape == delta_t.shape
    assert result.dtype == np.dtype(np.float64)
    assert not result.flags.writeable


def test_adiabatic_pressure_relation_preserves_complex_amplitude():
    delta_p = np.array([1.0 + 2.0j, -0.5j])
    result = adiabatic_pressure_density_perturbation(delta_p, 343.0)

    np.testing.assert_allclose(result, delta_p / 343.0**2)
    assert result.dtype == np.dtype(np.complex128)
    assert not result.flags.writeable


def test_array_inputs_are_defensively_copied():
    rho0 = np.array([1.0, 2.0])
    delta_t = np.array([3.0, 4.0])
    temperature = np.array([300.0, 400.0])
    result = isobaric_temperature_density_perturbation(rho0, delta_t, temperature)
    expected = result.copy()

    rho0[:] = 10.0
    delta_t[:] = 20.0
    temperature[:] = 30.0

    np.testing.assert_array_equal(result, expected)


def test_scalar_inputs_return_canonical_numpy_scalars():
    real = isobaric_temperature_density_perturbation(1.2, 2.0, 300.0)
    complex_value = adiabatic_pressure_density_perturbation(1.0 + 1.0j, 2.0)

    assert isinstance(real, np.float64)
    assert isinstance(complex_value, np.complex128)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: isobaric_temperature_density_perturbation(0.0, 1.0, 300.0),
            "rho0",
        ),
        (
            lambda: isobaric_temperature_density_perturbation(1.0, 1.0, np.inf),
            "T0",
        ),
        (
            lambda: adiabatic_pressure_density_perturbation(1.0, -343.0),
            "c_s",
        ),
        (
            lambda: adiabatic_pressure_density_perturbation(np.nan, 343.0),
            "delta_p",
        ),
    ],
)
def test_invalid_physical_values_are_rejected(call, message: str):
    with pytest.raises((TypeError, ValueError), match=message):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: isobaric_temperature_density_perturbation(True, 1.0, 300.0),
        lambda: isobaric_temperature_density_perturbation(1.0, False, 300.0),
        lambda: adiabatic_pressure_density_perturbation(True, 343.0),
        lambda: adiabatic_pressure_density_perturbation(1.0, True),
        lambda: adiabatic_pressure_density_perturbation(1.0, 343.0 + 0.0j),
    ],
)
def test_boolean_and_complex_background_inputs_are_rejected(call):
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: isobaric_temperature_density_perturbation(
            np.ones((2, 1)), np.ones((1, 2)), 300.0
        ),
        lambda: adiabatic_pressure_density_perturbation(
            np.ones((2, 1)), np.ones((1, 2))
        ),
    ],
)
def test_general_broadcasting_is_rejected(call):
    with pytest.raises(ValueError, match="shape"):
        call()


@pytest.mark.parametrize(
    "background",
    ["343", [True, 343.0], object()],
)
def test_backgrounds_require_strict_real_numeric_values(background):
    with pytest.raises(TypeError, match="real, positive, and finite"):
        adiabatic_pressure_density_perturbation(1.0, background)


def test_nonfinite_arithmetic_result_is_rejected_without_runtime_warning():
    with np.errstate(all="raise"), pytest.raises(ValueError, match="finite"):
        adiabatic_pressure_density_perturbation(1.0, 1e-300)
