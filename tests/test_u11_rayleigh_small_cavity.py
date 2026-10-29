"""U11 contracts and numerical regressions; no source-value lookup fixtures."""

import importlib
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gragra.waves.rayleigh import rayleigh_displacement, rayleigh_speed


def inputs(shape="sphere"):
    args = {
        "cavity_shape": shape,
        "center_m": [13.0, -7.0, -90.0],
        "density_kg_m3": 2300.0,
        "v_p_m_s": 5800.0,
        "v_s_m_s": 3200.0,
        "frequency_hz": 5.0,
        "vertical_surface_amplitude_m": (1 + 0.3j) * 1e-9,
        "direction": [1.0, 2.0, 0.0],
        "phase_origin_m": [-4.0, 9.0, 0.0],
    }
    args.update(
        {
            "sphere": {"radius_m": 2.0},
            "cuboid": {"half_lengths_m": [2.0, 1.5, 1.0]},
            "cylinder": {"radius_m": 2.0, "height_m": 3.0},
        }[shape]
    )
    return np.array([[13.0, -7.0, -90.0], [13.6, -7.3, -89.8]]), args


def api():
    return importlib.import_module(
        "gragra.analytic.u11_rayleigh_small_cavity"
    ).rayleigh_small_cavity_acceleration_g_free


def independent_center(args):
    # Two elastic potentials in the adopted z-up +i convention. This is a
    # numerical eigenfunction check, not the independent gravity integral.
    c = np.asarray(args["center_m"])
    d = np.asarray(args["direction"])
    d = d / np.linalg.norm(d)
    k = 2 * np.pi * args["frequency_hz"] / rayleigh_speed(5800.0, 3200.0)
    speed = rayleigh_speed(5800.0, 3200.0)
    q = np.sqrt(1 - (speed / 5800.0) ** 2)
    s = np.sqrt(1 - (speed / 3200.0) ** 2)
    zeta = 2 * q / (1 + s * s)
    amp = args["vertical_surface_amplitude_m"] / (k * (q - zeta))
    phase = np.exp(1j * k * np.dot(d, c - np.asarray(args["phase_origin_m"])))
    horizontal = 1j * k * amp * (np.exp(k * q * c[2]) - s * zeta * np.exp(k * s * c[2]))
    out = horizontal * d.astype(complex)
    out[2] = k * amp * (q * np.exp(k * q * c[2]) - zeta * np.exp(k * s * c[2]))
    return out * phase


def test_import_contract():
    assert callable(api())


@pytest.mark.parametrize("mutation", ["sign", "phase", "target_displacement"])
def test_numeric_counterexamples(mutation):
    targets, args = inputs()
    expected = np.broadcast_to(-4 * np.pi * 2300 / 3 * independent_center(args), (2, 3))
    if os.environ.get("U11_RED_MUTANTS") == "1":
        # Deliberately wrong adapters, exercised before implementation. These
        # produce numerical assertion failures independently of API import Red.
        k = 2 * np.pi * 5 / rayleigh_speed(5800.0, 3200.0)
        d = np.array([1.0, 2.0, 0.0]) / np.sqrt(5)
        uc = independent_center(args)
        if mutation == "sign":
            actual = -expected
        elif mutation == "phase":
            actual = expected * np.exp(1j * k * np.dot(d, args["phase_origin_m"]))
        else:
            u = rayleigh_displacement(targets, d, k, 5800.0, 3200.0)
            u = (
                u
                * args["vertical_surface_amplitude_m"]
                * np.exp(-1j * k * np.dot(d, args["phase_origin_m"]))
            )
            actual = -4 * np.pi * 2300 / 3 * u
        assert np.isfinite(uc).all()
    else:
        actual = api()(targets, **args)["cavity_wall_sca"]
    assert_allclose(actual, expected, rtol=1e-10, atol=4 * np.pi * 2300e-21)


@pytest.mark.parametrize("shape", ["sphere", "cuboid", "cylinder"])
def test_parts_units_series_and_input_ownership(shape):
    from gragra.analytic.rayleigh_literature import (
        rayleigh_halfspace_allspace_observables_g_free,
    )
    from gragra.constants import G_SI

    si = importlib.import_module(
        "gragra.observables.u11_rayleigh_small_cavity"
    ).rayleigh_small_cavity_acceleration
    targets, args = inputs(shape)
    original = targets.copy()
    args.update(
        frequency_hz=[3.0, 5.0], vertical_surface_amplitude_m=[1e-9, (1 + 0.3j) * 1e-9]
    )
    parts, diag = api()(targets, **args, return_diagnostics=True)
    converted, si_diag = si(targets, **args, return_diagnostics=True)
    assert set(parts) == {
        "background_halfspace",
        "cavity_wall_sca",
        "total_leading_sca",
    }
    for key, value in parts.items():
        assert value.shape == (2, 2, 3) and value.dtype == np.complex128
        assert not value.flags.writeable
        assert_allclose(converted[key], G_SI * value, rtol=1e-15, atol=0)
    assert_allclose(diag["u_c"], si_diag["u_c"], rtol=0, atol=0)
    assert_allclose(targets, original, rtol=0, atol=0)
    assert targets.flags.writeable
    assert_allclose(
        parts["total_leading_sca"],
        parts["background_halfspace"] + parts["cavity_wall_sca"],
        rtol=0,
        atol=0,
    )
    keys = [
        "density_kg_m3",
        "v_p_m_s",
        "v_s_m_s",
        "frequency_hz",
        "vertical_surface_amplitude_m",
        "direction",
        "phase_origin_m",
    ]
    background = rayleigh_halfspace_allspace_observables_g_free(
        targets, **{k: args[k] for k in keys}
    )
    assert_allclose(
        parts["background_halfspace"], background.acceleration, rtol=1e-13, atol=1e-20
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("frequency_hz", 0.0),
        ("frequency_hz", []),
        ("frequency_hz", [5.0, True]),
        ("frequency_hz", "5"),
        ("frequency_hz", [5 + 0j]),
        ("vertical_surface_amplitude_m", "1e-9"),
        ("vertical_surface_amplitude_m", [1e-9]),
        ("density_kg_m3", True),
        ("v_s_m_s", 5800.0),
        ("v_p_m_s", 3500.0),
        ("phase_origin_m", [0, 0, 1]),
        ("center_m", [13, True, -90]),
        ("direction", [1, 0, 1]),
        ("direction", ["1", 0, 0]),
        ("radius_m", -1.0),
        ("height_m", 1.0),
        ("max_sca_parameter", True),
        ("max_sca_parameter", 0.0),
        ("rotation", np.diag([1, 1, -1])),
        ("rotation", np.eye(3) * 2),
        ("return_diagnostics", 1),
    ],
)
def test_invalid_inputs_even_at_zero(key, value):
    targets, args = inputs()
    args["density_kg_m3"] = 0.0
    args[key] = value
    with pytest.raises((TypeError, ValueError)):
        api()(targets, **args)


@pytest.mark.parametrize(
    "targets",
    [
        [],
        np.empty((0, 3)),
        [13.0, -7.0, -90.0],
        [[13.0, False, -90.0]],
        [[13.0, -7.0, np.nan]],
        [[15.0, -7.0, -90.0]],
        [[16.0, -7.0, -90.0]],
    ],
)
def test_invalid_targets(targets):
    _, args = inputs()
    args["vertical_surface_amplitude_m"] = 0
    with pytest.raises((ValueError, TypeError)):
        api()(targets, **args)


@pytest.mark.parametrize("shape", ["sphere", "cuboid", "cylinder"])
def test_zero_linearity_and_domain(shape):
    targets, args = inputs(shape)
    original = api()(targets, **args)
    args["vertical_surface_amplitude_m"] *= -0.3 + 2j
    scaled = api()(targets, **args)
    for key in original:
        assert_allclose(
            scaled[key], (-0.3 + 2j) * original[key], rtol=1e-13, atol=1e-20
        )
    args["vertical_surface_amplitude_m"] = 0
    for value in api()(targets, **args).values():
        assert np.count_nonzero(value) == 0
    args["max_sca_parameter"] = 1e-4
    with pytest.raises(ValueError, match="sample"):
        api()(targets, **args)
    args["max_sca_parameter"] = 0.1
    args["center_m"] = [13, -7, -0.5]
    with pytest.raises(ValueError, match="underground"):
        api()(np.array([[13, -7, -0.5]]), **args)


def test_scale_safe_direction_and_horizontal_covariance():
    targets, args = inputs("cuboid")
    base = api()(targets, **args)
    args["direction"] = [1e200, 2e200, 0]
    for key, val in api()(targets, **args).items():
        assert_allclose(val, base[key], rtol=1e-13, atol=1e-20)
    theta = 0.7
    r = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    shift = np.array([5.0, 11.0, 0])
    args["direction"] = r @ np.array([1.0, 2.0, 0])
    args["rotation"] = r
    args["center_m"] = r @ np.array(args["center_m"]) + shift
    args["phase_origin_m"] = r @ np.array(args["phase_origin_m"]) + shift
    result = api()(targets @ r.T + shift, **args)
    for key in base:
        assert_allclose(result[key], base[key] @ r.T, rtol=2e-12, atol=1e-20)


@pytest.mark.parametrize("shape", ["sphere", "cuboid", "cylinder"])
def test_rotated_geometry_tensor_connection(shape):
    from gragra.analytic.cylinder_sca import cylinder_sca_wall_coupling_tensor_g_free
    from gragra.analytic.sca_tensor import (
        cuboid_sca_wall_coupling_tensor_g_free,
        sphere_sca_wall_coupling_tensor_g_free,
    )

    targets, args = inputs(shape)
    theta = 0.6
    r = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    args["rotation"] = r
    parts, diag = api()(targets, **args, return_diagnostics=True)
    c = args["center_m"]
    if shape == "sphere":
        tensor = sphere_sca_wall_coupling_tensor_g_free(c, 2.0, targets)
    elif shape == "cuboid":
        tensor = cuboid_sca_wall_coupling_tensor_g_free(
            c, [4.0, 3.0, 2.0], targets, rotation_matrix=r
        )
    else:
        tensor = cylinder_sca_wall_coupling_tensor_g_free(
            c, 2.0, 3.0, targets, rotation_matrix=r
        )
    expected = -2300 * np.einsum("mij,j->mi", tensor, diag["u_c"])
    assert_allclose(parts["cavity_wall_sca"], expected, rtol=1e-14, atol=1e-20)
    if shape != "sphere":
        assert np.max(np.abs(tensor[:, 0, 1:])) > 1e-3
    # The upper support of a tilted cavity must be checked, not just its center.
    args["center_m"] = [0, 0, -0.9]
    with pytest.raises(ValueError, match="underground"):
        api()([[0, 0, -0.9]], **args)


@pytest.mark.parametrize(
    "key,value",
    [
        ("center_m", np.array([13, -7, -90], dtype=object)),
        ("center_m", [13, np.bool_(False), -90]),
        ("center_m", [13 + 0j, -7, -90]),
        ("frequency_hz", [[5]]),
        ("vertical_surface_amplitude_m", [1e-9, np.bool_(False)]),
        ("vertical_surface_amplitude_m", complex(np.inf, 0)),
        ("max_sca_parameter", "0.1"),
        ("rotation", [[True, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ],
)
def test_strict_raw_types(key, value):
    targets, args = inputs()
    args[key] = value
    with pytest.raises((TypeError, ValueError)):
        api()(targets, **args)


def test_numpy_only_import_and_paths():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','sympy','mpmath','numba','pytest'}:
            raise ImportError('validation dependency forbidden: '+fullname)
sys.meta_path.insert(0,Block())
import gragra
import gragra.analytic.u11_rayleigh_small_cavity as analytic
import gragra.observables.u11_rayleigh_small_cavity as si
root=Path(sys.argv[1])
assert Path(gragra.__file__).resolve().is_relative_to(root/'src')
assert Path(analytic.__file__).resolve()==(
    root/'src/gragra/analytic/u11_rayleigh_small_cavity.py')
assert Path(si.__file__).resolve()==(
    root/'src/gragra/observables/u11_rayleigh_small_cavity.py')
assert Path(sys.executable).parent==Path(sys.prefix)/'bin'
result=analytic.rayleigh_small_cavity_acceleration_g_free([[0,0,-100]],cavity_shape='sphere',center_m=[0,0,-100],radius_m=1,density_kg_m3=2300,v_p_m_s=5800,v_s_m_s=3200,frequency_hz=5,vertical_surface_amplitude_m=1e-9)
assert len(result)==3
print(sys.executable,sys.path,gragra.__file__,analytic.__file__,si.__file__)
"""
    subprocess.run([sys.executable, "-c", code, str(root)], check=True)


def test_reference_constant_density_volume_transform():
    from tests.oracles.u11_rayleigh_small_cavity_reference import (
        removed_volume,
        wall_nodes,
    )

    targets, args = inputs()
    points, normals, weights = wall_nodes(args, 48)
    for target in targets:
        actual = removed_volume(
            target, points, normals, weights, lambda p: np.ones(len(p)), 8
        )
        expected = -4 * np.pi / 3 * (target - np.asarray(args["center_m"]))
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


def test_reference_gate_keeps_components_nonfinite_and_high_order_failure():
    from scripts.validate_u11_rayleigh_small_cavity import gate

    reference = np.array([[1.0, 0.0, 0.0]])
    assert gate(np.zeros((1, 3)), reference, 1.0)["pass"] == [True]
    # A low-order agreement cannot overwrite a later failed refinement.
    high = gate(np.array([[0.0, 2e-12, 0.0]]), reference, 1.0, 0.1)
    assert high["pass"] == [False]
    assert high["failed_target_component"] == [[0, 1]]
    assert gate(np.full((1, 3), np.nan), reference, 1.0)["pass"] == [False]


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["sphere", "cuboid", "cylinder"])
def test_independent_gravity_a_and_b(shape):
    from scripts.validate_u11_rayleigh_small_cavity import gate
    from tests.oracles.u11_rayleigh_small_cavity_reference import evaluate

    targets, args = inputs(shape)
    low = evaluate(targets, args, 40, 12)
    high = evaluate(targets, args, 64, 20)
    result = api()(targets, **args)
    scale = 4 * np.pi * 2300e-9
    for key in result:
        assert all(gate(high[key] - low[key], high[key], scale, 0.1)["pass"])
        assert all(gate(result[key] - high[key], high[key], scale)["pass"])
    assert_allclose(
        high["model_error"],
        high["removed_bulk"] - high["wall_variation"],
        rtol=1e-12,
        atol=1e-20,
    )
