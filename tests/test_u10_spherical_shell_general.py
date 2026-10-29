"""U10 regression and contract tests; references do not reuse shell series."""

import numpy as np
import pytest

from gragra.analytic.u10_spherical_shell_general import (
    spherical_shell_observables_g_free,
)


def parameters(wave="P", a=2.0, k=0.08):
    return {
        "inner_radius_m": a,
        "outer_radius_m": 10.0,
        "density_kg_m3": 2300.0,
        "wavenumber_rad_m": k,
        "displacement_m": 1e-9 * (1 + 0.4j),
        "propagation_direction": [1.0, 2.0, 3.0],
        "polarization": [1.0, 2.0, 3.0] if wave == "P" else [2.0, -1.0, 0.0],
        "wave_type": wave,
    }


def test_thin_center_bulk_potential():
    # Independent one-dimensional exact integral, evaluated at 70 digits.
    import mpmath as mp

    kw = parameters(a=9.999, k=0.001)
    with mp.workdps(70):
        a, b, k = map(mp.mpf, [kw["inner_radius_m"], 10.0, kw["wavenumber_rad_m"]])
        integral = mp.quad(lambda r: r * mp.sin(k * r) / (k * r), [a, b])
        expected = complex(
            4 * mp.pi * 1j * mp.mpf(2300) * k * mp.mpc(kw["displacement_m"]) * integral
        )
    got = spherical_shell_observables_g_free([[0.0, 0.0, 0.0]], **kw)["bulk"].potential[
        0
    ]
    scale = (10.0 - kw["inner_radius_m"]) / 10.0 * 4 * np.pi * 2300 * 1e-9 * 10
    assert abs(got - expected) <= 1e-12 * scale + 1e-10 * abs(expected)


@pytest.mark.parametrize(
    "wave,part,observable,k",
    [
        ("P", "bulk", "potential", 0.001),
        ("P", "total", "acceleration", 0.08),
        ("S", "total", "acceleration", 0.08),
    ],
)
def test_extremely_thin_center_against_independent_integrals(wave, part, observable, k):
    """Finite volume and two sheets, integrated without shell coefficients."""
    import mpmath as mp

    kw = parameters(wave, a=9.99999999999, k=k)
    d = np.asarray(kw["propagation_direction"], dtype=float)
    d /= np.linalg.norm(d)
    p = np.asarray(kw["polarization"], dtype=float)
    p /= np.linalg.norm(p)
    with mp.workdps(80):
        a = mp.mpf(kw["inner_radius_m"])
        b = mp.mpf(kw["outer_radius_m"])
        km = mp.mpf(k)
        rho = mp.mpf(2300)
        u = mp.mpc(kw["displacement_m"])

        def j0(z):
            return mp.sin(z) / z

        if observable == "potential":
            integral = mp.quad(lambda r: r * j0(km * r), [a, b])
            expected = complex(4 * mp.pi * 1j * rho * km * u * integral)
            got = spherical_shell_observables_g_free([[0, 0, 0]], **kw)[part].potential[
                0
            ]
            error = abs(got - expected)
            reference_norm = abs(expected)
            signal_scale = (
                (10 - kw["inner_radius_m"]) / 10 * 4 * np.pi * 2300 * 1e-9 * 10
            )
        else:

            def tensor_coefficient(radius, longitudinal):
                return (
                    2
                    * mp.pi
                    * mp.quad(
                        lambda mu: (
                            (mu * mu if longitudinal else (1 - mu * mu) / 2)
                            * mp.exp(1j * km * radius * mu)
                        ),
                        [-1, 1],
                    )
                )

            longitudinal = wave == "P"
            wall = (
                rho
                * u
                * (
                    tensor_coefficient(b, longitudinal)
                    - tensor_coefficient(a, longitudinal)
                )
            )
            bulk = (
                4 * mp.pi * rho * u * (j0(km * a) - j0(km * b)) if longitudinal else 0
            )
            axis = d if longitudinal else p
            expected = np.asarray(
                [complex((bulk + wall) * mp.mpf(float(v))) for v in axis]
            )
            got = spherical_shell_observables_g_free([[0, 0, 0]], **kw)[
                part
            ].acceleration[0]
            error = np.linalg.norm(got - expected)
            reference_norm = np.linalg.norm(expected)
            signal_scale = (10 - kw["inner_radius_m"]) / 10 * 4 * np.pi * 2300 * 1e-9
    acceptance = 1e-12 * signal_scale + 1e-10 * reference_norm
    assert error <= acceptance


@pytest.mark.parametrize("wave", ["P", "S"])
def test_shapes_closure_and_poisson(wave):
    kw = parameters(wave)
    targets = np.array(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, 0.0, 6.0], [20.0, 0.0, 0.0]]
    )
    before = targets.copy()
    fields = spherical_shell_observables_g_free(targets, **kw)
    assert tuple(fields) == ("bulk", "inner_wall", "outer_boundary", "total")
    for part, value in fields.items():
        assert [a.shape for a in value] == [(4,), (4, 3), (4, 3, 3)]
        for arr in value:
            assert arr.dtype == np.complex128 and not arr.flags.writeable
        expected = np.zeros(4, complex)
        if wave == "P" and part in ("bulk", "total"):
            d = np.array(kw["propagation_direction"]) / np.sqrt(14)
            delta = (
                -1j
                * 2300
                * 0.08
                * kw["displacement_m"]
                * np.exp(1j * 0.08 * (d @ targets[2]))
            )
            expected[2] = -4 * np.pi * delta
        np.testing.assert_allclose(
            np.trace(value.gradient, axis1=-2, axis2=-1), expected, atol=1e-16
        )
    for index in range(3):
        np.testing.assert_allclose(
            list(fields["total"])[index],
            sum(
                list(fields[p])[index] for p in ("bulk", "inner_wall", "outer_boundary")
            ),
            atol=1e-18,
        )
    np.testing.assert_array_equal(targets, before)
    assert targets.flags.writeable


@pytest.mark.parametrize("wave", ["P", "S"])
@pytest.mark.parametrize("a,k", [(2.0, 0.001), (8.0, 0.08), (9.9, 0.5), (9.999, 0.001)])
def test_center_connections(wave, a, k):
    from gragra.analytic.cavity_sphere import (
        spherical_shell_p_acceleration_components_g_free,
        spherical_shell_s_acceleration_components_g_free,
    )

    kw = parameters(wave, a, k)
    got = spherical_shell_observables_g_free([[0, 0, 0]], **kw)
    if wave == "P":
        # Unit direction avoids the separately reported legacy overflow bug.
        direction = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
        ref = spherical_shell_p_acceleration_components_g_free(
            a, 10.0, k, 2300.0, kw["displacement_m"], direction
        )
    else:
        ref = spherical_shell_s_acceleration_components_g_free(
            a, 10.0, k, 2300.0, kw["displacement_m"], [1, 2, 3], [2, -1, 0]
        )
    for part in got:
        # Legacy rounded differences are a connection test, not thin-shell truth.
        np.testing.assert_allclose(
            got[part].acceleration, getattr(ref, part), atol=2e-20, rtol=1e-10
        )


@pytest.mark.parametrize("wave", ["P", "S"])
def test_rotation_translation_phase_linearity_and_series(wave):
    kw = parameters(wave)
    x = np.array([[0.3, -0.2, 0.1], [2.0, 3.0, 5.0], [20.0, -5.0, 2.0]])
    values = spherical_shell_observables_g_free(x, **kw)
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2.0, 1, 2], [2, 3, -1.0]]))
    assert np.linalg.det(rotation) > 0
    shift = np.array([1.7, -2.1, 0.8])
    transformed = spherical_shell_observables_g_free(
        x @ rotation.T + shift,
        **dict(
            kw,
            center_m=shift,
            phase_origin_m=shift,
            propagation_direction=rotation @ kw["propagation_direction"],
            polarization=rotation @ kw["polarization"],
        ),
    )
    shifted = spherical_shell_observables_g_free(x, **dict(kw, phase_origin_m=shift))
    linear = spherical_shell_observables_g_free(
        x, **dict(kw, displacement_m=kw["displacement_m"] * (-0.3 + 0.7j))
    )
    series = spherical_shell_observables_g_free(
        x,
        **dict(
            kw,
            wavenumber_rad_m=[0.08, 0.08],
            displacement_m=[kw["displacement_m"], 2j * kw["displacement_m"]],
        ),
    )
    phase = np.exp(-1j * 0.08 * np.array([1, 2, 3]) @ shift / np.sqrt(14))
    for part in values:
        v, t = values[part], transformed[part]
        np.testing.assert_allclose(t.potential, v.potential, rtol=1e-12, atol=1e-19)
        np.testing.assert_allclose(
            t.acceleration, v.acceleration @ rotation.T, rtol=1e-12, atol=1e-19
        )
        np.testing.assert_allclose(
            t.gradient, rotation @ v.gradient @ rotation.T, rtol=1e-12, atol=1e-19
        )
        for original, phased, scaled, samples in zip(
            v, shifted[part], linear[part], series[part], strict=True
        ):
            np.testing.assert_allclose(phased, phase * original, rtol=1e-12, atol=1e-19)
            np.testing.assert_allclose(
                scaled, (-0.3 + 0.7j) * original, rtol=1e-12, atol=1e-19
            )
            np.testing.assert_allclose(samples[0], original, rtol=1e-12, atol=1e-19)
            np.testing.assert_allclose(
                samples[1], 2j * original, rtol=1e-12, atol=1e-19
            )


def test_si_once_and_readonly():
    from gragra.constants import G_SI
    from gragra.observables.analytic import GravityObservables
    from gragra.observables.u10_spherical_shell_general import (
        spherical_shell_observables,
    )

    kw = parameters()
    atol = np.array([1e-17, 1e-18, 1e-19])
    free = spherical_shell_observables_g_free([[0, 0, 0]], atol=atol, **kw)
    si, diag = spherical_shell_observables(
        [[0, 0, 0]], atol_si=atol * G_SI, return_diagnostics=True, **kw
    )
    assert diag["units"] == ["kg/m", "kg/m^2", "kg/m^3"]
    for part in free:
        assert isinstance(si[part], GravityObservables)
        for name in ("potential", "acceleration", "gradient"):
            arr = getattr(si[part], name)
            assert arr.dtype == np.complex128 and not arr.flags.writeable
            np.testing.assert_array_equal(arr, G_SI * getattr(free[part], name))
    with pytest.raises(TypeError):
        spherical_shell_observables([[0, 0, 0]], atol=atol, **kw)
    with pytest.raises(TypeError):
        spherical_shell_observables([[0, 0, 0]], atol_si=[True, 0, 0], **kw)


@pytest.mark.parametrize(
    "key",
    [
        "inner_radius_m",
        "outer_radius_m",
        "density_kg_m3",
        "wavenumber_rad_m",
        "displacement_m",
    ],
)
@pytest.mark.parametrize(
    "bad", [True, np.bool_(True), "1", np.array(1, dtype=object), np.nan, np.inf]
)
def test_bad_scalar_types(key, bad):
    kw = parameters()
    kw[key] = bad
    with pytest.raises((TypeError, ValueError)):
        spherical_shell_observables_g_free([[0, 0, 0]], **kw)


@pytest.mark.parametrize(
    "bad",
    [
        [[True, 0.0, 0.0]],
        [[1 + 0j, 0, 0]],
        [["1", 0, 0]],
        np.array([[1, 0, 0]], dtype=object),
        [[np.nan, 0, 0]],
        [[np.inf, 0, 0]],
        [],
        np.empty((0, 3)),
        [1, 2, 3],
    ],
)
def test_bad_targets_before_zero_shortcut(bad):
    with pytest.raises((TypeError, ValueError)):
        spherical_shell_observables_g_free(bad, **dict(parameters(), density_kg_m3=0))


@pytest.mark.parametrize(
    "key",
    [
        "center_m",
        "phase_origin_m",
        "propagation_direction",
        "polarization",
        "rtol",
        "atol",
    ],
)
@pytest.mark.parametrize(
    "bad",
    [
        [True, 1.0, 1.0],
        ["1", 1, 1],
        [1 + 0j, 1, 1],
        np.array([1, 1, 1], dtype=object),
        [np.nan, 1, 1],
    ],
)
def test_bad_vector_types(key, bad):
    kw = dict(parameters(), displacement_m=0)
    kw[key] = bad
    with pytest.raises((TypeError, ValueError)):
        spherical_shell_observables_g_free([[0, 0, 0]], **kw)


@pytest.mark.parametrize(
    "changes",
    [
        {"inner_radius_m": 0},
        {"inner_radius_m": 10},
        {"outer_radius_m": 1},
        {"density_kg_m3": -1},
        {"wavenumber_rad_m": 0},
        {"wavenumber_rad_m": []},
        {"wavenumber_rad_m": [0.1, 0.2], "displacement_m": [1, 2, 3]},
        {"displacement_m": [1]},
        {"propagation_direction": [0, 0, 0]},
        {"polarization": [1, 0, 0]},
        {"wave_type": "p"},
        {"rtol": 1e-10},
        {"atol": [1, -1, 1]},
        {"max_degree": True},
        {"max_degree": -1},
        {"return_diagnostics": 1},
        {"inner_radius_m": [2]},
    ],
)
def test_invalid_geometry_and_shapes(changes):
    with pytest.raises((TypeError, ValueError)):
        spherical_shell_observables_g_free([[0, 0, 0]], **dict(parameters(), **changes))


@pytest.mark.parametrize("radius", [2.0, 10.0])
@pytest.mark.parametrize("zero", [{"displacement_m": 0}, {"density_kg_m3": 0}])
def test_boundary_rejected_for_zero_input(radius, zero):
    with pytest.raises(ValueError, match="boundary"):
        spherical_shell_observables_g_free(
            [[radius, 0, 0]], **dict(parameters(), **zero)
        )


def test_zero_and_extreme_direction():
    kw = parameters()
    got = spherical_shell_observables_g_free(
        [[0, 0, 0]],
        **dict(
            kw,
            propagation_direction=[1e308, 1e308, 1e308],
            polarization=[-1e308, -1e308, -1e308],
        ),
    )
    ref = spherical_shell_observables_g_free(
        [[0, 0, 0]],
        **dict(kw, propagation_direction=[1, 1, 1], polarization=[-1, -1, -1]),
    )
    zero = spherical_shell_observables_g_free([[0, 0, 0]], **dict(kw, displacement_m=0))
    for part in got:
        for value, expected, z in zip(got[part], ref[part], zero[part], strict=True):
            np.testing.assert_array_equal(value, expected)
            assert np.all(z == 0)


def test_failure_diagnostics_and_no_stale_success():
    from gragra.analytic.u10_spherical_shell_general import ShellConvergenceError

    kw = parameters()
    _, success = spherical_shell_observables_g_free(
        [[0, 0, 0]], return_diagnostics=True, **kw
    )
    with pytest.raises(ShellConvergenceError) as caught:
        spherical_shell_observables_g_free([[0, 0, 0], [20, 0, 0]], max_degree=2, **kw)
    failed = caught.value.diagnostics
    assert failed["failed_index"] == [0, 0]
    assert failed["queries"][0]["status"] == "NOT_CONVERGED"
    assert failed["queries"][1]["status"] == "PENDING"
    assert success["status"] == "CONVERGED"
    with pytest.raises(ShellConvergenceError):
        spherical_shell_observables_g_free(
            [[0, 0, 0]], rtol=[0, 0, 0], atol=[0, 0, 0], **kw
        )


@pytest.mark.slow
@pytest.mark.parametrize("wave", ["P", "S"])
@pytest.mark.parametrize("a", [2.0, 9.999])
def test_independent_volume_and_surfaces(wave, a):
    from tests.oracles.u10_spherical_shell_general_reference import reference

    kw = parameters(wave, a, 0.5)
    axis = np.array([2.0, -3.0, 1.0])
    axis /= np.linalg.norm(axis)
    x = np.array([np.zeros(3), axis * 0.4 * a, axis * (a + 10) / 2, axis * 20])
    refs = [
        reference(x, **kw, nphi=n, nradial=radial, nmu=nmu)[0]
        for n, radial, nmu in [(48, 24, 96), (64, 32, 128)]
    ]
    got = spherical_shell_observables_g_free(x, **kw)
    for part in got:
        for j, (values, sl) in enumerate(
            zip(got[part], [slice(0, 1), slice(1, 4), slice(4, 13)], strict=True)
        ):
            for target in range(len(x)):
                scale = 4 * np.pi * 2300 * 1e-9 * [10, 1, 0.1][j]
                if part in ("bulk", "total") and target != 2:
                    scale *= (10 - a) / 10
                low, high = (ref[part][target, sl] for ref in refs)
                budget = 1e-12 * scale + [1e-10, 1e-10, 1e-9][j] * np.linalg.norm(high)
                assert np.linalg.norm(high - low) <= budget / 10, (
                    "REFERENCE_NOT_CONVERGED"
                )
                assert np.linalg.norm(values[target].ravel() - high) <= budget
                component_budget = 1e-12 * scale + [1e-10, 1e-10, 1e-9][j] * abs(high)
                assert np.all(abs(high - low) <= component_budget / 10), (
                    "REFERENCE_NOT_CONVERGED"
                )
                assert np.all(abs(values[target].ravel() - high) <= component_budget)


@pytest.mark.parametrize("wave", ["P", "S"])
def test_small_k_center_limit_and_far_monopole(wave):
    kw = parameters(wave, k=1e-6)
    first = spherical_shell_observables_g_free([[0, 0, 0]], **kw)["total"].acceleration
    second = spherical_shell_observables_g_free(
        [[0, 0, 0]], **dict(kw, wavenumber_rad_m=2e-6)
    )["total"].acceleration
    np.testing.assert_allclose(second, 4 * first, rtol=1e-9, atol=1e-27)
    axis = np.array(kw["polarization"])
    axis /= np.linalg.norm(axis)
    far = spherical_shell_observables_g_free(
        np.array([1000 * axis, 2000 * axis]), **parameters(wave)
    )
    ratio = abs(far["total"].potential[0] / far["total"].potential[1])
    assert 3.9 < ratio < 4.1  # A monopole would approach 2 instead of 4.


@pytest.mark.parametrize("wave", ["P", "S"])
def test_mass_conservation_from_direct_source_integral(wave):
    kw = parameters(wave)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    # Rotate integration axis to d; only p.d survives the azimuth integral.
    projection = 1 if wave == "P" else 0
    a, b, k = 2.0, 10.0, 0.08
    radius = a + (b - a) * (nodes + 1) / 2
    radial_weight = weights * (b - a) / 2
    angular = np.exp(1j * k * radius[:, None] * nodes)
    bulk = (
        -1j
        * k
        * projection
        * 2
        * np.pi
        * np.sum(radial_weight * radius**2 * (angular @ weights))
    )
    inner = (
        -projection
        * a
        * a
        * 2
        * np.pi
        * np.sum(weights * nodes * np.exp(1j * k * a * nodes))
    )
    outer = (
        projection
        * b
        * b
        * 2
        * np.pi
        * np.sum(weights * nodes * np.exp(1j * k * b * nodes))
    )
    assert abs(bulk + inner + outer) < 2e-12
    assert kw["wave_type"] == wave


def test_finite_distance_near_both_walls_and_small_inner_radius():
    kw = parameters()
    x = np.array(
        [[2 - 1e-6, 0, 0], [2 + 1e-6, 0, 0], [10 - 1e-6, 0, 0], [10 + 1e-6, 0, 0]]
    )
    fields = spherical_shell_observables_g_free(x, **kw)
    for part in fields:
        assert all(np.isfinite(arr).all() for arr in fields[part])
        np.testing.assert_allclose(
            fields[part].gradient,
            fields[part].gradient.swapaxes(-1, -2),
            rtol=1e-13,
            atol=1e-19,
        )
    small = spherical_shell_observables_g_free(
        [[0, 0, 0], [5, 0, 0]], **dict(kw, inner_radius_m=1e-5)
    )
    assert all(np.isfinite(arr).all() for arr in small["total"])


def test_import_and_evaluation_without_optional_scientific_dependencies():
    import subprocess
    import sys
    from pathlib import Path

    source_path = str(Path(__file__).resolve().parents[1] / "src")
    script = f"""
import importlib.abc
import sys
from pathlib import Path

sys.meta_path = [
    finder for finder in sys.meta_path
    if type(finder).__module__ != "_editable_skbc_gragra"
]
sys.path.insert(0, {source_path!r})

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {"scipy", "sympy", "mpmath"}:
            raise ImportError('verification dependency blocked')
sys.meta_path.insert(0,Block())
import gragra
assert Path(gragra.__file__).resolve() == Path(
    {source_path!r}, "gragra", "__init__.py"
).resolve()
from gragra.analytic import u10_spherical_shell_general as shell
result=shell.spherical_shell_observables_g_free([[0,0,0]],inner_radius_m=2,outer_radius_m=10,density_kg_m3=2300,wavenumber_rad_m=.08,displacement_m=1e-9,propagation_direction=[0,0,1],polarization=[0,0,1],wave_type='P')
assert result['total'].acceleration.shape==(1,3)
assert not set(("scipy", "sympy", "mpmath")).intersection(sys.modules)
print('NUMPY_ONLY_PASS')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "NUMPY_ONLY_PASS" in result.stdout
