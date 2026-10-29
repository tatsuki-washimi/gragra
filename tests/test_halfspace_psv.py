"""General P/SV: independent Cartesian traction and Green-function checks."""

import numpy as np
import pytest

from gragra.analytic.bodywave_literature import (
    halfspace_reflected_p_above_observables_g_free,
)
from gragra.analytic.halfspace_psv import (
    halfspace_psv_observables_g_free,
    halfspace_psv_reflection,
)

MATERIALS = [(5800.0, 3200.0), (1500.0 * np.sqrt(3), 1500.0)]


def independent_modes(vp, vs, theta, incident, omega):
    """Solve stresses of Cartesian displacement vectors, not potentials."""
    kp, ks = omega / vp, omega / vs
    q = (kp if incident == "P" else ks) * np.sin(theta)
    p, s = np.sqrt(kp**2 - q**2 + 0j), np.sqrt(ks**2 - q**2 + 0j)
    if incident == "P":
        p = kp * np.cos(theta)
    else:
        s = ks * np.cos(theta)
        if theta == np.arcsin(vs / vp):
            p = 0j
    kin = np.array([q, p if incident == "P" else s])
    ein = kin / kp if incident == "P" else np.array([s, -q]) / ks
    kpr, ksr = np.array([q, -p]), np.array([q, -s])
    epr, esr = kpr / kp, np.array([s, q]) / ks
    lam, mu = vp**2 - 2 * vs**2, vs**2

    def traction(k, e):
        return 1j * np.array(
            [mu * (k[1] * e[0] + k[0] * e[1]), lam * (k @ e) + 2 * mu * k[1] * e[1]]
        )

    rp, rs = np.linalg.solve(
        np.column_stack([traction(kpr, epr), traction(ksr, esr)]), -traction(kin, ein)
    )
    return [(kin, ein), (kpr, rp * epr), (ksr, rs * esr)]


@pytest.mark.parametrize("vp,vs", MATERIALS)
@pytest.mark.parametrize("incident", ["P", "SV"])
def test_reflection_stress_flux_and_cartesian_solve(vp, vs, incident):
    critical = np.arcsin(vs / vp)
    angles = [0.0, 0.2, critical - 1e-8, critical + 1e-8, 1.3, np.pi / 2 - 1e-7]
    if incident == "SV":
        angles.append(critical)
    for theta in angles:
        result = halfspace_psv_reflection(
            wave_type=incident, v_p_m_s=vp, v_s_m_s=vs, incidence_angle_rad=theta
        )
        q, p, s = result.q_over_ks, result.p_over_ks, result.s_over_ks
        a, c = (1 / (vs / vp), 0) if incident == "P" else (0, -1)
        b, d = result.reflected_p_potential_ratio, result.reflected_sv_potential_ratio
        # Ratios are against incident scalar potential, whose sign for SV
        # differs from its displacement amplitude.
        b, d = (b * a, d * a) if incident == "P" else (b * c, d * c)
        normal = -(1 - 2 * q * q) * (a + b) + 2 * q * s * (d - c)
        shear = 2 * q * p * (b - a) + (1 - 2 * q * q) * (c + d)
        assert abs(normal) < 2e-12
        assert abs(shear) < 2e-12
        # Flux of propagating P is proportional to Re(p); evanescent P
        # carries no depth-directed energy, regardless of |B|.
        incoming = p.real * abs(a) ** 2 if incident == "P" else s.real * abs(c) ** 2
        outgoing = p.real * abs(b) ** 2 + s.real * abs(d) ** 2
        assert abs(outgoing / incoming - 1) < 2e-9
        if theta not in (critical,):
            modes = independent_modes(vp, vs, theta, incident, 1.0)
            assert np.allclose(modes[1][1], np.array([q, -p]) * b, atol=2e-9)
            assert np.allclose(modes[2][1], np.array([s, q]) * d, atol=2e-9)


@pytest.mark.parametrize("theta", [0.0, 0.01, 0.5, 1.2, np.pi / 2 - 1e-8])
def test_p_reduces_to_existing_physical_api(theta):
    kw = {
        "density_kg_m3": 2700.0,
        "v_p_m_s": 5800.0,
        "v_s_m_s": 3200.0,
        "frequency_hz": [2.0, 7.0],
        "incidence_angle_rad": theta,
        "azimuth_rad": 0.7,
        "incident_displacement_m": [1e-6 + 2e-6j, 3e-6 - 1e-6j],
        "phase_origin_m": [3.0, 4.0, -5.0],
    }
    points = [[10.0, -7.0, 1.0], [1.0, 4.0, 50.0]]
    actual = halfspace_psv_observables_g_free(points, wave_type="P", **kw)
    ref = halfspace_reflected_p_above_observables_g_free(points, **kw)
    for field in ("potential", "acceleration", "gradient"):
        np.testing.assert_allclose(
            getattr(actual, field), getattr(ref, field), rtol=1e-10, atol=1e-20
        )


def green_reference(
    points, *, vp, vs, theta, incident, omega, rho, amplitude, azimuth, tolerance=2e-12
):
    """Numerical z-source convolution, with Cartesian elastic inputs."""
    quad = pytest.importorskip("scipy.integrate").quad
    modes = independent_modes(vp, vs, theta, incident, omega)
    q = modes[0][0][0].real
    sigma = rho * sum(e[1] for _, e in modes) * amplitude

    def density(z):
        return (
            sum(-1j * rho * (k @ e) * np.exp(1j * k[1] * z) for k, e in modes)
            * amplitude
        )

    results = []
    for x, y, z in points:
        values = []
        for derivative in (0, 1):

            def integrand(t, z=z, derivative=derivative):
                g = -np.exp(-q * abs(z - t)) / (2 * q)
                if derivative:
                    g *= -q * np.sign(z - t)
                return g * density(t)

            intervals = [(-np.inf, z), (z, 0.0)] if z < 0 else [(-np.inf, 0.0)]
            value = sum(
                quad(
                    lambda t: integrand(t).real,
                    lower,
                    upper,
                    epsabs=tolerance / 10,
                    epsrel=tolerance,
                    limit=500,
                )[0]
                + 1j
                * quad(
                    lambda t: integrand(t).imag,
                    lower,
                    upper,
                    epsabs=tolerance / 10,
                    epsrel=tolerance,
                    limit=500,
                )[0]
                for lower, upper in intervals
            )
            sheet = -np.exp(-q * abs(z)) / (2 * q) * sigma
            if derivative:
                sheet *= -q * np.sign(z)
            values.append(4 * np.pi * (value + sheet))
        f, df = values
        ddf = q * q * f + (4 * np.pi * density(z) if z < 0 else 0)
        h = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        phase = np.exp(1j * q * (x * h[0] + y * h[1]))
        acc = -1j * q * h * f - ez * df
        grad = (
            q * q * np.outer(h, h) * f
            - 1j * q * (np.outer(h, ez) + np.outer(ez, h)) * df
            - np.outer(ez, ez) * ddf
        )
        results.append((phase * f, phase * acc, phase * grad))
    return tuple(np.array([row[j] for row in results]) for j in range(3))


@pytest.mark.parametrize("vp,vs", MATERIALS)
@pytest.mark.parametrize(
    "incident,offset",
    [
        ("P", 0.1),
        ("SV", -0.01),
        ("SV", -1e-8),
        ("SV", 0.0),
        ("SV", 1e-8),
        ("SV", 0.01),
        ("SV", 0.4),
    ],
)
def test_independent_green_three_observables(vp, vs, incident, offset):
    theta = np.arcsin(vs / vp) + offset
    points = np.array([[3.0, -4.0, 20.0], [-10.0, 30.0, -25.0], [4.0, 5.0, -80.0]])
    ref = green_reference(
        points,
        vp=vp,
        vs=vs,
        theta=theta,
        incident=incident,
        omega=2 * np.pi * 5,
        rho=2700.0,
        amplitude=1e-6 + 2e-6j,
        azimuth=0.7,
    )
    actual = halfspace_psv_observables_g_free(
        points,
        wave_type=incident,
        density_kg_m3=2700.0,
        v_p_m_s=vp,
        v_s_m_s=vs,
        frequency_hz=5.0,
        incidence_angle_rad=theta,
        azimuth_rad=0.7,
        incident_displacement_m=1e-6 + 2e-6j,
    )
    for name, expected, scale in zip(
        ("potential", "acceleration", "gradient"),
        ref,
        (
            2700 * abs(1e-6 + 2e-6j) / (2 * np.pi * 5 / vs),
            2700 * abs(1e-6 + 2e-6j),
            2700 * abs(1e-6 + 2e-6j) * (2 * np.pi * 5 / vs),
        ),
        strict=True,
    ):
        errors = np.linalg.norm(
            (getattr(actual, name) - expected).reshape(3, -1), axis=1
        )
        norms = np.linalg.norm(expected.reshape(3, -1), axis=1)
        rtol = 1e-7 if incident == "SV" and abs(offset) < 1e-7 else 1e-10
        assert np.all(errors <= 1e-11 * scale + rtol * norms)


def test_normal_critical_and_input_contracts():
    kw = {
        "density_kg_m3": 2700.0,
        "v_p_m_s": 5800.0,
        "v_s_m_s": 3200.0,
        "frequency_hz": 5.0,
        "incidence_angle_rad": 0.0,
        "azimuth_rad": 0.0,
        "incident_displacement_m": 1e-6,
    }
    points = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -20.0]])
    sv = halfspace_psv_observables_g_free(points, wave_type="SV", **kw)
    assert np.count_nonzero(sv.gradient) == 0
    p = halfspace_psv_observables_g_free(points, wave_type="P", **kw)
    k = 2 * np.pi * 5 / 5800
    np.testing.assert_allclose(
        p.acceleration[1, 2], 8 * np.pi * 2700 * 1e-6 * np.cos(k * 20), rtol=1e-14
    )
    for name in ("potential", "acceleration", "gradient"):
        assert not getattr(p, name).flags.writeable
    with pytest.raises(ValueError, match="surface"):
        halfspace_psv_observables_g_free([[0, 0, 0]], wave_type="P", **kw)
    with pytest.raises(ValueError, match="grazing"):
        halfspace_psv_observables_g_free(
            points, wave_type="P", **(kw | {"incidence_angle_rad": np.pi / 2})
        )
    critical = np.arcsin(3200 / 5800)
    at = halfspace_psv_observables_g_free(
        points, wave_type="SV", **(kw | {"incidence_angle_rad": critical})
    )
    assert np.isfinite(at.gradient).all()
    with pytest.raises(ValueError, match="degenerate"):
        halfspace_psv_reflection(
            wave_type="SV",
            v_p_m_s=np.sqrt(2),
            v_s_m_s=1.0,
            incidence_angle_rad=np.pi / 4,
        )


@pytest.mark.parametrize("incident", ["P", "SV"])
def test_frequency_origin_rotation_si_and_poisson(incident):
    from gragra import G_SI
    from gragra.observables import halfspace_psv_observables

    points = np.array([[12.0, -3.0, 20.0], [1.0, 9.0, -30.0]])
    kw = {
        "wave_type": incident,
        "density_kg_m3": 2700.0,
        "v_p_m_s": 5800.0,
        "v_s_m_s": 3200.0,
        "frequency_hz": np.array([2.0, 7.0]),
        "incidence_angle_rad": 0.9,
        "azimuth_rad": 0.7,
        "incident_displacement_m": np.array([1e-6 + 2e-6j, -1e-6j]),
    }
    saved = points.copy()
    actual = halfspace_psv_observables_g_free(points, **kw)
    si = halfspace_psv_observables(points, **kw)
    origin = np.array([7.0, -8.0, 9.0])
    speed = kw["v_p_m_s"] if incident == "P" else kw["v_s_m_s"]
    kinc = 2 * np.pi * kw["frequency_hz"] / speed
    projection = (
        np.sin(0.9) * (origin[0] * np.cos(0.7) + origin[1] * np.sin(0.7))
        + np.cos(0.9) * origin[2]
    )
    shifted = halfspace_psv_observables_g_free(
        points,
        **(
            kw
            | {
                "phase_origin_m": origin,
                "incident_displacement_m": kw["incident_displacement_m"]
                * np.exp(1j * kinc * projection),
            }
        ),
    )
    beta = 1.1
    rot = np.array(
        [
            [np.cos(beta), -np.sin(beta), 0.0],
            [np.sin(beta), np.cos(beta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = halfspace_psv_observables_g_free(
        points @ rot.T, **(kw | {"azimuth_rad": 0.7 + beta})
    )
    for name in ("potential", "acceleration", "gradient"):
        arr = getattr(actual, name)
        np.testing.assert_allclose(getattr(si, name), G_SI * arr, rtol=2e-15)
        np.testing.assert_allclose(getattr(shifted, name), arr, rtol=2e-14)
        assert not getattr(si, name).flags.writeable
        for j in range(2):
            scalar = halfspace_psv_observables_g_free(
                points,
                **(
                    kw
                    | {
                        "frequency_hz": kw["frequency_hz"][j],
                        "incident_displacement_m": kw["incident_displacement_m"][j],
                    }
                ),
            )
            np.testing.assert_allclose(getattr(scalar, name), arr[j], rtol=2e-14)
        expected = (
            arr
            if name == "potential"
            else (arr @ rot.T if name == "acceleration" else rot @ arr @ rot.T)
        )
        np.testing.assert_allclose(
            getattr(rotated, name), expected, rtol=2e-14, atol=1e-20
        )
    np.testing.assert_array_equal(points, saved)
    trace = np.trace(actual.gradient, axis1=-2, axis2=-1)
    np.testing.assert_allclose(trace[:, 0], 0, atol=1e-18)
    for j, freq in enumerate(kw["frequency_hz"]):
        modes = independent_modes(5800.0, 3200.0, 0.9, incident, 2 * np.pi * freq)
        horizontal = points[1, 0] * np.cos(0.7) + points[1, 1] * np.sin(0.7)
        density = (
            sum(
                -1j
                * 2700
                * (k @ e)
                * np.exp(1j * (k[0] * horizontal + k[1] * points[1, 2]))
                for k, e in modes
            )
            * kw["incident_displacement_m"][j]
        )
        np.testing.assert_allclose(
            trace[j, 1], -4 * np.pi * density, rtol=2e-14, atol=1e-19
        )


@pytest.mark.parametrize("incident", ["P", "SV"])
def test_interface_continuity_jump_and_analytic_derivatives(incident):
    kw = {
        "wave_type": incident,
        "density_kg_m3": 2700.0,
        "v_p_m_s": 5800.0,
        "v_s_m_s": 3200.0,
        "frequency_hz": 5.0,
        "incidence_angle_rad": 0.9,
        "azimuth_rad": 0.0,
        "incident_displacement_m": 1e-6 + 2e-6j,
    }
    eps = 1e-6
    values = halfspace_psv_observables_g_free([[0.0, 0.0, eps], [0.0, 0.0, -eps]], **kw)
    modes = independent_modes(5800.0, 3200.0, 0.9, incident, 2 * np.pi * 5)
    sheet = 2700 * sum(e[1] for _, e in modes) * kw["incident_displacement_m"]
    assert abs(values.potential[0] - values.potential[1]) < 1e-7
    np.testing.assert_allclose(
        values.acceleration[0, 2] - values.acceleration[1, 2],
        -4 * np.pi * sheet,
        rtol=1e-7,
        atol=1e-12,
    )
    # Auxiliary differentiation only; production never uses finite differences.
    point = np.array([[3.0, 4.0, -20.0]])
    center = halfspace_psv_observables_g_free(point, **kw)
    for axis in range(3):
        shift = np.zeros((1, 3))
        shift[0, axis] = 1e-3
        plus = halfspace_psv_observables_g_free(point + shift, **kw)
        minus = halfspace_psv_observables_g_free(point - shift, **kw)
        np.testing.assert_allclose(
            -(plus.potential - minus.potential) / (2e-3),
            center.acceleration[:, axis],
            rtol=2e-8,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            (plus.acceleration - minus.acceleration) / (2e-3),
            center.gradient[:, :, axis],
            rtol=2e-8,
            atol=1e-13,
        )


@pytest.mark.parametrize(
    "key,value,exception",
    [
        ("wave_type", "SH", ValueError),
        ("density_kg_m3", -1, ValueError),
        ("frequency_hz", 0, ValueError),
        ("frequency_hz", [1.0, float("nan")], ValueError),
        ("incident_displacement_m", [1.0, 2.0], ValueError),
        ("incidence_angle_rad", True, TypeError),
        ("azimuth_rad", 1j, TypeError),
        ("phase_origin_m", [0.0, float("inf"), 0.0], ValueError),
        ("v_p_m_s", 3400.0, ValueError),
        ("incident_displacement_m", complex("nan"), ValueError),
    ],
)
def test_invalid_arguments(key, value, exception):
    kw = {
        "wave_type": "SV",
        "density_kg_m3": 2700.0,
        "v_p_m_s": 5800.0,
        "v_s_m_s": 3200.0,
        "frequency_hz": 5.0,
        "incidence_angle_rad": 0.9,
        "azimuth_rad": 0.0,
        "incident_displacement_m": 1e-6,
    }
    with pytest.raises(exception):
        halfspace_psv_observables_g_free([[0.0, 0.0, 1.0]], **(kw | {key: value}))
