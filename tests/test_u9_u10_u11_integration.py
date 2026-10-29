"""Small connection regression for the adopted U9, U10 and U11 modules."""

import importlib
import json

import numpy as np
import pytest

from gragra.constants import G_SI


def _rotation():
    a, b, c = 0.31, -0.47, 0.23
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


@pytest.mark.parametrize(
    ("shape", "dimensions", "target_scale"),
    [
        ("sphere", {"radius_m": 2.0}, np.array([2.0, 2.0, 2.0])),
        (
            "cuboid",
            {"half_lengths_m": np.array([2.0, 1.5, 1.0])},
            np.array([2.0, 1.5, 1.0]),
        ),
        (
            "cylinder",
            {"radius_m": 2.0, "height_m": 3.0},
            np.array([2.0, 2.0, 1.5]),
        ),
    ],
)
def test_u9_wall_acceleration_connects_to_u11_center_sca(
    shape, dimensions, target_scale, record_property
):
    """Compare U9 with U11's unchanged center displacement on ordinary points.

    The geometry and target fractions follow the U11 design fixture. The G-free
    acceptance width is the existing U11 numeric-counterexample width; the SI
    width is exactly that value multiplied by G_SI. This is an implementation
    connection test, not a new independent physical oracle.
    """
    center = np.array([13.0, -7.0, -90.0])
    rotation = _rotation()
    local_targets = np.array([[0.0, 0.0, 0.0], [0.23, -0.17, 0.31]])
    local_targets *= target_scale
    targets = local_targets @ rotation.T + center
    phase_origin = np.array([-4.0, 9.0, 0.0])
    eta = (1.0 + 0.3j) * 1e-9
    density = 2300.0

    u11_analytic = importlib.import_module(
        "gragra.analytic.u11_rayleigh_small_cavity"
    ).rayleigh_small_cavity_acceleration_g_free
    u11_si = importlib.import_module(
        "gragra.observables.u11_rayleigh_small_cavity"
    ).rayleigh_small_cavity_acceleration
    u9_analytic_module = importlib.import_module(
        "gragra.analytic.u9_sca_wall_observables"
    )
    u9_si_module = importlib.import_module("gragra.observables.u9_sca_wall_observables")
    u9_analytic = getattr(u9_analytic_module, f"{shape}_sca_wall_observables_g_free")
    u9_si = getattr(u9_si_module, f"{shape}_sca_wall_observables")

    u11_kwargs = {
        "cavity_shape": shape,
        "center_m": center,
        "density_kg_m3": density,
        "v_p_m_s": 5800.0,
        "v_s_m_s": 3200.0,
        "frequency_hz": 5.0,
        "vertical_surface_amplitude_m": eta,
        "direction": [1.0, 2.0, 0.0],
        "phase_origin_m": phase_origin,
        "rotation": rotation,
        **dimensions,
    }
    u11_gfree, diagnostics = u11_analytic(
        targets, **u11_kwargs, return_diagnostics=True
    )
    u11_si_values, si_diagnostics = u11_si(
        targets, **u11_kwargs, return_diagnostics=True
    )

    # Pass the U11 diagnostic value through unchanged; it is the center
    # displacement in meters, for every target in this comparison.
    center_displacement_m = diagnostics["u_c"]
    assert diagnostics["u_c_units"] == "m"
    np.testing.assert_array_equal(center_displacement_m, si_diagnostics["u_c"])
    u9_kwargs = {
        "center_m": center,
        "density_kg_m3": density,
        "displacement_m": center_displacement_m,
        **dimensions,
    }
    if shape != "sphere":
        u9_kwargs["rotation"] = rotation

    u9_gfree = u9_analytic(targets, **u9_kwargs)
    u9_si_values = u9_si(targets, **u9_kwargs)
    u11_wall_gfree = u11_gfree["cavity_wall_sca"]
    u11_wall_si = u11_si_values["cavity_wall_sca"]
    u9_wall_gfree = u9_gfree.acceleration
    u9_wall_si = u9_si_values.acceleration

    # Existing U11 test_numeric_counterexamples contract: rtol=1e-10 and
    # atol=4*pi*rho*1e-21 kg/m^2. Convert that absolute width once for SI.
    rtol = 1e-10
    atol_gfree = 4.0 * np.pi * density * 1e-21
    atol_si = G_SI * atol_gfree
    for label, actual, reference, atol in (
        ("gfree", u9_wall_gfree, u11_wall_gfree, atol_gfree),
        ("si", u9_wall_si, u11_wall_si, atol_si),
    ):
        component_error = np.abs(actual - reference)
        component_width = atol + rtol * np.abs(reference)
        vector_error = np.linalg.norm(actual - reference, axis=-1)
        vector_width = atol + rtol * np.linalg.norm(reference, axis=-1)
        component_ratio = component_error / component_width
        vector_ratio = vector_error / vector_width
        record_property(
            f"{shape}_{label}_comparison",
            json.dumps(
                {
                    "units": "kg/m^2" if label == "gfree" else "m/s^2",
                    "sample_index": 0,
                    "target_indices": list(range(targets.shape[0])),
                    "rtol": rtol,
                    "atol": atol,
                    "target_world_m": targets.tolist(),
                    "absolute_component_difference": component_error.tolist(),
                    "reference_component_magnitude": np.abs(reference).tolist(),
                    "component_tolerance_width": component_width.tolist(),
                    "component_error_over_width": component_ratio.tolist(),
                    "absolute_vector_difference": vector_error.tolist(),
                    "reference_vector_norm": np.linalg.norm(
                        reference, axis=-1
                    ).tolist(),
                    "vector_tolerance_width": vector_width.tolist(),
                    "vector_error_over_width": vector_ratio.tolist(),
                },
                sort_keys=True,
            ),
        )
        record_property(
            f"{shape}_{label}_max_component_error",
            float(np.max(component_error)),
        )
        record_property(
            f"{shape}_{label}_max_component_width", float(np.max(component_width))
        )
        record_property(
            f"{shape}_{label}_max_component_error_over_width",
            float(np.max(component_ratio)),
        )
        record_property(
            f"{shape}_{label}_max_vector_error", float(np.max(vector_error))
        )
        record_property(
            f"{shape}_{label}_max_vector_width", float(np.max(vector_width))
        )
        record_property(
            f"{shape}_{label}_max_vector_error_over_width",
            float(np.max(vector_ratio)),
        )
        assert np.all(component_ratio <= 1.0), (shape, label, component_ratio)
        assert np.all(vector_ratio <= 1.0), (shape, label, vector_ratio)

    np.testing.assert_allclose(
        u9_wall_si,
        G_SI * u9_wall_gfree,
        rtol=1e-15,
        atol=0,
    )
    np.testing.assert_allclose(
        u11_wall_si,
        G_SI * u11_wall_gfree,
        rtol=1e-15,
        atol=0,
    )
