from __future__ import annotations

from pathlib import Path

import numpy as np

import gragra
from scripts.research_wave5_environment_transport import (
    CASE_SPECS,
    TARGETS_M,
    _case_commands,
    _parse_native_scalar_field,
    analytic_total_mass_kg,
    cosine_bell_density,
    generate_case,
    periodic_center_of_mass,
    summarize_density_difference,
)
from scripts.research_wave5_environment_transport_centroids import (
    cartesian_signed_mass_centroid,
)


def test_imports_this_worktree_source():
    expected = Path(__file__).resolve().parents[1] / "src/gragra/__init__.py"
    assert Path(gragra.__file__).resolve() == expected


def test_case_matrix_matches_approved_resolution_and_time_step_sweep():
    assert [(item.level, item.delta_t_s) for item in CASE_SPECS] == [
        (24, 0.01),
        (48, 0.01),
        (48, 0.005),
    ]
    assert TARGETS_M.shape == (3, 3)


def test_generated_case_uses_function_object_and_has_no_future_density_reinjection(
    tmp_path: Path,
):
    case = tmp_path / "case"
    record = generate_case(case, level=4, delta_t_s=0.01)

    control = (case / "system/controlDict").read_text(encoding="ascii")
    functions = (case / "system/functions").read_text(encoding="ascii")
    schemes = (case / "system/fvSchemes").read_text(encoding="ascii")
    solution = (case / "system/fvSolution").read_text(encoding="ascii")
    momentum = (case / "constant/momentumTransport").read_text(encoding="ascii")

    assert "solver          incompressibleFluid;" in control
    assert "type            scalarTransport;" in functions
    assert 'libs            ("libsolverFunctionObjects.so");' in functions
    assert "field           delta_rho;" in functions
    assert "diffusivity     none;" in functions
    assert "executeControl  timeStep;" in functions
    assert "writeControl    writeTime;" in functions
    assert "div(phi,delta_rho) Gauss upwind;" in schemes
    assert "default         Euler;" in schemes
    assert "pRefCell        0;" in solution
    assert "pRefValue       0;" in solution
    assert "simulationType  laminar;" in momentum
    assert sorted(path.name for path in case.iterdir() if path.is_dir()) == [
        "0",
        "constant",
        "system",
    ]
    assert record["initial_density"]["amplitude_kg_m3"] == -0.01
    assert record["velocity_m_s"] == [0.2, 0.0, 0.0]


def test_runtime_commands_use_real_solver_and_precision_export(tmp_path: Path):
    case = tmp_path / "case"
    commands = _case_commands(case)

    assert commands[2] == [
        "foamRun",
        "-solver",
        "incompressibleFluid",
        "-case",
        str(case),
    ]
    exports = commands[-3:]
    assert [command[command.index("-time") + 1] for command in exports] == [
        "0",
        "0.5",
        "1",
    ]
    assert all(command[-2:] == ["-fields", "(delta_rho C Vc)"] for command in exports)


def test_native_scalar_parser_expands_uniform_cartesian_volume(tmp_path: Path):
    path = tmp_path / "Vc"
    path.write_text(
        """FoamFile
{
    version 2.0;
    format ascii;
    class volScalarField;
    object Vc;
}
dimensions [0 3 0 0 0 0 0];
internalField uniform 0.125;
boundaryField {}
""",
        encoding="ascii",
    )

    values = _parse_native_scalar_field(
        path,
        object_name="Vc",
        expected_count=8,
        expected_dimensions=(0, 3, 0, 0, 0, 0, 0),
    )

    np.testing.assert_array_equal(values, np.full(8, 0.125))


def test_cosine_bell_translates_periodically_without_changing_support_value():
    points = np.array([[-0.25, 0.0, 0.0], [-0.10, 0.0, 0.0]])
    at_zero = cosine_bell_density(points, 0.0)
    translated = cosine_bell_density(points + [0.2, 0.0, 0.0], 1.0)

    np.testing.assert_allclose(at_zero, [-0.01, 0.0], atol=1e-18)
    np.testing.assert_array_equal(translated, at_zero)


def test_actual_profile_has_parent_cas_verified_continuous_mass():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    radius = 0.5 * 0.15 * (nodes + 1.0)
    radial_weights = 0.5 * 0.15 * weights
    density = -0.01 / 2.0 * (1.0 + np.cos(np.pi * radius / 0.15))
    independent = np.sum(4.0 * np.pi * radial_weights * radius**2 * density)

    np.testing.assert_allclose(analytic_total_mass_kg(), independent, rtol=2e-15)
    np.testing.assert_allclose(analytic_total_mass_kg(), -2.7714000070958602e-05)


def test_density_summary_uses_volume_weighted_l1_and_l2():
    observed = np.array([-2.0, 1.0])
    reference = np.array([-1.0, -1.0])
    volumes = np.array([2.0, 1.0])

    summary = summarize_density_difference(observed, reference, volumes)

    assert summary["l1_kg"] == 4.0
    assert summary["l2_kg_m_neg_3_2"] == 6.0**0.5
    assert summary["linf_kg_m3"] == 2.0


def test_periodic_signed_center_of_mass_handles_boundary_crossing():
    positions = np.array([[0.49, 0.0, 0.0], [-0.49, 0.0, 0.0]])
    signed_mass = np.array([-1.0, -1.0])

    center = periodic_center_of_mass(positions, signed_mass)

    assert abs(abs(center[0]) - 0.5) < 1e-15
    np.testing.assert_allclose(center[1:], 0.0, atol=1e-15)


def test_cartesian_and_periodic_centers_are_explicitly_distinct_statistics():
    positions = np.array([[-0.49, 0.0, 0.0], [0.49, 0.0, 0.0]])
    mass = np.array([-3.0, -1.0])

    cartesian = cartesian_signed_mass_centroid(positions, mass)
    circular = periodic_center_of_mass(positions, mass)

    np.testing.assert_allclose(cartesian, [-0.245, 0.0, 0.0])
    assert abs(circular[0] - cartesian[0]) > 0.2
    assert not cartesian.flags.writeable
