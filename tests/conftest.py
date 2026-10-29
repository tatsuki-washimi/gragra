"""Shared pytest fixtures for gragra tests.

Provides a single source of truth for the cube surface/volume mesh used by the
surface-field sign-pinning tests and the orientation diagnostic tests, so the
diagnostic is verified against the same mesh that fixes the bulk/surface
mass-conservation sign.
"""

import os
from pathlib import Path

import numpy as np
import pytest

_REFERENCE_MODULES = frozenset(
    {
        "tests/test_gen1_b1_formal_generator.py",
        "tests/test_gen1_b1_nonzero_generator.py",
        "tests/test_stochastic_formal_acoustic.py",
        "tests/test_stochastic_formal_seismic.py",
    }
)
_REFERENCE_NODES = frozenset(
    {
        "tests/test_gen1_stochastic_acoustic_oracle.py::"
        "test_runner_refuses_formal_generation_and_runtime_denies_production",
        "tests/test_stochastic_formal_comparator.py::"
        "test_connection_phase_attempt_failure_still_sealed",
    }
)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run production-blind assertions separately from mixed-suite imports."""
    nodeid = pyfuncitem.nodeid
    if (
        nodeid.split("::", 1)[0] not in _REFERENCE_MODULES
        and nodeid not in _REFERENCE_NODES
    ):
        return None
    # Return to pytest's normal fixture/function dispatch inside the exact child.
    # Do not hide imports from the original production-denial guards.
    child = os.environ.get("GRAGRA_REFERENCE_TEST_CHILD_NODE")
    if child is not None:
        if child != nodeid:
            pytest.fail(
                "reference child node differs from its admission", pytrace=False
            )
        return None

    import json
    import tempfile

    from tests._reference_test_isolation import ReferenceChildError, run_node

    diagnostic = os.environ.get("A2_TEST_CHILD_ACCOUNTING")
    directory = Path(diagnostic).parent if diagnostic else None
    parent = Path(tempfile.mkdtemp(prefix="reference-test-", dir=directory))
    try:
        receipt = run_node(pyfuncitem.config.rootpath, nodeid, parent / "child")
    except ReferenceChildError as error:
        pytest.fail(str(error), pytrace=False)
    pyfuncitem.user_properties.append(("reference_child", json.dumps(receipt)))
    return True

# Cube: 8 vertices of [-h, h]^3, 12 outward-wound triangles, 6 Kuhn tetra.
_CUBE_TRIANGLES = np.array(
    [
        [0, 3, 2],
        [0, 2, 1],  # bottom z=-h  (n = -z)
        [4, 5, 6],
        [4, 6, 7],  # top    z=+h  (n = +z)
        [0, 1, 5],
        [0, 5, 4],  # front  y=-h  (n = -y)
        [3, 7, 6],
        [3, 6, 2],  # back   y=+h  (n = +y)
        [0, 4, 7],
        [0, 7, 3],  # left   x=-h  (n = -x)
        [1, 2, 6],
        [1, 6, 5],  # right  x=+h  (n = +x)
    ],
    dtype=np.int64,
)

# 6-tetra Kuhn decomposition sharing the main diagonal v0-v6 (total vol = cube).
_CUBE_TETRA = np.array(
    [
        [0, 1, 2, 6],
        [0, 1, 5, 6],
        [0, 3, 2, 6],
        [0, 3, 7, 6],
        [0, 4, 5, 6],
        [0, 4, 7, 6],
    ],
    dtype=np.int64,
)


@pytest.fixture
def rayleigh_green_oracle_case():
    """Shared physical inputs for the Rayleigh Green-oracle checks."""
    angle_z = 0.37
    angle_x = -0.29
    rz = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0.0],
            [np.sin(angle_z), np.cos(angle_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_x), -np.sin(angle_x)],
            [0.0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    return {
        "surface_origin_world_m": np.array([7.0, -4.0, 2.5]),
        "surface_basis_world": rz @ rx,
        "local_targets_m": np.array([[3.2, -1.4, 2.1], [-2.6, 4.3, -1.7]]),
        "target_ids": ("ground-side", "underground"),
        "density_kg_m3": 2300.0,
        "frequency_hz": 2.5,
        "amplitude_m": 1e-9 * (0.7 - 0.2j),
        "phase_origin_local_m": np.array([1.1, -0.8, 0.0]),
        "direction_local": np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0]),
        "reference_time_gps": (1_600_000_000, 250_000_000),
        "vp_m_s": np.sqrt(3.0) * 1500.0,
        "vs_m_s": 1500.0,
    }


def cube_vertices_array(h=1.0):
    """Return the 8 cube vertices of the box [-h, h]^3 (edge length 2h)."""
    return np.array(
        [
            [-h, -h, -h],  # 0
            [+h, -h, -h],  # 1
            [+h, +h, -h],  # 2
            [-h, +h, -h],  # 3
            [-h, -h, +h],  # 4
            [+h, -h, +h],  # 5
            [+h, +h, +h],  # 6
            [-h, +h, +h],  # 7
        ],
        dtype=np.float64,
    )


@pytest.fixture
def cube_half():
    """Half edge length of the cube fixture (edge length = 2 * cube_half)."""
    return 1.0


@pytest.fixture
def cube_vertices(cube_half):
    """8 cube vertices of the box [-cube_half, cube_half]^3."""
    return cube_vertices_array(cube_half)


@pytest.fixture
def cube_triangles():
    """12 outward-wound triangles of the cube surface (copy)."""
    return _CUBE_TRIANGLES.copy()


@pytest.fixture
def cube_tetra():
    """6 Kuhn tetrahedra filling the cube volume (copy)."""
    return _CUBE_TETRA.copy()


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Fail optional-backend runs when their selected tests were skipped."""
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = reporter.stats.get("skipped", [])
    require = any(
        os.environ.get(name) == "1"
        for name in (
            "GRAGRA_REQUIRE_CPP",
            "GRAGRA_REQUIRE_CUPY",
            "GRAGRA_REQUIRE_EXTRAS",
        )
    )
    if require and skipped:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter.write_line(
            "Selected optional backend/extra tests were skipped in a strict run:"
        )
        for report in skipped:
            reporter.write_line(f"  {report.nodeid}")


_RESEARCH_TEST_BASENAMES = frozenset(
    {
        "test_reproduce_report.py",
        "test_anna_comparison_contract.py",
        "test_wave4_environment.py",
        "test_wave5_environment_transport.py",
        "test_wave6_environment_acoustic.py",
        "test_cupy_registry_contract.py",
    }
)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Tag research-only modules so public CI can exclude them."""
    for item in items:
        path = Path(str(item.fspath))
        name = path.name
        rel = path.as_posix()
        if name.startswith("test_research_"):
            item.add_marker(pytest.mark.research)
            continue
        if name in _RESEARCH_TEST_BASENAMES:
            item.add_marker(pytest.mark.research)
            continue
        if rel.endswith(
            (
                "tests/oracles/gen1/test_frozen_refinement_trace_audit.py",
                "tests/oracles/gen1/test_refinement_trace_observer_v3.py",
            )
        ):
            item.add_marker(pytest.mark.research)
