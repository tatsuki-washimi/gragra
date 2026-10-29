"""Phase 16-b Stage A & A-3: half-space fixture-contract pilot and Rayleigh extension.

Design: docs/design/phase16b_anna_halfspace_pilot.md &
docs/design/phase16b_anna_fullspace.md §8.

Self-consistency check for Phase 13 half-space golden values and published
closed-form Rayleigh wave expressions (ANNA paper arXiv:2603.15157v4 eqs.75-76).
"""

import numpy as np

from gragra.analytic.halfspace import rayleigh_depth_reduction
from gragra.waves import (
    evanescent_complex_factor,
    free_surface_reflection_vertical,
    rayleigh_speed,
    surface_displacement_factor_vertical,
)

VS = 1.0
VP = np.sqrt(3.0)  # Poisson solid
J_ANNA_TO_GRAGRA = np.diag([1.0, 1.0, -1.0])

# Same golden values as tests/test_analytic_halfspace.py (independently
# derived from the exact 2-exponential Rayleigh eigenfunction, 2026-07-04).
R_KH_05 = 0.899944442512
R_KH_1 = 0.837384637935


def _map_rayleigh_anna_vector(v_anna: np.ndarray) -> np.ndarray:
    """Map ANNA's z-down complex vector to gragra's z-up convention."""
    return -np.conj(J_ANNA_TO_GRAGRA @ v_anna)


def _compute_rayleigh_non_poisson_gragra():
    vp = 5000.0
    vs = 2500.0
    cr = rayleigh_speed(vp, vs)
    q = np.sqrt(1.0 - (cr / vp) ** 2)
    s = np.sqrt(1.0 - (cr / vs) ** 2)
    gamma = (1.0 - np.sqrt(q / s)) / (q - np.sqrt(q / s))
    f = 5.0
    h = 20.0
    kr = 2.0 * np.pi * f / cr
    decay = evanescent_complex_factor(
        [[0.0, 0.0, -h]],
        [[1.0, 0.0, 0.0]],
        [kr],
        kr,
    )[0, 0, 0]
    a_coeff = 2.0 * np.pi * gamma * decay
    a_anna = np.array([1j * a_coeff, 0.0, -a_coeff], dtype=np.complex128)
    return _map_rayleigh_anna_vector(a_anna)


FIXTURE_CASES = [
    {
        "case_id": "rayleigh_depth_reduction_kh_0.5_poisson",
        "description": (
            "Rayleigh depth reduction R(kh=0.5) for a Poisson solid "
            "(VP=sqrt(3), VS=1), vertical incidence."
        ),
        "gragra_value": lambda: float(rayleigh_depth_reduction(0.5, 1.0, VP, VS)),
        "reference_value": R_KH_05,
        "reference_source": (
            "hand-computed (Stage A self-consistency, not ANNA output)"
        ),
        "rtol": 1e-9,
    },
    {
        "case_id": "rayleigh_depth_reduction_kh_1.0_poisson",
        "description": (
            "Rayleigh depth reduction R(kh=1.0) for a Poisson solid "
            "(VP=sqrt(3), VS=1), vertical incidence."
        ),
        "gragra_value": lambda: float(rayleigh_depth_reduction(1.0, 1.0, VP, VS)),
        "reference_value": R_KH_1,
        "reference_source": (
            "hand-computed (Stage A self-consistency, not ANNA output)"
        ),
        "rtol": 1e-9,
    },
    {
        "case_id": "free_surface_vertical_p_identity",
        "description": (
            "Vertical-incidence free-surface identity: "
            "1 + R_PP == surface displacement factor == 2.0."
        ),
        "gragra_value": lambda: 1.0 + free_surface_reflection_vertical("P")["R_PP"],
        "reference_value": surface_displacement_factor_vertical(),
        "reference_source": (
            "hand-computed (Stage A self-consistency, not ANNA output)"
        ),
        "rtol": 1e-15,
    },
    {
        "case_id": "rayleigh_halfspace_eq75_eq76_non_poisson",
        "description": (
            "Half-space Rayleigh wave Newtonian noise acceleration for a "
            "non-Poisson solid (ANNA eqs.75-76)."
        ),
        "gragra_value": _compute_rayleigh_non_poisson_gragra,
        "reference_value": np.array(
            [1j * 3.9853956605002006, 0.0, -3.9853956605002006],
            dtype=np.complex128,
        ),
        "reference_source": (
            "published closed-form (ANNA paper arXiv:2603.15157v4 "
            "eqs.75-76 = Harms 2019)"
        ),
        "rtol": 1e-12,
    },
]


def test_fixture_contract_shape():
    """Every fixture case exposes the Sec.4 contract fields."""
    required_keys = {
        "case_id",
        "description",
        "gragra_value",
        "reference_value",
        "reference_source",
        "rtol",
    }
    for case in FIXTURE_CASES:
        assert required_keys.issubset(case.keys())
        source = case["reference_source"]
        assert any(
            source.startswith(prefix)
            for prefix in ("hand-computed", "published closed-form")
        )


def test_fixture_cases_match_reference():
    """Self-consistency check (Stage A & A-3)."""
    for case in FIXTURE_CASES:
        gragra_value = case["gragra_value"]()
        np.testing.assert_allclose(
            gragra_value,
            case["reference_value"],
            rtol=case["rtol"],
            err_msg=f"fixture case {case['case_id']} failed",
        )


def test_rayleigh_non_poisson_assertions():
    """Verify Rayleigh non-Poisson component properties and reference match."""
    val = FIXTURE_CASES[-1]["gragra_value"]()
    np.testing.assert_allclose(val[0].real, 0.0, atol=1e-15)
    assert val[0].imag > 0.0
    np.testing.assert_allclose(val[2].imag, 0.0, atol=1e-15)
    assert val[2].real < 0.0
    ref = FIXTURE_CASES[-1]["reference_value"]
    np.testing.assert_allclose(val, ref, rtol=1e-12, atol=1e-15)


def test_rayleigh_anna_position_mapping_uses_z_reflection():
    """A-3 maps ANNA z-down positions to gragra z-up positions with J."""
    x_anna = np.array([1.25, -2.5, -3.75], dtype=np.float64)

    np.testing.assert_array_equal(
        J_ANNA_TO_GRAGRA @ x_anna,
        np.array([1.25, -2.5, 3.75], dtype=np.float64),
    )


def test_rayleigh_anna_displacement_mapping_includes_normalization_sign():
    """A-3 maps general complex displacement with its z-up normalization sign."""
    u_anna = np.array([1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j])

    np.testing.assert_array_equal(
        _map_rayleigh_anna_vector(u_anna),
        np.array([-1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j]),
    )


def test_rayleigh_anna_acceleration_mapping_includes_normalization_sign():
    """A-3 maps general complex acceleration independently of the golden case."""
    a_anna = np.array([-2.0 + 3.0j, 4.0 - 5.0j, 6.0 + 7.0j])

    np.testing.assert_array_equal(
        _map_rayleigh_anna_vector(a_anna),
        np.array([2.0 + 3.0j, -4.0 - 5.0j, 6.0 - 7.0j]),
    )


def test_no_cavity_coupling_dependency():
    """This module's own imports must not reference cavity_coupling."""
    import ast
    import inspect

    source = inspect.getsource(inspect.getmodule(test_no_cavity_coupling_dependency))
    tree = ast.parse(source)
    referenced_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            referenced_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                referenced_names.add(node.module)
            referenced_names.update(alias.name for alias in node.names)
    assert not any("cavity_coupling" in name for name in referenced_names)
