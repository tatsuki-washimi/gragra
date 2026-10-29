"""Fast availability and sanity checks for the `validation` extra.

Unlike ``tests/test_halfspace_gravity_oracle.py`` / ``test_rayleigh_gravity_
oracle.py`` (pytest.mark.slow, the full cutoff extrapolation contracts),
this module is unmarked so it runs in ``ci.yml``'s normal extras matrix
under GRAGRA_REQUIRE_EXTRAS=1, catching a silently-skipped scipy/mpmath
import for the `validation` extra the same way the other extras'
`*_available.py` files do for theirs.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("mpmath")

from _provenance_contract import verify_content_addressed_provenance  # noqa: E402
from oracles.generate_halfspace_oracle import (  # noqa: E402
    bulk_finite,
    disk_az,
    generator_sha256,
    sheet_finite,
)
from oracles.generate_halfspace_oracle import (
    derivation_input_sha256 as halfspace_derivation_input_sha256,
)
from oracles.generate_rayleigh_oracle import (  # noqa: E402
    derivation_input_sha256 as rayleigh_derivation_input_sha256,
)
from oracles.generate_rayleigh_oracle import (
    rayleigh_gamma,
    solve_rayleigh,
)


def test_disk_az_infinite_plane_limit():
    # As R -> infinity, the finite-disk formula must recover the standard
    # infinite-plane result a_z = -2*pi*G*sigma*sign(d) (G=1 in the oracle's
    # dimensionless units).
    d, sigma = 3.0, 2.0
    a_large_r = disk_az(d, 1.0e6, sigma)
    expected = -2.0 * np.pi * sigma
    assert a_large_r == pytest.approx(expected, rel=1e-4)


def test_disk_az_zero_radius_vanishes():
    assert disk_az(3.0, 0.0, 2.0) == pytest.approx(0.0, abs=1e-12)


def test_bulk_and_sheet_finite_values_are_sane():
    h, r_cut = 1.0, 20.0
    bulk_val, quad_err = bulk_finite(h, r_cut)
    sheet_val = sheet_finite(h, r_cut)
    assert np.isfinite(bulk_val)
    assert np.isfinite(sheet_val)
    assert quad_err < 1e-4


def test_oracle_data_file_present_and_loadable():
    path = Path(__file__).parent / "oracles" / "data" / "halfspace_v1.json"
    assert path.exists(), "run tests/oracles/generate_halfspace_oracle.py first"
    with path.open() as f:
        data = json.load(f)
    assert data["schema_version"] == 1
    assert len(data["points"]) >= 3


def test_oracle_generator_hash_matches_saved_data():
    # Fast, unmarked check: catches a generator edit that wasn't followed by
    # regenerating tests/oracles/data/halfspace_v1.json (the slow suite
    # re-derives the physics; this only guards the provenance contract).
    path = Path(__file__).parent / "oracles" / "data" / "halfspace_v1.json"
    with path.open() as f:
        data = json.load(f)
    assert data["provenance"]["source_content_sha256"] == generator_sha256(), (
        "generate_halfspace_oracle.py changed since halfspace_v1.json was "
        "generated -- rerun the generator and commit the updated JSON."
    )


def test_halfspace_oracle_content_addressed_provenance_is_verified():
    """Fail-closed, content-addressed provenance check with no git history or network
    access required. This protects the independently-derived scientific oracle.
    See tests/_provenance_contract.py for the numerical contract."""
    path = Path(__file__).parent / "oracles" / "data" / "halfspace_v1.json"
    with path.open() as f:
        data = json.load(f)
    generator_path = Path(__file__).parent / "oracles" / "generate_halfspace_oracle.py"
    verify_content_addressed_provenance(
        data,
        oracle_name="halfspace_v1",
        generator_path=generator_path,
        derivation_input_sha256=halfspace_derivation_input_sha256(),
    )


def test_rayleigh_secular_solve_matches_poisson_golden():
    # Sanity check against the published Poisson-solid golden values
    # (docs/architecture/halfspace_conventions.md Sec.3.1), independent of
    # gragra.waves.rayleigh_speed.
    c_over_vs, q, s = solve_rayleigh(np.sqrt(3.0))
    assert c_over_vs == pytest.approx(0.9194016868, rel=1e-6)
    assert q == pytest.approx(0.847487, rel=1e-5)
    assert s == pytest.approx(0.393320, rel=1e-5)


def test_rayleigh_gamma_is_finite_and_positive():
    _, q, s = solve_rayleigh(np.sqrt(3.0))
    gamma = rayleigh_gamma(q, s)
    assert np.isfinite(gamma)
    assert gamma > 0.0


def test_rayleigh_oracle_data_file_present_and_loadable():
    path = Path(__file__).parent / "oracles" / "data" / "rayleigh_v1.json"
    assert path.exists(), "run tests/oracles/generate_rayleigh_oracle.py first"
    with path.open() as f:
        data = json.load(f)
    assert data["schema_version"] == 1
    assert len(data["points"]) >= 3


def test_rayleigh_oracle_content_addressed_provenance_is_verified():
    """Fail-closed, content-addressed provenance check with no git history or network
    access required. This protects the independently-derived scientific oracle.
    See tests/_provenance_contract.py for the numerical contract."""
    path = Path(__file__).parent / "oracles" / "data" / "rayleigh_v1.json"
    with path.open() as f:
        data = json.load(f)
    generator_path = Path(__file__).parent / "oracles" / "generate_rayleigh_oracle.py"
    verify_content_addressed_provenance(
        data,
        oracle_name="rayleigh_v1",
        generator_path=generator_path,
        derivation_input_sha256=rayleigh_derivation_input_sha256(),
    )
