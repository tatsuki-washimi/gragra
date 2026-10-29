"""Optional scientific checks against a real SPECFEM3D pilot export.

The checks run ``check_a1``..``check_a4`` from ``tests/_tier_a_checks.py``
against real data, in addition to the synthetic fixtures used by
``tests/test_specfem_tier_a.py``. They are intentionally not part of the
ordinary CI matrix because a pilot HDF5 file is local research input. Run
them manually when that input is available:

.. code-block:: bash

    GRAGRA_SPECFEM_PILOT_H5=/path/to/pilot_static.h5 \\
    GRAGRA_SPECFEM_PILOT_EXPECT=/path/to/pilot_expect.json \\
    conda run -n gragra pytest tests/test_specfem_pilot_gate.py -v

Optional additional env vars:

- ``GRAGRA_SPECFEM_SOURCE``: path to a checked-out SPECFEM3D Cartesian
  source tree -- when set, also cross-checks ``NGLLX`` from
  ``setup/constants.h`` (or ``constants.h.in``) against the pilot HDF5's
  recorded ``/metadata/ngll_shape`` (mirrors
  ``tests/test_specfem_schema.py``'s 2-stage upstream-equivalence design,
  applied to a real *exported file* rather than the schema module).
- ``GRAGRA_SPECFEM_PILOT_SEMD_DIR``: directory of ``*.semd`` seismogram
  files -- when set, cross-checks ``/time`` against each ``*.semd`` file's
  first (time) column via ``provenance.crosscheck_t0_semd`` (only
  meaningful for a ``time_sampled`` pilot export; skipped for a static one).

``GRAGRA_SPECFEM_PILOT_EXPECT`` sidecar JSON shape (A1's direction-fixed
check has no generic default -- it needs the pilot mesh's own expected
end-to-end axis displacement vectors, which only whoever prepared the pilot
mesh can supply):

.. code-block:: json

    {
      "axis_vectors": {"0": {"i": [dx, dy, dz], "j": [...], "k": [...]}},
      "rho_by_material": {"1": 2650.0},
      "rtol_by_material": {"1": 2e-7},
      "profile_density_range": [2000.0, 3000.0]
    }
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Final

import pytest

_PILOT_H5_ENV: Final[str] = "GRAGRA_SPECFEM_PILOT_H5"
_PILOT_EXPECT_ENV: Final[str] = "GRAGRA_SPECFEM_PILOT_EXPECT"
_PILOT_SOURCE_ENV: Final[str] = "GRAGRA_SPECFEM_SOURCE"
_PILOT_SEMD_DIR_ENV: Final[str] = "GRAGRA_SPECFEM_PILOT_SEMD_DIR"

_pilot_h5_env = os.environ.get(_PILOT_H5_ENV)
if not _pilot_h5_env:
    pytest.skip(
        f"{_PILOT_H5_ENV} is not set; this file runs only against a real "
        "SPECFEM3D pilot export and is not part of the ordinary CI matrix.",
        allow_module_level=True,
    )

_PILOT_H5_PATH: Final[Path] = Path(_pilot_h5_env)

# E402: these imports must come after the module-level env-availability skip
# above, matching tests/test_cupy_backend_available.py's two-stage-skip style.
from _tier_a_checks import (  # noqa: E402
    check_a1_gll_ordering,
    check_a2_quadrature_volume,
    check_a3_material_mass,
    check_a4_masks_scalars_provenance,
)

from gragra.adapters.specfem import provenance as _provenance  # noqa: E402


def _h5py():
    return pytest.importorskip("h5py")


def _load_pilot_expect() -> dict[str, object]:
    """Load the expected-values sidecar JSON for the pilot HDF5 input.

    The sidecar is required because axis vectors and material densities are
    properties of the supplied research mesh rather than generic defaults.
    A missing sidecar is an input error, not an optional-dependency skip.
    """
    expect_env = os.environ.get(_PILOT_EXPECT_ENV)
    if not expect_env:
        pytest.fail(
            f"{_PILOT_EXPECT_ENV} must be set alongside {_PILOT_H5_ENV} -- "
            "A1's direction-fixed check and A3's material density/tolerance "
            "declarations have no generic default for an arbitrary pilot "
            "mesh (see this module's docstring for the expected JSON shape)."
        )
    raw = json.loads(Path(expect_env).read_text())
    axis_vectors = {
        int(row): {axis: tuple(vec) for axis, vec in axes.items()}
        for row, axes in raw["axis_vectors"].items()
    }
    rho_by_material = {int(k): float(v) for k, v in raw["rho_by_material"].items()}
    rtol_by_material = {int(k): float(v) for k, v in raw["rtol_by_material"].items()}
    profile_density_range = tuple(raw["profile_density_range"])
    return {
        "axis_vectors": axis_vectors,
        "rho_by_material": rho_by_material,
        "rtol_by_material": rtol_by_material,
        "profile_density_range": profile_density_range,
    }


def test_pilot_a1_gll_ordering():
    _h5py()
    expect = _load_pilot_expect()
    check_a1_gll_ordering(_PILOT_H5_PATH, expected_axis_vectors=expect["axis_vectors"])


def test_pilot_a2_quadrature_volume():
    _h5py()
    check_a2_quadrature_volume(_PILOT_H5_PATH)


def test_pilot_a3_material_mass():
    _h5py()
    expect = _load_pilot_expect()
    check_a3_material_mass(
        _PILOT_H5_PATH,
        rho_by_material=expect["rho_by_material"],
        rtol_by_material=expect["rtol_by_material"],
    )


def test_pilot_a4_masks_scalars_provenance():
    _h5py()
    expect = _load_pilot_expect()
    check_a4_masks_scalars_provenance(
        _PILOT_H5_PATH, profile_density_range=expect["profile_density_range"]
    )


# ===========================================================================
# Optional: upstream constants.h cross-check (GRAGRA_SPECFEM_SOURCE)
# ===========================================================================


def test_pilot_upstream_ngllx_equivalence():
    """When GRAGRA_SPECFEM_SOURCE points at a checked-out SPECFEM3D source
    tree, cross-check its setup/constants.h (or constants.h.in) NGLLX value
    against the pilot HDF5's own recorded /metadata/ngll_shape[0] -- the
    same optional-local-input, CI-skips 2-stage design as
    tests/test_specfem_schema.py's upstream-equivalence test, applied here
    to a real exported file's provenance rather than to schema.py's
    hardcoded expectations.
    """
    h5py = _h5py()
    source_env = os.environ.get(_PILOT_SOURCE_ENV)
    if not source_env:
        pytest.skip(
            f"{_PILOT_SOURCE_ENV} is not set; skipping upstream NGLLX "
            "cross-check (optional local check)"
        )
    source_root = Path(source_env)
    header = source_root / "setup" / "constants.h"
    if not header.is_file():
        header = source_root / "setup" / "constants.h.in"
    if not header.is_file():
        pytest.fail(
            f"Neither setup/constants.h nor constants.h.in found under {source_root}"
        )

    content = header.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"integer\s*,\s*parameter\s*::\s*NGLLX\s*=\s*(\d+)",
        content,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(matches) != 1:
        pytest.fail(f"expected exactly 1 NGLLX match in {header}, found {len(matches)}")
    upstream_ngllx = int(matches[0])

    with h5py.File(_PILOT_H5_PATH, "r") as f:
        ngll_shape = f["/metadata"].attrs["ngll_shape"]
    assert int(ngll_shape[0]) == upstream_ngllx, (
        f"pilot export's ngll_shape[0]={int(ngll_shape[0])} does not match "
        f"upstream NGLLX={upstream_ngllx} from {header}"
    )


# ===========================================================================
# Optional: semd time crosscheck (GRAGRA_SPECFEM_PILOT_SEMD_DIR)
# ===========================================================================


def test_pilot_semd_time_crosscheck():
    """When GRAGRA_SPECFEM_PILOT_SEMD_DIR is set, cross-check /time against
    every *.semd file's first (time) column via
    provenance.crosscheck_t0_semd (specfem_import.md §t0の取得契約: bit
    equality is not expected -- float32 storage + list-directed ASCII
    rounding, atol~2e-5 s). Only meaningful for a time_sampled pilot export.
    """
    h5py = _h5py()
    semd_dir_env = os.environ.get(_PILOT_SEMD_DIR_ENV)
    if not semd_dir_env:
        pytest.skip(
            f"{_PILOT_SEMD_DIR_ENV} is not set; skipping semd time crosscheck "
            "(optional local check)"
        )
    semd_dir = Path(semd_dir_env)
    semd_paths = sorted(semd_dir.glob("*.semd"))
    if not semd_paths:
        pytest.fail(f"no *.semd files found under {semd_dir}")

    with h5py.File(_PILOT_H5_PATH, "r") as f:
        meta = f["/metadata"]
        output_sampling_kind = meta.attrs.get("output_sampling_kind")
        if output_sampling_kind != "time_sampled":
            pytest.skip(
                "semd time crosscheck only applies to a time_sampled pilot "
                f"export, got output_sampling_kind={output_sampling_kind!r}"
            )
        dt_s = float(meta.attrs["dt_s"])
        t0 = float(meta.attrs["t0"])

    for semd_path in semd_paths:
        max_deviation = _provenance.crosscheck_t0_semd(semd_path, dt=dt_s, t0=t0)
        assert max_deviation >= 0.0  # crosscheck_t0_semd raises on failure
