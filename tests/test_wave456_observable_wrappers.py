from pathlib import Path

import numpy as np

import gragra
from gragra.analytic.wave456_reference import point_mass_observables_g_free
from gragra.constants import G_SI
from gragra.observables.wave456_reference import point_mass_observables


def test_wrapper_imports_this_worktree_and_applies_g_once_read_only():
    root = Path(__file__).resolve().parents[1]
    assert Path(gragra.__file__).resolve().is_relative_to(root / "src")
    args = ([[0.0, 0.0, 0.0]], [2.0 + 3.0j], [[1.0, 2.0, 3.0]])
    raw = point_mass_observables_g_free(*args)
    scaled = point_mass_observables(*args)
    for actual, expected in zip(scaled, raw, strict=True):
        np.testing.assert_array_equal(actual, G_SI * expected)
        assert actual.dtype == np.complex128
        assert not actual.flags.writeable


def test_wave456_names_are_not_top_level_exports():
    assert not hasattr(gragra, "point_mass_observables")
    assert not hasattr(gragra, "GFreeObservables")
