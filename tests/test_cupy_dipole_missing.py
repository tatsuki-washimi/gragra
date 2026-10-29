"""Tests for the dipole cupy backend under missing-cupy / bad-input conditions.

Covers two contracts specific to ``direct_point_dipole_sum`` /
``displacement_transfer_acceleration``'s cupy branch:

1. Target normalization must happen *before* the cupy-branch host
   validation runs, so ``TargetPoints`` instances and bare ``(3,)``
   single-vector targets are accepted (not rejected as malformed
   ``(N, 3)`` tables) — see the ``point_direct.py`` fix that passes the
   normalized ``t_xyz`` (not the raw ``targets``) into
   ``_validate_dipole_inputs``.
2. Host-side input validation (softening_m, chunk_size, etc.) still
   precedes the cupy import, so a malformed input raises ``ValueError``
   without ever touching the ``ImportError`` hint path
   (validate-before-import contract, mirrored from
   ``test_cupy_backend_missing.py``).

These tests run in every environment (no GPU/cupy needed): the
``hide_cupy`` fixture below is a local copy of the pattern used in
``test_cupy_backend_missing.py`` (not imported, per this file's
write-scope).
"""

import sys

import pytest

from gragra.observables.dipole import displacement_transfer_acceleration
from gragra.targets.points import TargetPoints


@pytest.fixture
def hide_cupy(monkeypatch):
    """Simulate an environment where cupy is not installed.

    Order-independent: also drops the wrapper module from the import
    cache and the parent-package attribute, so a wrapper imported by an
    earlier test cannot leak through (mirrors the cpp missing fixture).
    """
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.delitem(
        sys.modules, "gragra.kernels._cupy_inverse_square", raising=False
    )
    kernels_pkg = sys.modules.get("gragra.kernels")
    if kernels_pkg is not None and hasattr(kernels_pkg, "_cupy_inverse_square"):
        monkeypatch.delattr(kernels_pkg, "_cupy_inverse_square")


def test_cupy_dipole_missing_raises_import_error_with_target_points(hide_cupy):  # noqa: ARG001
    """backend='cupy' with a TargetPoints target reaches the ImportError hint.

    Regression guard: before the point_direct.py fix, the raw (unnormalized)
    ``targets`` argument was passed to ``_validate_dipole_inputs``, which
    rejects anything that is not already an (N, 3) array-like — including
    ``TargetPoints`` instances — with a ValueError, masking the intended
    ImportError.
    """
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets = TargetPoints([[1.0, 0.0, 0.0]])

    with pytest.raises(ImportError) as excinfo:
        displacement_transfer_acceleration(
            sources_xyz, mass_kg, displacement_m, targets, backend="cupy"
        )

    msg = str(excinfo.value)
    assert "cupy is not installed" in msg
    # Back-compat dual inheritance: ImportError is also a ValueError.
    assert isinstance(excinfo.value, ValueError)


def test_cupy_dipole_missing_raises_import_error_with_single_vector_target(
    hide_cupy,  # noqa: ARG001
):
    """backend='cupy' with a bare (3,) single-vector target reaches ImportError.

    Regression guard: before the fix, a (3,) target was promoted to
    (1, 3) by ``_normalize_targets`` but that promoted array was then
    discarded in favor of the raw (3,) ``targets`` argument passed to
    ``_validate_dipole_inputs`` — which strictly requires (N, 3) and
    rejects (3,) — so this case was misclassified as a ValueError instead
    of reaching the cupy import.
    """
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets = [1.0, 0.0, 0.0]  # single (3,) vector, not a table

    with pytest.raises(ImportError) as excinfo:
        displacement_transfer_acceleration(
            sources_xyz, mass_kg, displacement_m, targets, backend="cupy"
        )

    msg = str(excinfo.value)
    assert "cupy is not installed" in msg
    assert isinstance(excinfo.value, ValueError)


def test_cupy_dipole_input_validation_precedes_import(hide_cupy):  # noqa: ARG001
    """Host-side input validation raises ValueError before the cupy import.

    chunk_size=0 is invalid; this must be caught by
    ``_validate_dipole_inputs`` before ``_import_cupy_kernels`` runs, so
    the raised error is a plain ValueError, not the ImportError/ValueError
    dual-inheritance instance used for the missing-cupy hint.
    """
    sources_xyz = [[0.0, 0.0, 0.0]]
    mass_kg = [1.0]
    displacement_m = [[1.0, 0.0, 0.0]]
    targets = TargetPoints([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="chunk_size must be positive") as excinfo:
        displacement_transfer_acceleration(
            sources_xyz,
            mass_kg,
            displacement_m,
            targets,
            chunk_size=0,
            backend="cupy",
        )

    # Validate-before-import contract: this must not be the ImportError
    # subclass raised by the missing-cupy hint path.
    assert not isinstance(excinfo.value, ImportError)
