"""Coherent gravitational observables."""

import importlib

import numpy as np

from gragra._arrays import as_xyz_vector
from gragra.constants import G_SI
from gragra.observables.projection import project_vectors


def _resolve_coherent_parametric_contract(backend: str, contract_name: str):
    """Select a parametric contraction (by name) for the given backend.

    The lazy-import error messages mirror the point-direct dispatch
    (``integrators/point_direct.py``) so that backend-availability errors are
    consistent across the coherent and point-direct paths. Shared-resolver
    extraction across the point-direct and coherent dispatches stays a
    backlog item (re-evaluated with the coherent GPU port).
    """
    # NOTE: the backend modules MUST be resolved with importlib.import_module
    # (dotted-path resolution through sys.modules), not
    # ``from gragra.kernels import <mod>``: the from-import form returns the
    # module object cached as a *package attribute* even after the entry has
    # been removed from sys.modules, which silently bypasses the unbuilt-
    # backend simulation in tests/test_cpp_backend_missing.py (regression
    # caught by CI test-cpp on PR #37).
    if backend == "numpy":
        module = importlib.import_module("gragra.kernels.parametric")
        return getattr(module, contract_name)
    if backend == "numba":
        try:
            import numba  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "numba is not installed in the current environment. "
                "Please install numba (or use `pip install gragra[numba]`) "
                "to use the numba backend."
            ) from e
        module = importlib.import_module("gragra.kernels._numba_parametric")
        return getattr(module, contract_name)
    if backend == "cpp":
        module = importlib.import_module("gragra.kernels._cpp_parametric")
        return getattr(module, contract_name)
    if backend == "cupy":
        try:
            import cupy  # noqa: F401
        except ImportError as e:

            class CupyImportValueError(ImportError, ValueError):
                pass

            raise CupyImportValueError(
                "cupy is not installed in the current environment. Install the "
                "CuPy wheel matching your CUDA/ROCm runtime (e.g. `pip install "
                "'cupy-cuda12x[ctk]'` for CUDA 12.x without a system CUDA "
                "Toolkit, or plain `cupy-cuda12x` when one is installed) to use "
                "the cupy backend. Note that a CUDA-capable GPU and driver are "
                "required at runtime; see docs/design/cupy_backend.md."
            ) from e
        module = importlib.import_module("gragra.kernels._cupy_parametric")
        return getattr(module, contract_name)
    raise ValueError(
        "Unsupported backend: must be 'numpy', 'numba', 'cpp', or 'cupy', "
        f"got '{backend}'"
    )


def _resolve_coherent_acceleration_contract(backend: str):
    """Select the parametric acceleration contraction for the given backend."""
    return _resolve_coherent_parametric_contract(
        backend, "acceleration_contract_parametric"
    )


def coherent_point_acceleration(
    source_positions_m: list | tuple | np.ndarray,
    weights_kg: list | tuple | np.ndarray,
    targets_m: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Calculate coherent gravitational acceleration at target points.

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    ``backend`` selects the parametric acceleration contraction implementation
    (``"numpy"`` default, ``"numba"``, ``"cpp"``, or ``"cupy"``); all backends
    are numerically equivalent (the G-free geometric factor is identical) and
    the output dtype follows the weights. Raises ``ImportError`` if the
    selected compiled backend (``"numba"``, ``"cpp"``, or ``"cupy"``) is not
    installed; raises ``RuntimeError`` if ``"cupy"`` is installed but no usable
    CUDA device/driver is present at call time.
    """
    if backend == "cupy":
        # Host-side input validation must precede the lazy cupy import inside
        # the resolver below (validate-before-import contract, mirrored from
        # ``integrators/point_direct.py``'s dipole cupy branch). Otherwise a
        # malformed input in a cupy-less environment surfaces as ImportError
        # instead of the intended ValueError.
        from gragra.kernels.parametric import _validate_inputs_parametric

        _validate_inputs_parametric(
            source_positions_m, weights_kg, targets_m, softening_m, chunk_size
        )

    contract = _resolve_coherent_acceleration_contract(backend)

    # Compute G-free contraction
    acc_gfree = contract(
        sources_xyz=source_positions_m,
        weights=weights_kg,
        targets_xyz=targets_m,
        softening_m=softening_m,
        chunk_size=chunk_size,
    )
    # Multiply by G_SI to get physical acceleration
    acc = G_SI * acc_gfree

    acc = np.asarray(acc)
    acc = acc.copy()
    acc.setflags(write=False)
    return acc


def coherent_point_gravity_gradient(
    source_positions_m: list | tuple | np.ndarray,
    weights_kg: list | tuple | np.ndarray,
    targets_m: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Calculate the coherent gravity gradient tensor at target points.

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    Output shape is ``(M, *P, 3, 3)`` — the tensor axes trail the parametric
    axes (component-axis-last convention). The output dtype follows the
    weights (real -> float64, complex -> complex128) and the returned array is
    read-only. ``tr(T) = 0`` (the vacuum Laplace property) holds only for
    ``softening_m = 0``; the softened tensor is an approximation.

    ``backend`` selects the parametric gradient contraction implementation
    (``"numpy"`` default, ``"numba"``, ``"cpp"``, or ``"cupy"``); all backends
    are numerically equivalent (the G-free geometric factor is identical).
    Raises ``ImportError`` if the selected compiled backend (``"numba"``,
    ``"cpp"``, or ``"cupy"``) is not installed; raises ``RuntimeError`` if
    ``"cupy"`` is installed but no usable CUDA device/driver is present at
    call time.
    """
    if backend == "cupy":
        # Host-side input validation must precede the lazy cupy import inside
        # the resolver below (validate-before-import contract, mirrored from
        # ``integrators/point_direct.py``'s dipole cupy branch). Otherwise a
        # malformed input in a cupy-less environment surfaces as ImportError
        # instead of the intended ValueError.
        from gragra.kernels.parametric import _validate_inputs_parametric

        _validate_inputs_parametric(
            source_positions_m, weights_kg, targets_m, softening_m, chunk_size
        )

    contract = _resolve_coherent_parametric_contract(
        backend, "gradient_contract_parametric"
    )

    # Compute G-free contraction; only this observable layer multiplies G_SI.
    grad_gfree = contract(
        sources_xyz=source_positions_m,
        weights=weights_kg,
        targets_xyz=targets_m,
        softening_m=softening_m,
        chunk_size=chunk_size,
    )
    grad = G_SI * grad_gfree

    grad = np.asarray(grad)
    grad = grad.copy()
    grad.setflags(write=False)
    return grad


def coherent_projected_acceleration(
    source_positions_m: list | tuple | np.ndarray,
    weights_kg: list | tuple | np.ndarray,
    targets_m: list | tuple | np.ndarray,
    direction: list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Calculate coherent projected gravitational acceleration.

    Complex outputs represent linear complex amplitudes or transfer-function-like
    responses, not directly real-valued time-domain gravitational fields.

    ``backend`` selects the implementation (``"numpy"`` default, ``"numba"``,
    ``"cpp"``, or ``"cupy"``). Raises ``ImportError`` if the selected compiled
    backend (``"numba"``, ``"cpp"``, or ``"cupy"``) is not installed; raises
    ``RuntimeError`` if ``"cupy"`` is installed but no usable CUDA
    device/driver is present at call time.
    """
    dir_vec = as_xyz_vector("direction", direction)
    norm = np.linalg.norm(dir_vec)
    if norm == 0.0:
        raise ValueError("direction vector cannot be a zero vector")
    unit_dir = dir_vec / norm

    # Calculate full 3D coherent acceleration
    acc = coherent_point_acceleration(
        source_positions_m=source_positions_m,
        weights_kg=weights_kg,
        targets_m=targets_m,
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )

    # Project along unit direction
    res = project_vectors(acc, unit_dir)

    res = np.asarray(res)
    res = res.copy()
    res.setflags(write=False)
    return res
