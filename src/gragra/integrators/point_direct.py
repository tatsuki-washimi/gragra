"""Direct point-to-point integration engine."""

from collections.abc import Callable

import numpy as np

from gragra._arrays import as_xyz_table
from gragra.array_types import ComplexArray, FloatArray
from gragra.gravity_sources import PointMass, PointMassCloud, WeightedPointSource
from gragra.kernels.inverse_square import (
    acceleration_contract,
    dipole_contract,
    displacement_gradient_contract,
    displacement_potential_contract,
    gradient_contract,
    potential_contract,
)
from gragra.targets.points import TargetPoints


def _normalize_source(
    source: WeightedPointSource | PointMass | PointMassCloud,
) -> tuple[FloatArray, FloatArray | ComplexArray]:
    """Extract and normalize coordinates and weights from any source type.

    Parameters
    ----------
    source : WeightedPointSource, PointMass, or PointMassCloud
        The gravitational perturbation source.

    Returns
    -------
    positions : FloatArray (N, 3)
    weights : FloatArray or ComplexArray (N,)
    """
    if isinstance(source, WeightedPointSource):
        return source.positions_m, source.weights_kg
    if isinstance(source, (PointMass, PointMassCloud)):
        wps = source.as_weighted_source()
        return wps.positions_m, wps.weights_kg
    raise TypeError(
        "Unsupported source type: must be WeightedPointSource, "
        f"PointMass, or PointMassCloud, got {type(source)}"
    )


def _normalize_targets(targets: TargetPoints | list | tuple | np.ndarray) -> FloatArray:
    """Normalize and validate target coordinates.

    Supports TargetPoints or array-like.
    If a 1D vector of shape (3,) is passed, it is automatically promoted
    to shape (1, 3).

    Parameters
    ----------
    targets : TargetPoints or array-like
        The targets coordinates.

    Returns
    -------
    FloatArray
        A read-only float64 array of shape (M, 3).
    """
    if isinstance(targets, TargetPoints):
        return targets.positions_m

    # Handle array-like input
    try:
        val_arr = np.asarray(targets)
        # Promote single point vector (3,) to (1, 3) table
        if val_arr.ndim == 1 and val_arr.shape == (3,):
            val_arr = val_arr[np.newaxis, :]
        return as_xyz_table("targets", val_arr)
    except (ValueError, TypeError) as e:
        raise TypeError(f"targets must be TargetPoints or array-like: {e}") from e


def _import_numba_kernels():
    """Lazy import helper for Numba kernels with a clear error message."""
    try:
        import numba  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "numba is not installed in the current environment. "
            "Please install numba (or use `pip install gragra[numba]`) "
            "to use the numba backend."
        ) from e

    import gragra.kernels._numba_inverse_square as numba_kernels

    return numba_kernels


def _import_cpp_kernels():
    """Lazy import helper for C++ kernels with a clear error message."""
    try:
        import gragra.kernels._cpp_inverse_square as cpp_kernels
    except ImportError:
        raise

    return cpp_kernels


def _import_cupy_kernels():
    """Lazy import helper for CuPy kernels with a clear error message."""
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

    import gragra.kernels._cupy_inverse_square as cupy_kernels

    return cupy_kernels


def direct_point_sum(
    source: WeightedPointSource | PointMass | PointMassCloud,
    targets: TargetPoints | list | tuple | np.ndarray,
    *,
    observable: str,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Execute G-free integration by dispatching to the appropriate kernel contract.

    This function is internal to the package.

    Parameters
    ----------
    source : WeightedPointSource, PointMass, or PointMassCloud
        The source of gravitational perturbation.
    targets : TargetPoints or array-like
        The target observation points.
    observable : str
        The observable type: 'potential', 'acceleration', or 'gradient'.
    softening_m : float, optional
        Plummer softening parameter. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop. Defaults to None.
    backend : str, optional
        The calculation backend: 'numpy' (default), 'numba', 'cpp', or
        'cupy' (GPU; requires a CuPy wheel matching the local CUDA/ROCm
        runtime and a working CUDA device).

    Returns
    -------
    FloatArray or ComplexArray
        Calculated G-free observable result.
    """
    s_xyz, w = _normalize_source(source)
    t_xyz = _normalize_targets(targets)

    if backend == "numpy":
        dispatch_table: dict[str, Callable] = {
            "potential": potential_contract,
            "acceleration": acceleration_contract,
            "gradient": gradient_contract,
        }
    elif backend == "numba":
        numba_kernels = _import_numba_kernels()
        dispatch_table = {
            "potential": numba_kernels.potential_contract,
            "acceleration": numba_kernels.acceleration_contract,
            "gradient": numba_kernels.gradient_contract,
        }
    elif backend == "cpp":
        cpp_kernels = _import_cpp_kernels()
        dispatch_table = {
            "potential": cpp_kernels.potential_contract,
            "acceleration": cpp_kernels.acceleration_contract,
            "gradient": cpp_kernels.gradient_contract,
        }
    elif backend == "cupy":
        cupy_kernels = _import_cupy_kernels()
        dispatch_table = {
            "potential": cupy_kernels.potential_contract,
            "acceleration": cupy_kernels.acceleration_contract,
            "gradient": cupy_kernels.gradient_contract,
        }
    else:
        raise ValueError(
            "Unsupported backend: must be 'numpy', 'numba', 'cpp', or 'cupy', "
            f"got '{backend}'"
        )

    if observable not in dispatch_table:
        raise ValueError(
            f"Unsupported observable: must be one of {list(dispatch_table.keys())}, "
            f"got {observable}"
        )

    contract_fn = dispatch_table[observable]
    return contract_fn(s_xyz, w, t_xyz, softening_m=softening_m, chunk_size=chunk_size)


def direct_point_displacement_sum(
    mass_kg: list | tuple | np.ndarray,
    displacement_m: list | tuple | np.ndarray,
    sources_xyz: list | tuple | np.ndarray,
    targets: TargetPoints | list | tuple | np.ndarray,
    *,
    observable: str = "acceleration",
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Execute a G-free source-displacement derivative integration.

    Unlike ``direct_point_sum``, this takes raw ``mass_kg``/``displacement_m``
    arrays directly rather than a ``WeightedPointSource``, because the
    displacement vector weight is not a ``weights_kg``-compatible quantity
    (``weights_kg`` is mass-element-only; see kernel_contract.md §2.1).

    This function is internal to the package.

    Parameters
    ----------
    mass_kg : array-like
        Background mass element at each source point, shape (N,). See
        ``gragra.kernels.inverse_square.dipole_contract`` for the full
        validation contract (real, finite, non-negative only).
    displacement_m : array-like
        Displacement vector at each source point, shape (N, 3) (real or
        complex).
    sources_xyz : array-like
        Source coordinates table of shape (N, 3) in meters.
    targets : TargetPoints or array-like
        The target observation points.
    observable : str, optional
        Source-displacement derivative to calculate: ``'potential'``,
        ``'acceleration'`` (default), or ``'gradient'``.
    softening_m : float, optional
        Plummer softening parameter. Defaults to 0.0.
    chunk_size : int, optional
        Chunk size for source-axis loop. Defaults to None.
    backend : str, optional
        The calculation backend. Potential and gradient support ``'numpy'``
        and ``'numba'``. Acceleration preserves the existing ``'numpy'``,
        ``'numba'``, ``'cpp'``, and ``'cupy'`` implementations. C++ and CuPy
        remain unsupported for the new potential and gradient derivatives.

    Returns
    -------
    FloatArray or ComplexArray
        Calculated G-free displacement-transfer result.
    """
    t_xyz = _normalize_targets(targets)

    supported_observables = ("potential", "acceleration", "gradient")
    if observable not in supported_observables:
        raise ValueError(
            "Unsupported observable: must be one of "
            f"{list(supported_observables)}, got {observable}"
        )

    supported_backends = ("numpy", "numba", "cpp", "cupy")
    if backend not in supported_backends:
        raise ValueError(
            "Unsupported backend: must be 'numpy', 'numba', 'cpp', or 'cupy', "
            f"got '{backend}'"
        )

    if observable != "acceleration":
        if backend == "numba":
            # Validate before the optional dependency import so malformed
            # inputs retain the shared contract even when Numba is absent.
            from gragra.kernels.inverse_square import _validate_dipole_inputs

            _validate_dipole_inputs(
                sources_xyz,
                mass_kg,
                displacement_m,
                t_xyz,
                softening_m,
                chunk_size,
            )
            numba_kernels = _import_numba_kernels()
            contract_fn = {
                "potential": numba_kernels.displacement_potential_contract,
                "gradient": numba_kernels.displacement_gradient_contract,
            }[observable]
            return contract_fn(
                sources_xyz,
                mass_kg,
                displacement_m,
                t_xyz,
                softening_m=softening_m,
                chunk_size=chunk_size,
            )
        if backend != "numpy":
            raise NotImplementedError(
                f"backend='{backend}' does not implement displacement-transfer "
                f"{observable}; use backend='numpy' or 'numba'"
            )
        contract_fn = {
            "potential": displacement_potential_contract,
            "gradient": displacement_gradient_contract,
        }[observable]
        return contract_fn(
            sources_xyz,
            mass_kg,
            displacement_m,
            t_xyz,
            softening_m=softening_m,
            chunk_size=chunk_size,
        )

    if backend == "numpy":
        contract_fn = dipole_contract
    elif backend == "numba":
        contract_fn = _import_numba_kernels().dipole_contract
    elif backend == "cpp":
        contract_fn = _import_cpp_kernels().dipole_contract
    elif backend == "cupy":
        from gragra.kernels.inverse_square import _validate_dipole_inputs

        _validate_dipole_inputs(
            sources_xyz, mass_kg, displacement_m, t_xyz, softening_m, chunk_size
        )
        contract_fn = _import_cupy_kernels().dipole_contract

    return contract_fn(
        sources_xyz,
        mass_kg,
        displacement_m,
        t_xyz,
        softening_m=softening_m,
        chunk_size=chunk_size,
    )


def direct_point_dipole_sum(
    mass_kg: list | tuple | np.ndarray,
    displacement_m: list | tuple | np.ndarray,
    sources_xyz: list | tuple | np.ndarray,
    targets: TargetPoints | list | tuple | np.ndarray,
    *,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> FloatArray | ComplexArray:
    """Execute G-free vector-weight (dipole/total) integration.

    This compatibility entry point preserves the argument order, backend
    dispatch, and exception behavior that predate
    :func:`direct_point_displacement_sum`. It takes raw ``mass_kg`` and
    ``displacement_m`` arrays because the displacement vector is not a
    ``WeightedPointSource.weights_kg``-compatible quantity.

    Parameters
    ----------
    mass_kg : array-like
        Finite, real, non-negative background mass elements, shape ``(N,)``.
    displacement_m : array-like
        Real or complex displacement vectors, shape ``(N, 3)``.
    sources_xyz : array-like
        Source coordinates, shape ``(N, 3)``.
    targets : TargetPoints or array-like
        Target coordinates.
    softening_m : float, optional
        Plummer softening length in meters.
    chunk_size : int, optional
        Source-axis chunk size.
    backend : str, optional
        ``'numpy'``, ``'numba'``, ``'cpp'``, or ``'cupy'``.

    Returns
    -------
    FloatArray or ComplexArray
        G-free acceleration response with shape ``(M, 3)``.
    """
    return direct_point_displacement_sum(
        mass_kg,
        displacement_m,
        sources_xyz,
        targets,
        observable="acceleration",
        softening_m=softening_m,
        chunk_size=chunk_size,
        backend=backend,
    )
