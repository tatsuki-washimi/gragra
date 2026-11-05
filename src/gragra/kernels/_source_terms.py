"""Bounded NumPy per-source terms from the existing point kernels."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from gragra.kernels.inverse_square import (
    _acceleration_pairwise_terms,
    _check_chunk_collision,
    _gradient_pairwise_terms,
    _potential_pairwise_terms,
    _validate_inputs,
    _validate_softening,
)

_ALIASES = {
    "potential": "potential",
    "acceleration": "acceleration",
    "gradient": "gradient",
    "gravity_gradient": "gradient",
    "GG": "gradient",
    "gg": "gradient",
}


def _canonical_observable(observable: str) -> str:
    if not isinstance(observable, str):
        raise TypeError("observable must be a string")
    try:
        return _ALIASES[observable]
    except KeyError as error:
        raise ValueError(
            "observable must be 'potential', 'acceleration', or "
            "'gravity_gradient' ('gradient' and 'GG' are aliases)"
        ) from error


def iter_source_kernel_terms(
    sources_xyz,
    weights,
    targets_xyz,
    *,
    observable: str,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield read-only G-free source terms in bounded source blocks.

    Each yielded array has shape ``(M, B, components...)`` and applies the
    real or complex source weight to the same Plummer-softened point kernel
    used by the established contraction. The source axis is kept in input
    order. Softening must be finite and non-negative; zero-softening
    collisions raise the established kernel ``ValueError``.
    """
    canonical = _canonical_observable(observable)
    eps = _validate_softening(softening_m)
    sources, normalized_weights, targets, eps, c_size = _validate_inputs(
        sources_xyz, weights, targets_xyz, eps, chunk_size
    )
    yield from _normalized_source_term_blocks(
        sources, normalized_weights, targets, canonical, eps, c_size
    )


def _normalized_source_term_blocks(
    sources: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    canonical: str,
    eps: float,
    chunk_size: int,
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield terms from already normalized, read-only input arrays."""
    n_sources = len(sources)
    for start in range(0, n_sources, chunk_size):
        stop = min(start + chunk_size, n_sources)
        source_block = sources[start:stop]
        weight_block = weights[start:stop]
        diff = source_block[np.newaxis, :, :] - targets[:, np.newaxis, :]
        with np.errstate(over="ignore", invalid="ignore"):
            r2 = np.sum(diff**2, axis=-1)
        if eps == 0.0:
            _check_chunk_collision(np, r2, start)
        if canonical == "potential":
            terms = _potential_pairwise_terms(np, r2, eps)
        elif canonical == "acceleration":
            terms = _acceleration_pairwise_terms(np, diff, r2, eps)
        else:
            terms = _gradient_pairwise_terms(np, diff, r2, eps)
        weight_shape = (1, stop - start) + (1,) * (terms.ndim - 2)
        values = np.asarray(terms * weight_block.reshape(weight_shape))
        values.setflags(write=False)
        yield slice(start, stop), values


def source_kernel_terms(
    sources_xyz,
    weights,
    targets_xyz,
    *,
    observable: str,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Materialize G-free source terms as ``(M, N, components...)``.

    The result scales with all targets and sources; use
    :func:`iter_source_kernel_terms` when that allocation is too large.
    """
    canonical = _canonical_observable(observable)
    sources, normalized_weights, targets, eps, c_size = _validate_inputs(
        sources_xyz,
        weights,
        targets_xyz,
        _validate_softening(softening_m),
        chunk_size,
    )
    if canonical == "potential":
        component_shape: tuple[int, ...] = ()
    elif canonical == "acceleration":
        component_shape = (3,)
    else:
        component_shape = (3, 3)
    dtype = np.complex128 if np.iscomplexobj(normalized_weights) else np.float64
    result = np.empty((len(targets), len(sources), *component_shape), dtype=dtype)
    if len(sources) == 0:
        result.setflags(write=False)
        return result

    for block_slice, values in _normalized_source_term_blocks(
        sources, normalized_weights, targets, canonical, eps, c_size
    ):
        result[:, block_slice, ...] = values
    result.setflags(write=False)
    return result
