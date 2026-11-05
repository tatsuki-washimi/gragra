"""G-once observable wrappers for bounded per-source point terms."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from gragra.constants import G_SI
from gragra.kernels._source_terms import (
    iter_source_kernel_terms,
    source_kernel_terms,
)


def iter_point_observable_source_terms(
    sources_xyz,
    weights_kg,
    targets_xyz,
    *,
    observable: str,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield source-resolved potential, acceleration, or gravity-gradient terms.

    The underlying point kernel remains G-free. This observable wrapper
    multiplies each yielded source term by ``G_SI`` exactly once and preserves
    the source axis and complex dtype.
    """
    for source_slice, values in iter_source_kernel_terms(
        sources_xyz,
        weights_kg,
        targets_xyz,
        observable=observable,
        softening_m=softening_m,
        chunk_size=chunk_size,
    ):
        observable_values = np.asarray(G_SI * values)
        observable_values.setflags(write=False)
        yield source_slice, observable_values


def point_observable_source_terms(
    sources_xyz,
    weights_kg,
    targets_xyz,
    *,
    observable: str,
    softening_m: float = 0.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Materialize per-source observable terms in ``(M,N,components...)`` order."""
    values = source_kernel_terms(
        sources_xyz,
        weights_kg,
        targets_xyz,
        observable=observable,
        softening_m=softening_m,
        chunk_size=chunk_size,
    )
    result = np.asarray(G_SI * values)
    result.setflags(write=False)
    return result
