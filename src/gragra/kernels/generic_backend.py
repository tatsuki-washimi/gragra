"""Internal NumPy contraction backend for generic source blocks.

The contractions in this module are G-free.  They accept source-major
parameterized weights and add their result to a caller-owned accumulator.
"""

from __future__ import annotations

from operator import index
from typing import Protocol, runtime_checkable

import numpy as np

from gragra.kernels.parametric import (
    _acceleration_impl_parametric,
    _check_chunk_collision_parametric,
    _gradient_impl_parametric,
)


@runtime_checkable
class ContractionBackend(Protocol):
    """Internal interface for parameterized G-free contractions."""

    def potential_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        """Add potential contributions into ``out``."""

    def acceleration_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        """Add acceleration contributions into ``out``."""

    def gradient_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        """Add gravity-gradient contributions into ``out``."""

    def displacement_potential_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        """Add the source-displacement potential derivative into ``out``."""

    def dipole_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        """Add the source-displacement acceleration derivative into ``out``."""

    def displacement_gradient_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        """Add the source-displacement gradient derivative into ``out``."""


@runtime_checkable
class BlockedContractionBackend(ContractionBackend, Protocol):
    """Optional kernel-tile contractions used by the blocked strategy."""

    def potential_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None: ...

    def acceleration_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None: ...

    def gradient_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None: ...

    def displacement_potential_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None: ...

    def dipole_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None: ...

    def displacement_gradient_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None: ...


def _as_real_coordinates(name: str, value: object) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric coordinate table") from exc

    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain real coordinates, not boolean values")
    if raw.dtype.kind == "c":
        raise TypeError(f"{name} must contain real coordinates, not complex values")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric coordinates")
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {raw.shape}")

    coordinates = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return coordinates


def _as_source_major_weights(value: object, source_count: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("weights_kg must be a numeric source-major array") from exc

    if raw.dtype.kind == "b":
        raise TypeError("weights_kg must be numeric, not boolean")
    if raw.dtype.kind not in "iufc":
        raise TypeError("weights_kg must be real or complex numeric values")
    if raw.ndim != 2 or raw.shape[0] != source_count:
        raise ValueError(
            "weights_kg must have shape (Bs, Bp) with Bs matching sources; "
            f"got {raw.shape} for {source_count} sources"
        )

    dtype = np.complex128 if raw.dtype.kind == "c" else np.float64
    weights = np.asarray(raw, dtype=dtype)
    if not np.isfinite(weights).all():
        raise ValueError("weights_kg must contain only finite values")
    return weights


def _as_background_mass(value: object, source_count: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("mass_kg must be a real numeric vector") from exc

    if raw.dtype.kind == "b":
        raise TypeError("mass_kg must be numeric, not boolean")
    if raw.dtype.kind == "c":
        raise TypeError("mass_kg must be real-valued")
    if raw.dtype.kind not in "iuf":
        raise TypeError("mass_kg must be real numeric values")
    if raw.ndim != 1 or raw.shape[0] != source_count:
        raise ValueError(
            f"mass_kg must have shape ({source_count},), got {raw.shape}"
        )

    mass = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(mass).all():
        raise ValueError("mass_kg must contain only finite values")
    if (mass < 0.0).any():
        raise ValueError("mass_kg must be non-negative")
    return mass


def _as_source_major_displacement(value: object, source_count: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "displacement_m must be a real or complex numeric array"
        ) from exc

    if raw.dtype.kind == "b":
        raise TypeError("displacement_m must be numeric, not boolean")
    if raw.dtype.kind not in "iufc":
        raise TypeError("displacement_m must be real or complex numeric values")
    if raw.ndim != 3 or raw.shape[0] != source_count or raw.shape[2] != 3:
        raise ValueError(
            "displacement_m must have shape (Bs, Bp, 3) with Bs matching "
            f"sources; got {raw.shape} for {source_count} sources"
        )

    dtype = np.complex128 if raw.dtype.kind == "c" else np.float64
    displacement = np.asarray(raw, dtype=dtype)
    if not np.isfinite(displacement).all():
        raise ValueError("displacement_m must contain only finite values")
    return displacement


def _as_softening(value: object) -> float:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("softening_m must be a finite real scalar") from exc
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise TypeError("softening_m must be a finite real scalar")
    softening = float(raw)
    if not np.isfinite(softening):
        raise ValueError("softening_m must be finite")
    if softening < 0.0:
        raise ValueError("softening_m must be non-negative")
    return softening


def _as_chunk_size(value: object, source_count: int) -> int:
    if value is None:
        return max(source_count, 1)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("chunk_size must be a positive integer")
    try:
        chunk_size = index(value)
    except TypeError as exc:
        raise ValueError("chunk_size must be a positive integer") from exc
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return chunk_size


def _validate_out(
    out: object,
    expected_shape: tuple[int, ...],
    result_driver: np.ndarray,
    *inputs: np.ndarray,
) -> np.ndarray:
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a writable numpy.ndarray")
    if out.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}, got {out.shape}")
    if out.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError("out must have dtype float64 or complex128")
    if np.iscomplexobj(result_driver) and out.dtype != np.dtype(np.complex128):
        raise TypeError("complex inputs require a complex128 out accumulator")
    if not out.flags.writeable:
        raise ValueError("out must be writable")
    if not np.isfinite(out).all():
        raise ValueError("out must contain only finite values")
    if any(np.shares_memory(out, array) for array in inputs):
        raise ValueError("out must not share memory with inputs")
    return out


def _validated_inputs(
    sources_m: object,
    targets_m: object,
    weights_kg: object,
    *,
    softening_m: object,
    chunk_size: object,
    out: object,
    tail_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, np.ndarray]:
    sources_input = np.asarray(sources_m)
    targets_input = np.asarray(targets_m)
    weights_input = np.asarray(weights_kg)
    sources = _as_real_coordinates("sources_m", sources_input)
    targets = _as_real_coordinates("targets_m", targets_input)
    weights = _as_source_major_weights(weights_input, len(sources))
    softening = _as_softening(softening_m)
    actual_chunk_size = _as_chunk_size(chunk_size, len(sources))
    output = _validate_out(
        out,
        (len(targets), weights.shape[1]) + tail_shape,
        weights,
        sources_input,
        targets_input,
        weights_input,
    )
    return sources, targets, weights, softening, actual_chunk_size, output


def _validated_displacement_inputs(
    sources_m: object,
    targets_m: object,
    mass_kg: object,
    displacement_m: object,
    *,
    softening_m: object,
    chunk_size: object,
    out: object,
    tail_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, np.ndarray]:
    sources_input = np.asarray(sources_m)
    targets_input = np.asarray(targets_m)
    mass_input = np.asarray(mass_kg)
    displacement_input = np.asarray(displacement_m)
    sources = _as_real_coordinates("sources_m", sources_input)
    targets = _as_real_coordinates("targets_m", targets_input)
    mass = _as_background_mass(mass_input, len(sources))
    displacement = _as_source_major_displacement(displacement_input, len(sources))
    softening = _as_softening(softening_m)
    actual_chunk_size = _as_chunk_size(chunk_size, len(sources))
    output = _validate_out(
        out,
        (len(targets), displacement.shape[1]) + tail_shape,
        displacement,
        sources_input,
        targets_input,
        mass_input,
        displacement_input,
    )
    return (
        sources,
        targets,
        mass,
        displacement,
        softening,
        actual_chunk_size,
        output,
    )


def _preflight_collisions(
    sources: np.ndarray,
    targets: np.ndarray,
    softening_m: float,
    chunk_size: int,
) -> None:
    """Check every source tile before any caller-owned output is changed."""
    if softening_m != 0.0:
        return
    for start in range(0, len(sources), chunk_size):
        end = min(start + chunk_size, len(sources))
        diff = sources[start:end][np.newaxis, :, :] - targets[:, np.newaxis, :]
        r2 = np.sum(diff**2, axis=-1)
        _check_chunk_collision_parametric(np, r2, start)


def _displacement_potential_impl(
    sources: np.ndarray,
    targets: np.ndarray,
    mass: np.ndarray,
    displacement: np.ndarray,
    eps: float,
    c_size: int,
    out: np.ndarray,
) -> None:
    eps2 = eps**2
    for start in range(0, len(sources), c_size):
        end = min(start + c_size, len(sources))
        diff = sources[start:end][np.newaxis, :, :] - targets[:, np.newaxis, :]
        r2 = np.sum(diff**2, axis=-1)
        q = r2 + eps2
        z = np.einsum("mci,cpi->mcp", diff, displacement[start:end])
        contribution = z / (q[:, :, np.newaxis] * np.sqrt(q[:, :, np.newaxis]))
        out += np.sum(
            contribution * mass[np.newaxis, start:end, np.newaxis], axis=1
        )


def _scalar_kernel_tile(
    sources: np.ndarray,
    targets: np.ndarray,
    observable: str,
    softening: float,
) -> np.ndarray:
    """Build one bounded scalar kernel tile for reuse across parameter weights."""
    diff = sources[np.newaxis, :, :] - targets[:, np.newaxis, :]
    q = np.sum(diff * diff, axis=-1) + softening**2
    root_q = np.sqrt(q)
    if observable == "potential":
        return -1.0 / root_q
    q3 = q * root_q
    if observable == "acceleration":
        return diff / q3[:, :, np.newaxis]
    if observable == "gradient":
        q5 = q3 * q
        outer = diff[:, :, :, np.newaxis] * diff[:, :, np.newaxis, :]
        return 3.0 * outer / q5[:, :, np.newaxis, np.newaxis] - np.eye(
            3, dtype=np.float64
        )[np.newaxis, np.newaxis, :, :] / q3[:, :, np.newaxis, np.newaxis]
    raise ValueError(f"unsupported scalar observable: {observable}")


def _displacement_kernel_tile(
    sources: np.ndarray,
    targets: np.ndarray,
    observable: str,
    softening: float,
) -> np.ndarray:
    """Build a bounded source-derivative kernel tile for displacement reuse."""
    diff = sources[np.newaxis, :, :] - targets[:, np.newaxis, :]
    q = np.sum(diff * diff, axis=-1) + softening**2
    root_q = np.sqrt(q)
    q3 = q * root_q
    if observable == "potential":
        return diff / q3[:, :, np.newaxis]

    q5 = q3 * q
    outer = diff[:, :, :, np.newaxis] * diff[:, :, np.newaxis, :]
    if observable == "acceleration":
        return np.eye(3, dtype=np.float64)[np.newaxis, np.newaxis, :, :] / q3[
            :, :, np.newaxis, np.newaxis
        ] - 3.0 * outer / q5[:, :, np.newaxis, np.newaxis]
    if observable != "gradient":
        raise ValueError(f"unsupported displacement observable: {observable}")

    q7 = q5 * q
    inverse_q5 = 3.0 / q5
    inverse_q7 = 15.0 / q7
    kernel = np.empty((*diff.shape[:2], 3, 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            for source_axis in range(3):
                symmetric = (
                    diff[:, :, column] if row == source_axis else 0.0
                ) + (diff[:, :, row] if column == source_axis else 0.0) + (
                    diff[:, :, source_axis] if row == column else 0.0
                )
                kernel[:, :, row, column, source_axis] = (
                    inverse_q5 * symmetric
                    - inverse_q7
                    * diff[:, :, row]
                    * diff[:, :, column]
                    * diff[:, :, source_axis]
                )
    return kernel


def _blocked_scalar_contract(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    observable: str,
    softening: float,
    chunk_size: int,
    out: np.ndarray,
) -> None:
    equations = {
        "potential": "ts,sp->tp",
        "acceleration": "tsi,sp->tpi",
        "gradient": "tsij,sp->tpij",
    }
    for start in range(0, len(sources), chunk_size):
        end = min(start + chunk_size, len(sources))
        kernel = _scalar_kernel_tile(
            sources[start:end], targets, observable, softening
        )
        out += np.einsum(
            equations[observable], kernel, weights[start:end], optimize=False
        )


def _blocked_displacement_contract(
    sources: np.ndarray,
    targets: np.ndarray,
    mass: np.ndarray,
    displacement: np.ndarray,
    observable: str,
    softening: float,
    chunk_size: int,
    out: np.ndarray,
) -> None:
    equations = {
        "potential": "tsj,spj->tp",
        "acceleration": "tsij,spj->tpi",
        "gradient": "tsijk,spk->tpij",
    }
    for start in range(0, len(sources), chunk_size):
        end = min(start + chunk_size, len(sources))
        kernel = _displacement_kernel_tile(
            sources[start:end], targets, observable, softening
        )
        weighted_displacement = (
            mass[start:end, np.newaxis, np.newaxis]
            * displacement[start:end]
        )
        out += np.einsum(
            equations[observable],
            kernel,
            weighted_displacement,
            optimize=False,
        )


def _dipole_impl_parameterized(
    sources: np.ndarray,
    targets: np.ndarray,
    mass: np.ndarray,
    displacement: np.ndarray,
    eps: float,
    c_size: int,
    out: np.ndarray,
) -> None:
    eps2 = eps**2
    for start in range(0, len(sources), c_size):
        end = min(start + c_size, len(sources))
        d_chunk = displacement[start:end]
        diff = sources[start:end][np.newaxis, :, :] - targets[:, np.newaxis, :]
        r2 = np.sum(diff**2, axis=-1)
        q = r2 + eps2
        sqrt_q = np.sqrt(q)
        inverse_q3 = 1.0 / (q * sqrt_q)
        inverse_q5 = inverse_q3 / q
        z = np.einsum("mci,cpi->mcp", diff, d_chunk)
        first = d_chunk[np.newaxis, :, :, :] * inverse_q3[:, :, np.newaxis, np.newaxis]
        second = (
            3.0
            * diff[:, :, np.newaxis, :]
            * z[:, :, :, np.newaxis]
            * inverse_q5[:, :, np.newaxis, np.newaxis]
        )
        contribution = first - second
        out += np.sum(
            contribution * mass[np.newaxis, start:end, np.newaxis, np.newaxis],
            axis=1,
        )


def _displacement_gradient_impl(
    sources: np.ndarray,
    targets: np.ndarray,
    mass: np.ndarray,
    displacement: np.ndarray,
    eps: float,
    c_size: int,
    out: np.ndarray,
) -> None:
    eps2 = eps**2
    eye3 = np.eye(3)
    for start in range(0, len(sources), c_size):
        end = min(start + c_size, len(sources))
        d_chunk = displacement[start:end]
        diff = sources[start:end][np.newaxis, :, :] - targets[:, np.newaxis, :]
        r2 = np.sum(diff**2, axis=-1)
        q = r2 + eps2
        q5_2 = q**2 * np.sqrt(q)
        q7_2 = q5_2 * q
        z = np.einsum("mci,cpi->mcp", diff, d_chunk)
        r_col = diff[:, :, np.newaxis, :, np.newaxis]
        r_row = diff[:, :, np.newaxis, np.newaxis, :]
        d_col = d_chunk[np.newaxis, :, :, :, np.newaxis]
        d_row = d_chunk[np.newaxis, :, :, np.newaxis, :]
        z_identity = (
            z[:, :, :, np.newaxis, np.newaxis]
            * eye3[np.newaxis, np.newaxis, np.newaxis, :, :]
        )
        symmetric = d_col * r_row + r_col * d_row + z_identity
        rr_z = r_col * r_row * z[:, :, :, np.newaxis, np.newaxis]
        contribution = (
            3.0 * symmetric / q5_2[:, :, np.newaxis, np.newaxis, np.newaxis]
            - 15.0 * rr_z / q7_2[:, :, np.newaxis, np.newaxis, np.newaxis]
        )
        out += np.sum(
            contribution
            * mass[np.newaxis, start:end, np.newaxis, np.newaxis, np.newaxis],
            axis=1,
        )


class NumpyContractionBackend:
    """NumPy implementation of the internal scalar contraction protocol."""

    def potential_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        sources, targets, weights, eps, c_size, accumulator = _validated_inputs(
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        eps2 = eps**2

        for start in range(0, len(sources), c_size):
            end = min(start + c_size, len(sources))
            diff = sources[start:end][np.newaxis, :, :] - targets[:, np.newaxis, :]
            r2 = np.sum(diff**2, axis=-1)
            kernel = -1.0 / np.sqrt(r2 + eps2)
            accumulator += np.sum(
                kernel[:, :, np.newaxis] * weights[np.newaxis, start:end, :],
                axis=1,
            )

    def acceleration_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        sources, targets, weights, eps, c_size, accumulator = _validated_inputs(
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(3,),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _acceleration_impl_parametric(
            np,
            sources,
            weights,
            targets,
            eps,
            c_size,
            (weights.shape[1],),
            accumulator,
        )

    def gradient_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        sources, targets, weights, eps, c_size, accumulator = _validated_inputs(
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(3, 3),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _gradient_impl_parametric(
            np,
            sources,
            weights,
            targets,
            eps,
            c_size,
            (weights.shape[1],),
            accumulator,
        )

    def displacement_potential_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        (
            sources,
            targets,
            mass,
            displacement,
            eps,
            c_size,
            accumulator,
        ) = _validated_displacement_inputs(
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _displacement_potential_impl(
            sources, targets, mass, displacement, eps, c_size, accumulator
        )

    def dipole_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        (
            sources,
            targets,
            mass,
            displacement,
            eps,
            c_size,
            accumulator,
        ) = _validated_displacement_inputs(
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(3,),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _dipole_impl_parameterized(
            sources, targets, mass, displacement, eps, c_size, accumulator
        )

    def displacement_gradient_contract(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        (
            sources,
            targets,
            mass,
            displacement,
            eps,
            c_size,
            accumulator,
        ) = _validated_displacement_inputs(
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(3, 3),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _displacement_gradient_impl(
            sources, targets, mass, displacement, eps, c_size, accumulator
        )

    def potential_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        sources, targets, weights, eps, c_size, accumulator = _validated_inputs(
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _blocked_scalar_contract(
            sources, targets, weights, "potential", eps, c_size, accumulator
        )

    def acceleration_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        sources, targets, weights, eps, c_size, accumulator = _validated_inputs(
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(3,),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _blocked_scalar_contract(
            sources, targets, weights, "acceleration", eps, c_size, accumulator
        )

    def gradient_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        weights_kg: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        sources, targets, weights, eps, c_size, accumulator = _validated_inputs(
            sources_m,
            targets_m,
            weights_kg,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(3, 3),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _blocked_scalar_contract(
            sources, targets, weights, "gradient", eps, c_size, accumulator
        )

    def displacement_potential_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        (
            sources,
            targets,
            mass,
            displacement,
            eps,
            c_size,
            accumulator,
        ) = _validated_displacement_inputs(
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _blocked_displacement_contract(
            sources,
            targets,
            mass,
            displacement,
            "potential",
            eps,
            c_size,
            accumulator,
        )

    def dipole_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        (
            sources,
            targets,
            mass,
            displacement,
            eps,
            c_size,
            accumulator,
        ) = _validated_displacement_inputs(
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(3,),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _blocked_displacement_contract(
            sources,
            targets,
            mass,
            displacement,
            "acceleration",
            eps,
            c_size,
            accumulator,
        )

    def displacement_gradient_contract_blocked(
        self,
        sources_m: np.ndarray,
        targets_m: np.ndarray,
        mass_kg: np.ndarray,
        displacement_m: np.ndarray,
        *,
        softening_m: float = 0.0,
        chunk_size: int | None = None,
        out: np.ndarray,
    ) -> None:
        (
            sources,
            targets,
            mass,
            displacement,
            eps,
            c_size,
            accumulator,
        ) = _validated_displacement_inputs(
            sources_m,
            targets_m,
            mass_kg,
            displacement_m,
            softening_m=softening_m,
            chunk_size=chunk_size,
            out=out,
            tail_shape=(3, 3),
        )
        _preflight_collisions(sources, targets, eps, c_size)
        _blocked_displacement_contract(
            sources,
            targets,
            mass,
            displacement,
            "gradient",
            eps,
            c_size,
            accumulator,
        )
